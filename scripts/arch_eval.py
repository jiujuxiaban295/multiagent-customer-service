"""Answer-quality evaluation of this LangChain v3 on EchoMind's 30-case set, scored exactly like EchoMind.

generate  Every case runs through the real /chat in-process (isolated SQLite, own business scope, retrieval
          cache off like EchoMind P9); multi-turn cases are sent turn by turn in one session. Records use
          EchoMind's generations.jsonl schema with mode "v3lc".
judge     EchoMind evaluation/arch_eval.py judge (same prompt, parser, local judge model). Only the client
          adapter differs: anthropic 1.x takes sampling parameters through extra_body.
report    EchoMind's report; --baseline copies an earlier EchoMind run (v1/v2/v3) in for a four-column table.

Run (unset a host ANTHROPIC_BASE_URL so .env's DeepSeek address is used):
  env -u ANTHROPIC_BASE_URL .venv/bin/python -m scripts.arch_eval --out data/eval/runs/<name>
Calls real Jev, DeepSeek, the local embedding service and the local judge (EVAL_JUDGE_BASE_URL).
"""
import argparse
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import anthropic
import httpx

from app.config import ROOT, Settings
from app.main import create_app

ECHOMIND = Path("/Users/yiyang/Downloads/EchoMind所有代码+详细文档+简历/EchoMind")
MODE = "v3lc"


def load_echomind_eval(echomind: Path):
    sys.path.insert(0, str(echomind))
    spec = importlib.util.spec_from_file_location("echomind_arch_eval", echomind / "evaluation/arch_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODES = (*module.MODES, MODE)
    module.MODE_NAMES[MODE] = "V3 LangChain 重构"
    module.make_client = make_client
    return module


def make_client(api_key, base_url=None):
    """Same calls as EchoMind's make_client (thinking disabled); sampling params moved into extra_body."""
    client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=0, timeout=600)
    create = client.messages.create

    async def compatible(*args, **kwargs):
        extra = dict(kwargs.pop("extra_body", None) or {})
        for key in ("temperature", "top_p", "top_k"):
            if key in kwargs:
                extra[key] = kwargs.pop(key)
        extra.setdefault("thinking", {"type": "disabled"})
        return await create(*args, extra_body=extra, **kwargs)

    client.messages.create = compatible
    return client


def copy_baseline(ev, baseline: Path, out_dir: Path) -> None:
    generations = out_dir / "generations.jsonl"
    present = {mode for _, mode in ev.latest_by_key(ev.read_jsonl(generations))}
    if present & {"v1", "v2", "v3"}:
        return
    for name in ("generations.jsonl", "judgments.jsonl"):
        for record in ev.read_jsonl(baseline / name):
            ev.append_jsonl(out_dir / name, record)
    ev.update_meta(out_dir, {"baseline_run": str(baseline), "baseline_meta": ev.load_meta(baseline)})


def turn_record(message: str, data: dict | None, error: str = "") -> dict:
    if data is None:
        return {"message": message, "response": "", "success": False, "error": error, "escalated": False,
                "tools_used": [], "tool_traces": [], "rag_calls": [], "retrieved_titles": [],
                "latency_ms": 0.0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "extra": {}}
    titles = list(dict.fromkeys(item.get("title") for item in data.get("retrieved", []) if item.get("title")))
    rag_calls = [{"query": (trace.get("input") or {}).get("query"), "success": bool(trace.get("success")), "titles": titles}
                 for trace in data.get("tool_traces", []) if trace.get("tool_name") == "search_knowledge_base"]
    return {"message": message, "response": data["response"], "success": data["success"], "error": error,
            "escalated": data["escalated"], "tools_used": data["tools_used"], "tool_traces": data["tool_traces"],
            "rag_calls": rag_calls, "retrieved_titles": titles, "latency_ms": data["latency_ms"],
            "llm_calls": data["llm_calls"], "input_tokens": data["input_tokens"], "output_tokens": data["output_tokens"],
            "extra": data.get("extra", {})}


async def generate(ev, cases, out_dir: Path, concurrency: int) -> None:
    path = out_dir / "generations.jsonl"
    done = {case_id for case_id, mode in ev.latest_by_key(ev.read_jsonl(path)) if mode == MODE}
    jobs = [case for case in cases if case["id"] not in done]
    if not jobs:
        print("生成：没有待跑的题", flush=True)
        return
    settings = Settings.from_env()
    settings.business_scope = f"eval-{out_dir.name}-{uuid4().hex[:6]}"
    settings.rag_cache_ttl = 0
    with tempfile.TemporaryDirectory(prefix="arch-eval-") as directory:
        settings.sqlite_path = str(Path(directory) / "records.sqlite3")
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services = app.state.services
            kb_chunks = await asyncio.to_thread(services["index"].knowledge.count)
            ev.update_meta(out_dir, {
                "model": settings.model, "base_url": settings.model_base_url,
                "kb_collection": settings.knowledge_collection, "kb_chunks": kb_chunks,
                "embedding_model": settings.embedding_model, "rag_recall_k": settings.rag_recall_k,
                "concurrency": concurrency, "generate_started_at": datetime.now().isoformat(timespec="seconds"),
                "v3lc_system": "LangChain v3 (多agent编排客服), Jev routing, retrieval cache off",
            })
            print(f"生成：{len(jobs)} 题，模型 {settings.model}，知识库 {settings.knowledge_collection}（{kb_chunks} 个片段）", flush=True)
            semaphore, lock, finished, created = asyncio.Semaphore(concurrency), asyncio.Lock(), 0, []
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://eval", timeout=600) as api:

                    async def run_case(case):
                        nonlocal finished
                        owner = f"eval-{case['id'].lower()}-{MODE}"
                        async with semaphore:
                            started = time.monotonic()
                            response = await api.post("/sessions", json={"user_id": owner},
                                                      headers={"X-Dev-Key": settings.dev_api_key})
                            response.raise_for_status()
                            session = response.json()
                            created.append((session["conv_id"], owner))
                            auth = {"Authorization": "Bearer " + session["session_token"]}
                            turns = []
                            for message in case["turns"]:
                                try:
                                    response = await api.post("/chat", json={"message": message}, headers=auth)
                                    response.raise_for_status()
                                    turns.append(turn_record(message, response.json()))
                                except Exception as exc:   # a failed turn is recorded, later turns still run
                                    turns.append(turn_record(message, None, f"{type(exc).__name__}: {exc}"))
                        record = {"case_id": case["id"], "mode": MODE, "category": case["category"], "user_id": owner,
                                  "conv_id": session["conv_id"], "turns": turns,
                                  "finished_at": datetime.now().isoformat(timespec="seconds")}
                        async with lock:
                            ev.append_jsonl(path, record)
                            finished += 1
                            failed = sum(not turn["success"] for turn in turns)
                            print(f"[{finished}/{len(jobs)}] {case['id']} {time.monotonic() - started:.1f}s"
                                  + (f" 失败 {failed} 轮" if failed else ""), flush=True)

                    await asyncio.gather(*(run_case(case) for case in jobs))
            finally:
                for conv_id, owner in created:
                    await services["redis"].delete(services["store"]._key(conv_id, owner, settings.business_scope))


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain v3 on EchoMind's architecture evaluation set")
    parser.add_argument("--out", help="结果目录；续跑时传同一个目录")
    parser.add_argument("--stage", choices=("all", "generate", "judge", "report"), default="all")
    parser.add_argument("--echomind", type=Path, default=ECHOMIND)
    parser.add_argument("--baseline", type=Path, default=ECHOMIND / "data/eval/runs/full-20261004",
                        help="EchoMind 已完成的 v1/v2/v3 结果目录；传空字符串则不对比")
    parser.add_argument("--cases", help="只跑这些题，逗号分隔")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--judge-concurrency", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("EVAL_JUDGE_BASE_URL", "http://127.0.0.1:1234")
    ev = load_echomind_eval(args.echomind)
    out_dir = Path(args.out or ROOT / "data/eval/runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.echomind / "data/eval/arch_cases.json"
    cases = ev.load_cases(cases_path, args.cases.split(",") if args.cases else None)
    if str(args.baseline) not in ("", "."):
        copy_baseline(ev, args.baseline, out_dir)
    if args.stage in ("all", "generate"):
        asyncio.run(generate(ev, cases, out_dir, args.concurrency))
    if args.stage in ("all", "judge"):
        asyncio.run(ev.judge(cases, [MODE], out_dir, args.judge_concurrency))
    if args.stage in ("all", "report"):
        ev.report(cases_path, cases, list(ev.MODES), out_dir)


if __name__ == "__main__":
    main()
