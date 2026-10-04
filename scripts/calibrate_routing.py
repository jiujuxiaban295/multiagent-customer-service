"""Collect live Jev answers for routing calibration.

Single-turn samples: one Jev request with empty history, the same context as a first turn in production.
Multi-turn samples: earlier turns run through the real /chat (Jev, DeepSeek, knowledge base and session
store, with an isolated SQLite file and business scope); the last turn is then routed with the stored
snapshot, so Jev receives exactly the recent_messages / session_summary production would send.

Run: .venv/bin/python -m scripts.calibrate_routing [--eval PATH]
Calls real Jev, DeepSeek and the local embedding service. Writes docs/jev-routing-calibration.json.
"""
import argparse
import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from app.config import ROOT, Settings
from app.main import create_app
from app.routing import _questions

# Original EchoMind 30-case set; override with --eval.
DEFAULT_EVAL = Path("/Users/yiyang/Downloads/EchoMind所有代码+详细文档+简历/EchoMind/data/eval/arch_cases.json")
INTENT_DOMAIN = {
    "technical": "technical", "technical_login": "technical", "technical_crash": "technical",
    "billing": "billing", "refund": "billing", "invoice": "billing", "payment_issue": "billing",
    "human_handoff": "escalation", "escalation": "escalation", "account_security": "escalation",
}
H401 = ["App 登录报 401"]
# (id, message, earlier user turns, acceptable outcomes, expect escalation: True / False / None = either)
EXTRA = [
    ("X01", "我要投诉，你们服务太差了", [], {"escalation"}, True),
    ("X02", "找你们经理", [], {"escalation"}, True),
    ("X03", "能不能帮我转人工", [], {"escalation"}, True),
    ("X04", "紧急！支付失败了", [], {"escalation"}, True),
    ("X05", "我的账号被盗了", [], {"escalation"}, True),
    ("X06", "怎么投诉商家？", [], {"general"}, False),
    ("X07", "我不是要投诉，只想问发票", [], {"billing"}, False),
    ("X08", "不用转人工，帮我查退款", [], {"billing"}, False),
    ("X09", "页面显示“联系人工客服”，但我的问题是退款", [], {"billing"}, False),
    ("X10", "怎么防止充电宝起火", [], {"general"}, False),
    ("X11", "如何防止账号被盗", [], {"general", "technical"}, False),
    ("X12", "不紧急，退款慢慢处理就好", [], {"billing"}, False),
    ("X13", "能立刻发货吗", [], {"general"}, False),
    ("X14", "我想立刻取消订单", [], {"general", "billing"}, False),
    ("X15", "你好", [], {"clarify", "general"}, False),
    ("X16", "你好，我想退款", [], {"billing"}, False),
    ("X17", "你好，App 登录报错了", [], {"technical"}, False),
    ("X18", "退换货政策里退款多久到账", [], {"billing", "general"}, False),
    ("X19", "会员积分怎么用", [], {"general"}, False),
    ("X20", "怎么换绑手机号", [], {"general"}, False),
    ("X21", "我想注销账户", [], {"general"}, False),
    ("X22", "登录后怎么查积分", [], {"general"}, False),
    ("X23", "我这边页面一直不对劲", [], {"technical", "clarify"}, False),
    ("X24", "页面一直转圈打不开", [], {"technical"}, False),
    ("X25", "App 支付页面报错", [], {"technical", "billing"}, False),
    ("X26", "登录报错401，而且被重复扣款了", [], {"technical", "billing"}, False),
    ("X27", "App 登录失败，顺便问下订阅怎么取消", [], {"technical", "billing"}, False),
    ("X28", "订单物流不更新，退款还没到账", [], {"billing", "general"}, False),
    ("X29", "我买的衣服尺码不对怎么办", H401, {"general"}, False),
    ("X30", "还是不行", H401 + ["安卓 14，昨天开始的"], {"technical"}, False),
    ("X31", "帮我看看", [], {"clarify"}, False),
    ("X32", "那个东西有点奇怪", [], {"clarify"}, False),
    ("X33", "订单 A10050 重复扣款 500 元，而且 App 登录报 401", [], {"technical", "billing"}, False),
    ("X34", "有人用我的银行卡在你们平台盗刷了 3000 块", [], {"escalation"}, True),
    ("X35", "我只是问问，你们的人工客服几点上班？", [], {"general"}, False),
    ("X36", "紧急！订单 A10050 还没发货", [], {"escalation"}, True),
    ("X37", "不急，物流慢点也没关系，就想问下大概几天到", [], {"general"}, False),
    ("X38", "Urgent: my payment failed twice", [], {"escalation"}, True),
    ("X39", "怎么防止被骗", [], {"general"}, False),
    ("X40", "我要找律师起诉你们", [], {"escalation"}, True),
    # Multi-turn probes: earlier answers often mention 人工客服; a resolved or switched topic must not carry over.
    ("M01", "那大概几天能退回来？", ["我的订单 A1005 重复扣款了"], {"billing"}, False),
    ("M02", "算了不用人工了，我自己查下退款进度就行", ["帮我转人工"], {"billing"}, False),
    ("M03", "还是不行，给我转人工", ["App 登录一直报 401"], {"escalation"}, True),
    ("M04", "密码已经改好了，现在想问被盗刷的钱怎么退", ["我的账号被盗了"], {"billing", "escalation"}, None),
    ("M05", "现在支付成功了，发票怎么开", ["紧急！支付失败了"], {"billing"}, False),
    ("M06", "退回来的运费谁出？", ["我想退货", "订单是 A1012，衣服尺码不对"], {"billing", "general"}, False),
    ("M07", "已经解决了。另外想问下会员积分怎么用", H401, {"general"}, False),
    ("M08", "订单号是 A1002，什么时候能到？", ["我要投诉你们的快递太慢了"], {"general", "escalation"}, None),
]


def samples(eval_path):
    data = json.loads(eval_path.read_text())
    for case in data["cases"]:
        accept = {INTENT_DOMAIN.get(case["intent"], "general")} | {INTENT_DOMAIN.get(a, "general") for a in case["intent_alt"]}
        escalate = INTENT_DOMAIN.get(case["intent"]) == "escalation" or case["id"] in {"D03", "D04"}
        yield case["id"], case["turns"][-1], case["turns"][:-1], accept, escalate
    yield from EXTRA


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.business_scope = "routing-calibration-" + uuid4().hex[:8]
    out = ROOT / "docs/jev-routing-calibration.json"
    semaphore = asyncio.Semaphore(4)
    with tempfile.TemporaryDirectory(prefix="routing-calibration-") as directory:
        # Isolated session records; the real knowledge base keeps earlier answers realistic.
        settings.sqlite_path = str(Path(directory) / "records.sqlite3")
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            services = app.state.services
            store, pipeline = services["store"], services["pipeline"]
            roles = set(pipeline.agents) | {"escalation"}
            created = []
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url="http://calibration", timeout=300) as api:

                    async def one(sample):
                        sid, message, prior, accept, escalate = sample
                        history, summary, earlier = [], "", []
                        async with semaphore:
                            if prior:
                                owner = "calibration-" + sid.lower()
                                response = await api.post("/sessions", json={"user_id": owner},
                                                          headers={"X-Dev-Key": settings.dev_api_key})
                                response.raise_for_status()
                                session = response.json()
                                created.append((session["conv_id"], owner))
                                auth = {"Authorization": "Bearer " + session["session_token"]}
                                for text in prior:
                                    response = await api.post("/chat", json={"message": text}, headers=auth)
                                    response.raise_for_status()
                                    data = response.json()
                                    if not data["success"]:
                                        # A failure reply is not a realistic context; stop before writing the file.
                                        errors = {r.get("error") for r in data["extra"].get("agent_results", [])}
                                        raise RuntimeError(f"{sid}: earlier turn failed ({', '.join(filter(None, errors))}); "
                                                           "fix the generation model before calibrating multi-turn samples")
                                    earlier.append({"user": text, "assistant": data["response"],
                                                    "primary_agent": data["primary_agent"], "escalated": data["escalated"],
                                                    "llm_calls": data["llm_calls"]})
                                snapshot = await store.snapshot(session["conv_id"], owner, settings.business_scope)
                                history, summary = snapshot["history"], snapshot["summary"]
                            # Same call the pipeline makes for this turn.
                            decision = await pipeline.router.route(message, history, summary, available_roles=roles)
                        jev = decision.jev
                        print(f"{sid} {jev['choice']} p={jev['choice_probabilities'].get(jev['choice'])} "
                              f"noul={jev['domain_probabilities']} urgent={jev['urgent_probability']} "
                              f"{jev['latency_ms']:.0f}ms {decision.degraded_reason}", flush=True)
                        return {"id": sid, "message": message, "accept": sorted(accept), "expect_escalation": escalate,
                                "context_source": "live_chat" if prior else "first_turn",
                                "earlier_turns": earlier, "history": history, "session_summary": summary,
                                "jev_context": {
                                    "recent_messages": [{"role": item.get("role", ""), "content": str(item.get("content", ""))[:2000]}
                                                        for item in history[-6:]],
                                    "session_summary": summary[:4000]},
                                "jev_available": jev["available"], "degraded_reason": decision.degraded_reason,
                                "model": jev["model"], "latency_ms": jev["latency_ms"], "choice": jev["choice"],
                                "choice_probabilities": jev["choice_probabilities"],
                                "domain_probabilities": jev["domain_probabilities"],
                                "urgent_probability": jev["urgent_probability"], "usage": jev.get("usage")}

                    rows = await asyncio.gather(*(one(s) for s in samples(args.eval)))
            finally:
                for conv_id, owner in created:
                    await services["redis"].delete(store._key(conv_id, owner, settings.business_scope))
    out.write_text(json.dumps({
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "live_jev_routing_calibration",
        "scope": "One live Jev request per routed sample. First-turn samples use empty context; multi-turn samples "
                 "run earlier turns through the real /chat (Jev, DeepSeek, knowledge base, isolated session store) and "
                 "route the last turn with the stored snapshot. 30 cases from EchoMind arch_cases.json plus 48 "
                 "hand-written samples. Labels are routing expectations, not answer quality.",
        "requested_model": settings.jev_model,
        "questions": _questions(),
        "samples": rows,
    }, ensure_ascii=False, indent=2))
    print("saved", out, len(rows))


asyncio.run(main())
