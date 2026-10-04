"""Live routing check: every calibration sample once through real Jev, once through keyword fallback.

Run: .venv/bin/python -m scripts.verify_routing   (after scripts.calibrate_routing)
Writes docs/routing-verification.json. Calls the real Jev API.
"""
import asyncio
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import ROOT, Settings
from app.routing import Router

CAL = json.loads((ROOT / "docs/jev-routing-calibration.json").read_text())
# Samples that need both specialists, and single-domain samples where Jev relevance runs high.
NEED_BOTH = {"C01", "X25", "X26", "X27", "X33"}
NO_SUPPORT = {"A03", "A01", "A04", "B02", "X14", "X20", "X28", "F04"}


def outcome(decision):
    if decision.escalated:
        return "escalation"
    return "clarify" if decision.needs_clarification else decision.primary_agent


def judge(sample, decision):
    got = outcome(decision)
    ok = got in sample["accept"] or (got == "clarify" and "clarify" in sample["accept"])
    if sample["expect_escalation"] is True:
        ok = decision.escalated
    elif sample["expect_escalation"] is False and decision.escalated:
        ok = False
    roles = {decision.primary_agent, *decision.supporting_agents}
    if sample["id"] in NEED_BOTH and not decision.escalated:
        ok = ok and {"technical", "billing"} <= roles
    if sample["id"] in NO_SUPPORT:
        ok = ok and not decision.supporting_agents
    return ok, got


async def main():
    settings = Settings.from_env()
    semaphore = asyncio.Semaphore(4)
    async with httpx.AsyncClient() as client:
        live = Router(client, settings.jev_api_key, settings.jev_model, settings.jev_endpoint, 20.0)
        keyword = Router(client)

        async def one(sample):
            # Recorded production context: the real /chat snapshot for multi-turn samples.
            history, summary = sample["history"], sample["session_summary"]
            async with semaphore:
                jev_decision = await live.route(sample["message"], history, summary)
            kw_decision = await keyword.route(sample["message"], history, summary)
            row = {"id": sample["id"], "message": sample["message"], "accept": sample["accept"],
                   "expect_escalation": sample["expect_escalation"], "context_source": sample["context_source"]}
            for mode, decision in (("jev", jev_decision), ("keyword", kw_decision)):
                ok, got = judge(sample, decision)
                row[mode] = {"ok": ok, "outcome": got, "primary": decision.primary_agent,
                             "supporting": decision.supporting_agents, "reason": decision.routing_reason,
                             "source": decision.primary_source, "degraded": decision.degraded_reason,
                             "urgent": decision.urgent}
            row["jev"].update(choice=jev_decision.jev["choice"],
                              choice_probability=jev_decision.jev["choice_probabilities"].get(jev_decision.jev["choice"]),
                              relevance=jev_decision.jev["domain_probabilities"],
                              urgent_probability=jev_decision.jev["urgent_probability"],
                              latency_ms=jev_decision.jev["latency_ms"])
            return row

        rows = await asyncio.gather(*(one(s) for s in CAL["samples"]))
    summary = {}
    for mode in ("jev", "keyword"):
        stat = collections.Counter("ok" if r[mode]["ok"] else "wrong" for r in rows)
        esc = [r for r in rows if r["expect_escalation"] is True]
        summary[mode] = {
            "correct": stat["ok"], "total": len(rows),
            "escalation_recall": f"{sum(r[mode]['outcome'] == 'escalation' for r in esc)}/{len(esc)}",
            "false_escalation": sum(r[mode]["outcome"] == "escalation" for r in rows if r["expect_escalation"] is False),
            "multi_turn_correct": f"{sum(r[mode]['ok'] for r in rows if r['context_source'] == 'live_chat')}/"
                                  f"{sum(r['context_source'] == 'live_chat' for r in rows)}",
            "clarify": sum(r[mode]["outcome"] == "clarify" for r in rows),
            "degraded": sum(bool(r[mode]["degraded"]) for r in rows) if mode == "jev" else None,
        }
        print(mode, summary[mode])
        for r in rows:
            if not r[mode]["ok"]:
                print(f"   ✗ {r['id']} {r['message'][:24]} -> {r[mode]['outcome']} sup={r[mode]['supporting']} "
                      f"(accept {r['accept']}{' + escalation' if r['expect_escalation'] is True else ''})")
    latencies = sorted(r["jev"]["latency_ms"] for r in rows)
    summary["jev_latency_ms"] = {"p50": latencies[len(latencies) // 2], "p95": latencies[int(len(latencies) * 0.95)]}
    print("latency", summary["jev_latency_ms"])
    (ROOT / "docs/routing-verification.json").write_text(json.dumps({
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "live_routing_check",
        "scope": "Same samples and recorded /chat context as jev-routing-calibration.json; Jev mode with live Jev, keyword mode offline. "
                 "Thresholds were set on these samples, so this is a consistency check, not a held-out accuracy estimate.",
        "requested_model": settings.jev_model, "summary": summary, "rows": rows,
    }, ensure_ascii=False, indent=2))


asyncio.run(main())
