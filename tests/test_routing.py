"""Offline tests use the official Jev response shape, not a fake live model."""

import copy
import json
import unittest
from types import SimpleNamespace

import httpx

from app.routing import DEFAULT_MODEL, Router


def official_response(intent="technical", domains=None, confidence=0.9, probabilities=None, urgent=None):
    if probabilities is None:
        probabilities = {domain: 0.025 for domain in ("general", "technical", "billing", "escalation", "unknown")}
        probabilities[intent] = 0.9
    relevance = dict.fromkeys(("general", "technical", "billing", "escalation"), 0.02)
    relevance[intent if intent != "unknown" else "general"] = 0.98
    if domains:
        relevance.update(domains)
    answers = {
        "intent": {"type": "choice", "choice": intent, "confidence": confidence, "probabilities": probabilities},
        **{domain: {"type": "noul", "noul": score} for domain, score in relevance.items()},
    }
    if urgent is not None:
        answers["urgent"] = {"type": "noul", "noul": urgent}
    return {"model": DEFAULT_MODEL, "answers": answers, "usage": {"input_tokens": 180, "output_tokens": 25}}


async def offline(message, history=None, summary="", available_roles=None):
    """Keyword fallback: no Jev credential, and any HTTP request fails the test."""
    def unexpected(_):
        raise AssertionError("keyword fallback must not call Jev")
    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        return await Router(client).route(message, history, summary, available_roles)


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def evaluate(self, message, payload=None, status=200, history=None, summary="", error=None, available_roles=None):
        calls = []

        def transport(request):
            calls.append(request)
            if error:
                raise error
            return httpx.Response(status, json=payload if payload is not None else official_response())

        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            decision = await Router(client, api_key="test-key").route(message, history, summary, available_roles)
        return decision, calls

    async def test_official_protocol_one_request_and_raw_model(self):
        payload = official_response()
        decision, calls = await self.evaluate("登录失败，错误码 401", payload)
        self.assertEqual(len(calls), 1)
        body = json.loads(calls[0].content)
        self.assertEqual(body["model"], DEFAULT_MODEL)
        self.assertEqual(set(body["questions"]), {"intent", "general", "technical", "billing", "escalation", "urgent"})
        self.assertEqual(body["questions"]["urgent"]["type"], "noul")
        self.assertEqual(body["questions"]["intent"]["type"], "choice")
        self.assertEqual(body["questions"]["billing"]["type"], "noul")
        self.assertEqual(calls[0].headers["authorization"], "Bearer test-key")
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.jev["raw"], payload)
        self.assertEqual(decision.jev["model_version"], DEFAULT_MODEL)
        self.assertEqual(decision.jev["request_count"], 1)
        self.assertFalse(decision.degraded)
        self.assertGreaterEqual(decision.jev["latency_ms"], 0)

    async def test_confidence_does_not_replace_probabilities(self):
        payload = official_response("billing", confidence=0.1)
        decision, _ = await self.evaluate("退款", payload)
        self.assertEqual(decision.jev["choice_confidence"], 0.1)
        self.assertEqual(decision.jev["choice_probabilities"]["billing"], 0.9)
        self.assertEqual(decision.primary_agent, "billing")

    async def test_distinct_compound_domains_use_two_agents(self):
        payload = official_response("billing", {"technical": 0.95, "billing": 0.98})
        decision, _ = await self.evaluate("退款一直没到账，而且登录出现 401 报错", payload)
        self.assertEqual(decision.primary_agent, "billing")
        self.assertEqual(decision.supporting_agents, ["technical"])
        self.assertFalse(decision.needs_clarification)

    async def test_multiple_billing_requests_do_not_add_agent(self):
        decision, _ = await self.evaluate("退款没到账，还需要发票", official_response("billing"))
        self.assertEqual(decision.primary_agent, "billing")
        self.assertEqual(decision.supporting_agents, [])

    async def test_legacy_collaboration_keywords_apply_only_in_keyword_mode(self):
        decision = await offline("登录报错401，另外要核对订阅")
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.supporting_agents, ["billing"])
        # With Jev available, keywords do not add a role Jev considers irrelevant.
        decision, calls = await self.evaluate("登录报错401，另外要核对订阅", official_response("technical"))
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.supporting_agents, [])
        self.assertEqual(len(calls), 1)

    async def test_keyword_score_rule_adds_support_without_legacy_keyword(self):
        decision = await offline("App 安装出问题，金额显示也不对")
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.supporting_agents, ["billing"])
        self.assertEqual((await offline("App 安装出问题")).supporting_agents, [])

    async def test_jev_relevance_threshold_decides_support(self):
        for relevance, expected_support in ((0.86, ["billing"]), (0.84, [])):
            decision, _ = await self.evaluate("请分析目前问题", official_response("technical", {"billing": relevance}))
            self.assertEqual(decision.primary_agent, "technical")
            self.assertEqual(decision.supporting_agents, expected_support)
            self.assertEqual(decision.routing_scores["billing"], relevance)

    async def test_confident_jev_choice_sets_primary_and_relevance_adds_support(self):
        decision, _ = await self.evaluate("技术报错401，同时退款", official_response("billing", {"technical": 0.98, "billing": 0.40}))
        self.assertGreater(decision.routing_scores["technical"], decision.routing_scores["billing"])
        self.assertEqual(decision.primary_agent, "billing")
        self.assertEqual(decision.primary_source, "jev_choice")
        self.assertEqual(decision.supporting_agents, ["technical"])

    async def test_explicit_compound_keywords_survive_small_score_margin_offline(self):
        decision = await offline("报错401，同时退款")
        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.supporting_agents, ["billing"])

    async def test_weak_jev_choice_falls_back_to_domain_relevance(self):
        # Live sample X26: choice billing at 0.36, both domains at 0.97.
        weak = {"general": 0.2, "technical": 0.34, "billing": 0.36, "escalation": 0.05, "unknown": 0.05}
        decision, _ = await self.evaluate("登录报错401，而且被重复扣款了", official_response(
            "billing", {"technical": 0.97, "billing": 0.97}, probabilities=weak))
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.primary_source, "jev_relevance")
        self.assertEqual(decision.supporting_agents, ["billing"])
        # Live sample X39: weak choice and no relevant domain.
        weak = {"general": 0.42, "technical": 0.1, "billing": 0.3, "escalation": 0.08, "unknown": 0.1}
        decision, _ = await self.evaluate("怎么防止被骗", official_response(
            "general", {"general": 0.14, "billing": 0.2}, probabilities=weak))
        self.assertTrue(decision.needs_clarification)
        # Live sample X35: a confident choice is used even when its relevance is low.
        decision, _ = await self.evaluate("人工客服几点上班？", official_response("general", {"general": 0.32}))
        self.assertEqual(decision.primary_agent, "general")
        self.assertFalse(decision.needs_clarification)

    async def test_unavailable_support_and_primary_are_filtered(self):
        decision, _ = await self.evaluate("登录报错401，另外订阅", official_response("technical"), available_roles={"general", "technical", "escalation"})
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.supporting_agents, [])
        decision, _ = await self.evaluate("登录报错401", official_response("technical"), available_roles={"general", "billing", "escalation"})
        self.assertEqual(decision.primary_agent, "general")
        self.assertEqual(decision.intent, "technical")
        self.assertFalse(decision.needs_clarification)
        self.assertTrue(decision.degraded)
        self.assertIn("primary_role_unavailable:technical", decision.degraded_reason)
        decision, _ = await self.evaluate("登录报错401，退款没到账", official_response("technical", {"billing": 0.95}), available_roles={"general", "billing"})
        self.assertEqual(decision.primary_agent, "billing")
        self.assertEqual(decision.supporting_agents, [])

    async def test_missing_escalation_role_preserves_human_request(self):
        decision, calls = await self.evaluate("请转人工处理退款", official_response("escalation"), available_roles={"general", "billing"})
        self.assertEqual(decision.primary_agent, "general")
        self.assertEqual(decision.intent, "escalation")
        self.assertTrue(decision.escalated)
        self.assertTrue(decision.degraded)
        self.assertEqual(len(calls), 1)

    async def test_unknown_still_clarifies_with_restricted_roles(self):
        decision, _ = await self.evaluate("帮我看看", official_response("unknown", {"general": 0.2}), available_roles={"general", "billing"})
        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.primary_agent, "general")
        self.assertEqual(decision.supporting_agents, [])

    async def test_general_registration_is_required_for_safe_fallback(self):
        async with httpx.AsyncClient() as client:
            with self.assertRaisesRegex(ValueError, "general role must be registered"):
                await Router(client).route("帮我看看", available_roles=set())

    async def test_unknown_clarifies_but_close_scores_do_not(self):
        decision, _ = await self.evaluate("请帮我看看", official_response("unknown", {"general": 0.2, "technical": 0.2, "billing": 0.2}))
        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.primary_agent, "general")
        decision, _ = await self.evaluate("请帮我看看", official_response("technical", {"general": 0.8, "technical": 0.82, "billing": 0.1}))
        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.primary_agent, "technical")

    async def test_more_relevance_never_turns_into_clarification(self):
        expected = {0.90: ["technical"], 0.85: ["technical"], 0.70: [], 0.60: []}
        for relevance, supporting in expected.items():
            decision, _ = await self.evaluate("我这边页面一直不对劲", official_response("general", {"general": 0.9, "technical": relevance}))
            self.assertFalse(decision.needs_clarification, relevance)
            self.assertEqual(decision.primary_agent, "general", relevance)
            self.assertEqual(decision.supporting_agents, supporting, relevance)
        seen_support = False
        for step in range(21):
            decision, _ = await self.evaluate("我这边页面一直不对劲", official_response("general", {"general": 0.9, "technical": step / 20}))
            self.assertFalse(decision.needs_clarification, step)
            if seen_support:
                self.assertIn("technical", decision.supporting_agents, step)
            seen_support = seen_support or "technical" in decision.supporting_agents
        self.assertTrue(seen_support)

    async def test_offline_greeting_is_not_domain_evidence_and_ties_go_to_specialists(self):
        async with httpx.AsyncClient() as client:
            router = Router(client)
            for message, primary in (("你好，我想退款", "billing"), ("你好，App 登录报错了", "technical"),
                                     ("退换货政策里退款多久到账", "billing")):
                decision = await router.route(message)
                self.assertFalse(decision.needs_clarification, message)
                self.assertEqual(decision.primary_agent, primary, message)
            self.assertTrue((await router.route("你好")).needs_clarification)

    async def test_offline_vocabulary_follows_knowledge_base_sections(self):
        samples = {
            "A1005 这个订单我怎么被扣了两次钱？": "billing",
            "我想退掉 A1001，另外退了以后我的积分会被扣回去吗？": "billing",
            "我把密码告诉你，你帮我登录看看哪里有问题": "technical",
            "取消自动续费之后，这个月扣的钱会退给我吗？": "billing",
            "那多出来的那笔什么时候能退给我？": "billing",
            "买个 100 块的东西最多能抵多少？": "general",
            "新地址是北京市朝阳区建国路 88 号": "general",
            "能退多少钱？": "billing",
            "怎么换绑手机号": "general",
            "我想注销账户": "general",
            "登录后怎么查积分": "general",
        }
        async with httpx.AsyncClient() as client:
            router = Router(client)
            for message, primary in samples.items():
                decision = await router.route(message)
                self.assertFalse(decision.needs_clarification, message)
                self.assertEqual(decision.primary_agent, primary, message)
            compound = await router.route("订单 A1005 被扣了两次钱，而且我现在登录 App 一直报 401")
            self.assertEqual({compound.primary_agent, *compound.supporting_agents}, {"technical", "billing"})

    async def test_keyword_mode_escalation_signals_follow_knowledge_base(self):
        for message, reason in (
            ("我要投诉，你们服务太差了", "complaint"),
            ("找你们经理", "complaint"),
            ("能不能找你们经理", "complaint"),
            ("能不能帮我转人工", "explicit_human_request"),
            ("可不可以转人工", "explicit_human_request"),
            ("收到的充电宝冒烟了，差点着火！", "product_safety"),
            ("有人冒充你们客服让我转账，我已经转了 2000 块！", "fraud"),
            ("紧急！支付失败了", "urgent"),
            ("紧急！请转人工", "explicit_human_request"),
        ):
            decision = await offline(message)
            self.assertTrue(decision.escalated, message)
            self.assertEqual(decision.primary_agent, "escalation", message)
            self.assertEqual(decision.routing_reason, reason, message)
            self.assertEqual(decision.urgent, "紧急" in message, message)
        for message in ("怎么投诉商家？", "我不是要投诉，只想问发票", "怎么防止充电宝起火",
                        "如何防止被骗", "不紧急，退款慢慢处理就好", "能立刻发货吗", "我想立刻取消订单"):
            decision = await offline(message)
            self.assertFalse(decision.escalated, message)
            self.assertEqual(decision.signals, {}, message)

    async def test_jev_mode_escalation_and_urgency_follow_jev_only(self):
        # Keyword signals never override a valid Jev answer.
        decision, _ = await self.evaluate("我要投诉，你们服务太差了", official_response("general"))
        self.assertFalse(decision.escalated)
        self.assertIn("complaint", decision.signals)
        decision, _ = await self.evaluate("不用转人工，帮我查退款", official_response("escalation"))
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.routing_reason, "jev_escalation")
        # Escalation needs both the choice probability and the escalation relevance.
        decision, _ = await self.evaluate("找你们经理", official_response("escalation", {"escalation": 0.59}))
        self.assertFalse(decision.escalated)
        # Explicit urgency (live X04: urgent 0.98 while the choice is billing).
        decision, _ = await self.evaluate("紧急！支付失败了", official_response("billing", urgent=0.98))
        self.assertTrue(decision.escalated)
        self.assertTrue(decision.urgent)
        self.assertEqual(decision.routing_reason, "jev_urgent")
        decision, _ = await self.evaluate("能立刻发货吗", official_response("general", urgent=0.07))
        self.assertFalse(decision.escalated)
        self.assertFalse(decision.urgent)
        self.assertEqual(decision.jev["urgent_probability"], 0.07)

    async def test_continuation_looks_back_but_fresh_questions_do_not_inherit(self):
        history = [{"role": "user", "content": "App 登录报 401"}, {"role": "assistant", "content": "请重新登录"},
                   {"role": "user", "content": "安卓 14，昨天开始的"}, {"role": "assistant", "content": "好的"}]
        async with httpx.AsyncClient() as client:
            router = Router(client)
            decision = await router.route("还是不行", history)
            self.assertEqual(decision.primary_agent, "technical")
            self.assertEqual(decision.inherited_from, "user_message:-2")
            decision = await router.route("我买的衣服尺码不对怎么办", history[:2])
            self.assertEqual(decision.primary_agent, "general")
            self.assertEqual(decision.inherited_from, "")

    async def test_http_errors_fallback_without_retry(self):
        for status in (401, 429, 503):
            decision, calls = await self.evaluate("退款", {"detail": "unavailable"}, status=status)
            self.assertTrue(decision.degraded)
            self.assertEqual(decision.degraded_reason, f"jev_http_{status}")
            self.assertEqual(decision.primary_agent, "billing")
            self.assertEqual(len(calls), 1)

    async def test_timeout_and_network_fallback(self):
        for error, reason in ((httpx.ReadTimeout("slow"), "jev_timeout"), (httpx.ConnectError("offline"), "jev_network_error")):
            decision, calls = await self.evaluate("报错 401", error=error)
            self.assertEqual(decision.primary_agent, "technical")
            self.assertEqual(decision.degraded_reason, reason)
            self.assertEqual(len(calls), 1)

    async def test_numeric_and_schema_validation(self):
        original = official_response()
        malformed = []
        for invalid in (-0.1, 1.1, float("nan"), float("inf"), "0.98", True):
            payload = copy.deepcopy(original)
            payload["answers"]["technical"]["noul"] = invalid
            malformed.append(payload)
        payload = copy.deepcopy(original)
        del payload["answers"]["billing"]
        malformed.append(payload)
        payload = copy.deepcopy(original)
        payload["answers"]["technical"] = {"type": "score", "score": 0.98}
        malformed.append(payload)
        payload = copy.deepcopy(original)
        payload["answers"]["intent"]["probabilities"]["technical"] = 0.5
        malformed.append(payload)
        for payload in malformed:
            # HTTPX itself rejects non-finite JSON serialization; return encoded
            # bytes so the provider-response validation receives these values.
            calls = []
            def transport(request):
                calls.append(request)
                return httpx.Response(200, content=json.dumps(payload), headers={"content-type": "application/json"})
            async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
                decision = await Router(client, api_key="test-key").route("退款")
            self.assertEqual(decision.degraded_reason, "jev_invalid_response")
            self.assertEqual(decision.primary_agent, "billing")
            self.assertEqual(decision.jev["model"], DEFAULT_MODEL)
            self.assertEqual(len(calls), 1)
            json.dumps(decision.jev, allow_nan=False)

    async def test_non_json_response_fallback_retains_body(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not json"))) as client:
            decision = await Router(client, api_key="test-key").route("发票")
        self.assertEqual(decision.degraded_reason, "jev_invalid_response")
        self.assertEqual(decision.jev["raw"], {"unparsed_body": "not json"})

    async def test_no_key_does_not_send_request(self):
        def unexpected(_):
            self.fail("missing credentials must not trigger a request")
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
            decision = await Router(client).route("退款和 401 报错")
        self.assertTrue(decision.degraded)
        self.assertEqual(decision.degraded_reason, "jev_api_key_missing")
        self.assertEqual(decision.jev["request_count"], 0)
        self.assertEqual(set([decision.primary_agent, *decision.supporting_agents]), {"technical", "billing"})

    async def test_restored_keywords_route_offline_without_repeat_weight(self):
        def unexpected(_):
            self.fail("keyword routing must not send a request without credentials")
        samples = [("验证码", "technical"), ("退货", "billing"),
                   ("订阅", "billing"), ("多扣", "billing"),
                   ("会员", "general"), ("积分", "general"),
                   ("咨询", "general"), ("帮助", "general")]
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
            router = Router(client)
            for keyword, domain in samples:
                with self.subTest(keyword=keyword):
                    single = await router.route(keyword)
                    repeated = await router.route(keyword * 3)
                    self.assertEqual(single.primary_agent, domain)
                    self.assertFalse(single.needs_clarification)
                    self.assertEqual(single.keyword_hits[domain], [keyword])
                    self.assertEqual(single.jev["request_count"], 0)
                    self.assertEqual(repeated.primary_agent, domain)
                    self.assertEqual(repeated.routing_scores, single.routing_scores)
                    self.assertEqual(repeated.keyword_hits[domain], [keyword])
                    self.assertEqual(repeated.jev["request_count"], 0)

    async def test_no_evidence_fallback_clarifies(self):
        async with httpx.AsyncClient() as client:
            decision = await Router(client).route("那个东西有点奇怪")
        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.primary_agent, "general")

    async def test_keyword_mode_human_request_overrides_domain(self):
        decision = await offline("退款失败，现在请转人工")
        self.assertTrue(decision.escalated)
        self.assertEqual(decision.routing_reason, "explicit_human_request")

    async def test_keyword_mode_negated_quoted_and_resolved_requests_do_not_escalate(self):
        for message in (
            "不用转人工，帮我查退款",
            "我不是要转人工，我要退款",
            "页面显示“联系人工客服”，但我的问题是退款",
            "之前要求转人工，已经解决了。现在想查退款",
            "人工客服不用了，我想查退款",
            "我不转人工，我想查退款",
            "页面显示 '联系人工客服'，但我的问题是退款",
            "转人工是什么意思？我的问题是退款",
        ):
            decision = await offline(message)
            self.assertFalse(decision.escalated, message)
            self.assertEqual(decision.primary_agent, "billing", message)

    async def test_keyword_mode_current_affirmative_request_after_resolved_context(self):
        self.assertTrue((await offline("上次转人工的问题已经解决。现在退款失败，请转人工")).escalated)
        self.assertTrue((await offline("退款已解决现在请转人工处理登录问题")).escalated)

    async def test_keyword_mode_security_risk_and_prevention_are_different(self):
        risk = await offline("我的账号被盗了")
        self.assertTrue(risk.escalated)
        self.assertEqual(risk.routing_reason, "security_risk")
        for message in ("如何防止账号被盗", "我的账号没有被盗", "页面写着“账号被盗”，是什么意思"):
            self.assertFalse((await offline(message)).escalated, message)

    async def test_continuation_uses_latest_customer_context(self):
        history = [{"role": "user", "content": "退款"}, {"role": "user", "content": "登录报错 401"}, {"role": "assistant", "content": "试试重新登录"}]
        async with httpx.AsyncClient() as client:
            decision = await Router(client).route("还是不行", history)
        self.assertEqual(decision.primary_agent, "technical")
        self.assertEqual(decision.supporting_agents, [])
        self.assertIn("context:401", decision.keyword_hits["technical"])

    async def test_continuation_can_use_summary_without_history(self):
        async with httpx.AsyncClient() as client:
            decision = await Router(client).route("仍然如此", summary="当前问题是发票未收到")
        self.assertEqual(decision.primary_agent, "billing")

    async def test_new_issue_does_not_reuse_resolved_history(self):
        async with httpx.AsyncClient() as client:
            decision = await Router(client).route("那个东西是什么", [{"role": "user", "content": "退款"}])
        self.assertTrue(decision.needs_clarification)

    async def test_entities_include_reusable_codes_and_order(self):
        decision, _ = await self.evaluate("订单号 AB123456 退款 HK$100，报错 401", official_response("billing", {"technical": 0.95}))
        self.assertEqual(decision.entities["order_ids"], ["AB123456"])
        self.assertEqual(decision.entities["error_codes"], ["401"])
        self.assertEqual(decision.entities["amounts"], ["HK$100"])

    async def test_existing_billing_handler_recognizes_supplied_fields(self):
        from app.business_tools import check_billing_fields
        decision, _ = await self.evaluate("订单号 AB123456 在 2026年10月4日 付款 500元，现在要求退款", official_response("billing"))
        fields = check_billing_fields(SimpleNamespace(entities=decision.entities), {"payment_channel": "微信"})
        self.assertEqual(fields["missing_fields"], [])
        self.assertEqual(decision.entities["amount"], ["500元"])
        self.assertEqual(decision.entities["date"], ["2026年10月4日"])
        self.assertEqual(decision.entities["order_id"], ["AB123456"])
        self.assertEqual(decision.entities["error_codes"], [])
        self.assertEqual(decision.supporting_agents, [])

    async def test_chinese_adjacent_error_code_and_order_number_are_distinct(self):
        decision, _ = await self.evaluate("订单号 ABC401999 退款，登录报错401怎么办", official_response("technical", {"billing": 0.98}))
        self.assertEqual(decision.entities["error_code"], ["401"])
        self.assertEqual(decision.entities["order_id"], ["ABC401999"])
        def unexpected(_):
            self.fail("numeric keyword checks must remain offline")
        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
            router = Router(client)
            for message in ("退款500元", "订单号 ABC401999 退款"):
                with self.subTest(message=message):
                    decision = await router.route(message)
                    self.assertEqual(decision.primary_agent, "billing")
                    self.assertEqual(decision.supporting_agents, [])
                    self.assertEqual(decision.keyword_hits["technical"], [])
                    self.assertEqual(decision.routing_scores["technical"], 0)
                    self.assertEqual(decision.entities["error_codes"], [])
                    self.assertEqual(decision.jev["request_count"], 0)


if __name__ == "__main__":
    unittest.main()
