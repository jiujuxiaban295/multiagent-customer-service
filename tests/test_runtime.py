"""Exercise the actual LangChain graph with a scripted ChatModel, never a mock graph."""
import asyncio
import json
from typing import Any
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from app.contracts import PipelineInput
from app.models import UsageCallback
from app.profiles import PROFILES
from app.routing import RouteDecision
from app.runtime import V3Pipeline
from app.tools import build_tools, TOOL_SCOPES


class ScriptModel(BaseChatModel):
    behavior: Any
    events: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-runtime-test"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("the runtime should use async model calls")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = str(messages[0].content)
        role = "composer"
        for name in PROFILES:
            if "[角色]\n" + name + "\n" in prompt:
                role = name
                break
        user = next(message.content for message in reversed(messages) if isinstance(message, HumanMessage))
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        visible = {getattr(tool, "name", "") for tool in kwargs.get("tools", [])}
        event = {"role": role, "user": user, "prompt": prompt, "tools": visible,
                 "tool_messages": tool_messages, "temperature": kwargs.get("temperature"),
                 "max_tokens": kwargs.get("max_tokens")}
        self.events.append(event)
        await asyncio.sleep(0.01)
        answer = self.behavior(event)
        if isinstance(answer, Exception):
            raise answer
        answer.usage_metadata = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        return ChatResult(generations=[ChatGeneration(message=answer)])


class FixedRouter:
    def __init__(self, primary="general", supporting=None):
        self.calls = 0
        self.decision = RouteDecision(
            intent=primary, primary_agent=primary, supporting_agents=supporting or [],
            routing_scores={primary: 1.0}, entities={}, keyword_hits={}, jev={},
            degraded=False, degraded_reason="", routing_reason="test fixture", needs_clarification=False,
            escalated=primary == "escalation",
        )

    async def route(self, message, history, context, available_roles=None):
        self.calls += 1
        self.available_roles = available_roles
        return self.decision


class FakeSkills:
    def prompt_for(self, message, role):
        return f"fixture skill for {role}"


class FakeIndex:
    def __init__(self):
        self.calls = []

    async def search_knowledge(self, query, top_k):
        self.calls.append((query, top_k))
        return {"success": True, "results": [{"id": "fixture", "content": "policy"}]}


class FakeCases:
    async def search(self, query, scope, top_k, **kwargs):
        return []


def inp(message="当前问题", user="user-a", context="", history=None):
    return PipelineInput(message, user, "conversation-" + user, "trusted-scope", context, history)


def tool_call(name, args, content="", call_id="fixture-call"):
    return AIMessage(content=content, tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, behavior, primary="general", supporting=None, call_limit=4):
        model = ScriptModel(behavior=behavior, callbacks=[UsageCallback()])
        router, index = FixedRouter(primary, supporting), FakeIndex()
        pipeline = V3Pipeline(model, router, build_tools(index, FakeCases()), FakeSkills(), call_limit, 5)
        return pipeline, model, router, index

    async def test_role_whitelist_is_filtered_and_enforced_with_paired_tool_id(self):
        def behavior(event):
            if not event["tool_messages"]:
                return tool_call("compare_amounts", {"amount_a": 10, "amount_b": 2}, call_id="denied-amount")
            return AIMessage(content="当前技术角色不能执行金额核验。")

        pipeline, model, router, _ = self.pipeline(behavior, primary="technical")
        result = await pipeline.run(inp())
        self.assertTrue(result.success)
        self.assertEqual(model.events[0]["tools"], TOOL_SCOPES["technical"])
        self.assertNotIn("compare_amounts", model.events[0]["tools"])
        self.assertEqual(router.calls, 1)
        self.assertEqual(result.tool_traces[0]["tool_call_id"], "denied-amount")
        self.assertFalse(result.tool_traces[0]["success"])
        self.assertEqual(result.tool_traces[0]["error"], "PermissionError")
        tool_result = model.events[1]["tool_messages"][0]
        self.assertEqual(tool_result.tool_call_id, "denied-amount")
        self.assertEqual(tool_result.status, "error")
        self.assertNotIn("difference", tool_result.content)

    async def test_tool_round_prose_and_final_answer_are_preserved_without_old_history(self):
        def behavior(event):
            if not event["tool_messages"]:
                return tool_call("lookup_error_code", {"error_code": "401"}, "请先记录发生时间。")
            return AIMessage(content="401 表示认证失败，请检查凭证是否过期。")

        pipeline, _, _, _ = self.pipeline(behavior, primary="technical")
        result = await pipeline.run(inp(history=[{"role": "assistant", "content": "OLD_HISTORY_SHOULD_NOT_REPEAT"}]))
        self.assertTrue(result.success)
        self.assertIn("请先记录发生时间。", result.response)
        self.assertIn("401 表示认证失败", result.response)
        self.assertNotIn("OLD_HISTORY_SHOULD_NOT_REPEAT", result.response)
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.input_tokens, 6)

    async def test_last_allowed_model_round_has_no_tools(self):
        pipeline, model, _, _ = self.pipeline(lambda event: AIMessage(content="根据已有信息完成回答。"), call_limit=1)
        result = await pipeline.run(inp())
        self.assertTrue(result.success)
        self.assertEqual(len(model.events), 1)
        self.assertEqual(model.events[0]["tools"], set())

    async def test_model_cannot_execute_tools_or_continue_after_call_limit(self):
        pipeline, model, _, index = self.pipeline(
            lambda event: tool_call("search_knowledge_base", {"query": "rogue extra call"}), call_limit=1)
        result = await pipeline.run(inp())
        self.assertFalse(result.success)
        self.assertLessEqual(len(model.events), 1)
        self.assertEqual(index.calls, [])

    async def test_concurrent_requests_keep_context_tool_arguments_and_usage_isolated(self):
        def behavior(event):
            if not event["tool_messages"]:
                return tool_call("inspect_request_context", {"focus": event["user"]}, call_id="call-" + event["user"])
            return AIMessage(content="完成:" + event["user"])

        pipeline, model, router, _ = self.pipeline(behavior)
        result_a, result_b = await asyncio.gather(
            pipeline.run(inp("request-a", "a", "PRIVATE_CONTEXT_A")),
            pipeline.run(inp("request-b", "b", "PRIVATE_CONTEXT_B")),
        )
        self.assertEqual(router.calls, 2)
        for result, name in ((result_a, "a"), (result_b, "b")):
            self.assertTrue(result.success)
            self.assertEqual(result.response, "完成:request-" + name)
            self.assertEqual(result.tool_traces[0]["input"], {"focus": "request-" + name})
            self.assertEqual(result.tool_traces[0]["tool_call_id"], "call-request-" + name)
            self.assertEqual(result.llm_calls, 2)
            self.assertEqual(result.input_tokens, 6)
        self.assertNotEqual(result_a.request_id, result_b.request_id)
        for event in model.events:
            own = "A" if event["user"] == "request-a" else "B"
            other = "B" if own == "A" else "A"
            self.assertIn("PRIVATE_CONTEXT_" + own, event["prompt"])
            self.assertNotIn("PRIVATE_CONTEXT_" + other, event["prompt"])

    async def test_failed_specialist_falls_back_to_general(self):
        pipeline, _, router, _ = self.pipeline(
            lambda event: RuntimeError("specialist fixture failure") if event["role"] == "technical"
            else AIMessage(content="通用客服已接续处理。"), primary="technical")
        result = await pipeline.run(inp())
        self.assertTrue(result.success)
        self.assertEqual(result.response, "通用客服已接续处理。")
        self.assertEqual(result.extra["agent_types"], ["technical", "general"])
        self.assertEqual(result.extra["failed_model_calls"], 1)
        self.assertEqual(router.calls, 1)

    async def test_composer_failure_preserves_primary_and_supporting_answers(self):
        def behavior(event):
            if event["role"] == "composer":
                return RuntimeError("composer fixture failure")
            return AIMessage(content="technical-answer" if event["role"] == "technical" else "billing-answer")

        pipeline, _, router, _ = self.pipeline(behavior, primary="technical", supporting=["billing"])
        result = await pipeline.run(inp())
        self.assertTrue(result.success)
        self.assertIn("technical-answer", result.response)
        self.assertIn("billing-answer", result.response)
        self.assertTrue(result.extra["composition_degraded"])
        self.assertEqual(router.calls, 1)
        self.assertEqual(result.llm_calls, 3)

    async def test_contract_packet_and_role_settings_are_applied_without_mutating_shared_model(self):
        pipeline, model, router, _ = self.pipeline(
            lambda event: AIMessage(content="技术处理完成。" if event['role'] == 'technical' else "账单处理完成。"),
            primary="technical", supporting=["billing"])
        router.decision.entities = {'error_code': ['401'], 'amount': ['20元']}
        result = await pipeline.run(inp())
        self.assertTrue(result.success)
        for event in model.events:
            if event['role'] == 'composer':
                self.assertIn('如果结论冲突', event['prompt'])
                self.assertIn('fixture skill for general', event['prompt'])
                packet = json.loads(event['user'])
                self.assertEqual(packet['selected_primary'], 'technical')
                self.assertEqual(packet['primary_agent'], 'technical')
                self.assertEqual((event['temperature'], event['max_tokens']), (0.1, 1000))
                continue
            profile = PROFILES[event['role']]
            self.assertIn(profile.mission, event['prompt'])
            self.assertIn(' -> '.join(profile.workflow), event['prompt'])
            self.assertIn('；'.join(profile.output_contract), event['prompt'])
            self.assertIn('；'.join(profile.handoff_conditions), event['prompt'])
            self.assertEqual((event['temperature'], event['max_tokens']),
                             (profile.temperature, profile.max_tokens))
            packet = json.loads(event['prompt'].split('[角色输入包]\n', 1)[1])
            self.assertIsNone(packet['urgency'])
            self.assertNotIn('user_profile', packet)
            if event['role'] == 'technical':
                self.assertEqual(packet['diagnostic_fields']['error_codes'], ['401'])
            else:
                self.assertEqual(packet['verification_fields']['missing_fields'], ['订单号或交易号'])
        self.assertEqual(router.available_roles, set(PROFILES))

    async def test_failed_supporting_specialist_falls_back_and_retains_its_tool_trace(self):
        def behavior(event):
            if event['role'] == 'billing':
                if not event['tool_messages']:
                    return tool_call('compare_amounts', {'amount_a': 20, 'amount_b': 10}, call_id='billing-before-failure')
                return RuntimeError('billing fixture failure')
            if event['role'] == 'composer':
                packet = json.loads(event['user'])
                self.assertEqual([answer['requested_role'] for answer in packet['answers']], ['technical', 'billing'])
                self.assertEqual([answer['role'] for answer in packet['answers']], ['technical', 'general'])
                return AIMessage(content='技术答复与通用接续。')
            return AIMessage(content=event['role'] + '-answer')

        pipeline, model, router, _ = self.pipeline(behavior, primary='technical', supporting=['billing'])
        result = await pipeline.run(inp())
        self.assertTrue(result.success)
        self.assertEqual(result.response, '技术答复与通用接续。')
        self.assertEqual(result.extra['agent_types'], ['technical', 'billing', 'general'])
        self.assertEqual(result.extra['role_results'][1]['role'], 'general')
        self.assertEqual(result.tool_traces[0]['tool_call_id'], 'billing-before-failure')
        self.assertEqual(result.tool_traces[0]['agent_type'], 'billing')
        self.assertEqual(router.calls, 1)
        general = next(event for event in model.events if event['role'] == 'general')
        self.assertNotIn('compare_amounts', general['tools'])
        self.assertEqual(general['tool_messages'], [])

    async def test_each_failed_specialist_gets_one_fallback_and_fallback_failure_stops(self):
        pipeline, model, router, _ = self.pipeline(lambda event: RuntimeError('fixture failure'),
            primary='technical', supporting=['billing'])
        result = await pipeline.run(inp())
        self.assertFalse(result.success)
        self.assertEqual([event['role'] for event in model.events].count('general'), 2)
        self.assertEqual(len(model.events), 4)
        self.assertTrue(all(not output['success'] for output in result.extra['role_results']))
        self.assertEqual(router.calls, 1)

    async def test_escalation_produces_structured_handoff_without_generation(self):
        pipeline, model, router, index = self.pipeline(
            lambda event: RuntimeError('no generation should happen'), primary='escalation')
        router.decision.entities = {'order_id': ['FIXTURE-1001']}
        result = await pipeline.run(inp('请转人工'))
        self.assertTrue(result.success)
        self.assertTrue(result.escalated)
        self.assertEqual(result.llm_calls, 0)
        self.assertEqual(model.events, [])
        self.assertEqual(index.calls, [])
        self.assertEqual(result.tools_used, [])
        handoff = result.extra['handoff_summaries'][0]
        self.assertEqual(handoff['request_id'], result.request_id)
        self.assertEqual(handoff['entities'], {'order_id': ['FIXTURE-1001']})
        self.assertEqual(handoff['urgency'], 'UNKNOWN')
        self.assertIn('官方人工客服入口', result.response)
        self.assertEqual(router.calls, 1)

    async def test_handoff_adds_kb_stop_loss_and_hides_internal_fields(self):
        class Knowledge:
            def __init__(self, fail=False):
                self.titles, self.fail = [], fail

            async def section(self, title):
                self.titles.append(title)
                if self.fail:
                    raise RuntimeError('chroma unavailable')
                return '请立即通过「忘记密码」重置密码，并联系人工客服申请临时冻结账户。'

        pipeline, model, router, _ = self.pipeline(
            lambda event: RuntimeError('no generation should happen'), primary='escalation')
        pipeline.knowledge = Knowledge()
        router.decision.routing_reason = 'security_risk'
        router.decision.urgent = True
        router.decision.entities = {'order_ids': ['A10050'], 'order_id': ['A10050'], 'amount': ['299 元'],
                                    'date': [], 'emails': ['a@example.com']}
        result = await pipeline.run(inp('紧急！我的账号被盗了'))
        self.assertEqual(pipeline.knowledge.titles, ['账号被盗与异常登录'])
        self.assertIn('升级原因：账号或资金存在安全风险', result.response)
        self.assertIn('优先级：紧急', result.response)
        self.assertIn('已记录信息：订单号 A10050；金额 299 元', result.response)
        self.assertIn('重置密码', result.response)
        for internal in ('security_risk', 'order_ids', 'emails', 'a@example.com', 'UNKNOWN'):
            self.assertNotIn(internal, result.response)
        self.assertEqual(result.extra['handoff_summaries'][0]['urgency'], 'CRITICAL')
        self.assertEqual(result.extra['role_results'][0]['handoff_kb_title'], '账号被盗与异常登录')
        self.assertEqual(model.events, [])

        pipeline.knowledge = Knowledge(fail=True)
        result = await pipeline.run(inp('我的账号被盗了'))
        self.assertTrue(result.success)
        self.assertNotIn('请先参考', result.response)
        self.assertIn('官方人工客服入口', result.response)

    async def test_conditional_handoff_suggestion_does_not_restore_old_p17_false_positive(self):
        pipeline, _, _, _ = self.pipeline(lambda event: AIMessage(content='如果之后仍有问题，可以联系人工客服核验。'),
                                         primary='technical')
        result = await pipeline.run(inp())
        self.assertFalse(result.escalated)
        self.assertEqual(result.extra['handoff_summaries'], [])
        self.assertEqual(result.llm_calls, 1)

    async def test_missing_specialist_falls_back_to_registered_general(self):
        pipeline, model, _, _ = self.pipeline(lambda event: AIMessage(content='通用客服接续。'), primary='technical')
        del pipeline.agents['technical']
        result = await pipeline.run(inp())
        self.assertTrue(result.success)
        self.assertEqual([event['role'] for event in model.events], ['general'])
        self.assertEqual(result.extra['role_results'][0]['requested_role'], 'technical')


if __name__ == "__main__":
    unittest.main()
