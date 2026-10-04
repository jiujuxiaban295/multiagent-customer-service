import asyncio
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.storage import CaseService, SessionStore


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.sets = 0
        self.failed = False

    async def get(self, key):
        if self.failed:
            raise ConnectionError("fixture outage")
        return self.values.get(key)

    async def set(self, key, value, ex):
        if self.failed:
            raise ConnectionError("fixture outage")
        self.values[key] = value
        self.sets += 1


class FakeIndex:
    def __init__(self):
        self.values = {}
        self.upserts = 0
        self.fail_write = False
        self.fail_delete = False

    async def upsert_case(self, case_id, payload, scope):
        if self.fail_write:
            raise ConnectionError("fixture outage")
        self.upserts += 1
        self.values[case_id] = {**payload, "scope": scope}

    async def delete_case(self, case_id):
        if self.fail_delete:
            raise ConnectionError("fixture outage")
        self.values.pop(case_id, None)

    async def search_cases(self, query, scope, top_k=5, **filters):
        return [{**value, "score": 0.9} for value in self.values.values() if value["scope"] == scope][:top_k]


class SummaryModel:
    def __init__(self):
        self.inputs = []

    async def ainvoke(self, messages):
        data = json.loads(messages[-1]["content"])
        self.inputs.append(data)
        return SimpleNamespace(content=data["previous_summary"] + "\n" + data["older_messages"])


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "records.sqlite3"
        self.redis, self.index = FakeRedis(), FakeIndex()
        self.store = SessionStore(self.path, self.redis)
        self.cases = CaseService(self.store, self.index)
        self.conv = self.store.create_session("alice@example.com", "shop") ["conv_id"]

    async def asyncTearDown(self):
        self.store.close()
        self.temp.cleanup()

    async def append(self, message="应用无法启动，错误码 E101", response="请清理缓存后重新启动", request_id="turn-1"):
        async with self.store.lock(self.conv, "alice@example.com", "shop"):
            await self.store.append_turn(self.conv, "alice@example.com", "shop", message, response, request_id, [])

    async def resolve(self, *, evidence_text="我已清理缓存并重新启动，问题已解决", steps=None):
        evidence = await self.store.record_confirmation(self.conv, "alice@example.com", "shop", evidence_text, "confirm-1")
        return await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence,
                                      steps or ["清理缓存", "重新启动"])

    async def test_turn_written_once_and_private_durable_repair(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        await self.append()
        self.assertEqual(self.redis.sets, 1)
        await self.append()
        self.assertEqual(len(self.store.raw_messages(self.conv, "alice@example.com", "shop")), 2)
        self.assertEqual(self.redis.sets, 1)
        with self.assertRaises(PermissionError):
            await self.store.snapshot(self.conv, "bob", "shop")
        with self.assertRaises(PermissionError):
            self.store.get_session(self.conv, "alice@example.com", "other-shop")
        self.redis.failed = True
        await self.append("第二个问题", "第二个回答", "turn-2")
        self.assertEqual(len(self.store.raw_messages(self.conv, "alice@example.com", "shop")), 4)
        self.store.close()
        self.store = SessionStore(self.path, self.redis)
        self.cases = CaseService(self.store, self.index)
        self.redis.failed = False
        snapshot = await self.store.snapshot(self.conv, "alice@example.com", "shop")
        self.assertEqual(len(snapshot["history"]), 4)
        self.assertEqual(self.redis.sets, 2)

    async def test_cumulative_summary_uses_shared_model_without_losing_raw(self):
        model = SummaryModel()
        self.store.model, self.store.recent_limit = model, 2
        for number in range(3):
            await self.append(f"问题{number}", f"回答{number}", f"turn-{number}")
        snapshot = await self.store.snapshot(self.conv, "alice@example.com", "shop")
        self.assertEqual(len(snapshot["history"]), 2)
        self.assertIn("问题0", snapshot["summary"])
        self.assertIn("问题1", snapshot["summary"])
        self.assertIn("问题0", model.inputs[-1]["previous_summary"])
        self.assertEqual(len(self.store.raw_messages(self.conv, "alice@example.com", "shop")), 6)

    async def test_summary_failure_keeps_unsummarized_context(self):
        class BrokenModel:
            async def ainvoke(self, messages):
                raise ConnectionError("fixture outage")

        self.store.model, self.store.recent_limit = BrokenModel(), 2
        await self.append()
        await self.append("问题二", "回答二", "turn-2")
        snapshot = await self.store.snapshot(self.conv, "alice@example.com", "shop")
        self.assertEqual(len(snapshot["history"]), 4)
        self.assertEqual(snapshot["summary"], "")

    async def test_evidence_publication_is_sanitized_and_scope_isolated(self):
        await self.append("我叫张三，邮箱 alice@example.com，订单号 AB123456，支付金额 ¥300，应用无法启动，错误码 E101")
        result = await self.resolve()
        self.assertEqual(result["publication_status"], "published")
        recalled = await self.cases.search("无法启动", "shop")
        self.assertEqual(len(recalled), 1)
        serialized = json.dumps(recalled, ensure_ascii=False)
        for private in ("张三", "alice@example.com", "AB123456", "300", self.conv, "evidence_message_id", "scope"):
            self.assertNotIn(private, serialized)
        self.assertEqual(recalled[0]["steps"], ["清理缓存", "重新启动"])
        self.assertEqual(recalled[0]["error_code"], "E101")
        self.assertEqual(await self.cases.search("无法启动", "other-shop"), [])
        self.assertEqual(await self.cases.search("无法启动", "shop", version="999"), [])
        other = self.store.create_session("bob", "shop")["conv_id"]
        with self.assertRaises(PermissionError):
            await self.cases.close(self.conv, "bob", "shop", "resolved")
        with self.assertRaises(PermissionError):
            self.store.raw_messages(self.conv, "bob", "shop")
        self.assertNotEqual(other, self.conv)

    async def test_praise_assistant_success_and_suggested_steps_do_not_publish(self):
        await self.append(response="我已清理缓存并重新启动，问题已解决")
        assistant = self.store.raw_messages(self.conv, "alice@example.com", "shop")[-1]["message_id"]
        result = await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", assistant, ["清理缓存"])
        self.assertEqual(result["outcome"], "unknown")
        self.assertEqual(result["publication_status"], "rejected")
        self.assertFalse(self.index.values)
        await self.cases.reopen(self.conv, "alice@example.com", "shop")
        result = await self.resolve(evidence_text="谢谢，回答很好", steps=["清理缓存"])
        self.assertEqual(result["publication_status"], "rejected")
        await self.cases.reopen(self.conv, "alice@example.com", "shop")
        evidence = await self.store.record_confirmation(self.conv, "alice@example.com", "shop", "问题已解决", "confirm-2")
        result = await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence, ["清理缓存"])
        self.assertEqual(result["publication_status"], "rejected")

    async def test_unknown_unresolved_and_negated_confirmation_do_not_publish(self):
        for outcome in ("unknown", "unresolved"):
            await self.append()
            result = await self.cases.close(self.conv, "alice@example.com", "shop", outcome)
            self.assertEqual(result["publication_status"], "rejected")
            await self.cases.reopen(self.conv, "alice@example.com", "shop")
        evidence = await self.store.record_confirmation(self.conv, "alice@example.com", "shop", "清理缓存后仍然没有解决", "negated")
        result = await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence, ["清理缓存"])
        self.assertEqual(result["publication_status"], "rejected")
        self.assertEqual(await self.cases.search("启动", "shop"), [])

    async def test_case_close_idempotent_retry_revoke_reopen(self):
        await self.append()
        self.index.fail_write = True
        result = await self.resolve()
        self.assertEqual(result["publication_status"], "pending")
        case_id = result["case_id"]
        self.assertEqual(await self.cases.search("启动", "shop"), [])
        result2 = await self.cases.close(self.conv, "alice@example.com", "shop", "unresolved")
        self.assertEqual(result2["case_id"], case_id)
        self.assertEqual(result2["outcome"], "resolved")
        self.index.fail_write = False
        result = await self.cases.retry(self.conv, "alice@example.com", "shop")
        self.assertEqual(result["publication_status"], "published")
        await self.cases.retry(self.conv, "alice@example.com", "shop")
        self.assertEqual(self.index.upserts, 1)
        with self.assertRaises(ValueError):
            await self.append("新问题", "回答", "turn-2")
        self.index.fail_delete = True
        await self.cases.revoke(self.conv, "alice@example.com", "shop")
        self.assertTrue(self.index.values)
        self.assertEqual(await self.cases.search("启动", "shop"), [])
        await self.cases.retry(self.conv, "alice@example.com", "shop")
        self.assertEqual(self.index.upserts, 1)
        await self.cases.reopen(self.conv, "alice@example.com", "shop")
        self.assertEqual(await self.cases.search("启动", "shop"), [])
        await self.append("后来又出现问题", "检查日志", "turn-2")
        self.assertEqual(self.store.get_session(self.conv, "alice@example.com", "shop")["status"], "open")

    async def test_lock_serializes_distinct_store_instances(self):
        other = SessionStore(self.path, self.redis)
        order = []

        async def first():
            async with self.store.lock(self.conv, "alice@example.com", "shop"):
                order.append("first-start")
                await asyncio.sleep(0.12)
                order.append("first-end")

        async def second():
            await asyncio.sleep(0.02)
            async with other.lock(self.conv, "alice@example.com", "shop"):
                order.append("second")

        try:
            await asyncio.gather(first(), second())
            self.assertEqual(order, ["first-start", "first-end", "second"])
        finally:
            other.close()

    async def test_reopen_requires_new_evidence_and_current_question(self):
        await self.append()
        evidence = await self.store.record_confirmation(self.conv, "alice@example.com", "shop", "清理缓存，问题已解决", "old-evidence")
        await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence, ["清理缓存"])
        await self.cases.reopen(self.conv, "alice@example.com", "shop")
        result = await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence, ["清理缓存"])
        self.assertEqual(result["publication_status"], "rejected")
        self.assertEqual(await self.cases.search("启动", "shop"), [])

    async def test_public_issue_omits_unlabeled_name_and_retains_product_numeric_error(self):
        await self.append("我是张三，张三的产品 EchoMind App 无法启动，错误码 401，版本 1.2")
        result = await self.resolve()
        self.assertEqual(result["publication_status"], "published")
        cases = await self.cases.search("启动", "shop", product="EchoMind", error_code="401", version="1.2")
        self.assertEqual(len(cases), 1)
        self.assertNotIn("张三", json.dumps(cases, ensure_ascii=False))
        self.assertEqual(cases[0]["product"], "EchoMind")

    async def test_personal_transaction_status_has_no_reusable_problem(self):
        await self.append("我是张三，我的退款已到账")
        result = await self.resolve()
        self.assertEqual(result["publication_status"], "rejected")
        self.assertEqual(await self.cases.search("退款", "shop"), [])

    async def test_model_candidate_rejects_unsupported_and_identifiable_output(self):
        class Extractor:
            def __init__(self, product):
                self.product = product

            async def ainvoke(self, messages):
                return SimpleNamespace(content=json.dumps({"problem": "应用无法启动", "environment": "",
                    "steps": ["清理缓存", "重新启动"], "product": self.product, "version": "", "error_code": "E101"}))

        await self.append()
        self.cases.model = Extractor("虚构产品")
        result = await self.resolve()
        self.assertEqual(result["publication_status"], "rejected")
        self.assertIn("没有用户记录支持", result["reason"])
        await self.cases.reopen(self.conv, "alice@example.com", "shop")
        await self.append("应用无法启动，账号 secret001，错误码 E101", request_id="fresh-turn")
        self.cases.model = Extractor("账号 secret001")
        evidence = await self.store.record_confirmation(self.conv, "alice@example.com", "shop", "清理缓存并重新启动，问题已解决", "fresh-evidence")
        result = await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence, ["清理缓存", "重新启动"])
        self.assertEqual(result["publication_status"], "rejected")
        self.assertIn("不可公开", result["reason"])

    async def test_unresolved_english_and_quoted_assistant_claim_are_not_confirmation(self):
        for number, text in enumerate(("重启后 still unresolved", "重启了，客服说问题已经解决")):
            if number:
                await self.cases.reopen(self.conv, "alice@example.com", "shop")
            await self.append(request_id=f"issue-{number}")
            evidence = await self.store.record_confirmation(self.conv, "alice@example.com", "shop", text, f"evidence-{number}")
            result = await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence, ["重启"])
            self.assertEqual(result["publication_status"], "rejected")

    async def test_spaced_order_identifier_cleaned_and_model_sees_only_public_candidate(self):
        class Extractor:
            def __init__(self):
                self.sent = ""

            async def ainvoke(self, messages):
                self.sent = messages[-1]["content"]
                safe = json.loads(self.sent)["safe_candidate"]
                return SimpleNamespace(content=json.dumps({key: safe[key] for key in
                    ("problem", "environment", "steps", "product", "version", "error_code")}))

        model = Extractor()
        self.cases.model = model
        await self.append("App 登录报401，订单 A10050，手机号 13812345678")
        result = await self.resolve()
        self.assertEqual(result["publication_status"], "published")
        public = json.dumps(await self.cases.search("登录", "shop", error_code="401"), ensure_ascii=False)
        for private in ("A10050", "13812345678", "alice@example.com", self.conv):
            self.assertNotIn(private, public)
            self.assertNotIn(private, model.sent)

    async def test_later_user_contradiction_or_new_problem_invalidates_earlier_evidence(self):
        class NoCloudCall:
            calls = 0

            async def ainvoke(self, messages):
                self.calls += 1
                raise AssertionError("stale evidence must fail before model extraction")

        model = NoCloudCall()
        self.cases.model = model
        for number, contradiction in enumerate(("清理缓存后还是没有解决", "App 又无法启动")):
            self.conv = self.store.create_session("alice@example.com", "shop")["conv_id"]
            await self.append()
            evidence = await self.store.record_confirmation(self.conv, "alice@example.com", "shop", "清理缓存后问题已解决", f"confirmation-{number}")
            await self.append(contradiction, "继续检查", f"later-{number}")
            result = await self.cases.close(self.conv, "alice@example.com", "shop", "resolved", evidence, ["清理缓存"])
            self.assertEqual(result["publication_status"], "rejected")
            self.assertEqual(result["outcome"], "unknown")
            self.assertIn("旧证据已失效", result["reason"])
        self.assertEqual(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
