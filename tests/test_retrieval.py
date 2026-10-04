import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from app.config import Settings
from app.retrieval import KnowledgeIndex, chunk_text, parse_markdown_docs


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.last_where = None
        self.query_count = 0

    def count(self):
        return len(self.records)

    def upsert(self, *, ids, documents, embeddings, metadatas):
        for row in zip(ids, documents, embeddings, metadatas):
            self.records[row[0]] = row

    def delete(self, *, ids):
        for key in ids:
            self.records.pop(key, None)

    @staticmethod
    def _matches(metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(FakeCollection._matches(metadata, child) for child in where["$and"])
        return all(metadata.get(key) == value for key, value in where.items())

    def get(self, *, where, include):
        return {"ids": [key for key, row in self.records.items() if self._matches(row[3], where)]}

    def query(self, *, query_embeddings, n_results, include, where=None):
        self.query_count += 1
        self.last_where = where
        rows = [row for row in self.records.values() if self._matches(row[3], where)][:n_results]
        return {
            "ids": [[row[0] for row in rows]], "documents": [[row[1] for row in rows]],
            "metadatas": [[row[3] for row in rows]], "distances": [[0.1] * len(rows)],
        }


class FakeChroma:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, *, name, embedding_function, metadata):
        assert embedding_function is None
        assert metadata["hnsw:space"] == "cosine"
        return self.collections.setdefault(name, FakeCollection())


class FakeModel:
    def __init__(self, rerank=None):
        self.calls = 0
        self.rerank = [2, 0, 1] if rerank is None else rerank

    async def ainvoke(self, messages):
        self.calls += 1
        if "改写" in messages[0][1]:
            content = ["first query", "second query", "third query"]
        else:
            content = self.rerank
        return SimpleNamespace(content=json.dumps(content))


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = TemporaryDirectory()
        self.settings = Settings(chroma_path=self.directory.name, embedding_base_url="http://embedding.test",
                                 embedding_model="bge-m3", embedding_api_key="fixture", rag_recall_k=5)
        self.requests = []

        def embed(request):
            payload = json.loads(request.content)
            self.requests.append((str(request.url), dict(request.headers), payload))
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": [1.0, 0.0]}
                for index in reversed(range(len(payload["input"])))
            ]})

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(embed))
        self.model = FakeModel()
        self.index = KnowledgeIndex(self.settings, self.http, self.model)
        with patch("app.retrieval.chromadb.PersistentClient", return_value=FakeChroma()):
            await self.index.initialize()

    async def asyncTearDown(self):
        await self.http.aclose()
        self.directory.cleanup()

    async def test_import_is_idempotent_and_replaces_changed_title(self):
        original = "## Refund\n" + "规则。" * 250
        self.assertEqual(len(parse_markdown_docs(original)), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunk_text("a" * 2000)))
        first = await self.index.import_documents(original)
        self.assertGreater(first, 1)
        await self.index.import_documents(original)
        self.assertEqual(self.index.knowledge.count(), first)
        await self.index.import_documents("## Refund\n修订后的唯一规则。")
        self.assertEqual(self.index.knowledge.count(), 1)
        self.assertEqual(self.requests[0][0], "http://embedding.test/v1/embeddings")
        self.assertEqual(self.requests[0][1]["authorization"], "Bearer fixture")
        self.settings.embedding_base_url = "http://embedding.test/v1/"
        await self.index._embed(["query"])
        self.assertEqual(self.requests[-1][0], "http://embedding.test/v1/embeddings")

    async def test_recall_deduplicates_chunks_and_reranks(self):
        await self.index.import_documents("## One\n第一条。\n## Two\n第二条。\n## Three\n第三条。")
        response = await self.index.search_knowledge("question", top_k=2)
        self.assertTrue(response["success"])
        self.assertTrue(response["reranked"])
        self.assertEqual([row["title"] for row in response["results"]], ["Three", "One"])
        self.assertEqual(len({row["id"] for row in response["results"]}), 2)
        self.assertEqual(self.index.knowledge.query_count, 4)
        self.assertEqual(self.model.calls, 2)

    async def test_malformed_rerank_falls_back_and_recall_cache_is_reused(self):
        await self.index.import_documents("## One\n第一条。\n## Two\n第二条。\n## Three\n第三条。")
        self.model.rerank = [0, 0, 99]
        result = await self.index.search_knowledge("question", top_k=2)
        self.assertTrue(result["success"])
        self.assertFalse(result["reranked"])
        self.assertTrue(result["degraded"])
        self.assertIn("reranking unavailable", result["degraded_reason"])
        self.assertEqual([row["title"] for row in result["results"]], ["One", "Two"])
        queries = self.index.knowledge.query_count
        await self.index.search_knowledge("question", top_k=2)
        self.assertEqual(self.index.knowledge.query_count, queries)

    async def test_cases_require_scope_and_lifecycle_filter_and_exclude_private_fields(self):
        payload = {"problem": "错误 401", "environment": "App", "steps": ["更新过期登录凭证"],
                   "result": "用户确认恢复", "product": "App", "version": "2", "error_code": "401",
                   "closed_at": "2026-10-04T12:00:00Z", "user_id": "PRIVATE", "evidence_reference": "PRIVATE"}
        await self.index.upsert_case("case-a", payload, "scope-a")
        await self.index.upsert_case("case-b", payload, "scope-b")
        await self.index.upsert_case("case-a", payload, "scope-a")
        self.assertEqual(self.index.cases.count(), 2)
        stored = self.index.cases.records["case-a"]
        self.assertNotIn("PRIVATE", stored[1])
        rows = await self.index.search_cases("401", "scope-a", product="App", version="2", error_code="401")
        self.assertEqual([row["case_id"] for row in rows], ["case-a"])
        self.assertNotIn("user_id", rows[0])
        self.assertEqual(self.index.cases.last_where, {"$and": [
            {"scope": "scope-a"}, {"status": "published"}, {"active": True},
            {"product": "App"}, {"version": "2"}, {"error_code": "401"},
        ]})
        await self.index.delete_case("case-a")
        self.assertEqual(await self.index.search_cases("401", "scope-a"), [])
        with self.assertRaises(ValueError):
            await self.index.search_cases("401", "")

    async def test_embedding_failure_is_explicit_and_circuit_opens(self):
        await self.index.import_documents("## One\n第一条。")
        await self.http.aclose()
        calls = []

        def unavailable(request):
            calls.append(request)
            return httpx.Response(503, json={"error": "fixture unavailable"})

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
        self.index.http = self.http
        for _ in range(5):
            result = await self.index._recall("uncached", 5)
            self.assertIn("unavailable", result[1])
        result = await self.index._recall("uncached", 5)
        self.assertEqual(len(calls), 5)
        self.assertIn("circuit is open", result[1])
        self.settings.embedding_base_url = ""
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            await self.index.upsert_case("case", {"problem": "fixture"}, "scope")

    async def test_invalid_embedding_indices_are_rejected(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"data": [{"index": 1, "embedding": [1.0, 0.0]}]},
        ))) as client:
            self.index.http = client
            with self.assertRaises(ValueError):
                await self.index._embed(["only item"])

    async def test_remote_kb_import_is_blocked(self):
        self.settings.chroma_host = "old-project-db"
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            await self.index.import_documents("## New\ncontent")

    async def test_real_chroma_explicit_vectors_and_compound_filter(self):
        index = KnowledgeIndex(self.settings, self.http, self.model)
        await index.initialize()
        await index.import_documents("## One\n第一条。")
        await index.upsert_case("real-case", {"problem": "401", "environment": "App", "steps": ["登录"],
            "result": "确认恢复", "product": "App", "version": "2", "error_code": "401"}, "scope-a")
        self.assertEqual(len(await index.search_cases("401", "scope-a", error_code="401")), 1)
        self.assertEqual(await index.search_cases("401", "scope-b"), [])
        await index.delete_case("real-case")
        self.assertEqual(await index.search_cases("401", "scope-a"), [])

    def test_reviewed_kb_has_26_documents(self):
        path = Path(__file__).resolve().parents[1] / "data/kb/电商客服知识库.md"
        self.assertEqual(len(parse_markdown_docs(path.read_text(encoding="utf-8"))), 26)


if __name__ == "__main__":
    unittest.main()
