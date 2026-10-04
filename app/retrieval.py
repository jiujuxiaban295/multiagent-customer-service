"""Explicit bge-m3 embeddings, knowledge retrieval, and a separate public case index."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import chromadb
import httpx


def parse_markdown_docs(text: str) -> list[dict[str, str]]:
    parts = re.split(r"^## +(.+?)\s*$", text, flags=re.MULTILINE)
    return [
        {"title": title.strip(), "content": body.strip()}
        for title, body in zip(parts[1::2], parts[2::2]) if body.strip()
    ]


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Keep the original sentence-based chunks, also bounding long single sentences."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks, current = [], ""
    for sentence in text.replace("\n", "。").split("。"):
        sentence = sentence.strip()
        if not sentence:
            continue
        for start in range(0, len(sentence), chunk_size - 1):
            piece = sentence[start:start + chunk_size - 1] + "。"
            if len(current) + len(piece) > chunk_size:
                if current:
                    chunks.append(current)
                current = ""
            current += piece
    if current:
        chunks.append(current)
    return chunks


def _message_text(reply: Any) -> str:
    content = reply.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _json_array(reply: Any) -> list:
    text = _message_text(reply).strip()
    value = json.loads(text[text.index("["):text.rindex("]") + 1])
    if not isinstance(value, list):
        raise ValueError("model must return a JSON array")
    return value


class KnowledgeIndex:
    """Cases always use new-project local storage; an optional remote KB is read-only."""

    PUBLIC_CASE_FIELDS = (
        "problem", "environment", "steps", "result", "product", "version",
        "error_code", "closed_at",
    )

    def __init__(self, settings: Any, http_client: httpx.AsyncClient, model: Any):
        self.settings = settings
        self.http = http_client
        self.model = model
        self.knowledge = None
        self.cases = None
        self._cache: dict[tuple[str, int], tuple[float, list[dict]]] = {}
        self._failures = 0
        self._open_until = 0.0
        self._probe_in_flight = False

    async def initialize(self) -> None:
        def initialize_collections() -> None:
            Path(self.settings.chroma_path).mkdir(parents=True, exist_ok=True)
            local = chromadb.PersistentClient(
                path=str(self.settings.chroma_path),
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self.cases = local.get_or_create_collection(
                name=self.settings.case_collection, embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
            if self.settings.chroma_host:
                remote = chromadb.HttpClient(
                    host=self.settings.chroma_host, port=self.settings.chroma_port,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                self.knowledge = remote.get_collection(
                    name=self.settings.knowledge_collection, embedding_function=None,
                )
            else:
                self.knowledge = local.get_or_create_collection(
                    name=self.settings.knowledge_collection, embedding_function=None,
                    metadata={"hnsw:space": "cosine"},
                )

        await asyncio.to_thread(initialize_collections)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        base = self.settings.embedding_base_url.strip().rstrip("/")
        if not base or not self.settings.embedding_model.strip():
            raise RuntimeError("embedding endpoint and model are not configured")
        url = base + ("/embeddings" if base.endswith("/v1") else "/v1/embeddings")
        headers = {}
        if self.settings.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.settings.embedding_api_key}"
        response = await self.http.post(
            url, headers=headers,
            json={"model": self.settings.embedding_model, "input": texts},
            timeout=self.settings.retrieval_timeout,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda row: row["index"])
        if [row["index"] for row in data] != list(range(len(texts))):
            raise ValueError("embedding response has missing or duplicate indices")
        vectors = [row["embedding"] for row in data]
        if not vectors or not vectors[0] or any(
            len(vector) != len(vectors[0]) or any(
                isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) for value in vector
            ) for vector in vectors
        ):
            raise ValueError("embedding response has invalid vectors")
        return vectors

    async def import_documents(self, text: str) -> int:
        """Import only into this project's own KB. Embed before replacing old chunks."""
        if self.settings.chroma_host:
            raise RuntimeError("remote knowledge collection is read-only; import into local storage")
        if self.knowledge is None:
            raise RuntimeError("knowledge index is not initialized")
        docs = parse_markdown_docs(text)
        ids, bodies, metadata = [], [], []
        for doc in docs:
            chunks = chunk_text(doc["content"])
            for index, chunk in enumerate(chunks):
                ids.append(hashlib.md5(f"{doc['title']}_{index}_{chunk[:50]}".encode()).hexdigest())
                bodies.append(chunk)
                metadata.append({"title": doc["title"], "chunk_index": index, "total_chunks": len(chunks)})
        if not ids:
            return 0
        vectors = []
        for start in range(0, len(bodies), 32):
            vectors.extend(await self._embed(bodies[start:start + 32]))

        def write() -> None:
            old_ids = set()
            for title in {doc["title"] for doc in docs}:
                old_ids.update(self.knowledge.get(where={"title": title}, include=[])["ids"])
            self.knowledge.upsert(ids=ids, documents=bodies, embeddings=vectors, metadatas=metadata)
            stale = old_ids.difference(ids)
            if stale:
                self.knowledge.delete(ids=list(stale))

        await asyncio.to_thread(write)
        self._cache.clear()
        return len(ids)

    async def _rewrite(self, query: str) -> tuple[list[str], str]:
        try:
            reply = await self.model.ainvoke([
                ("system", "将问题改写为 3 个不同角度的知识库搜索查询。只返回 JSON 字符串数组；不要添加用户未提供的事实。"),
                ("human", query),
            ])
            variants = _json_array(reply)
            if not variants or any(not isinstance(value, str) or not value.strip() for value in variants):
                raise ValueError("query rewrite must contain nonempty strings")
            return list(dict.fromkeys([query] + [value.strip()[:2000] for value in variants[:3]])), ""
        except Exception as exc:
            return [query], f"query rewrite unavailable: {type(exc).__name__}"

    async def _recall(self, query: str, top_k: int) -> tuple[list[dict], str]:
        cache_key = (query, top_k)
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1], ""
        if self._open_until and (now < self._open_until or self._probe_in_flight):
            return [], "knowledge circuit is open"
        probe = bool(self._open_until)
        if probe:
            self._probe_in_flight = True
        try:
            if self.knowledge is None:
                raise RuntimeError("knowledge index is not initialized")
            count = await asyncio.to_thread(self.knowledge.count)
            if not count:
                return [], ""
            vectors = await self._embed([query])
            result = await asyncio.wait_for(asyncio.to_thread(
                self.knowledge.query, query_embeddings=vectors,
                n_results=min(top_k, count), include=["documents", "metadatas", "distances"],
            ), timeout=self.settings.retrieval_timeout)
            items = []
            for case_id, body, meta, distance in zip(
                result["ids"][0], result["documents"][0],
                result["metadatas"][0], result["distances"][0],
            ):
                meta = meta or {}
                items.append({
                    "title": meta.get("title", ""), "content": body or "",
                    "score": round(1.0 - distance, 4), "chunk": meta.get("chunk_index", 0),
                    "id": case_id,
                })
            self._failures, self._open_until = 0, 0.0
            if self.settings.rag_cache_ttl > 0:
                if len(self._cache) >= 512:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = (now + self.settings.rag_cache_ttl, items)
            return items, ""
        except Exception as exc:
            self._failures += 1
            if self._failures >= 5 or probe:
                self._open_until = time.monotonic() + 60.0
            return [], f"knowledge recall unavailable: {type(exc).__name__}"
        finally:
            if probe:
                self._probe_in_flight = False

    async def search_knowledge(self, query: str, top_k: int = 5) -> dict:
        query = query.strip()
        if not query:
            return {"success": False, "results": [], "reranked": False, "error": "query is empty"}
        top_k = max(1, min(int(top_k), 20))
        queries, rewrite_error = await self._rewrite(query)
        recalled = await asyncio.gather(*(
            self._recall(value, max(top_k, self.settings.rag_recall_k)) for value in queries
        ))
        errors = [rewrite_error] if rewrite_error else []
        merged = {}
        for items, error in recalled:
            if error:
                errors.append(error)
            for item in items:
                merged.setdefault(item["id"], item)
        items, reranked = list(merged.values()), False
        if len(items) > top_k:
            try:
                candidates = [
                    {"index": i, "title": item["title"], "content": item["content"][:500]}
                    for i, item in enumerate(items)
                ]
                reply = await self.model.ainvoke([
                    ("system", "按与用户问题的相关性排列候选索引。只返回从最相关到最不相关的 JSON 整数数组。"),
                    ("human", json.dumps({"query": query, "candidates": candidates}, ensure_ascii=False)),
                ])
                order = _json_array(reply)
                if not order or len(set(order)) != len(order) or any(
                    type(index) is not int or not 0 <= index < len(items) for index in order
                ):
                    raise ValueError("reranking contains invalid or duplicate indices")
                order += [index for index in range(len(items)) if index not in order]
                items = [items[index] for index in order]
                reranked = True
            except Exception as exc:
                errors.append(f"reranking unavailable: {type(exc).__name__}")
        result = {"success": bool(items), "results": items[:top_k], "reranked": reranked}
        if not items:
            result["error"] = "no knowledge documents were retrieved"
        if errors:
            result.update(degraded=True, degraded_reason="; ".join(dict.fromkeys(errors)))
        return result

    async def section(self, title: str) -> str:
        """Read one knowledge-base section by exact title: no embedding, no model call."""
        if self.knowledge is None:
            return ""

        def read() -> str:
            rows = self.knowledge.get(where={"title": title}, include=["documents", "metadatas"])
            chunks = sorted(zip(rows["documents"], rows["metadatas"]),
                            key=lambda row: (row[1] or {}).get("chunk_index", 0))
            return "".join(body or "" for body, _ in chunks).strip()

        return await asyncio.to_thread(read)

    async def upsert_case(self, case_id: str, payload: dict, scope: str) -> None:
        if self.cases is None:
            raise RuntimeError("case index is not initialized")
        if not scope.strip() or not case_id.strip():
            raise ValueError("case ID and trusted business scope are required")
        public = {field: payload[field] for field in self.PUBLIC_CASE_FIELDS if field in payload}
        public["case_id"] = case_id
        body = json.dumps(public, ensure_ascii=False)
        vectors = await self._embed([body])
        metadata = {"scope": scope, "status": "published", "active": True}
        metadata.update({field: str(public.get(field, "")) for field in ("product", "version", "error_code", "closed_at")})
        await asyncio.to_thread(
            self.cases.upsert, ids=[case_id], documents=[body], embeddings=vectors, metadatas=[metadata],
        )

    async def delete_case(self, case_id: str) -> None:
        if self.cases is None:
            raise RuntimeError("case index is not initialized")
        await asyncio.to_thread(self.cases.delete, ids=[case_id])

    async def search_cases(
        self, query: str, scope: str, top_k: int = 5,
        product: str = "", version: str = "", error_code: str = "",
    ) -> list[dict]:
        if self.cases is None:
            raise RuntimeError("case index is not initialized")
        if not scope.strip():
            raise ValueError("trusted business scope is required")
        if not query.strip():
            return []
        count = await asyncio.to_thread(self.cases.count)
        if not count:
            return []
        filters = [{"scope": scope}, {"status": "published"}, {"active": True}]
        filters.extend({key: value} for key, value in (
            ("product", product), ("version", version), ("error_code", error_code),
        ) if value)
        vectors = await self._embed([query])
        result = await asyncio.wait_for(asyncio.to_thread(
            self.cases.query, query_embeddings=vectors, where={"$and": filters},
            n_results=min(max(1, min(int(top_k), 20)), count),
            include=["documents", "metadatas", "distances"],
        ), timeout=self.settings.retrieval_timeout)
        cases = []
        for case_id, body, distance in zip(result["ids"][0], result["documents"][0], result["distances"][0]):
            score = 1.0 - distance
            if score < self.settings.case_min_similarity:
                continue
            payload = json.loads(body)
            public = {field: payload[field] for field in self.PUBLIC_CASE_FIELDS if field in payload}
            public.update(case_id=case_id, score=round(score, 4))
            cases.append(public)
        return cases
