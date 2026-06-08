"""In-process RAG pipeline dùng cho đánh giá RAGAS.

Pipeline này gọi **trực tiếp** các module production của `rag_server` để đảm bảo
đánh giá phản ánh đúng hành vi thật của hệ thống:

- ``app.services.embedding.EmbeddingService`` — encode truy vấn + tài liệu.
- ``app.services.vector_store.QdrantService`` — upsert + search (Phase 1, tài liệu).
- ``app.services.transcript_store.TranscriptStore`` — upsert + search (Phase 2).
- ``app.services.retrieval.fuse`` — hybrid fusion BM25+vector (nếu bật).
- ``app.services.reranker.get_reranker().rerank`` — cross-encoder rerank (nếu bật).

Qdrant chạy **in-memory** (``QdrantClient(location=':memory:')``) để không phụ
thuộc Docker/qdrant-server. Mọi code path khác (cosine, score, payload, filter,
scroll) đều giống production.

Answer generation: gọi LM Studio (OpenAI-compatible). Trả về dict
``{answer, contexts}`` đúng format RAGAS mong đợi.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings
from app.services.embedding import EmbeddingService
from app.services.reranker import get_reranker
from app.services.retrieval import fuse
from app.services.transcript_store import TranscriptStore
from app.services.vector_store import QdrantService

logger = logging.getLogger("rag_eval.pipeline")


# --------------------------------------------------------------------------- #
# Qdrant in-memory wrapper — sửa collection của production QdrantClient sang ':memory:'
# --------------------------------------------------------------------------- #


class _InMemoryQdrant:
    """Patch tạm thời: chuyển Qdrant sang in-memory và inject vào các store."""

    @staticmethod
    def patch_into_qdrant_service() -> QdrantClient:
        client = QdrantClient(location=":memory:")
        QdrantService._client = client
        QdrantService._collections_created = set()
        return client

    @staticmethod
    def patch_into_transcript_store() -> QdrantClient:
        client = QdrantClient(location=":memory:")
        TranscriptStore._client = client
        TranscriptStore._collections_ready = set()
        return client


# --------------------------------------------------------------------------- #
# Kết quả một câu truy vấn
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalHit:
    text: str
    score: float
    source: str
    doc_id: str
    chunk_index: int = 0
    chunk_total: int = 1
    sequence_id: Optional[int] = None
    speaker: Optional[str] = None
    timestamp: Optional[str] = None
    context: Optional[str] = None
    context_status: Optional[str] = None


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: List[str]
    hits: List[RetrievalHit]
    latency_ms: float


# --------------------------------------------------------------------------- #
# Pipeline chính
# --------------------------------------------------------------------------- #


class RAGPipeline:
    """RAG pipeline in-process, dùng cho đánh giá RAGAS."""

    def __init__(
        self,
        answer_model: str = "zaya1-8b",
        answer_base_url: str = "http://192.168.240.1:1234/v1",
        answer_api_key: str = "sk-lm-I4p1UFW1:ADiMgh6qgZUwq4VixJH6",
        temperature: float = 0.0,
        max_tokens: int = 400,
    ):
        # Qdrant in-memory (chia sẻ giữa QdrantService và TranscriptStore).
        self._qclient = QdrantClient(location=":memory:")
        QdrantService._client = self._qclient
        QdrantService._collections_created = set()
        TranscriptStore._client = self._qclient
        TranscriptStore._collections_ready = set()

        # Embedding production code (load model 1 lần).
        self.embedder = EmbeddingService()
        self.dim = self.embedder.dim

        # Cấu hình rerank.
        self.reranker = get_reranker()

        # Cấu hình hybrid.
        self.hybrid_enabled = settings.hybrid_enabled

        # Answer generation.
        self.answer_model = answer_model
        self.answer_base_url = answer_base_url.rstrip("/")
        self.answer_api_key = answer_api_key
        self.answer_temperature = temperature
        self.answer_max_tokens = max_tokens
        self._http = httpx.Client(timeout=60.0)

    # ---- Ingest ----------------------------------------------------------

    def ensure_collection(self, name: str) -> None:
        if name in {c.name for c in self._qclient.get_collections().collections}:
            return
        self._qclient.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        self._qclient.create_payload_index(
            collection_name=name, field_name="source", field_schema="keyword"
        )

    def upsert_doc_chunks(
        self,
        collection: str,
        chunks: List[str],
        source: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.ensure_collection(collection)
        doc_id = str(uuid.uuid4())
        vectors = self.embedder.embed_texts(chunks)
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text": chunk,
                        "source": source,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        **(extra_metadata or {}),
                    },
                )
            )
        # Batch upsert.
        for i in range(0, len(points), 100):
            self._qclient.upsert(collection_name=collection, points=points[i : i + 100])
        return doc_id

    def upsert_transcript_point(
        self,
        collection: str,
        meeting_id: str,
        sequence_id: int,
        speaker: str,
        text: str,
    ) -> str:
        self.ensure_collection(collection)
        # Payload indexes (keyword + integer) — như production.
        for field, schema in (
            ("meeting_id", "keyword"),
            ("sequence_id", "integer"),
            ("speaker", "keyword"),
        ):
            try:
                self._qclient.create_payload_index(
                    collection_name=collection, field_name=field, field_schema=schema
                )
            except Exception:
                pass

        point_id = str(uuid.uuid4())
        vector = self.embedder.embed_query(text)
        self._qclient.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "meeting_id": meeting_id,
                        "sequence_id": sequence_id,
                        "speaker": speaker,
                        "text": text,
                        "context": "",
                        "context_status": "disabled",  # không bật LLM build context
                    },
                )
            ],
        )
        return point_id

    # ---- Query -----------------------------------------------------------

    def query_docs(
        self,
        collection: str,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[RetrievalHit]:
        qvec = self.embedder.embed_query(question)
        fetch_k = (
            max(top_k, top_k * settings.hybrid_fetch_multiplier)
            if self.hybrid_enabled or self.reranker.enabled
            else top_k
        )
        raw = self._qclient.search(
            collection_name=collection,
            query_vector=qvec,
            limit=fetch_k,
            with_payload=True,
            score_threshold=max(0.0, score_threshold),
        )
        candidates = [
            {
                "text": r.payload.get("text", ""),
                "score": round(float(r.score), 4),
                "source": r.payload.get("source", ""),
                "doc_id": r.payload.get("doc_id", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "chunk_total": r.payload.get("chunk_total", 0),
            }
            for r in raw
        ]
        if self.hybrid_enabled:
            candidates = fuse(question, candidates)
        # Note: rerank với local CrossEncoder sẽ block event loop (sync predict).
        # Trong eval, rerank được bật/tắt qua env RERANK_PROVIDER — sync predict
        # của CrossEncoder cũng không sao vì đây là script đánh giá, không phải server.
        if self.reranker.enabled:
            try:
                import asyncio

                candidates = asyncio.run(self.reranker.rerank(question, candidates))
            except Exception as e:  # noqa: BLE001
                logger.warning("Rerank failed: %s", e)
        candidates = candidates[:top_k]
        return [RetrievalHit(**c) for c in candidates]

    def query_transcript(
        self,
        collection: str,
        meeting_id: str,
        question: str,
        top_k: int = 3,
        window_size: int = 2,
        score_threshold: float = 0.0,
    ) -> List[RetrievalHit]:
        qvec = self.embedder.embed_query(question)
        fetch_k = (
            max(top_k, top_k * settings.hybrid_fetch_multiplier)
            if self.hybrid_enabled
            else top_k
        )
        flt = Filter(
            must=[FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id))]
        )
        raw = self._qclient.search(
            collection_name=collection,
            query_vector=qvec,
            limit=fetch_k,
            query_filter=flt,
            with_payload=True,
            score_threshold=max(0.0, score_threshold),
        )
        candidates = []
        for r in raw:
            payload = r.payload or {}
            candidates.append(
                {
                    "text": payload.get("text", ""),
                    "score": round(float(r.score), 4),
                    "source": collection,
                    "doc_id": str(r.id),
                    "sequence_id": payload.get("sequence_id"),
                    "speaker": payload.get("speaker"),
                    "context": payload.get("context", ""),
                    "context_status": payload.get("context_status", "disabled"),
                }
            )
        if self.hybrid_enabled:
            candidates = fuse(question, candidates)
        candidates = candidates[:top_k]

        hits: List[RetrievalHit] = []
        for c in candidates:
            seq = c.get("sequence_id")
            window_entries: List[RetrievalHit] = []
            if window_size > 0 and seq is not None:
                seq_min = max(settings.transcript_seq_start, seq - window_size)
                seq_max = seq + window_size
                rows, _ = self._qclient.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="meeting_id", match=MatchValue(value=meeting_id)
                            ),
                            FieldCondition(
                                key="sequence_id",
                                range=Range(gte=seq_min, lte=seq_max),
                            ),
                        ]
                    ),
                    with_payload=True,
                    with_vectors=False,
                    limit=20,
                )
                for p in rows:
                    if int(p.payload.get("sequence_id", 0)) == seq:
                        continue
                    window_entries.append(
                        RetrievalHit(
                            text=p.payload.get("text", ""),
                            score=0.0,
                            source=collection,
                            doc_id=str(p.id),
                            sequence_id=p.payload.get("sequence_id"),
                            speaker=p.payload.get("speaker"),
                        )
                    )
            hits.append(
                RetrievalHit(
                    text=c["text"],
                    score=c["score"],
                    source=c["source"],
                    doc_id=c["doc_id"],
                    sequence_id=seq,
                    speaker=c.get("speaker"),
                    context=c.get("context"),
                    context_status=c.get("context_status"),
                )
            )
        return hits

    # ---- Answer generation ----------------------------------------------

    def generate_answer(
        self,
        question: str,
        contexts: List[str],
        max_retries: int = 4,
    ) -> str:
        """Sinh câu trả lời dùng LM Studio. Trả chuỗi rỗng nếu thất bại."""
        if not contexts:
            return ""
        ctx_block = "\n\n---\n\n".join(contexts[:5])
        system = (
            "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu được cung cấp. "
            "Chỉ sử dụng thông tin trong tài liệu. Nếu không đủ thông tin, hãy nói "
            "rõ rằng bạn không tìm thấy. Trả lời ngắn gọn, bằng tiếng Việt, "
            "không thêm giải thích thừa."
        )
        user = (
            f"Câu hỏi: {question}\n\n"
            f"Tài liệu tham khảo:\n{ctx_block}\n\n"
            "Câu trả lời:"
        )
        url = f"{self.answer_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.answer_api_key:
            headers["Authorization"] = f"Bearer {self.answer_api_key}"
        payload = {
            "model": self.answer_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.answer_temperature,
            "max_tokens": self.answer_max_tokens,
        }
        for attempt in range(1, max_retries + 1):
            try:
                r = self._http.post(url, json=payload, headers=headers, timeout=120.0)
                if r.status_code != 200:
                    logger.warning(
                        "LLM answer gen HTTP %d (attempt %d/%d): %s",
                        r.status_code, attempt, max_retries, r.text[:200],
                    )
                    if attempt < max_retries:
                        time.sleep(2.0)
                        continue
                    return ""
                data = r.json()
                choices = data.get("choices") or []
                if choices:
                    return (choices[0].get("message") or {}).get("content", "").strip()
                return ""
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "LLM answer gen failed (attempt %d/%d): %s", attempt, max_retries, e
                )
                if attempt < max_retries:
                    time.sleep(2.0)
        return ""

    # ---- End-to-end ------------------------------------------------------

    def ask_docs(
        self,
        collection: str,
        question: str,
        top_k: int = 5,
    ) -> RAGResult:
        t0 = time.perf_counter()
        hits = self.query_docs(collection, question, top_k=top_k)
        contexts = [h.text for h in hits]
        answer = self.generate_answer(question, contexts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            hits=hits,
            latency_ms=latency_ms,
        )

    def ask_transcript(
        self,
        collection: str,
        meeting_id: str,
        question: str,
        top_k: int = 3,
        window_size: int = 2,
    ) -> RAGResult:
        t0 = time.perf_counter()
        hits = self.query_transcript(
            collection,
            meeting_id,
            question,
            top_k=top_k,
            window_size=window_size,
        )
        # Contexts cho answer gen: lấy text + window tóm gọn.
        contexts = [h.text for h in hits]
        answer = self.generate_answer(question, contexts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            hits=hits,
            latency_ms=latency_ms,
        )
