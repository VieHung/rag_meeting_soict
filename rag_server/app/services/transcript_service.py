"""TranscriptService — orchestrator cho luồng transcript (Phase 2) - Bản v2.

Gom các nghiệp vụ: embed transcript, query có window, get context, list segments.

Theo phase2plan_v2.md:
- meeting_id suy ra từ collection (prefix meeting-)
- Lazy init: tự khởi tạo ở lần embed đầu
- Context lưu trong metadata vector (không Redis)
- Endpoint gọp: context latest + by sequence_id

Design Decisions:
- D2: query trả kèm window ±N câu lân cận.
- D8: ingest đồng bộ (gán seq + lưu vector), build context chạy nền.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from app.config import settings
from app.schemas.transcript import (
    ContextResponse,
    SegmentEntry,
    SegmentListResponse,
    TranscriptEmbedRequest,
    TranscriptEmbedResponse,
    TranscriptQueryRequest,
    TranscriptQueryResponse,
    TranscriptQueryResult,
    WindowEntry,
    WindowResult,
)
from app.services.context_builder import ContextBuilder
from app.services.embedding import EmbeddingService
from app.services.sequence_manager import SequenceManager
from app.services.transcript_store import TranscriptStore
from app.utils.redis_client import RedisClient

logger = logging.getLogger("transcript_service")

MEETING_PREFIX = "meeting-"


class InvalidCollectionPrefix(Exception):
    pass


def _to_iso(dt: Optional[Any]) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    if isinstance(dt, str):
        return dt
    return None


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


class TranscriptService:
    def __init__(
        self,
        embedder: Optional[EmbeddingService] = None,
        sequence_manager: Optional[SequenceManager] = None,
        context_builder_factory=None,
        redis_client: Optional[RedisClient] = None,
    ):
        self._embedder = embedder or EmbeddingService()
        self._seq = sequence_manager or SequenceManager(redis_client)
        self._redis = redis_client or RedisClient()
        self._context_builder_factory = context_builder_factory or (
            lambda store: ContextBuilder(store=store, redis_client=self._redis)
        )

    # ---- helpers ----------------------------------------------------------

    def _store(self, collection: str) -> TranscriptStore:
        return TranscriptStore(collection_name=collection)

    def _resolve_collection(self, collection: Optional[str]) -> str:
        if collection:
            return collection
        return f"{MEETING_PREFIX}{settings.transcript_default_collection}"

    def _clamp_window(self, window_size: int) -> int:
        return max(0, min(window_size, settings.transcript_max_window_size))

    # ---- ingest -----------------------------------------------------------

    async def embed_transcript(
        self,
        collection: str,
        meeting_id: str,
        request: TranscriptEmbedRequest,
    ) -> Tuple[TranscriptEmbedResponse, ContextBuilder, int]:
        """Lưu 1 câu transcript đồng bộ (lazy init)."""
        text = (request.text or "").strip()
        if not text:
            raise ValueError("text must not be empty")

        col = self._resolve_collection(collection)
        store = self._store(col)
        await asyncio.to_thread(store.ensure_collection)

        sequence_id = await self._seq.next(col)

        # Câu transcript là PASSAGE (đối tượng được search tới), không phải query
        # → dùng embed_texts để nhận đúng prefix "passage:" cho model E5.
        vector = (await asyncio.to_thread(self._embedder.embed_texts, [text]))[0]
        timestamp = request.timestamp or datetime.now(timezone.utc)

        context_status = "pending"
        if settings.llm_provider == "none":
            context_status = "disabled"

        point_id = await asyncio.to_thread(
            store.upsert_point,
            vector=vector,
            meeting_id=meeting_id,
            sequence_id=sequence_id,
            speaker=request.speaker,
            speaker_id=request.speaker_id,
            text=text,
            timestamp=timestamp,
            lang=request.lang,
            context="",
            context_status=context_status,
            context_seq_base=None,
        )

        builder = self._context_builder_factory(store)
        return (
            TranscriptEmbedResponse(
                meeting_id=meeting_id,
                sequence_id=sequence_id,
                point_id=point_id,
                context_status=context_status,
            ),
            builder,
            sequence_id,
        )

    # ---- query + window ---------------------------------------------------

    async def query_transcript(
        self, request: TranscriptQueryRequest
    ) -> TranscriptQueryResponse:
        if not (request.query or "").strip():
            raise ValueError("query must not be empty")

        col = self._resolve_collection(request.collection)
        store = self._store(col)

        meeting_id = col.replace(MEETING_PREFIX, "")

        if not store.collection_exists():
            return TranscriptQueryResponse(query=request.query, results=[], count=0)

        window_size = self._clamp_window(request.window_size)

        vector = await asyncio.to_thread(self._embedder.embed_query, request.query)
        raw = await asyncio.to_thread(
            store.search,
            vector,
            request.top_k,
            meeting_id,
            request.speaker_filter,
            request.speaker_id_filter,
            request.score_threshold,
        )

        results: List[TranscriptQueryResult] = []
        for r in raw:
            seq = int(r.get("sequence_id", 0))
            window: Optional[WindowResult] = None
            if window_size > 0:
                window = await self._fetch_window(
                    store, meeting_id, seq, window_size
                )

            results.append(
                TranscriptQueryResult(
                    sequence_id=seq,
                    speaker=r.get("speaker", ""),
                    speaker_id=r.get("speaker_id"),
                    timestamp=_parse_dt(r.get("timestamp")),
                    text=r.get("text", ""),
                    score=float(r.get("score", 0.0)),
                    meeting_id=meeting_id,
                    context=(r.get("context") if request.include_context else None),
                    context_status=(
                        r.get("context_status") if request.include_context else None
                    ),
                    window=window,
                )
            )

        return TranscriptQueryResponse(
            query=request.query, results=results, count=len(results)
        )

    async def _fetch_window(
        self,
        store: TranscriptStore,
        meeting_id: str,
        center_seq: int,
        window_size: int,
    ) -> WindowResult:
        seq_min = max(settings.transcript_seq_start, center_seq - window_size)
        seq_max = center_seq + window_size
        rows = await asyncio.to_thread(
            store.scroll_window, meeting_id, seq_min, seq_max
        )
        before: List[WindowEntry] = []
        after: List[WindowEntry] = []
        for row in rows:
            seq = int(row.get("sequence_id", 0))
            if seq == center_seq:
                continue
            entry = WindowEntry(
                sequence_id=seq,
                speaker=row.get("speaker", ""),
                speaker_id=row.get("speaker_id"),
                timestamp=_parse_dt(row.get("timestamp")),
                text=row.get("text", ""),
            )
            if seq < center_seq:
                before.append(entry)
            else:
                after.append(entry)
        before.sort(key=lambda e: e.sequence_id)
        after.sort(key=lambda e: e.sequence_id)
        return WindowResult(before=before, after=after)

    # ---- context endpoint (merged) ----------------------------------------

    async def get_context(
        self,
        collection: str,
        meeting_id: str,
        sequence_id: Optional[int] = None,
    ) -> Optional[ContextResponse]:
        """Lấy context (mới nhất nếu sequence_id=None, hoặc tại sequence_id cụ thể)."""
        col = self._resolve_collection(collection)
        store = self._store(col)

        if not store.collection_exists():
            return None

        if sequence_id is None:
            sequence_id = await self._seq.current(col)
            if sequence_id <= 0:
                return None

        point = await asyncio.to_thread(store.find_by_seq, meeting_id, sequence_id)
        if point is None:
            return None
        _, payload = point
        return ContextResponse(
            meeting_id=meeting_id,
            sequence_id=sequence_id,
            context=payload.get("context", "") or "",
            context_status=payload.get("context_status", "pending"),
            context_seq_base=payload.get("context_seq_base"),
        )

    # ---- segments ---------------------------------------------------------

    async def list_segments(
        self,
        collection: str,
        meeting_id: str,
        from_seq: int,
        to_seq: Optional[int],
        limit: int,
    ) -> SegmentListResponse:
        col = self._resolve_collection(collection)
        store = self._store(col)

        if not store.collection_exists():
            return SegmentListResponse(
                meeting_id=meeting_id,
                collection=col,
                from_seq=from_seq,
                to_seq=to_seq or 0,
                count=0,
                segments=[],
            )

        if to_seq is None or to_seq <= 0:
            current = await self._seq.current(col)
            to_seq = current

        rows = await asyncio.to_thread(
            store.scroll_window, meeting_id, from_seq, to_seq
        )
        rows = rows[:limit]
        segments = [
            SegmentEntry(
                sequence_id=int(r.get("sequence_id", 0)),
                speaker=r.get("speaker", ""),
                speaker_id=r.get("speaker_id"),
                timestamp=_parse_dt(r.get("timestamp")),
                text=r.get("text", ""),
                context_status=r.get("context_status"),
            )
            for r in rows
        ]
        return SegmentListResponse(
            meeting_id=meeting_id,
            collection=col,
            from_seq=from_seq,
            to_seq=to_seq,
            count=len(segments),
            segments=segments,
        )
