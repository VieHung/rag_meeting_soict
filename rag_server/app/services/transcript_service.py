"""TranscriptService — orchestrator cho luồng transcript (Phase 2).

Gom các nghiệp vụ: init meeting, ingest transcript, query có window,
get/update context, list segments, delete meeting.

Design Decisions:
- D2: query trả kèm window ±N câu lân cận.
- D8: ingest đồng bộ (gán seq + lưu vector), build context chạy nền.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.schemas.transcript import (
    ContextResponse,
    DeleteMeetingResponse,
    MeetingInitResponse,
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


class MeetingNotInitialized(LookupError):
    pass


class MeetingAlreadyExists(FileExistsError):
    pass


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
        # context_builder_factory(store) → ContextBuilder ; cho phép inject ở test
        self._context_builder_factory = context_builder_factory or (
            lambda store: ContextBuilder(store=store, redis_client=self._redis)
        )

    # ---- helpers ----------------------------------------------------------

    def _store(self, collection: Optional[str]) -> TranscriptStore:
        return TranscriptStore(collection_name=collection)

    def _resolve_collection(self, collection: Optional[str]) -> str:
        return collection or settings.transcript_default_collection

    def _clamp_window(self, window_size: int) -> int:
        return max(0, min(window_size, settings.transcript_max_window_size))

    # ---- meeting lifecycle ------------------------------------------------

    async def init_meeting(
        self,
        collection: Optional[str],
        meeting_id: str,
        force_reset: bool,
    ) -> MeetingInitResponse:
        col = self._resolve_collection(collection)
        # Đảm bảo collection (sync trên thread pool để không block loop).
        store = self._store(col)
        await asyncio.to_thread(store.ensure_collection)

        try:
            meta = await self._seq.init(meeting_id, col, force_reset=force_reset)
        except FileExistsError as e:
            raise MeetingAlreadyExists(str(e)) from e

        # Nếu force_reset → xóa luôn các point cũ của meeting trên Qdrant.
        if force_reset:
            await asyncio.to_thread(store.delete_meeting, meeting_id)
            # Reset cache latest_ctx.
            try:
                await self._redis.client.delete(f"meeting:{meeting_id}:latest_ctx")
            except Exception:
                pass

        return MeetingInitResponse(
            meeting_id=meeting_id,
            collection=col,
            status="reset" if meta.get("reset") else "initialized",
            seq_counter=int(meta.get("seq_counter", 0)),
        )

    async def delete_meeting(
        self, collection: Optional[str], meeting_id: str
    ) -> DeleteMeetingResponse:
        col = self._resolve_collection(collection)
        store = self._store(col)
        if not await self._seq.exists(meeting_id) and not store.collection_exists():
            return DeleteMeetingResponse(
                success=False, meeting_id=meeting_id, message="Meeting not found"
            )
        deleted = 0
        if store.collection_exists():
            deleted = await asyncio.to_thread(store.delete_meeting, meeting_id)
        await self._seq.reset(meeting_id)
        return DeleteMeetingResponse(
            success=True,
            meeting_id=meeting_id,
            deleted_points=int(deleted),
            message=f"Deleted meeting '{meeting_id}'",
        )

    # ---- ingest -----------------------------------------------------------

    async def embed_transcript(
        self,
        collection: Optional[str],
        request: TranscriptEmbedRequest,
    ) -> Tuple[TranscriptEmbedResponse, ContextBuilder, int]:
        """Lưu 1 câu transcript đồng bộ. Trả thêm builder để router schedule background."""
        text = (request.text or "").strip()
        if not text:
            raise ValueError("text must not be empty")
        if not await self._seq.exists(request.meeting_id):
            raise MeetingNotInitialized(
                f"Meeting '{request.meeting_id}' not initialized"
            )

        col = self._resolve_collection(collection)
        store = self._store(col)
        await asyncio.to_thread(store.ensure_collection)

        # Atomic seq.
        sequence_id = await self._seq.next(request.meeting_id)

        # Embed & upsert (chạy trên thread để không block loop).
        vector = await asyncio.to_thread(self._embedder.embed_query, text)
        timestamp = request.timestamp or datetime.now(timezone.utc)
        point_id = await asyncio.to_thread(
            store.upsert_point,
            vector=vector,
            meeting_id=request.meeting_id,
            sequence_id=sequence_id,
            speaker=request.speaker,
            speaker_id=request.speaker_id,
            text=text,
            timestamp=timestamp,
            lang=request.lang,
            context="",
            context_status="pending",
            context_seq_base=None,
        )

        builder = self._context_builder_factory(store)
        return (
            TranscriptEmbedResponse(
                meeting_id=request.meeting_id,
                sequence_id=sequence_id,
                point_id=point_id,
                context_status="pending",
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
        if not store.collection_exists():
            return TranscriptQueryResponse(query=request.query, results=[], count=0)

        window_size = self._clamp_window(request.window_size)

        vector = await asyncio.to_thread(self._embedder.embed_query, request.query)
        raw = await asyncio.to_thread(
            store.search,
            vector,
            request.top_k,
            request.meeting_id,
            request.speaker_filter,
            request.speaker_id_filter,
            request.score_threshold,
        )

        results: List[TranscriptQueryResult] = []
        for r in raw:
            seq = int(r.get("sequence_id", 0))
            meeting_id = r.get("meeting_id") or (request.meeting_id or "")
            window: Optional[WindowResult] = None
            if window_size > 0 and meeting_id:
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

    # ---- context endpoints ------------------------------------------------

    async def get_latest_context(
        self, collection: Optional[str], meeting_id: str
    ) -> ContextResponse:
        # Ưu tiên Redis cache.
        try:
            raw = await self._redis.client.get(f"meeting:{meeting_id}:latest_ctx")
        except Exception:
            raw = None
        if raw:
            try:
                cached = json.loads(raw)
                return ContextResponse(
                    meeting_id=meeting_id,
                    sequence_id=int(cached.get("sequence_id", 0)),
                    context=cached.get("context", "") or "",
                    context_status=cached.get("context_status", "pending"),
                )
            except Exception:
                pass

        # Fallback: tìm câu có sequence_id lớn nhất trên Qdrant.
        store = self._store(collection)
        if not store.collection_exists():
            raise MeetingNotInitialized(f"Meeting '{meeting_id}' not found")
        current = await self._seq.current(meeting_id)
        if current <= 0:
            return ContextResponse(
                meeting_id=meeting_id,
                sequence_id=0,
                context="",
                context_status="pending",
            )
        point = await asyncio.to_thread(store.find_by_seq, meeting_id, current)
        if point is None:
            return ContextResponse(
                meeting_id=meeting_id,
                sequence_id=current,
                context="",
                context_status="pending",
            )
        _, payload = point
        return ContextResponse(
            meeting_id=meeting_id,
            sequence_id=current,
            context=payload.get("context", "") or "",
            context_status=payload.get("context_status", "pending"),
            context_seq_base=payload.get("context_seq_base"),
        )

    async def get_context_at(
        self,
        collection: Optional[str],
        meeting_id: str,
        sequence_id: int,
    ) -> Optional[ContextResponse]:
        store = self._store(collection)
        if not store.collection_exists():
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

    async def update_context(
        self,
        collection: Optional[str],
        sequence_id: int,
        meeting_id: str,
        context: str,
        context_status: str,
        context_seq_base: Optional[int],
    ) -> ContextResponse:
        store = self._store(collection)
        if not store.collection_exists():
            raise MeetingNotInitialized(f"Meeting '{meeting_id}' not found")
        point = await asyncio.to_thread(store.find_by_seq, meeting_id, sequence_id)
        if point is None:
            raise LookupError(
                f"Point not found meeting={meeting_id} seq={sequence_id}"
            )
        point_id, _ = point
        await asyncio.to_thread(
            store.update_context,
            point_id,
            context=context,
            context_status=context_status,
            context_seq_base=context_seq_base,
        )
        # Cập nhật Redis cache nếu là câu mới nhất.
        current = await self._seq.current(meeting_id)
        if sequence_id == current:
            try:
                import json

                await self._redis.client.set(
                    f"meeting:{meeting_id}:latest_ctx",
                    json.dumps(
                        {
                            "sequence_id": sequence_id,
                            "context": context,
                            "context_status": context_status,
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception:
                pass
        return ContextResponse(
            meeting_id=meeting_id,
            sequence_id=sequence_id,
            context=context,
            context_status=context_status,
            context_seq_base=context_seq_base,
        )

    # ---- segments ---------------------------------------------------------

    async def list_segments(
        self,
        collection: Optional[str],
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
            current = await self._seq.current(meeting_id)
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
