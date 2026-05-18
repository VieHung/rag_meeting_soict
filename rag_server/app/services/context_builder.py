"""ContextBuilder — build cumulative context using LLM (background task).

Theo D3 (plan.md):
    context[N] = LLM_summarize(context[N-1] + transcript[N-1])
Tức context tại câu N là *bối cảnh dẫn tới câu N* — chưa bao gồm chính câu N.

Luồng (mục 8.2 plan):
    1. Lấy `(prev_context, prev_text)` từ Redis hoặc Qdrant.
    2. Đánh dấu point[N].context_status = "processing".
    3. Gọi LLM.summarize(prev_context, prev_text).
    4. Update Qdrant payload + Redis `latest_ctx`.
    5. Retry tối đa CONTEXT_MAX_RETRY lần. Vẫn lỗi → fallback context = prev_context,
       status = "failed".

Câu đầu tiên (sequence_id == TRANSCRIPT_SEQ_START):
    → không có câu trước, set context = "", status = "ready" mà không gọi LLM.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional, Tuple

from app.config import settings
from app.services.llm_client import LLMClient, LLMError, get_llm_client
from app.services.transcript_store import TranscriptStore
from app.utils.redis_client import RedisClient

logger = logging.getLogger("context_builder")


def _latest_ctx_key(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:latest_ctx"


class ContextBuilder:
    def __init__(
        self,
        store: TranscriptStore,
        llm: Optional[LLMClient] = None,
        redis_client: Optional[RedisClient] = None,
    ):
        self._store = store
        self._llm = llm or get_llm_client()
        self._redis = redis_client or RedisClient()

    # ----- public ----------------------------------------------------------

    async def build(self, meeting_id: str, sequence_id: int) -> None:
        """Background entrypoint — không raise ra ngoài.

        Args:
            meeting_id: ID cuộc họp.
            sequence_id: sequence_id của câu vừa được lưu (= N).
        """
        try:
            await self._build_inner(meeting_id, sequence_id)
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "ContextBuilder.build failed meeting=%s seq=%s: %s",
                meeting_id,
                sequence_id,
                e,
            )

    # ----- core ------------------------------------------------------------

    async def _build_inner(self, meeting_id: str, sequence_id: int) -> None:
        # Câu đầu cuộc họp: không có previous → context rỗng, ready ngay.
        if sequence_id <= settings.transcript_seq_start:
            await self._mark_ready_empty(meeting_id, sequence_id)
            return

        # Tìm point hiện tại để biết point_id cần update.
        current = await asyncio.to_thread(
            self._store.find_by_seq, meeting_id, sequence_id
        )
        if current is None:
            logger.warning(
                "ContextBuilder: point not found meeting=%s seq=%s",
                meeting_id, sequence_id,
            )
            return
        point_id, _payload = current

        # Lấy (prev_context, prev_text) từ Redis (nhanh) → fallback Qdrant.
        prev_context, prev_text = await self._load_previous(meeting_id, sequence_id)
        if prev_text is None:
            logger.warning(
                "ContextBuilder: previous transcript missing meeting=%s seq=%s (need %s)",
                meeting_id, sequence_id, sequence_id - 1,
            )
            # Fallback: status failed, giữ context cũ (rỗng nếu không có).
            await asyncio.to_thread(
                self._store.update_context,
                point_id,
                context=(prev_context or ""),
                context_status="failed",
                context_seq_base=sequence_id - 1,
            )
            return

        # Mark processing
        await asyncio.to_thread(
            self._store.update_context,
            point_id,
            context="",
            context_status="processing",
            context_seq_base=sequence_id - 1,
        )

        new_context = await self._summarize_with_retry(prev_context, prev_text)
        if new_context is None:
            # Fallback giữ nguyên context trước đó.
            await asyncio.to_thread(
                self._store.update_context,
                point_id,
                context=prev_context or "",
                context_status="failed",
                context_seq_base=sequence_id - 1,
            )
            return

        # Update Qdrant + Redis cache.
        await asyncio.to_thread(
            self._store.update_context,
            point_id,
            context=new_context,
            context_status="ready",
            context_seq_base=sequence_id - 1,
        )
        await self._save_latest(meeting_id, sequence_id, new_context, "ready")

    # ----- helpers ---------------------------------------------------------

    async def _mark_ready_empty(self, meeting_id: str, sequence_id: int) -> None:
        current = await asyncio.to_thread(
            self._store.find_by_seq, meeting_id, sequence_id
        )
        if current is None:
            return
        point_id, _ = current
        await asyncio.to_thread(
            self._store.update_context,
            point_id,
            context="",
            context_status="ready",
            context_seq_base=None,
        )
        await self._save_latest(meeting_id, sequence_id, "", "ready")

    async def _load_previous(
        self, meeting_id: str, sequence_id: int
    ) -> Tuple[str, Optional[str]]:
        """Trả (prev_context, prev_text). prev_text=None nếu không có."""
        prev_seq = sequence_id - 1
        # Đọc cache latest_ctx — chứa context của câu prev_seq nếu vừa build xong.
        prev_context = ""
        try:
            raw = await self._redis.client.get(_latest_ctx_key(meeting_id))
        except Exception:
            raw = None
        if raw:
            try:
                cached = json.loads(raw)
                if int(cached.get("sequence_id", -1)) == prev_seq:
                    prev_context = cached.get("context", "") or ""
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Lấy text của câu prev_seq từ Qdrant (luôn cần).
        prev_point = await asyncio.to_thread(
            self._store.find_by_seq, meeting_id, prev_seq
        )
        if prev_point is None:
            return prev_context, None
        _, prev_payload = prev_point
        # Nếu cache miss, dùng context của câu N-1 lưu trên Qdrant (nếu đã ready).
        if not prev_context:
            prev_context = prev_payload.get("context", "") or ""
        return prev_context, prev_payload.get("text", "") or ""

    async def _save_latest(
        self,
        meeting_id: str,
        sequence_id: int,
        context: str,
        status: str,
    ) -> None:
        try:
            await self._redis.client.set(
                _latest_ctx_key(meeting_id),
                json.dumps(
                    {
                        "sequence_id": sequence_id,
                        "context": context,
                        "context_status": status,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to cache latest_ctx for %s: %s", meeting_id, e)

    async def _summarize_with_retry(
        self, prev_context: str, prev_text: str
    ) -> Optional[str]:
        attempts = max(1, settings.context_max_retry + 1)
        last_err: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                result = await self._llm.summarize(prev_context, prev_text)
                return (result or "").strip()
            except LLMError as e:
                last_err = e
                logger.warning(
                    "LLM summarize failed attempt=%d/%d: %s", attempt, attempts, e
                )
                if attempt < attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 5))
        logger.error("LLM summarize gave up after %d attempts: %s", attempts, last_err)
        return None
