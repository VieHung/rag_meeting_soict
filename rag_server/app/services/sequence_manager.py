"""SequenceManager — atomic sequence_id per meeting (Redis-backed).

Design Decision D1: server gán sequence_id, dùng atomic INCR.
Design Decision D5: state lưu trên Redis.

Key layout:
- meeting:{meeting_id}:seq_counter  → INTEGER (atomic INCR)
- meeting:{meeting_id}:meta         → JSON {created_at, collection, status}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings
from app.utils.redis_client import RedisClient


def _seq_key(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:seq_counter"


def _meta_key(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:meta"


class SequenceManager:
    """Quản lý sequence_id atomic + metadata cuộc họp."""

    def __init__(self, redis_client: Optional[RedisClient] = None):
        self._redis = redis_client or RedisClient()

    # ---- meta -------------------------------------------------------------

    async def init(
        self,
        meeting_id: str,
        collection: str,
        force_reset: bool = False,
    ) -> Dict[str, Any]:
        """Khởi tạo cuộc họp mới.

        - Nếu meeting đã tồn tại và `force_reset=False` → raise FileExistsError.
        - Counter được set về `TRANSCRIPT_SEQ_START - 1` để lần `next()` đầu tiên
          trả đúng `TRANSCRIPT_SEQ_START`.
        """
        client = self._redis.client
        meta_key = _meta_key(meeting_id)

        existed = await client.exists(meta_key)
        if existed and not force_reset:
            raise FileExistsError(f"Meeting '{meeting_id}' already initialized")

        meta = {
            "meeting_id": meeting_id,
            "collection": collection,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # counter base = SEQ_START - 1 để INCR đầu tiên cho ra SEQ_START
        base = max(0, settings.transcript_seq_start - 1)

        async with client.pipeline(transaction=True) as pipe:
            pipe.set(meta_key, json.dumps(meta))
            pipe.set(_seq_key(meeting_id), base)
            await pipe.execute()

        return {**meta, "seq_counter": base, "reset": bool(existed)}

    async def get_meta(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        raw = await self._redis.client.get(_meta_key(meeting_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def exists(self, meeting_id: str) -> bool:
        return bool(await self._redis.client.exists(_meta_key(meeting_id)))

    # ---- counter ----------------------------------------------------------

    async def next(self, meeting_id: str) -> int:
        """Atomic INCR. Yêu cầu meeting đã `init`."""
        if not await self.exists(meeting_id):
            raise LookupError(f"Meeting '{meeting_id}' is not initialized")
        return int(await self._redis.client.incr(_seq_key(meeting_id)))

    async def current(self, meeting_id: str) -> int:
        """Giá trị counter hiện tại (chưa INCR). Trả 0 nếu chưa có."""
        raw = await self._redis.client.get(_seq_key(meeting_id))
        if raw is None:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    async def reset(self, meeting_id: str) -> None:
        """Reset cả counter và meta (dùng khi xóa meeting)."""
        client = self._redis.client
        async with client.pipeline(transaction=True) as pipe:
            pipe.delete(_seq_key(meeting_id))
            pipe.delete(_meta_key(meeting_id))
            pipe.delete(f"meeting:{meeting_id}:latest_ctx")
            await pipe.execute()
