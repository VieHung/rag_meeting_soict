"""SequenceManager — atomic sequence_id per collection (Redis-backed).

- Key layout: rag:seq:{collection} → INTEGER
- Lazy init: tự khởi tạo ở lần /embed đầu tiên
- Self-healing: rebuild từ Qdrant nếu Redis miss
- TTL: 7 ngày, tự refresh mỗi lần embed

Design Decision D1: server gán sequence_id, dùng atomic INCR.
Design Decision D5: state lưu trên Redis.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.services.transcript_store import TranscriptStore
from app.utils.redis_client import RedisClient

logger = logging.getLogger("sequence_manager")


def _seq_key(collection: str) -> str:
    return f"rag:seq:{collection}"


class SequenceManager:
    """Quản lý sequence_id atomic theo collection (lazy init + self-healing)."""

    def __init__(self, redis_client: Optional[RedisClient] = None):
        self._redis = redis_client or RedisClient()

    async def next(self, collection: str) -> int:
        """Atomic INCR với lazy init và self-healing."""
        client = self._redis.client
        seq_key = _seq_key(collection)

        try:
            exists = await client.exists(seq_key)
            if not exists:
                await self._rebuild_from_qdrant(collection)
            new_val = await client.incr(seq_key)
            await client.expire(seq_key, settings.seq_key_ttl_seconds)
            return int(new_val)
        except Exception as e:
            logger.warning("Redis INCR failed: %s, trying to rebuild from Qdrant", e)
            return await self._rebuild_from_qdrant(collection)

    async def current(self, collection: str) -> int:
        """Giá trị counter hiện tại. Nếu miss → rebuild từ Qdrant."""
        client = self._redis.client
        seq_key = _seq_key(collection)

        raw = await client.get(seq_key)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 0

        return await self._rebuild_from_qdrant(collection)

    async def _rebuild_from_qdrant(self, collection: str) -> int:
        """Rebuild counter từ Qdrant (lấy max sequence_id của meeting đúng)."""
        try:
            store = TranscriptStore(collection_name=collection)
            max_seq = store.get_max_sequence_id()
            if max_seq is None:
                max_seq = settings.transcript_seq_start - 1
            client = self._redis.client
            seq_key = _seq_key(collection)
            await client.set(seq_key, max_seq)
            await client.expire(seq_key, settings.seq_key_ttl_seconds)
            logger.info("Rebuilt sequence counter for %s: %s", collection, max_seq)
            return int(max_seq)
        except Exception as e:
            logger.error("Failed to rebuild from Qdrant: %s", e)
            return settings.transcript_seq_start

    async def exists(self, collection: str) -> bool:
        """Kiểm tra collection có dữ liệu không (qua counter hoặc Qdrant)."""
        client = self._redis.client
        seq_key = _seq_key(collection)

        if await client.exists(seq_key):
            return True

        store = TranscriptStore(collection_name=collection)
        return store.collection_exists() and store.get_max_sequence_id() is not None
