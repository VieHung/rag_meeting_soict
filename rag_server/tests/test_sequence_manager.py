"""Unit test SequenceManager — khớp thiết kế v2 (lazy-init + self-healing).

API thực tế (app/services/sequence_manager.py):
- next(collection)    : atomic INCR, lazy-init, trả int. Lần đầu = transcript_seq_start.
- current(collection) : giá trị hiện tại, rebuild từ Qdrant nếu Redis miss.
- counter key = rag:seq:{collection}, TTL refresh mỗi lần next.

KHÔNG còn init()/reset()/next(meeting_id) như bản cũ — đã chuyển sang lazy-init
keyed theo collection (D1 + phase2plan_v2.md mục 5.3).

Cần Redis chạy (REDIS_HOST/PORT). Test rebuild cần cả Qdrant. Thiếu → skip.
"""
import asyncio
import uuid

import pytest

from app.config import settings
from app.services.sequence_manager import SequenceManager, _seq_key
from app.utils.redis_client import RedisClient

pytestmark = pytest.mark.asyncio


def _collection() -> str:
    return f"meeting-unittest-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def redis_up():
    client = RedisClient()
    try:
        ok = await client.ping()
    except Exception:
        ok = False
    if not ok:
        pytest.skip("Redis không khả dụng — bỏ qua test SequenceManager")
    return client


@pytest.fixture
async def manager(redis_up):
    yield SequenceManager(redis_up)


async def _cleanup(redis_client: RedisClient, collection: str):
    try:
        await redis_client.client.delete(_seq_key(collection))
    except Exception:
        pass


async def test_next_lazy_init_starts_at_seq_start(manager, redis_up):
    col = _collection()
    try:
        first = await manager.next(col)
        assert first == settings.transcript_seq_start
        second = await manager.next(col)
        assert second == first + 1
    finally:
        await _cleanup(redis_up, col)


async def test_current_reflects_last_next(manager, redis_up):
    col = _collection()
    try:
        await manager.next(col)
        await manager.next(col)
        cur = await manager.current(col)
        assert cur == settings.transcript_seq_start + 1
    finally:
        await _cleanup(redis_up, col)


async def test_concurrent_increments_are_unique(manager, redis_up):
    """100 lần next đồng thời → 100 giá trị duy nhất, liên tục."""
    col = _collection()
    try:
        tasks = [manager.next(col) for _ in range(100)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 100
        assert len(set(results)) == 100
        start = settings.transcript_seq_start
        assert sorted(results) == list(range(start, start + 100))
    finally:
        await _cleanup(redis_up, col)


async def test_rebuild_from_qdrant_after_redis_flush(manager, redis_up):
    """Xóa key Redis giữa chừng → next rebuild đúng từ max sequence_id trên Qdrant."""
    from app.services.transcript_store import TranscriptStore

    col = _collection()
    meeting_id = col.removeprefix("meeting-")
    store = TranscriptStore(collection_name=col)
    try:
        try:
            store.ensure_collection()
        except Exception:
            pytest.skip("Qdrant không khả dụng — bỏ qua test rebuild")

        dummy = [0.0] * settings.embedding_dim
        for seq in (1, 2, 3):
            store.upsert_point(
                vector=dummy,
                meeting_id=meeting_id,
                sequence_id=seq,
                speaker="tester",
                text=f"câu {seq}",
                timestamp=None,
            )

        # Mô phỏng Redis bị flush.
        await redis_up.client.delete(_seq_key(col))

        # next phải rebuild từ Qdrant (max=3) rồi INCR → 4.
        nxt = await manager.next(col)
        assert nxt == 4
    finally:
        await _cleanup(redis_up, col)
        try:
            store.client.delete_collection(col)
        except Exception:
            pass
