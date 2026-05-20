"""Unit test SequenceManager (cần Redis chạy tại REDIS_HOST/PORT)."""
import asyncio
import os
import uuid

import pytest

from app.config import settings
from app.services.sequence_manager import SequenceManager
from app.utils.redis_client import RedisClient


pytestmark = pytest.mark.asyncio


def _meeting() -> str:
    return f"unit_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def manager():
    mgr = SequenceManager(RedisClient())
    yield mgr


async def test_init_and_next_returns_seq_start(manager):
    mid = _meeting()
    try:
        await manager.init(mid, "test_collection")
        first = await manager.next(mid)
        assert first == settings.transcript_seq_start
        second = await manager.next(mid)
        assert second == first + 1
    finally:
        await manager.reset(mid)


async def test_init_duplicate_raises(manager):
    mid = _meeting()
    try:
        await manager.init(mid, "test_collection")
        with pytest.raises(FileExistsError):
            await manager.init(mid, "test_collection")
    finally:
        await manager.reset(mid)


async def test_force_reset_allows_reinit(manager):
    mid = _meeting()
    try:
        await manager.init(mid, "c")
        await manager.next(mid)
        await manager.next(mid)
        # reset
        meta = await manager.init(mid, "c", force_reset=True)
        assert meta["reset"] is True
        # counter quay lại SEQ_START
        next_val = await manager.next(mid)
        assert next_val == settings.transcript_seq_start
    finally:
        await manager.reset(mid)


async def test_next_without_init_raises(manager):
    mid = _meeting()
    with pytest.raises(LookupError):
        await manager.next(mid)


async def test_concurrent_increments_are_unique(manager):
    """100 lần INCR đồng thời → 100 giá trị duy nhất, liên tục."""
    mid = _meeting()
    try:
        await manager.init(mid, "c")
        tasks = [manager.next(mid) for _ in range(100)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 100
        assert len(set(results)) == 100
        assert sorted(results) == list(
            range(settings.transcript_seq_start, settings.transcript_seq_start + 100)
        )
    finally:
        await manager.reset(mid)
