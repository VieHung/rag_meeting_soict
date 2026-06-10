"""ContextWorker — hàng đợi FIFO build context (Phase 2, D9 + Phase 3 B1).

Phase 3 B1 thay đổi so với Phase 2:
- Thay 1 queue global bằng **dict {meeting_id: asyncio.Queue}** → nhiều cuộc họp build
  song song (giữ FIFO trong từng cuộc — D9 vẫn thoả).
- Semaphore `CONTEXT_WORKER_CONCURRENCY` giới hạn số LLM call đồng thời.
- Worker của cuộc họp idle quá `WORKER_IDLE_TTL` giây tự dọn.

Recovery scan (B1b) được thực hiện ở main.py lifespan bằng cách enqueue lại
các điểm `context_status ∈ {pending, processing}` sau khi server khởi động.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Tuple

from app.config import settings
from app.services.context_builder import ContextBuilder
from app.services.llm_client import LLMClient, get_llm_client
from app.services.transcript_store import TranscriptStore
from app.utils.redis_client import RedisClient

logger = logging.getLogger("context_worker")

# Job = (collection, meeting_id, sequence_id)
Job = Tuple[str, str, int]


class ContextWorker:
    """Per-meeting FIFO queues + global concurrency semaphore."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        redis_client: Optional[RedisClient] = None,
    ):
        self._llm = llm
        self._redis = redis_client
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._queues: Dict[str, asyncio.Queue] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    # ----- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Khởi động worker (gọi trong lifespan startup)."""
        if self._semaphore is not None:
            return
        self._llm = self._llm or get_llm_client()
        self._redis = self._redis or RedisClient()
        self._semaphore = asyncio.Semaphore(settings.context_worker_concurrency)
        self._running = True
        logger.info(
            "ContextWorker started (concurrency=%d, idle_ttl=%ds)",
            settings.context_worker_concurrency,
            settings.worker_idle_ttl,
        )

    async def stop(self, drain: bool = True) -> None:
        """Dừng worker. Nếu drain=True, đợi xử lý hết job còn trong hàng đợi."""
        self._running = False
        if drain and self._queues:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(q.join() for q in list(self._queues.values())),
                        return_exceptions=True,
                    ),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                logger.warning("ContextWorker drain timeout — cancelling tasks")
        for task in list(self._tasks.values()):
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._tasks.clear()
        self._queues.clear()
        logger.info("ContextWorker stopped")

    # ----- producer --------------------------------------------------------

    async def enqueue(self, collection: str, meeting_id: str, sequence_id: int) -> None:
        """Đẩy 1 job build context vào hàng đợi của meeting_id.

        Tắt LLM (LLM_PROVIDER=none) → bỏ qua; context_status đã là 'disabled'.
        """
        if settings.llm_provider == "none":
            return
        if self._semaphore is None:
            logger.warning("ContextWorker.enqueue called before start() — ignoring job")
            return
        if meeting_id not in self._queues:
            q: asyncio.Queue = asyncio.Queue()
            self._queues[meeting_id] = q
            self._tasks[meeting_id] = asyncio.create_task(
                self._drain(meeting_id, q),
                name=f"context-worker-{meeting_id}",
            )
        await self._queues[meeting_id].put((collection, meeting_id, sequence_id))

    # ----- consumer --------------------------------------------------------

    async def _drain(self, meeting_id: str, queue: asyncio.Queue) -> None:
        """Drain loop cho 1 meeting_id (FIFO). Tự dọn sau WORKER_IDLE_TTL giây idle."""
        idle_ttl = float(settings.worker_idle_ttl)
        while True:
            try:
                job = await asyncio.wait_for(queue.get(), timeout=idle_ttl)
            except asyncio.TimeoutError:
                # Idle hết hạn → dọn.
                self._queues.pop(meeting_id, None)
                self._tasks.pop(meeting_id, None)
                logger.debug("ContextWorker idle timeout for meeting=%s — cleaned up", meeting_id)
                return

            if job is None:  # sentinel
                queue.task_done()
                return

            collection, mid, sequence_id = job
            try:
                async with self._semaphore:
                    await self._process(collection, mid, sequence_id)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "ContextWorker job failed collection=%s meeting=%s seq=%s: %s",
                    collection, mid, sequence_id, e,
                )
            finally:
                queue.task_done()

    async def _process(self, collection: str, meeting_id: str, sequence_id: int) -> None:
        store = TranscriptStore(collection_name=collection)
        builder = ContextBuilder(store=store, llm=self._llm, redis_client=self._redis)
        await builder.build(collection, meeting_id, sequence_id)


# ----- singleton ----------------------------------------------------------

_worker_singleton: Optional[ContextWorker] = None


def get_context_worker() -> ContextWorker:
    """Singleton ContextWorker — chia sẻ LLM + Redis client."""
    global _worker_singleton
    if _worker_singleton is None:
        _worker_singleton = ContextWorker()
    return _worker_singleton
