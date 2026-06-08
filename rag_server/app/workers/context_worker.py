"""ContextWorker — hàng đợi FIFO build context (Phase 2, D9).

Vì sao cần file này:
    `POST /transcript/{collection}/embed` trước đây đăng ký build context qua
    FastAPI `BackgroundTasks`. BackgroundTasks **không đảm bảo thứ tự** giữa các
    request → khi nhiều câu đến gần đồng thời, `context[N]` có thể được build
    trước `context[N-1]`, phá định nghĩa:

        context[N] = LLM_summarize(context[N-1] + transcript[N-1])

    (phase2plan_v2.md, mục 8.2 + Design Decision D9).

Giải pháp:
    Một `asyncio.Queue` + đúng **một** worker task lấy job tuần tự → strict
    global FIFO. Vì `context[N-1]` luôn được xử lý xong (ready/failed) trước
    `context[N]` của cùng collection, D9 được đảm bảo. Server phòng họp serialize
    toàn cục là chấp nhận được (LLM build là điểm nghẽn, không nên chạy song song).

Worker tái dùng nguyên `ContextBuilder` (app/services/context_builder.py) —
KHÔNG sửa logic build. Mỗi job chỉ cần `(collection, meeting_id, sequence_id)`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

from app.config import settings
from app.services.context_builder import ContextBuilder
from app.services.llm_client import LLMClient, get_llm_client
from app.services.transcript_store import TranscriptStore
from app.utils.redis_client import RedisClient

logger = logging.getLogger("context_worker")

# Job = (collection, meeting_id, sequence_id)
Job = Tuple[str, str, int]


class ContextWorker:
    """Hàng đợi FIFO + 1 worker task build context tuần tự."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        redis_client: Optional[RedisClient] = None,
        maxsize: int = 0,
    ):
        self._queue: asyncio.Queue[Optional[Job]] = asyncio.Queue(maxsize=maxsize)
        self._llm = llm
        self._redis = redis_client
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ----- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Khởi động worker task (gọi trong lifespan startup)."""
        if self._task is not None:
            return
        # Khởi tạo client dùng chung khi start (đã có event loop).
        self._llm = self._llm or get_llm_client()
        self._redis = self._redis or RedisClient()
        self._running = True
        self._task = asyncio.create_task(self._drain(), name="context-worker")
        logger.info("ContextWorker started")

    async def stop(self, drain: bool = True) -> None:
        """Dừng worker. Nếu drain=True, đợi xử lý hết job còn trong hàng đợi."""
        if self._task is None:
            return
        self._running = False
        if drain:
            # Đợi các job đã enqueue được xử lý xong.
            try:
                await asyncio.wait_for(self._queue.join(), timeout=30)
            except asyncio.TimeoutError:
                logger.warning("ContextWorker drain timeout — cancelling")
        # Đẩy sentinel để worker thoát vòng lặp nếu đang chờ get().
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None
        logger.info("ContextWorker stopped")

    # ----- producer --------------------------------------------------------

    async def enqueue(self, collection: str, meeting_id: str, sequence_id: int) -> None:
        """Đẩy 1 job build context vào hàng đợi.

        Nếu LLM tắt (LLM_PROVIDER=none) thì bỏ qua — context_status đã được set
        'disabled' lúc upsert, không có gì để build.
        """
        if settings.llm_provider == "none":
            return
        await self._queue.put((collection, meeting_id, sequence_id))

    # ----- consumer --------------------------------------------------------

    async def _drain(self) -> None:
        while True:
            job = await self._queue.get()
            if job is None:  # sentinel
                self._queue.task_done()
                if not self._running:
                    return
                continue
            collection, meeting_id, sequence_id = job
            try:
                await self._process(collection, meeting_id, sequence_id)
            except Exception as e:  # noqa: BLE001 — worker không bao giờ chết vì 1 job
                logger.exception(
                    "ContextWorker job failed collection=%s meeting=%s seq=%s: %s",
                    collection, meeting_id, sequence_id, e,
                )
            finally:
                self._queue.task_done()

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
