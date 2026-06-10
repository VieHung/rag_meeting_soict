import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import embed, query, transcript
from app.services.embedding import EmbeddingService
from app.services.llm_client import get_llm_client, shutdown_llm_client
from app.services.reranker import get_reranker
from app.services.vector_store import QdrantService
from app.utils.redis_client import RedisClient
from app.workers.context_worker import ContextWorker, get_context_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rag_server")


async def _recovery_scan(worker: ContextWorker) -> None:
    """B1b: Re-enqueue điểm context_status ∈ {pending, processing} sau restart."""
    from app.services.transcript_store import TranscriptStore
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        store = TranscriptStore()
        collections = store.client.get_collections().collections
        prefix = settings.transcript_collection_prefix
        shared = settings.transcript_shared_collection

        scan_cols = [
            c.name for c in collections
            if c.name.startswith(prefix) or c.name == shared
        ]
        if not scan_cols:
            logger.info("Recovery scan: no transcript collections found")
            return

        total = 0
        for col_name in scan_cols:
            for status_val in ("pending", "processing"):
                flt = Filter(must=[
                    FieldCondition(key="context_status", match=MatchValue(value=status_val))
                ])
                offset = None
                while True:
                    points, offset = store.client.scroll(
                        collection_name=col_name,
                        scroll_filter=flt,
                        with_payload=True,
                        with_vectors=False,
                        limit=200,
                        offset=offset,
                    )
                    for p in points:
                        payload = p.payload or {}
                        meeting_id = payload.get("meeting_id")
                        sequence_id = payload.get("sequence_id")
                        if meeting_id and sequence_id is not None:
                            collection = f"{settings.transcript_collection_prefix}{meeting_id}"
                            await worker.enqueue(collection, meeting_id, int(sequence_id))
                            total += 1
                    if not offset:
                        break

        if total:
            logger.info("Recovery scan: re-enqueued %d pending/processing job(s)", total)
        else:
            logger.info("Recovery scan: no pending jobs found")
    except Exception as e:  # noqa: BLE001
        logger.warning("Recovery scan failed (non-fatal): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading embedding model...")
    EmbeddingService()
    logger.info("Embedding model ready")

    logger.info("Connecting to Qdrant...")
    QdrantService()
    logger.info("Qdrant ready")

    logger.info("Connecting to Redis...")
    redis_client = RedisClient()
    ok = await redis_client.ping()
    logger.info("Redis ready: %s", ok)

    logger.info("Initializing LLM client...")
    llm = get_llm_client()
    logger.info("LLM client ready (provider=%s, model=%s)", llm.provider, llm.model)

    reranker = get_reranker()
    reranker.warmup()
    logger.info(
        "Reranker ready (provider=%s, hybrid_enabled=%s)",
        settings.rerank_provider,
        settings.hybrid_enabled,
    )

    worker = get_context_worker()
    worker.start()

    # B1b: Recovery scan — re-enqueue các điểm bị interrupt trước khi restart.
    if settings.context_recovery_scan and settings.llm_provider != "none":
        await _recovery_scan(worker)

    app.state.redis = redis_client

    yield

    # Shutdown
    logger.info("Shutting down...")
    await worker.stop(drain=True)
    await shutdown_llm_client()
    await redis_client.close()


app = FastAPI(
    title="RAG Vector Store API",
    description=(
        "API embedding tài liệu và truy vấn ngữ nghĩa với Qdrant + MiniLM-L12-v2.\n\n"
        "Phase 2 bổ sung luồng transcript cho cuộc họp (BKMEETING).\n"
        "Phase 3 giải quyết too-many-open-files, worker durability, VN tokenizer."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(embed.router)
app.include_router(query.router)
app.include_router(transcript.router)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check có kiểm tra dependency (Qdrant, Redis, LLM provider).

    Giữ tương thích ngược: top-level vẫn có `status` = ok | degraded.
    Trả 503 nếu một dependency cốt lõi (Qdrant / Redis) chết.
    """
    deps = {}

    try:
        QdrantService.list_collections()
        deps["qdrant"] = "ok"
    except Exception as e:  # noqa: BLE001
        deps["qdrant"] = f"error: {e}"

    try:
        redis_client = getattr(app.state, "redis", None) or RedisClient()
        deps["redis"] = "ok" if await redis_client.ping() else "error: ping failed"
    except Exception as e:  # noqa: BLE001
        deps["redis"] = f"error: {e}"

    deps["llm"] = (
        "disabled" if settings.llm_provider == "none" else f"provider={settings.llm_provider}"
    )
    deps["storage_layout"] = settings.transcript_storage_layout

    core_ok = deps["qdrant"] == "ok" and deps["redis"] == "ok"
    body = {
        "status": "ok" if core_ok else "degraded",
        "service": "RAG Vector Store API",
        "version": "3.0.0",
        "dependencies": deps,
    }
    if not core_ok:
        return JSONResponse(status_code=503, content=body)
    return body
