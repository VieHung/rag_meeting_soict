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
from app.workers.context_worker import get_context_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rag_server")


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

    # Reranker (gated — chỉ load model khi RERANK_PROVIDER != none).
    reranker = get_reranker()
    reranker.warmup()
    logger.info(
        "Reranker ready (provider=%s, hybrid_enabled=%s)",
        settings.rerank_provider,
        settings.hybrid_enabled,
    )

    # Context worker (FIFO build context — D9).
    worker = get_context_worker()
    worker.start()

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
        "Phase 2 bổ sung luồng transcript cho cuộc họp (BKMEETING)."
    ),
    version="2.0.0",
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

    # Qdrant
    try:
        QdrantService.list_collections()
        deps["qdrant"] = "ok"
    except Exception as e:  # noqa: BLE001
        deps["qdrant"] = f"error: {e}"

    # Redis
    try:
        redis_client = getattr(app.state, "redis", None) or RedisClient()
        deps["redis"] = "ok" if await redis_client.ping() else "error: ping failed"
    except Exception as e:  # noqa: BLE001
        deps["redis"] = f"error: {e}"

    # LLM (chỉ báo cấu hình, không gọi mạng để health nhẹ).
    deps["llm"] = (
        "disabled" if settings.llm_provider == "none" else f"provider={settings.llm_provider}"
    )

    core_ok = deps["qdrant"] == "ok" and deps["redis"] == "ok"
    body = {
        "status": "ok" if core_ok else "degraded",
        "service": "RAG Vector Store API",
        "version": "2.0.0",
        "dependencies": deps,
    }
    if not core_ok:
        return JSONResponse(status_code=503, content=body)
    return body
