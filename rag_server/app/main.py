from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import embed, query, transcript
from app.services.embedding import EmbeddingService
from app.services.llm_client import get_llm_client, shutdown_llm_client
from app.services.vector_store import QdrantService
from app.utils.redis_client import RedisClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading embedding model...")
    EmbeddingService()
    print("Embedding model ready")

    print("Connecting to Qdrant...")
    QdrantService()
    print("Qdrant ready")

    print("Connecting to Redis...")
    redis_client = RedisClient()
    ok = await redis_client.ping()
    print(f"Redis ready: {ok}")

    print("Initializing LLM client...")
    llm = get_llm_client()
    print(f"LLM client ready (provider={llm.provider}, model={llm.model})")

    yield

    # Shutdown
    print("Shutting down...")
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
    return {"status": "ok", "service": "RAG Vector Store API", "version": "2.0.0"}
