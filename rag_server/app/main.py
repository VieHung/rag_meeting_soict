from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import embed, query
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    EmbeddingService()
    print("Embedding model ready")

    print("Connecting to Qdrant...")
    QdrantService()
    print("Qdrant ready")

    yield
    print("Shutting down...")


app = FastAPI(
    title="RAG Vector Store API",
    description="API embedding tài liệu và truy vấn ngữ nghĩa với Qdrant + MiniLM-L12-v2",
    version="1.0.0",
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


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "RAG Vector Store API"}