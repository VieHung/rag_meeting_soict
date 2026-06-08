from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.schemas.query import QueryRequest, QueryResponse, QueryResult
from app.schemas.transcript import (
    TranscriptQueryRequest,
    TranscriptQueryResponse,
)
from app.services.embedding import EmbeddingService
from app.services.reranker import get_reranker
from app.services.retrieval import fuse
from app.services.vector_store import QdrantService
from app.dependencies import get_transcript_service
from app.services.transcript_service import TranscriptService


router = APIRouter(prefix="/query", tags=["Query"])


def get_embedder() -> EmbeddingService:
    return EmbeddingService()


def get_qdrant() -> QdrantService:
    return QdrantService()


# ---- Phase 1: query tài liệu (giữ nguyên) ------------------------------------


@router.post("/", response_model=QueryResponse, summary="Truy vấn ngữ nghĩa (tài liệu)")
async def query_documents(
    request: QueryRequest,
    embedder: EmbeddingService = Depends(get_embedder),
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query không được rỗng")

    qdrant = QdrantService(request.collection)
    query_vector = embedder.embed_query(request.query)

    # Hybrid/rerank (RAGFlow-style) — gated. Khi cả hai TẮT, fetch_k == top_k
    # và luồng y hệt bản pure-vector cũ.
    reranker = get_reranker()
    need_more = settings.hybrid_enabled or reranker.enabled
    fetch_k = (
        max(request.top_k, request.top_k * settings.hybrid_fetch_multiplier)
        if need_more
        else request.top_k
    )

    raw_results = qdrant.search(
        query_vector=query_vector,
        top_k=fetch_k,
        source_filter=request.source_filter,
    )

    # score_threshold áp trên điểm vector gốc (giữ nguyên ngữ nghĩa) trước khi fuse.
    filtered = [r for r in raw_results if r["score"] >= request.score_threshold]

    if settings.hybrid_enabled:
        filtered = fuse(request.query, filtered)
    if reranker.enabled:
        filtered = await reranker.rerank(request.query, filtered)
    filtered = filtered[: request.top_k]

    results = [QueryResult(**r) for r in filtered]

    return QueryResponse(
        query=request.query,
        results=results,
        total_found=len(results),
    )


# ---- Phase 2: query transcript -----------------------------------------------

MEETING_PREFIX = "meeting-"


class InvalidCollectionPrefix(Exception):
    pass


def _validate_meeting_collection(collection: Optional[str]) -> str:
    if collection is None:
        raise InvalidCollectionPrefix("collection is required for transcript query")
    if not collection.startswith(MEETING_PREFIX):
        raise InvalidCollectionPrefix(
            f"collection must start with '{MEETING_PREFIX}' for transcript query"
        )
    return collection


@router.post(
    "/transcript",
    response_model=TranscriptQueryResponse,
    summary="Truy vấn ngữ nghĩa trên transcript (kèm window ±N câu)",
)
async def query_transcript(
    request: TranscriptQueryRequest,
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        _validate_meeting_collection(request.collection)
    except InvalidCollectionPrefix as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        return await service.query_transcript(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
