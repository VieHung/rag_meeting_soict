from fastapi import APIRouter, Depends, HTTPException

from app.schemas.query import QueryRequest, QueryResponse, QueryResult
from app.schemas.transcript import (
    TranscriptQueryRequest,
    TranscriptQueryResponse,
)
from app.services.embedding import EmbeddingService
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

    raw_results = qdrant.search(
        query_vector=query_vector,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    filtered = [
        r for r in raw_results
        if r["score"] >= request.score_threshold
    ]

    results = [QueryResult(**r) for r in filtered]

    return QueryResponse(
        query=request.query,
        results=results,
        total_found=len(results),
    )


# ---- Phase 2: query transcript ----------------------------------------------


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
        return await service.query_transcript(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
