from fastapi import APIRouter, HTTPException, Depends
from app.schemas.query import QueryRequest, QueryResponse, QueryResult
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService


router = APIRouter(prefix="/query", tags=["Query"])


def get_embedder() -> EmbeddingService:
    return EmbeddingService()


def get_qdrant() -> QdrantService:
    return QdrantService()


@router.post("/", response_model=QueryResponse, summary="Truy vấn ngữ nghĩa")
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