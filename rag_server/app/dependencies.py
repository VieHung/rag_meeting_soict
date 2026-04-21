from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService


def get_embedder() -> EmbeddingService:
    return EmbeddingService()


def get_qdrant() -> QdrantService:
    return QdrantService()