"""FastAPI dependencies (DI)."""
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService
from app.services.sequence_manager import SequenceManager
from app.services.transcript_service import TranscriptService
from app.services.llm_client import LLMClient, get_llm_client
from app.utils.redis_client import RedisClient


def get_embedder() -> EmbeddingService:
    return EmbeddingService()


def get_qdrant() -> QdrantService:
    return QdrantService()


def get_redis_client() -> RedisClient:
    return RedisClient()


def get_llm() -> LLMClient:
    return get_llm_client()


def get_sequence_manager() -> SequenceManager:
    return SequenceManager(get_redis_client())


_transcript_service_singleton: TranscriptService | None = None


def get_transcript_service() -> TranscriptService:
    """Singleton để chia sẻ EmbeddingService, Redis, LLM."""
    global _transcript_service_singleton
    if _transcript_service_singleton is None:
        _transcript_service_singleton = TranscriptService(
            embedder=get_embedder(),
            sequence_manager=get_sequence_manager(),
            redis_client=get_redis_client(),
        )
    return _transcript_service_singleton
