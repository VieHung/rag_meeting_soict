from sentence_transformers import SentenceTransformer
from app.config import settings
from typing import List


class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(settings.embedding_model)
            cls._instance._dim = settings.embedding_dim
        return cls._instance

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]