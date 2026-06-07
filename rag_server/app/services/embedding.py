from typing import List

from app.config import settings
from app.services.embedding_backends import create_backend


class EmbeddingService:
    """Singleton embedding service.

    Chọn backend (sentence_transformers | qaic) qua cấu hình và áp prefix
    theo kiểu E5: text lưu trữ (passage) và câu truy vấn (query) được gắn
    tiền tố khác nhau. Với model đối xứng (vd MiniLM cũ) đặt prefix rỗng
    trong .env để tắt.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._backend = create_backend()
            cls._instance._dim = cls._instance._backend.dim
        return cls._instance

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _apply_prefix(prefix: str, texts: List[str]) -> List[str]:
        if not prefix:
            return list(texts)
        return [f"{prefix}{t}" for t in texts]

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed các đoạn lưu trữ (passage) — dùng cho document chunks và transcript."""
        prefixed = self._apply_prefix(settings.embedding_passage_prefix, texts)
        return self._backend.encode(prefixed, batch_size=batch_size)

    def embed_query(self, query: str) -> List[float]:
        """Embed một câu truy vấn (query)."""
        prefixed = self._apply_prefix(settings.embedding_query_prefix, [query])
        return self._backend.encode(prefixed, batch_size=1)[0]
