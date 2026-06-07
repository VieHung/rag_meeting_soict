"""Embedding backends — tách phần "chạy transformer" khỏi EmbeddingService.

Hai backend, chọn qua `EMBEDDING_BACKEND`:
- `sentence_transformers`: dùng thư viện sentence-transformers (dev/CPU/GPU thường).
- `qaic`: chạy trên accelerator Qualcomm Cloud AI (server AI080). sentence-transformers
  KHÔNG chạy được trên NPU QAIC, nên backend này tự làm:
      tokenize (CPU) → forward trên NPU qua QPC → mean-pooling + L2-normalize (CPU).

Cả hai backend nhận vào danh sách text **đã được gắn prefix** (E5: "query: " / "passage: ")
bởi EmbeddingService, và trả về vector đã L2-normalize (để Qdrant dùng COSINE).
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from app.config import settings


class EmbeddingBackend:
    """Interface chung cho mọi backend embedding."""

    @property
    def dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def encode(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError


class SentenceTransformerBackend(EmbeddingBackend):
    """Backend mặc định — giữ nguyên hành vi cũ (dev/CPU/GPU)."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.embedding_model)
        self._dim = settings.embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class QaicEmbeddingBackend(EmbeddingBackend):
    """Backend chạy trên Qualcomm Cloud AI (AI080) qua QPC đã compile sẵn.

    Yêu cầu môi trường:
    - `transformers` (tokenizer chạy CPU).
    - Python package `qaic` do Qualcomm Apps SDK cung cấp (không có trên PyPI):
        pip install /opt/qti-aic/dev/lib/x86_64/qaic-*-py3-none-any.whl
    - `EMBEDDING_QPC_PATH` trỏ tới QPC build từ scripts/build_qpc.sh (thư mục chứa
      `programqpc.bin`, hoặc trỏ thẳng file `.bin`).
    - QPC compile với batch=1, seq_len = EMBEDDING_MAX_SEQ_LEN (shape TĨNH) → mỗi lần
      inference xử lý đúng 1 câu, độ dài cố định.

    API đã kiểm chứng trên AI080 (qaic 0.0.1):
        sess = qaic.Session(model_path='.../programqpc.bin')   # nạp QPC, KHÔNG compile=False
        sess.setup()
        out = sess.run({'input_ids': (1,128) int64, 'attention_mask': (1,128) int64})
        # out['last_hidden_state'] shape (1, 128, 768)
    """

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        if not settings.embedding_qpc_path:
            raise RuntimeError(
                "EMBEDDING_QPC_PATH chưa được cấu hình nhưng EMBEDDING_BACKEND=qaic"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
        self._dim = settings.embedding_dim
        self._max_seq_len = settings.embedding_max_seq_len
        self._session = self._open_session(settings.embedding_qpc_path)
        # Tên output lấy từ QPC (vd 'last_hidden_state').
        self._output_name = self._session.model_output_names[0]

    @staticmethod
    def _open_session(qpc_path: str):
        # Import muộn để máy dev (không có SDK) vẫn chạy được backend kia.
        import qaic  # type: ignore

        # qaic.Session nạp QPC khi model_path kết thúc bằng '.bin'. Nếu cấu hình trỏ vào
        # thư mục QPC, tự ghép 'programqpc.bin'.
        bin_path = qpc_path
        if not bin_path.endswith(".bin"):
            bin_path = os.path.join(bin_path, "programqpc.bin")
        # KHÔNG truyền compile=False: với QPC đã nạp, Session.compile() chỉ chạy
        # _model_init() (tạo context + io shapes), không biên dịch lại.
        session = qaic.Session(model_path=bin_path)
        session.setup()
        return session

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        # QPC có batch tĩnh = 1 → xử lý từng câu một (batch_size bỏ qua).
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> List[float]:
        enc = self._tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=self._max_seq_len,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        outputs = self._session.run(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        last_hidden = outputs[self._output_name]  # (1, seq, hidden)
        pooled = self._mean_pool(last_hidden, attention_mask)  # (1, hidden)
        normalized = self._l2_normalize(pooled)  # (1, hidden)
        return normalized[0].astype(np.float32).tolist()

    @staticmethod
    def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[..., None].astype(np.float32)  # [B, S, 1]
        summed = (last_hidden.astype(np.float32) * mask).sum(axis=1)  # [B, H]
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)  # [B, 1]
        return summed / counts

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return vectors / norms


def create_backend() -> EmbeddingBackend:
    """Factory chọn backend theo cấu hình."""
    backend = (settings.embedding_backend or "sentence_transformers").lower()
    if backend == "qaic":
        return QaicEmbeddingBackend()
    return SentenceTransformerBackend()
