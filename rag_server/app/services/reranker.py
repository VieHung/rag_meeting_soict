"""Reranker — cross-encoder rerank tùy chọn (gated như LLMClient).

Lấy cảm hứng từ RAGFlow (`rag/llm/rerank_model.py`): sau khi lấy ứng viên thô,
một cross-encoder chấm lại độ liên quan (query, document) để sắp xếp chính xác hơn
mô hình bi-encoder (embedding) đơn thuần.

Provider (settings.rerank_provider):
- "none"  : no-op — KHÔNG load model, KHÔNG thêm chi phí (mặc định).
- "local" : sentence_transformers.CrossEncoder (vd BAAI/bge-reranker-v2-m3,
            đa ngôn ngữ, hỗ trợ tiếng Việt). Chỉ load khi bật.
- "http"  : gọi endpoint rerank ngoài (TEI / Infinity / Xinference) — hợp tinh
            thần "server self-host mạnh", không bundle model vào image.

Áp lên danh sách ứng viên (dict có `text` + `score`), trả về danh sách đã sắp xếp
lại; `score` được thay bằng điểm rerank đã chuẩn hóa [0,1]. KHÔNG đổi hình dạng dict.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger("reranker")


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Reranker:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.provider = (provider or settings.rerank_provider or "none").lower()
        self.model = model or settings.rerank_model
        self.base_url = base_url if base_url is not None else settings.rerank_base_url
        self.api_key = api_key if api_key is not None else settings.rerank_api_key
        self.timeout = timeout or settings.rerank_timeout_seconds
        self._cross_encoder = None  # lazy

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    # ----- lifecycle -------------------------------------------------------

    def warmup(self) -> None:
        """Load model nếu provider=local (gọi 1 lần lúc startup)."""
        if self.provider != "local":
            return
        self._load_cross_encoder()

    def _load_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            logger.info("Loading CrossEncoder reranker: %s", self.model)
            self._cross_encoder = CrossEncoder(self.model)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load reranker '%s': %s — rerank disabled", self.model, e)
            self._cross_encoder = None
        return self._cross_encoder

    # ----- public ----------------------------------------------------------

    async def rerank(
        self,
        query: str,
        candidates: List[Dict],
        *,
        text_key: str = "text",
        score_key: str = "score",
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Sắp xếp lại candidates theo điểm cross-encoder. No-op nếu disabled.

        Nếu rerank lỗi → trả candidates nguyên trạng (degrade an toàn).
        """
        if not self.enabled or not candidates:
            return candidates

        docs = [str(c.get(text_key, "") or "") for c in candidates]
        try:
            if self.provider == "local":
                scores = await self._rerank_local(query, docs)
            elif self.provider == "http":
                scores = await self._rerank_http(query, docs)
            else:
                return candidates
        except Exception as e:  # noqa: BLE001
            logger.warning("Rerank failed (%s) — giữ thứ tự cũ: %s", self.provider, e)
            return candidates

        if not scores or len(scores) != len(candidates):
            return candidates

        for c, s in zip(candidates, _minmax([float(x) for x in scores])):
            c[score_key] = s
        candidates.sort(key=lambda c: c.get(score_key, 0.0), reverse=True)
        if top_k is not None:
            return candidates[:top_k]
        return candidates

    # ----- providers -------------------------------------------------------

    async def _rerank_local(self, query: str, docs: List[str]) -> List[float]:
        import asyncio

        ce = self._load_cross_encoder()
        if ce is None:
            raise RuntimeError("CrossEncoder not loaded")
        pairs = [[query, d] for d in docs]
        scores = await asyncio.to_thread(ce.predict, pairs)
        return [float(s) for s in scores]

    async def _rerank_http(self, query: str, docs: List[str]) -> List[float]:
        if not self.base_url:
            raise RuntimeError("rerank_base_url chưa cấu hình cho provider=http")
        url = self.base_url.rstrip("/") + "/rerank"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"query": query, "documents": docs, "model": self.model}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        # Hỗ trợ vài định dạng phổ biến (TEI/Infinity/Jina-like).
        items = data.get("results", data) if isinstance(data, dict) else data
        scores = [0.0] * len(docs)
        for it in items:
            idx = it.get("index")
            score = it.get("relevance_score", it.get("score"))
            if idx is not None and score is not None and 0 <= idx < len(scores):
                scores[idx] = float(score)
        return scores


# ----- singleton ----------------------------------------------------------

_reranker_singleton: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = Reranker()
    return _reranker_singleton
