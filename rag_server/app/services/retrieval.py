"""Hybrid retrieval — fusion điểm từ khóa (BM25) + điểm vector (cosine).

Lấy cảm hứng từ RAGFlow (`rag/nlp/search.py`): điểm cuối là tổ hợp có trọng số

    final = vec_w * vector_sim_norm + term_w * term_sim_norm

Bản tối giản cho rag_server:
- KHÔNG re-index Qdrant, KHÔNG thêm sparse vector. BM25 chạy **trên chính tập
  ứng viên** vector search trả về (bounded → rẻ). Đây là tinh thần "two-pass"
  của RAGFlow rút gọn: vector lọc thô → fuse từ khóa tinh chỉnh thứ tự.
- Dùng chung cho cả `/query/` (tài liệu) và `/query/transcript`.
- Mặc định TẮT (settings.hybrid_enabled=False) → không đụng gì tới luồng cũ.

Tokenizer tối giản, unicode-aware: lowercase + tách theo ký tự không phải chữ/số.
Đủ để khớp tên riêng / con số / từ khóa tiếng Việt (mỗi âm tiết = 1 token).
Không kéo `underthesea` (nặng/chậm) — có thể nâng cấp sau nếu cần segment từ ghép.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from app.config import settings

logger = logging.getLogger("retrieval")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_BM25 = True
except Exception:  # noqa: BLE001 — fallback nếu chưa cài rank-bm25
    _HAS_BM25 = False


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        # Tất cả bằng nhau → trả 1.0 (không phân biệt được, giữ trung tính).
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _bm25_scores(query: str, docs: List[str]) -> List[float]:
    """Điểm BM25 của query trên tập docs (đã tokenize nội bộ)."""
    tokenized = [tokenize(d) for d in docs]
    if _HAS_BM25:
        # Bỏ doc rỗng để BM25Okapi không chia 0 (avgdl=0).
        if not any(tokenized):
            return [0.0 for _ in docs]
        bm25 = BM25Okapi(tokenized)
        return list(bm25.get_scores(tokenize(query)))
    # Fallback: token-overlap đếm số token query xuất hiện trong doc.
    q = set(tokenize(query))
    return [float(sum(1 for t in toks if t in q)) for toks in tokenized]


def fuse(
    query: str,
    candidates: List[Dict],
    *,
    text_key: str = "text",
    score_key: str = "score",
    vector_weight: float | None = None,
    term_weight: float | None = None,
) -> List[Dict]:
    """Fuse điểm vector + BM25, ghi lại `score_key` = điểm cuối, sort giảm dần.

    Mỗi candidate là dict có sẵn `text` và `score` (cosine từ Qdrant). Trả về
    cùng danh sách (đã sort), `score` được thay bằng điểm fused.
    Không phá hình dạng dict (giữ mọi field khác nguyên vẹn).
    """
    if not candidates:
        return candidates

    vec_w = settings.hybrid_vector_weight if vector_weight is None else vector_weight
    term_w = settings.hybrid_term_weight if term_weight is None else term_weight

    vec_scores = [float(c.get(score_key, 0.0) or 0.0) for c in candidates]
    docs = [str(c.get(text_key, "") or "") for c in candidates]
    term_scores = _bm25_scores(query, docs)

    vec_norm = _minmax_norm(vec_scores)
    term_norm = _minmax_norm(term_scores)

    for c, vn, tn in zip(candidates, vec_norm, term_norm):
        c[score_key] = vec_w * vn + term_w * tn

    candidates.sort(key=lambda c: c.get(score_key, 0.0), reverse=True)
    return candidates
