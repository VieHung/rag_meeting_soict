"""Hybrid retrieval — fusion điểm từ khóa (BM25) + điểm vector (cosine).

Phase 3 A1: tokenize() trở thành pluggable qua HYBRID_TOKENIZER:
- simple      : regex \\w+ (zero-dep, mặc định)
- pyvi        : ViTokenizer.tokenize — segment từ ghép tiếng Việt; cần `pip install pyvi`
- underthesea : word_tokenize — nặng hơn; cần `pip install underthesea`
Thiếu lib → log warning + fallback về simple.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Optional

from app.config import settings

logger = logging.getLogger("retrieval")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:  # noqa: BLE001
    _HAS_BM25 = False

# Cached tokenizer function — set lazily on first call.
_tokenizer_fn: Optional[Callable[[str], List[str]]] = None


def _simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _get_tokenizer() -> Callable[[str], List[str]]:
    """Lazy-load và cache tokenizer theo HYBRID_TOKENIZER setting."""
    global _tokenizer_fn
    if _tokenizer_fn is not None:
        return _tokenizer_fn

    mode = settings.hybrid_tokenizer

    if mode == "pyvi":
        try:
            from pyvi import ViTokenizer  # type: ignore
            def _pyvi_tok(text: str) -> List[str]:
                return _TOKEN_RE.findall(ViTokenizer.tokenize(text or "").lower())
            _tokenizer_fn = _pyvi_tok
            logger.info("Tokenizer: pyvi (ViTokenizer)")
            return _tokenizer_fn
        except ImportError:
            logger.warning(
                "HYBRID_TOKENIZER=pyvi but 'pyvi' is not installed — falling back to simple. "
                "Install with: pip install pyvi"
            )

    elif mode == "underthesea":
        try:
            from underthesea import word_tokenize  # type: ignore
            def _uth_tok(text: str) -> List[str]:
                return _TOKEN_RE.findall(" ".join(word_tokenize(text or "")).lower())
            _tokenizer_fn = _uth_tok
            logger.info("Tokenizer: underthesea (word_tokenize)")
            return _tokenizer_fn
        except ImportError:
            logger.warning(
                "HYBRID_TOKENIZER=underthesea but 'underthesea' is not installed — falling back to simple. "
                "Install with: pip install underthesea"
            )

    _tokenizer_fn = _simple_tokenize
    if mode != "simple":
        pass  # warning already logged above for unknown modes
    return _tokenizer_fn


def tokenize(text: str) -> List[str]:
    return _get_tokenizer()(text)


def _minmax_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _bm25_scores(query: str, docs: List[str]) -> List[float]:
    """Điểm BM25 của query trên tập docs."""
    tokenized = [tokenize(d) for d in docs]
    if _HAS_BM25:
        if not any(tokenized):
            return [0.0 for _ in docs]
        bm25 = BM25Okapi(tokenized)
        return list(bm25.get_scores(tokenize(query)))
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
    """Fuse điểm vector + BM25, cập nhật `score_key`, sort giảm dần."""
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
