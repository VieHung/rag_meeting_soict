"""Custom retrieval evaluation (no LLM) — RAGAS-compatible metrics.

Bộ metric này:
- **Context Recall (token)**: |tokens(gold_answer) ∩ tokens(retrieved)| / |tokens(gold_answer)|
- **Context Precision (token)**: |tokens(retrieved) ∩ tokens(gold_answer)| / |tokens(retrieved)|
- **Hit Rate @ K**: tỉ lệ câu hỏi có ÍT NHẤT 1 gold chunk nằm trong top-K retrieved.
- **MRR**: Mean Reciprocal Rank của gold chunk đầu tiên trong top-K.
- **Latency**: mean / p50 / p95 / max (ms).

Các metric này tương đương các metric phổ biến trong benchmark retrieval (BEIR,
MIRACL) và bổ trợ cho RAGAS LLM-based metrics — cho kết quả nhanh, ổn định, không
cần API LLM.

Tokenizer đơn giản: lowercase + tách theo ``\\w+`` (Unicode-aware). Đủ tốt cho
tiếng Việt (mỗi âm tiết = 1 token).
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.services.embedding import EmbeddingService
from app.services.reranker import get_reranker
from app.services.retrieval import fuse
from app.config import settings


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> Set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class PerQueryResult:
    id: str
    question: str
    kind: str
    retrieved_indices: List[int] = field(default_factory=list)  # 0-based
    retrieved_scores: List[float] = field(default_factory=list)
    retrieved_sources: List[str] = field(default_factory=list)
    first_relevant_rank: Optional[int] = None  # 1-based, None nếu miss
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    hit_at_k: int = 0
    mrr: float = 0.0
    latency_ms: float = 0.0


@dataclass
class AggregateMetrics:
    n: int
    n_hit: int
    hit_rate_at_k: float
    mrr: float
    mean_recall_at_k: float
    mean_precision_at_k: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_max_ms: float
    per_kind: Dict[str, Dict[str, float]]


# --------------------------------------------------------------------------- #
# Gold chunk matching
# --------------------------------------------------------------------------- #


def _is_relevant(retrieved_text: str, gold_chunk_text: str, threshold: float = 0.5) -> bool:
    """Một retrieved chunk được coi là relevant nếu độ overlap token với gold ≥ threshold."""
    if not gold_chunk_text:
        return False
    rt = tokenize(retrieved_text)
    gt = tokenize(gold_chunk_text)
    if not gt:
        return False
    overlap = len(rt & gt) / len(gt)
    return overlap >= threshold


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class RetrievalEvaluator:
    """Đánh giá retrieval in-process, gọi trực tiếp production code."""

    def __init__(
        self,
        embedder: EmbeddingService,
        qclient,
        collection: str,
        gold_chunks: List[str],  # text vàng, 0-based indexing
        top_k: int = 5,
        use_hybrid: bool = False,
        rerank_provider: Optional[str] = None,
    ):
        self.embedder = embedder
        self.qclient = qclient
        self.collection = collection
        self.gold_chunks = gold_chunks
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        # Tạm thời override settings.hybrid_enabled cho đợt eval này.
        self._saved_hybrid = settings.hybrid_enabled
        settings.hybrid_enabled = use_hybrid
        if rerank_provider is not None:
            self._saved_rerank = settings.rerank_provider
            settings.rerank_provider = rerank_provider
            # Reset singleton để dùng provider mới.
            from app.services.reranker import _reranker_singleton
            import app.services.reranker as reranker_mod
            reranker_mod._reranker_singleton = None
        else:
            self._saved_rerank = None

    def restore(self) -> None:
        """Khôi phục settings về giá trị trước khi eval.

        Không dùng ``__del__`` — Python GC có thể gọi nó SAU khi evaluator kế
        tiếp đã khởi tạo xong, làm sai luồng hybrid (vô hiệu hoá fuse trong
        khi self.use_hybrid=True). Phải gọi explicit qua context manager
        hoặc hàm restore() sau khi evaluate() xong.
        """
        settings.hybrid_enabled = self._saved_hybrid
        if self._saved_rerank is not None:
            settings.rerank_provider = self._saved_rerank
            from app.services.reranker import _reranker_singleton
            import app.services.reranker as reranker_mod
            reranker_mod._reranker_singleton = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()

    def evaluate(
        self,
        qa_items: List[Dict[str, Any]],
        qa_id_field: Optional[str] = None,
        question_field: str = "question",
        kind_field: str = "kind",
        gold_field: str = "gold_chunks",
    ) -> tuple[List[PerQueryResult], AggregateMetrics]:
        results: List[PerQueryResult] = []
        for idx, qa in enumerate(qa_items, 1):
            qid = qa.get(qa_id_field) if qa_id_field else f"q_{idx:02d}"
            qid = qid or f"q_{idx:02d}"
            question = qa[question_field]
            kind = qa.get(kind_field, "unknown")
            golds = qa.get(gold_field, []) or []
            # Chuyển sang 0-based nếu là index.
            gold_indices = [int(g) for g in golds if isinstance(g, (int, str)) and str(g).isdigit()] or list(range(len(golds)))
            # Hỗ trợ 2 dạng gold: list[int] (index vào gold_chunks) hoặc list[dict] (text vàng).
            if golds and isinstance(golds[0], int):
                gold_texts = [self.gold_chunks[i] for i in golds if 0 <= i < len(self.gold_chunks)]
            else:
                gold_texts = [g for g in golds if isinstance(g, str)]

            t0 = time.perf_counter()
            hits = self._query(question)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Xếp hạng: chunk i là relevant nếu trùng với bất kỳ gold nào.
            first_rank: Optional[int] = None
            n_relevant_in_top = 0
            for rank_idx, hit in enumerate(hits, 1):
                if any(_is_relevant(hit["text"], gt) for gt in gold_texts):
                    n_relevant_in_top += 1
                    if first_rank is None:
                        first_rank = rank_idx

            recall = (
                sum(_is_relevant(h["text"], gt) for h in hits for gt in gold_texts)
                / max(1, len(gold_texts) * len(hits))
            ) if gold_texts and hits else 0.0
            # Token recall (RAGAS style).
            gold_tok = set().union(*[tokenize(gt) for gt in gold_texts]) if gold_texts else set()
            ret_tok = set().union(*[tokenize(h["text"]) for h in hits]) if hits else set()
            if gold_tok and ret_tok:
                token_recall = len(gold_tok & ret_tok) / len(gold_tok)
                token_precision = len(gold_tok & ret_tok) / len(ret_tok)
            else:
                token_recall = 0.0
                token_precision = 0.0

            hit = 1 if first_rank is not None else 0
            mrr = (1.0 / first_rank) if first_rank else 0.0

            results.append(
                PerQueryResult(
                    id=str(qid),
                    question=question,
                    kind=kind,
                    retrieved_indices=list(range(len(hits))),
                    retrieved_scores=[h["score"] for h in hits],
                    retrieved_sources=[h.get("source", "") for h in hits],
                    first_relevant_rank=first_rank,
                    recall_at_k=round(token_recall, 4),
                    precision_at_k=round(token_precision, 4),
                    hit_at_k=hit,
                    mrr=round(mrr, 4),
                    latency_ms=round(latency_ms, 2),
                )
            )

        metrics = self._aggregate(results)
        return results, metrics

    def _query(self, question: str) -> List[Dict[str, Any]]:
        qvec = self.embedder.embed_query(question)
        fetch_k = max(self.top_k, self.top_k * settings.hybrid_fetch_multiplier)
        raw = self.qclient.search(
            collection_name=self.collection,
            query_vector=qvec,
            limit=fetch_k,
            with_payload=True,
            score_threshold=0.0,
        )
        candidates = [
            {
                "text": r.payload.get("text", ""),
                "score": round(float(r.score), 4),
                "source": r.payload.get("source", ""),
            }
            for r in raw
        ]
        if settings.hybrid_enabled:
            candidates = fuse(question, candidates)
        return candidates[: self.top_k]

    @staticmethod
    def _aggregate(results: List[PerQueryResult]) -> AggregateMetrics:
        if not results:
            return AggregateMetrics(
                n=0, n_hit=0, hit_rate_at_k=0.0, mrr=0.0,
                mean_recall_at_k=0.0, mean_precision_at_k=0.0,
                latency_mean_ms=0.0, latency_p50_ms=0.0,
                latency_p95_ms=0.0, latency_max_ms=0.0,
                per_kind={},
            )
        n = len(results)
        n_hit = sum(r.hit_at_k for r in results)
        latencies = [r.latency_ms for r in results]
        per_kind: Dict[str, Dict[str, float]] = {}
        for r in results:
            d = per_kind.setdefault(
                r.kind,
                {
                    "n": 0, "n_hit": 0, "mrr_sum": 0.0,
                    "recall_sum": 0.0, "precision_sum": 0.0, "latency_sum": 0.0,
                },
            )
            d["n"] += 1
            d["n_hit"] += r.hit_at_k
            d["mrr_sum"] += r.mrr
            d["recall_sum"] += r.recall_at_k
            d["precision_sum"] += r.precision_at_k
            d["latency_sum"] += r.latency_ms
        for k, d in per_kind.items():
            n_k = d["n"] or 1
            per_kind[k] = {
                "n": int(d["n"]),
                "hit_rate_at_k": round(d["n_hit"] / n_k, 4),
                "mrr": round(d["mrr_sum"] / n_k, 4),
                "token_recall": round(d["recall_sum"] / n_k, 4),
                "token_precision": round(d["precision_sum"] / n_k, 4),
                "latency_mean_ms": round(d["latency_sum"] / n_k, 2),
            }
        return AggregateMetrics(
            n=n,
            n_hit=n_hit,
            hit_rate_at_k=round(n_hit / n, 4),
            mrr=round(sum(r.mrr for r in results) / n, 4),
            mean_recall_at_k=round(sum(r.recall_at_k for r in results) / n, 4),
            mean_precision_at_k=round(sum(r.precision_at_k for r in results) / n, 4),
            latency_mean_ms=round(statistics.mean(latencies), 2),
            latency_p50_ms=round(statistics.median(latencies), 2),
            latency_p95_ms=round(sorted(latencies)[max(0, int(0.95 * n) - 1)], 2),
            latency_max_ms=round(max(latencies), 2),
            per_kind=per_kind,
        )
