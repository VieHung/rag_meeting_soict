"""RAGAS evaluation script.

Script này:
1. Khởi tạo RAG pipeline in-process (dùng production code + Qdrant in-memory).
2. Ingest 2 corpus: tài liệu (Phase 1) + transcript (Phase 2).
3. Với mỗi câu hỏi vàng, gọi pipeline sinh ``(answer, contexts)``.
4. Build dataset RAGAS (``question, answer, contexts, ground_truth``).
5. Tính 4 metric RAGAS: ``faithfulness``, ``answer_relevancy``,
   ``context_precision``, ``context_recall``.
6. Chạy với 2 cấu hình retrieval (so sánh pure-vector vs hybrid) và dump kết quả
   ra ``eval/results.json`` + in bảng tóm tắt.

LLM judge dùng LM Studio (OpenAI-compatible). Embedding dùng ``HuggingFaceEmbeddings``
trỏ vào cùng model paraphrase-multilingual-MiniLM-L12-v2 của production.

Chạy:
    wsl -d Ubuntu bash /mnt/c/Users/navis/hungtv/run_eval.sh
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Đảm bảo import được app.* (rag_server).
ROOT = Path(__file__).resolve().parent
RAG_SERVER = ROOT.parent / "rag_server"
sys.path.insert(0, str(RAG_SERVER))

# Tránh load lại model 2 lần — tắt TF progress.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from eval.gold_dataset import DOCS_CORPUS, DOCS_QA, TRANSCRIPT_CORPUS, TRANSCRIPT_QA  # noqa: E402
from eval.pipeline import RAGPipeline  # noqa: E402

# RAGAS imports (sau khi đã load torch + qdrant ở trên).
from datasets import Dataset  # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("rag_eval")


# --------------------------------------------------------------------------- #
# Cấu hình chung
# --------------------------------------------------------------------------- #

LM_BASE = "http://192.168.240.1:1234/v1"
LM_KEY = "sk-lm-I4p1UFW1:ADiMgh6qgZUwq4VixJH6"
# Mô hình đang được LM Studio load sẵn (các model lớn hơn bị unload / timeout).
# e4b đủ nhỏ để chạy nhanh + chịu được nhiều call đồng thời.
LM_MODEL = "google/gemma-4-e4b"

DOCS_COLLECTION = "docs-eval-ragas"
TRANSCRIPT_COLLECTION = "meeting-bt2-eval"
# Phải khớp với ``TRANSCRIPT_CORPUS[0]["meeting_id"]`` (xem eval/gold_dataset.py).
MEETING_ID = "meeting-bt2-eval"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_llm_judge() -> ChatOpenAI:
    return ChatOpenAI(
        model=LM_MODEL,
        base_url=LM_BASE,
        api_key=LM_KEY,
        temperature=0.0,
        max_tokens=2048,
        timeout=120,
    )


def make_embeddings() -> HuggingFaceEmbeddings:
    """Embedding cho RAGAS (dùng cùng model production để so sánh công bằng)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


# --------------------------------------------------------------------------- #
# Ingest + chạy
# --------------------------------------------------------------------------- #


def ingest(pipeline: RAGPipeline) -> None:
    log.info("Ingesting %d document corpus...", len(DOCS_CORPUS))
    for doc in DOCS_CORPUS:
        pipeline.upsert_doc_chunks(
            DOCS_COLLECTION, doc["chunks"], source=doc["source"]
        )

    log.info("Ingesting %d meeting transcripts...", len(TRANSCRIPT_CORPUS))
    for mt in TRANSCRIPT_CORPUS:
        for ut in mt["utterances"]:
            pipeline.upsert_transcript_point(
                collection=TRANSCRIPT_COLLECTION,
                meeting_id=mt["meeting_id"],
                sequence_id=ut["sequence_id"],
                speaker=ut["speaker"],
                text=ut["text"],
            )


def collect_answers(
    pipeline: RAGPipeline, top_k_docs: int = 5, top_k_transcript: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """Chạy cả 2 bộ Q&A, trả về dict {phase: [rows]}.

    Mỗi row có: question, answer, contexts, ground_truth, gold_ref.
    """
    out: Dict[str, List[Dict[str, Any]]] = {"docs": [], "transcript": []}

    log.info("Generating answers for %d docs Q&A...", len(DOCS_QA))
    for i, qa in enumerate(DOCS_QA, 1):
        r = pipeline.ask_docs(
            DOCS_COLLECTION, qa["question"], top_k=top_k_docs
        )
        out["docs"].append(
            {
                "id": f"docs_{i:02d}",
                "question": qa["question"],
                "answer": r.answer,
                "contexts": r.contexts,
                "ground_truth": qa["ground_truth_answer"],
                "kind": qa["kind"],
                "doc_idx": qa["doc_idx"],
                "gold_chunks": qa["gold_chunks"],
                "hit_sources": [h.source for h in r.hits],
                "hit_scores": [h.score for h in r.hits],
                "latency_ms": r.latency_ms,
            }
        )
        if i % 3 == 0:
            log.info("  docs Q&A %d/%d", i, len(DOCS_QA))

    log.info("Generating answers for %d transcript Q&A...", len(TRANSCRIPT_QA))
    for i, qa in enumerate(TRANSCRIPT_QA, 1):
        r = pipeline.ask_transcript(
            TRANSCRIPT_COLLECTION,
            MEETING_ID,
            qa["question"],
            top_k=top_k_transcript,
            window_size=2,
        )
        # Thêm context window vào contexts để RAGAS chấm trên "đầy đủ" thông tin.
        contexts = [h.text for h in r.hits]
        out["transcript"].append(
            {
                "id": f"trans_{i:02d}",
                "question": qa["question"],
                "answer": r.answer,
                "contexts": contexts,
                "ground_truth": qa["ground_truth_answer"],
                "kind": qa["kind"],
                "gold_sequence_ids": qa["gold_sequence_ids"],
                "hit_sequence_ids": [
                    h.sequence_id for h in r.hits if h.sequence_id is not None
                ],
                "hit_speakers": [h.speaker for h in r.hits],
                "hit_scores": [h.score for h in r.hits],
                "latency_ms": r.latency_ms,
            }
        )
        if i % 3 == 0:
            log.info("  transcript Q&A %d/%d", i, len(TRANSCRIPT_QA))

    return out


# --------------------------------------------------------------------------- #
# RAGAS run
# --------------------------------------------------------------------------- #


def rows_to_ragas_dataset(
    rows: List[Dict[str, Any]],
    max_contexts: int = 3,
    max_context_chars: int = 600,
) -> Dataset:
    """Convert list of dicts sang HuggingFace Dataset đúng schema RAGAS.

    Truncate số lượng + độ dài context để prompt RAGAS (faithfulness /
    context_*) vừa context 8K của LM Studio model nhỏ (gemma-4-e4b).
    """
    return Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": [
                    c[:max_context_chars]
                    for c in (r["contexts"] or [])[:max_contexts]
                ] or ["(không có ngữ cảnh)"],
                "ground_truth": r["ground_truth"],
            }
            for r in rows
        ]
    )


def run_ragas(
    rows: List[Dict[str, Any]],
    llm,
    embeddings,
    phase_label: str,
    metrics_to_run: List[str] | None = None,
) -> Dict[str, Any]:
    """Chạy RAGAS trên tập rows. Trả về dict kết quả (phase, n, metrics, per_query).

    ``metrics_to_run``: subset của {"faithfulness", "answer_relevancy",
    "context_precision", "context_recall"}. None = chạy cả 4 (mặc định).
    """
    if not rows:
        return {"phase": phase_label, "n": 0, "metrics": {}}
    ds = rows_to_ragas_dataset(rows)
    log.info("Running RAGAS on %s (%d rows)...", phase_label, len(rows))
    t0 = time.perf_counter()
    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=1, timeout=180)
    except Exception:  # noqa: BLE001
        run_config = None

    metric_objs = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    if metrics_to_run is None:
        metrics_to_run = list(metric_objs.keys())
    sel = [metric_objs[m] for m in metrics_to_run if m in metric_objs]

    kwargs = dict(
        metrics=sel,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    if run_config is not None:
        kwargs["run_config"] = run_config
    result = evaluate(ds, **kwargs)
    dt = time.perf_counter() - t0
    metrics: Dict[str, float] = {}
    per_query: List[Dict[str, Any]] = []
    try:
        df = result.to_pandas()
        for col in metrics_to_run:
            if col in df.columns:
                # mean() bỏ qua NaN
                metrics[col] = float(df[col].mean())
        # Lưu per-row để debug.
        for col in metrics_to_run:
            if col not in df.columns:
                df[col] = None
        for _, row in df.iterrows():
            per_query.append(
                {
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    **{m: (None if row.get(m) is None or (isinstance(row.get(m), float) and row.get(m) != row.get(m)) else float(row.get(m)))
                       for m in metrics_to_run},
                }
            )
    except Exception as e:  # noqa: BLE001
        log.warning("Could not parse RAGAS result: %s", e)
    return {
        "phase": phase_label,
        "n": len(rows),
        "duration_sec": round(dt, 2),
        "metrics": {k: (None if v is None or v != v else round(v, 4)) for k, v in metrics.items()},
        "per_query": per_query,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ragas-subset",
        type=int,
        default=0,
        help="Nếu >0: chỉ lấy N câu đầu của mỗi phase (để tiết kiệm thời gian LLM judge).",
    )
    parser.add_argument(
        "--ragas-metrics",
        type=str,
        default="answer_relevancy,faithfulness",
        help="Comma-separated metric names (subset của {faithfulness,answer_relevancy,context_precision,context_recall}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "metrics_ragas.json"),
        help="File JSON xuất kết quả RAGAS.",
    )
    args = parser.parse_args()

    metrics_to_run = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]
    cfg = {
        "hybrid_enabled": False,
        "rerank_provider": "none",
        "top_k_docs": 5,
        "top_k_transcript": 3,
        "answer_model": LM_MODEL,
        "ragas_subset": args.ragas_subset,
        "ragas_metrics": metrics_to_run,
    }
    log.info("=== Config: %s", json.dumps(cfg, ensure_ascii=False))

    pipeline = RAGPipeline(answer_model=LM_MODEL)
    log.info("Embedding dim: %d", pipeline.dim)
    ingest(pipeline)

    answers = collect_answers(
        pipeline,
        top_k_docs=cfg["top_k_docs"],
        top_k_transcript=cfg["top_k_transcript"],
    )

    # Subset nếu được yêu cầu.
    if args.ragas_subset > 0:
        for phase in list(answers.keys()):
            answers[phase] = answers[phase][: args.ragas_subset]
        log.info("Subset -> %d rows/phase", args.ragas_subset)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_answers.json"
    raw_path.write_text(
        json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Wrote raw answers -> %s", raw_path)

    # RAGAS.
    log.info("Loading RAGAS LLM judge + embeddings (model=%s)...", LM_MODEL)
    llm = make_llm_judge()
    emb = make_embeddings()

    results: List[Dict[str, Any]] = []
    for phase, rows in answers.items():
        res = run_ragas(rows, llm, emb, phase, metrics_to_run=metrics_to_run)
        results.append(res)
        log.info("  %s -> %s", phase, res.get("metrics"))

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"config": cfg, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote metrics -> %s", out_path)

    # Tóm tắt cuối.
    print("\n========= RAGAS SUMMARY =========")
    for r in results:
        print(f"--- {r['phase']} (n={r['n']}, {r['duration_sec']}s) ---")
        for k, v in r["metrics"].items():
            if v is None:
                print(f"  {k:22s} = NaN")
            else:
                print(f"  {k:22s} = {v:.4f}")
    print("=================================\n")


if __name__ == "__main__":
    main()
