"""Comprehensive RAG benchmark — retrieval matrix + RAGAS full (4 metrics) × config.

Khác với ``run_eval.py`` / ``run_ragas.py`` (chạy subset 4 câu, 2 metric, model 8K):
script này chạy **toàn diện**:

1. **Retrieval matrix** (deterministic, no LLM): docs + transcript × {pure_vector, hybrid}.
2. **RAGAS đầy đủ**: TOÀN BỘ gold dataset (12 docs + 10 transcript) × {pure_vector, hybrid},
   chạy CẢ 4 metric ``faithfulness, answer_relevancy, context_precision, context_recall``
   bằng model context lớn (mặc định lấy ``LLM_MODEL`` từ ``.env`` — vd gemma-4-26b-a4b-it).

Creds (base url / api key / model) ĐỌC TỪ ``rag_server/.env`` — KHÔNG hardcode key.

Chạy (WSL):
    cd /mnt/c/Users/navis/hungtv/rag_base && source venv/bin/activate
    python -m eval.run_comprehensive --out eval/results/comprehensive.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
RAG_SERVER = ROOT.parent / "rag_server"
sys.path.insert(0, str(RAG_SERVER))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Bảo vệ: env lạ không làm vỡ pydantic Settings (đã có extra=ignore, nhưng chắc chắn).
os.environ.pop("OPENROUTER_API_KEY", None)

import torch  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.embedding import EmbeddingService  # noqa: E402

from eval.gold_dataset import (  # noqa: E402
    DOCS_CORPUS,
    DOCS_QA,
    TRANSCRIPT_CORPUS,
    TRANSCRIPT_QA,
)
from eval.retrieval_eval import RetrievalEvaluator  # noqa: E402
from eval import run_eval as RE  # reuse ingest + gold-chunk builders  # noqa: E402
from eval.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("rag_comprehensive")

DOCS_COLLECTION = "docs-eval-ragas"
TRANSCRIPT_COLLECTION = "meeting-bt2-eval"
MEETING_ID = TRANSCRIPT_CORPUS[0]["meeting_id"]  # khớp tuyệt đối, tránh bug filter cũ

CONFIGS = {
    "pure_vector": {"hybrid_enabled": False},
    "hybrid": {"hybrid_enabled": True},
}


# --------------------------------------------------------------------------- #
# Đọc creds từ .env (không hardcode key)
# --------------------------------------------------------------------------- #


def read_env_creds() -> Dict[str, str]:
    env_path = RAG_SERVER / ".env"
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

    def _get(key: str, default: str = "") -> str:
        m = re.search(rf"^{key}=(.*)$", text, re.M)
        return (m.group(1).strip() if m else os.environ.get(key, default)).strip()

    base = _get("LLM_BASE_URL", "http://192.168.240.1:1234/v1").rstrip("/")
    return {
        "base_url": base,
        "api_key": _get("LLM_API_KEY", ""),
        "model": _get("LLM_MODEL", "gemma-4-26b-a4b-it"),
    }


# --------------------------------------------------------------------------- #
# Retrieval matrix
# --------------------------------------------------------------------------- #


def run_retrieval_matrix(embedder: EmbeddingService, top_k_docs: int, top_k_transcript: int) -> Dict[str, Any]:
    qclient = QdrantClient(location=":memory:")
    RE.ingest_docs(qclient, embedder, DOCS_COLLECTION)
    RE.ingest_transcripts(qclient, embedder, TRANSCRIPT_COLLECTION)

    docs_gold = RE.build_docs_gold_chunks()
    docs_qa = [
        {"id": f"docs_{i+1:02d}", "question": qa["question"], "kind": qa["kind"], "gold_chunks": qa["gold_chunks"]}
        for i, qa in enumerate(DOCS_QA)
    ]
    trans_gold = RE.build_transcript_gold_chunks()
    trans_qa = []
    for qa in TRANSCRIPT_QA:
        gold_indices, cursor = [], 0
        for mt in TRANSCRIPT_CORPUS:
            for i, ut in enumerate(mt["utterances"]):
                if ut["sequence_id"] in qa["gold_sequence_ids"]:
                    gold_indices.append(cursor + i)
            cursor += len(mt["utterances"])
        trans_qa.append({"question": qa["question"], "kind": qa["kind"], "gold_chunks": gold_indices})

    out: Dict[str, Any] = {"docs": [], "transcript": []}
    for phase, collection, qa_items, gold, top_k in [
        ("docs", DOCS_COLLECTION, docs_qa, docs_gold, top_k_docs),
        ("transcript", TRANSCRIPT_COLLECTION, trans_qa, trans_gold, top_k_transcript),
    ]:
        for cfg_name, use_hybrid in [("pure_vector", False), ("hybrid", True)]:
            with RetrievalEvaluator(
                embedder=embedder, qclient=qclient, collection=collection,
                gold_chunks=gold, top_k=top_k, use_hybrid=use_hybrid,
            ) as ev:
                per_query, agg = ev.evaluate(qa_items)
            out[phase].append({
                "config": cfg_name, "top_k": top_k,
                "aggregate": asdict(agg),
                "per_query": [asdict(r) for r in per_query],
            })
            log.info("[retrieval] %s/%s Hit@%d=%.4f MRR=%.4f TokenR=%.4f",
                     phase, cfg_name, top_k, agg.hit_rate_at_k, agg.mrr, agg.mean_recall_at_k)
    return out


# --------------------------------------------------------------------------- #
# Answer generation per config
# --------------------------------------------------------------------------- #


def collect_answers(creds: Dict[str, str], hybrid: bool, top_k_docs: int, top_k_transcript: int) -> Dict[str, List[Dict[str, Any]]]:
    # QUAN TRỌNG: set settings TRƯỚC khi tạo pipeline (RAGPipeline chốt hybrid_enabled lúc init).
    settings.hybrid_enabled = hybrid
    pipe = RAGPipeline(
        answer_model=creds["model"],
        answer_base_url=creds["base_url"],
        answer_api_key=creds["api_key"],
    )
    pipe.hybrid_enabled = hybrid  # chắc chắn

    for doc in DOCS_CORPUS:
        pipe.upsert_doc_chunks(DOCS_COLLECTION, doc["chunks"], source=doc["source"])
    for mt in TRANSCRIPT_CORPUS:
        for ut in mt["utterances"]:
            pipe.upsert_transcript_point(
                collection=TRANSCRIPT_COLLECTION, meeting_id=mt["meeting_id"],
                sequence_id=ut["sequence_id"], speaker=ut["speaker"], text=ut["text"],
            )

    out: Dict[str, List[Dict[str, Any]]] = {"docs": [], "transcript": []}
    log.info("[answers/%s] docs: generating %d...", "hybrid" if hybrid else "pure", len(DOCS_QA))
    for i, qa in enumerate(DOCS_QA, 1):
        r = pipe.ask_docs(DOCS_COLLECTION, qa["question"], top_k=top_k_docs)
        out["docs"].append({
            "question": qa["question"], "answer": r.answer, "contexts": r.contexts,
            "ground_truth": qa["ground_truth_answer"], "kind": qa["kind"],
            "hit_sources": [h.source for h in r.hits], "latency_ms": r.latency_ms,
        })
        if i % 4 == 0:
            log.info("  docs %d/%d", i, len(DOCS_QA))
    log.info("[answers/%s] transcript: generating %d...", "hybrid" if hybrid else "pure", len(TRANSCRIPT_QA))
    for i, qa in enumerate(TRANSCRIPT_QA, 1):
        r = pipe.ask_transcript(TRANSCRIPT_COLLECTION, MEETING_ID, qa["question"],
                                top_k=top_k_transcript, window_size=2)
        out["transcript"].append({
            "question": qa["question"], "answer": r.answer, "contexts": r.contexts,
            "ground_truth": qa["ground_truth_answer"], "kind": qa["kind"],
            "hit_sequence_ids": [h.sequence_id for h in r.hits if h.sequence_id is not None],
            "latency_ms": r.latency_ms,
        })
        if i % 4 == 0:
            log.info("  transcript %d/%d", i, len(TRANSCRIPT_QA))
    return out


# --------------------------------------------------------------------------- #
# RAGAS
# --------------------------------------------------------------------------- #


def run_ragas(rows: List[Dict[str, Any]], llm, emb, metrics_to_run: List[str],
              max_contexts: int, max_context_chars: int, max_workers: int, timeout: int) -> Dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(max_workers=max_workers, timeout=timeout)
    except Exception:  # noqa: BLE001
        run_config = None

    metric_objs = {
        "faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
        "context_precision": context_precision, "context_recall": context_recall,
    }
    sel = [metric_objs[m] for m in metrics_to_run if m in metric_objs]
    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"] or "(không sinh được câu trả lời)",
            "contexts": [c[:max_context_chars] for c in (r["contexts"] or [])[:max_contexts]] or ["(không có ngữ cảnh)"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])
    t0 = time.perf_counter()
    kwargs = dict(metrics=sel, llm=llm, embeddings=emb, raise_exceptions=False)
    if run_config is not None:
        kwargs["run_config"] = run_config
    result = evaluate(ds, **kwargs)
    dt = time.perf_counter() - t0

    metrics: Dict[str, Any] = {}
    per_query: List[Dict[str, Any]] = []
    try:
        df = result.to_pandas()
        for col in metrics_to_run:
            if col in df.columns and not df[col].isna().all():
                metrics[col] = round(float(df[col].mean(skipna=True)), 4)
            else:
                metrics[col] = None
        for _, row in df.iterrows():
            per_query.append({
                "question": row.get("question", ""),
                **{m: (None if (row.get(m) is None or (isinstance(row.get(m), float) and row.get(m) != row.get(m)))
                       else round(float(row.get(m)), 4)) for m in metrics_to_run},
            })
    except Exception as e:  # noqa: BLE001
        log.warning("parse RAGAS result failed: %s", e)
    return {"n": len(rows), "duration_sec": round(dt, 2), "metrics": metrics, "per_query": per_query}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Override answer+judge model (mặc định LLM_MODEL trong .env)")
    p.add_argument("--configs", default="pure_vector,hybrid")
    p.add_argument("--metrics", default="answer_relevancy,faithfulness,context_precision,context_recall")
    p.add_argument("--top-k-docs", type=int, default=5)
    p.add_argument("--top-k-transcript", type=int, default=3)
    p.add_argument("--max-contexts", type=int, default=4)
    p.add_argument("--max-context-chars", type=int, default=1000)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--skip-ragas", action="store_true", help="Chỉ chạy retrieval matrix")
    p.add_argument("--out", default=str(ROOT / "results" / "comprehensive.json"))
    args = p.parse_args()

    creds = read_env_creds()
    if args.model:
        creds["model"] = args.model
    metrics_to_run = [m.strip() for m in args.metrics.split(",") if m.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    log.info("Model=%s base=%s configs=%s metrics=%s", creds["model"], creds["base_url"], configs, metrics_to_run)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Loading embedding model ===")
    embedder = EmbeddingService()
    log.info("  dim=%d", embedder.dim)

    summary: Dict[str, Any] = {
        "config": {
            "model": creds["model"], "embedding_model": settings.embedding_model,
            "hybrid_vector_weight": settings.hybrid_vector_weight,
            "hybrid_term_weight": settings.hybrid_term_weight,
            "top_k_docs": args.top_k_docs, "top_k_transcript": args.top_k_transcript,
            "max_contexts": args.max_contexts, "max_context_chars": args.max_context_chars,
            "metrics": metrics_to_run,
        },
        "retrieval": {},
        "ragas": {},
    }

    # 1) Retrieval matrix (nhanh).
    log.info("=== Retrieval matrix ===")
    summary["retrieval"] = run_retrieval_matrix(embedder, args.top_k_docs, args.top_k_transcript)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote partial (retrieval) -> %s", args.out)

    if args.skip_ragas:
        log.info("--skip-ragas → done.")
        return

    # 2) RAGAS matrix.
    log.info("=== Loading RAGAS judge (%s) + embeddings ===", creds["model"])
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=creds["model"], base_url=creds["base_url"], api_key=creds["api_key"],
                     temperature=0.0, max_tokens=2048, timeout=args.timeout)
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    for cfg_name in configs:
        hybrid = CONFIGS[cfg_name]["hybrid_enabled"]
        log.info("=== Answers + RAGAS for config=%s (hybrid=%s) ===", cfg_name, hybrid)
        answers = collect_answers(creds, hybrid, args.top_k_docs, args.top_k_transcript)
        # Lưu raw answers (để backfill metric mà không gọi lại LLM).
        (out_dir / f"raw_answers_{cfg_name}.json").write_text(
            json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")

        summary["ragas"][cfg_name] = {}
        for phase, rows in answers.items():
            log.info("  RAGAS %s/%s (%d rows, model=%s)...", cfg_name, phase, len(rows), creds["model"])
            res = run_ragas(rows, llm, emb, metrics_to_run,
                            args.max_contexts, args.max_context_chars, args.max_workers, args.timeout)
            summary["ragas"][cfg_name][phase] = res
            log.info("  -> %s", res["metrics"])
            # checkpoint sau mỗi (config, phase).
            Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Wrote final -> %s", args.out)

    # Tóm tắt.
    print("\n" + "=" * 72)
    print("COMPREHENSIVE RAG BENCHMARK SUMMARY")
    print("=" * 72)
    print("\n[RETRIEVAL]")
    for phase, rows in summary["retrieval"].items():
        for r in rows:
            a = r["aggregate"]
            print(f"  {phase:<10s} {r['config']:<12s} Hit@{r['top_k']}={a['hit_rate_at_k']:.4f} "
                  f"MRR={a['mrr']:.4f} TokenR={a['mean_recall_at_k']:.4f} TokenP={a['mean_precision_at_k']:.4f}")
    print("\n[RAGAS]")
    for cfg_name, phases in summary["ragas"].items():
        for phase, res in phases.items():
            print(f"  {cfg_name:<12s} {phase:<10s} n={res['n']:<3d} {res['metrics']}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
