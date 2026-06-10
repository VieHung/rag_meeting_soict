"""Main evaluation script — retrieval + RAGAS.

Workflow:
1. Khởi tạo Qdrant in-memory + EmbeddingService (production code).
2. Ingest 2 corpus: DOCS_CORPUS + TRANSCRIPT_CORPUS.
3. Chạy **retrieval eval** (deterministic, no LLM) cho cả 4 config:
     - docs / transcript × {pure vector, hybrid BM25+vector}
4. Chạy **answer generation + RAGAS LLM judge** trên subset nhỏ (LLM judge dùng
   LM Studio ``google/gemma-4-e4b`` — model nhỏ duy nhất đang load sẵn).
5. Dump tất cả kết quả ra ``eval/results/`` + in bảng tóm tắt.

Có thể chạy:
    python -m eval.run_eval --phase retrieval            # retrieval eval nhanh (~30s)
    python -m eval.run_eval --phase all --ragas-subset 4 # retrieval + RAGAS subset

Để chạy benchmark đầy đủ (4 metric × 2 config), dùng ``eval.run_comprehensive``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
RAG_SERVER = ROOT.parent / "rag_server"
sys.path.insert(0, str(RAG_SERVER))

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, VectorParams  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.embedding import EmbeddingService  # noqa: E402
from eval.gold_dataset import (  # noqa: E402
    DOCS_CORPUS,
    DOCS_QA,
    TRANSCRIPT_CORPUS,
    TRANSCRIPT_QA,
)
from eval.retrieval_eval import RetrievalEvaluator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("rag_eval")


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def _ensure_collection(qclient: QdrantClient, name: str, dim: int) -> None:
    if name in {c.name for c in qclient.get_collections().collections}:
        return
    qclient.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    qclient.create_payload_index(
        collection_name=name, field_name="source", field_schema="keyword"
    )


def ingest_docs(qclient: QdrantClient, embedder: EmbeddingService, collection: str) -> None:
    import uuid
    from qdrant_client.models import PointStruct

    _ensure_collection(qclient, collection, embedder.dim)
    for doc in DOCS_CORPUS:
        vecs = embedder.embed_texts(doc["chunks"])
        points = []
        for i, (chunk, vec) in enumerate(zip(doc["chunks"], vecs)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "text": chunk,
                        "source": doc["source"],
                        "doc_id": str(uuid.uuid4()),
                        "chunk_index": i,
                        "chunk_total": len(doc["chunks"]),
                    },
                )
            )
        qclient.upsert(collection_name=collection, points=points)
    log.info("Ingested %d docs into '%s'", len(DOCS_CORPUS), collection)


def ingest_transcripts(
    qclient: QdrantClient, embedder: EmbeddingService, collection: str
) -> None:
    import uuid
    from qdrant_client.models import PointStruct, FieldCondition, Filter, MatchValue, Range

    _ensure_collection(qclient, collection, embedder.dim)
    # Index cho filter meeting_id, sequence_id, speaker.
    for field, schema in (
        ("meeting_id", "keyword"),
        ("sequence_id", "integer"),
        ("speaker", "keyword"),
    ):
        try:
            qclient.create_payload_index(
                collection_name=collection, field_name=field, field_schema=schema
            )
        except Exception:
            pass
    for mt in TRANSCRIPT_CORPUS:
        for ut in mt["utterances"]:
            vec = embedder.embed_query(ut["text"])
            qclient.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "meeting_id": mt["meeting_id"],
                            "sequence_id": ut["sequence_id"],
                            "speaker": ut["speaker"],
                            "text": ut["text"],
                            "context": "",
                            "context_status": "disabled",
                        },
                    )
                ],
            )
    log.info("Ingested %d transcript meetings into '%s'", len(TRANSCRIPT_CORPUS), collection)


# --------------------------------------------------------------------------- #
# Build gold_chunks list (text vàng) cho retrieval eval
# --------------------------------------------------------------------------- #


def build_docs_gold_chunks() -> List[str]:
    """Flatten DOCS_CORPUS → list text chunks theo (doc_idx, chunk_idx)."""
    out: List[str] = []
    for doc in DOCS_CORPUS:
        out.extend(doc["chunks"])
    return out


def build_transcript_gold_chunks() -> List[str]:
    """Flatten TRANSCRIPT_CORPUS → list text utterances theo (mt_idx*1000 + seq)."""
    out: List[str] = []
    for mt in TRANSCRIPT_CORPUS:
        for ut in mt["utterances"]:
            out.append(ut["text"])
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run_retrieval_phase(
    qclient: QdrantClient,
    embedder: EmbeddingService,
    phase: str,
    collection: str,
    qa_items: List[Dict[str, Any]],
    gold_chunks: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Chạy 2 config (pure-vector, hybrid) cho 1 phase. Trả list kết quả để dump."""
    out: List[Dict[str, Any]] = []
    for cfg_name, use_hybrid in [("pure_vector", False), ("hybrid_bm25_vector", True)]:
        log.info("[%s] Running retrieval: %s (top_k=%d)...", phase, cfg_name, top_k)
        with RetrievalEvaluator(
            embedder=embedder,
            qclient=qclient,
            collection=collection,
            gold_chunks=gold_chunks,
            top_k=top_k,
            use_hybrid=use_hybrid,
        ) as ev:
            per_query, agg = ev.evaluate(qa_items)
        out.append(
            {
                "phase": phase,
                "config": cfg_name,
                "top_k": top_k,
                "aggregate": asdict(agg),
                "per_query": [asdict(r) for r in per_query],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        default="all",
        choices=["retrieval", "ragas", "all"],
    )
    parser.add_argument("--top-k-docs", type=int, default=5)
    parser.add_argument("--top-k-transcript", type=int, default=3)
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
    args = parser.parse_args()

    log.info("=== Loading embedding model ===")
    t0 = time.perf_counter()
    embedder = EmbeddingService()
    log.info("  Loaded in %.1fs, dim=%d", time.perf_counter() - t0, embedder.dim)

    log.info("=== Setting up Qdrant in-memory ===")
    qclient = QdrantClient(location=":memory:")

    DOCS_COLLECTION = "docs-eval-ragas"
    TRANSCRIPT_COLLECTION = "meeting-bt2-eval"
    ingest_docs(qclient, embedder, DOCS_COLLECTION)
    ingest_transcripts(qclient, embedder, TRANSCRIPT_COLLECTION)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "config": {
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "hybrid_vector_weight": settings.hybrid_vector_weight,
            "hybrid_term_weight": settings.hybrid_term_weight,
            "top_k_docs": args.top_k_docs,
            "top_k_transcript": args.top_k_transcript,
        },
        "phases": {},
    }

    # ------------------------------------------------------------------ #
    # Phase 1: Retrieval eval (deterministic, no LLM)
    # ------------------------------------------------------------------ #
    if args.phase in ("retrieval", "all"):
        log.info("=== Phase: Retrieval eval (no LLM) ===")
        # Docs phase: build gold_chunks list.
        docs_gold = build_docs_gold_chunks()
        # Q&A items cần gold_chunks ở dạng index.
        docs_qa_for_eval = []
        for qa in DOCS_QA:
            docs_qa_for_eval.append(
                {
                    "id": f"docs_{len(docs_qa_for_eval) + 1:02d}",
                    "question": qa["question"],
                    "kind": qa["kind"],
                    "gold_chunks": qa["gold_chunks"],  # index
                }
            )
        docs_results = run_retrieval_phase(
            qclient, embedder, "docs", DOCS_COLLECTION,
            docs_qa_for_eval, docs_gold, args.top_k_docs,
        )
        summary["phases"]["docs"] = docs_results

        # Transcript phase: gold là sequence_id, cần map sang index trong gold_chunks.
        trans_gold = build_transcript_gold_chunks()
        trans_qa_for_eval = []
        for qa in TRANSCRIPT_QA:
            # Map sequence_id -> index trong trans_gold.
            # trans_gold layout: meeting[0] utters (0..n-1), meeting[1] utters (n..)
            gold_indices = []
            cursor = 0
            for mt in TRANSCRIPT_CORPUS:
                for i, ut in enumerate(mt["utterances"]):
                    if ut["sequence_id"] in qa["gold_sequence_ids"]:
                        gold_indices.append(cursor + i)
                cursor += len(mt["utterances"])
            trans_qa_for_eval.append(
                {
                    "id": f"trans_{len(trans_qa_for_eval) + 1:02d}",
                    "question": qa["question"],
                    "kind": qa["kind"],
                    "gold_chunks": gold_indices,
                }
            )
        trans_results = run_retrieval_phase(
            qclient, embedder, "transcript", TRANSCRIPT_COLLECTION,
            trans_qa_for_eval, trans_gold, args.top_k_transcript,
        )
        summary["phases"]["transcript"] = trans_results

    # ------------------------------------------------------------------ #
    # Phase 2: RAGAS LLM judge (subset nhỏ vì LLM judge chậm với LM Studio)
    # ------------------------------------------------------------------ #
    if args.phase in ("ragas", "all"):
        log.info("=== Phase: RAGAS LLM judge (subset=%d) ===", args.ragas_subset)
        try:
            import torch
            from datasets import Dataset
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_openai import ChatOpenAI
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except Exception as e:  # noqa: BLE001
            log.error("Skip RAGAS — import error: %s", e)
        else:
            LM_BASE = "http://192.168.240.1:1234/v1"
            LM_KEY = "sk-lm-I4p1UFW1:ADiMgh6qgZUwq4VixJH6"
            LM_MODEL = "google/gemma-4-e4b"

            llm = ChatOpenAI(
                model=LM_MODEL, base_url=LM_BASE, api_key=LM_KEY,
                temperature=0.0, max_tokens=2048, timeout=180,
            )
            emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
            )

            from eval.pipeline import RAGPipeline  # noqa: WPS433

            rag_pipeline = RAGPipeline(answer_model=LM_MODEL)
            # Ingest lại (QdrantPipeline có instance riêng, đã share settings)
            rag_pipeline.upsert_doc_chunks  # touch
            ingest_docs(rag_pipeline._qclient, embedder, DOCS_COLLECTION := "docs-eval-ragas")
            ingest_transcripts(rag_pipeline._qclient, embedder, "meeting-bt2-eval")

            ragas_results: List[Dict[str, Any]] = []
            # meeting_id phải khớp với gold_dataset.TRANSCRIPT_CORPUS[0]["meeting_id"]
            for phase, qa, col, mid in [
                ("docs", DOCS_QA, "docs-eval-ragas", None),
                ("transcript", TRANSCRIPT_QA, "meeting-bt2-eval", "meeting-bt2-eval"),
            ]:
                rows = []
                for i, qa_item in enumerate(qa[: args.ragas_subset], 1):
                    if phase == "docs":
                        r = rag_pipeline.ask_docs(col, qa_item["question"], top_k=args.top_k_docs)
                    else:
                        r = rag_pipeline.ask_transcript(
                            col, mid, qa_item["question"],
                            top_k=args.top_k_transcript, window_size=2,
                        )
                    rows.append(
                        {
                            "question": qa_item["question"],
                            "answer": r.answer or "(không sinh được câu trả lời)",
                            "contexts": r.contexts or ["(không có ngữ cảnh)"],
                            "ground_truth": qa_item["ground_truth_answer"],
                        }
                    )
                ds = Dataset.from_list(rows)
                log.info(
                    "  RAGAS %s: %d rows (this may take several minutes)...",
                    phase, len(rows),
                )
                t0 = time.perf_counter()
                try:
                    metrics_to_run = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]
                    metric_objs = {
                        "faithfulness": faithfulness,
                        "answer_relevancy": answer_relevancy,
                        "context_precision": context_precision,
                        "context_recall": context_recall,
                    }
                    sel = [metric_objs[m] for m in metrics_to_run if m in metric_objs]
                    # Truncate contexts để prompt RAGAS vừa 8K context của e4b.
                    for r in rows:
                        r["contexts"] = [c[:600] for c in (r.get("contexts") or [])[:3]] or ["(không có ngữ cảnh)"]
                    ds = Dataset.from_list(rows)
                    out = evaluate(
                        ds,
                        metrics=sel,
                        llm=llm,
                        embeddings=emb,
                        raise_exceptions=False,
                    )
                    df = out.to_pandas()
                    metrics = {
                        col: (None if col not in df.columns else (
                            None if df[col].isna().all() else float(df[col].mean(skipna=True))
                        ))
                        for col in metrics_to_run
                    }
                    metrics = {
                        k: (None if v is None else round(v, 4))
                        for k, v in metrics.items()
                    }
                except Exception as e:  # noqa: BLE001
                    log.error("  RAGAS %s failed: %s", phase, e)
                    metrics = {"error": str(e)[:200]}
                dt = time.perf_counter() - t0
                log.info("  RAGAS %s done in %.1fs -> %s", phase, dt, metrics)
                ragas_results.append(
                    {"phase": phase, "n": len(rows), "duration_sec": round(dt, 2), "metrics": metrics}
                )
            summary["phases"]["ragas"] = ragas_results

    # ------------------------------------------------------------------ #
    # Save + print
    # ------------------------------------------------------------------ #
    out_path = out_dir / "metrics.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote metrics -> %s", out_path)

    # Pretty print summary.
    print("\n" + "=" * 70)
    print("RAG EVALUATION SUMMARY")
    print("=" * 70)
    for phase, results in summary["phases"].items():
        if phase == "ragas":
            print(f"\n[{phase.upper()}] (LLM judge, subset)")
            for r in results:
                print(f"  {r['phase']:<10s}  n={r['n']:<3d}  {r['metrics']}")
        else:
            print(f"\n[{phase.upper()}]")
            for r in results:
                agg = r["aggregate"]
                print(
                    f"  {r['config']:<24s}  "
                    f"n={agg['n']:<3d}  "
                    f"Hit@{r['top_k']}={agg['hit_rate_at_k']:.4f}  "
                    f"MRR={agg['mrr']:.4f}  "
                    f"TokenR={agg['mean_recall_at_k']:.4f}  "
                    f"TokenP={agg['mean_precision_at_k']:.4f}  "
                    f"lat_p50={agg['latency_p50_ms']:.1f}ms"
                )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
