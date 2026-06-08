# Evaluation harness

RAGAS + deterministic retrieval benchmark for the RAG server. Runs **in-process** against an
in-memory Qdrant (`:memory:`) using the real production modules — no Docker / Qdrant server /
Redis required. Results and methodology are summarized in [../docs/evaluation.md](../docs/evaluation.md).

## Layout

| File | Role |
|------|------|
| `gold_dataset.py` | Vietnamese gold corpus + Q&A (12 docs, 10 transcript). |
| `pipeline.py` | `RAGPipeline` — in-memory Qdrant + production embedding/search/fuse/rerank + answer generation. |
| `retrieval_eval.py` | `RetrievalEvaluator` — Hit@K, MRR, token recall/precision, latency. |
| `run_eval.py` | Retrieval matrix + optional RAGAS subset (provides ingest helpers reused below). |
| `run_comprehensive.py` | **Main runner** — retrieval matrix + full RAGAS (4 metrics) × {pure_vector, hybrid}. |
| `gen_report.py` | Renders `results/comprehensive.json` → `REPORT_COMPREHENSIVE.md`. |
| `REPORT.md` / `REPORT_COMPREHENSIVE.md` | Human-readable reports. |
| `results/` | Generated JSON (git-ignored). |

## Prerequisites

- Python env with `ragas`, `datasets`, `langchain-openai`, `langchain-community`,
  `sentence-transformers`, `torch`, `qdrant-client` (the project `venv`).
- An OpenAI-compatible LLM endpoint for the judge + answer generation. Credentials are read from
  `../rag_server/.env` (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) — **no keys are hardcoded**.
  A large-context model (e.g. 26B / 32K) is needed for `faithfulness` and `context_*`.

## Run

```bash
# from rag_base/, with the venv active
unset OPENROUTER_API_KEY            # avoid stray env vars

# retrieval matrix only (fast, deterministic, ~30s)
python -m eval.run_comprehensive --skip-ragas

# full benchmark: retrieval + RAGAS (4 metrics × 2 configs), ~1h on a 26B judge
python -m eval.run_comprehensive --out eval/results/comprehensive.json

# render the markdown report
python -m eval.gen_report
```

Useful flags: `--model <id>` (override judge), `--configs pure_vector,hybrid`,
`--metrics answer_relevancy,faithfulness,context_precision,context_recall`,
`--max-workers`, `--timeout`. The runner checkpoints `comprehensive.json` after each
(config, phase) and writes `results/raw_answers_{config}.json` for metric back-fill.
