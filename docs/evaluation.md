# Evaluation

The retrieval and end-to-end answer quality of the server are measured with two complementary
suites:

1. **Deterministic retrieval metrics** (no LLM) — Hit@K, MRR, token recall/precision, latency.
2. **RAGAS** (LLM-as-judge) — answer relevancy, faithfulness, context precision, context recall.

The harness lives in [`eval/`](../eval/) and runs **in-process** against an in-memory Qdrant
(`:memory:`), exercising the real production code paths (embedding, search, `fuse`, rerank) —
so no Docker, Qdrant server, or Redis is required to evaluate. See [eval/README.md](../eval/README.md)
to run it.

---

## Dataset

A Vietnamese gold dataset ([`eval/gold_dataset.py`](../eval/gold_dataset.py)):

- **Documents** — 3 Markdown files × 4 chunks = 12 chunks (leave policy, travel allowance,
  AI Meeting Box specs). **12 Q&A** (factual / lookup_number / lookup_spec), each with gold
  chunks + ground-truth answer.
- **Transcript** — 1 meeting × 12 utterances (Q4 budget + hiring). **10 Q&A**
  (factual / lookup_speaker / lookup_number / multi_hop), each with gold `sequence_id`s +
  ground-truth answer.

Two retrieval configurations are compared: **pure_vector** and **hybrid** (BM25 + vector, weights 0.7 / 0.3).

---

## Latest comprehensive results

> Judge model **Gemma-4-26B** (large context), full dataset, all four RAGAS metrics.
> Full report: [eval/REPORT_COMPREHENSIVE.md](../eval/REPORT_COMPREHENSIVE.md).

### Retrieval (deterministic)

| Phase | Config | Hit@K | MRR | TokenRecall | TokenPrec |
|-------|--------|-------|-----|-------------|-----------|
| docs | pure_vector | 0.5833 | 0.3819 | 0.7410 | 0.2000 |
| docs | **hybrid** | **0.6667** | **0.3986** | **0.7925** | **0.2145** |
| transcript | pure_vector | 1.0000 | 0.8833 | 0.9653 | 0.4870 |
| transcript | **hybrid** | 1.0000 | **0.9333** | **1.0000** | **0.5261** |

Hybrid retrieval is **≥ pure-vector on every metric**.

### RAGAS (full dataset)

| Phase | Config | answer_relevancy | faithfulness | context_precision | context_recall |
|-------|--------|------------------|--------------|-------------------|----------------|
| docs | pure_vector | 0.761 | **1.00** | 0.903 | **1.00** |
| docs | hybrid | 0.739 | **1.00** | **0.917** | **1.00** |
| transcript | pure_vector | 0.661 | 0.90 | 0.767 | 0.75 |
| transcript | hybrid | 0.663 | 0.867 | **0.817** | **0.85** |

---

## Key findings

- **Document RAG is near-perfect on grounding**: faithfulness 1.00 and context recall 1.00 —
  answers stay faithful to retrieved context and the retrieved context contains the gold facts.
- **Transcript RAG is solid but lower**, with `answer_relevancy ≈ 0.66` the weakest metric across
  the system — a candidate for a better answer-generation prompt.
- **Hybrid improves the transcript context metrics** (context recall +0.10, precision +0.05),
  consistent with the retrieval results. `answer_relevancy` / `faithfulness` stay roughly flat
  between configs: once relevant context is present, answer quality depends more on the generator
  LLM than on retrieval order.
- An earlier run with an 8K-context judge could not compute `faithfulness` / `context_*`
  (prompt overflow); the large-context judge resolves this.

---

## Notes on metric validity

- The deterministic `_is_relevant` uses a token-Jaccard threshold (≥ 0.3), which can
  **under-estimate** retrieval quality for heavily paraphrased questions (the cosine match is
  correct but token overlap is low). A semantic relevance check (cosine ≥ 0.7) would tighten this.
- Raw answers are saved to `eval/results/raw_answers_{config}.json` so new metrics can be
  back-filled without re-calling the LLM.

## Bugs found during evaluation

Evaluation surfaced and fixed three real bugs in the server/harness:
1. A GC-ordering bug where an evaluator's `__del__` reset `settings.hybrid_enabled` after the next
   evaluator had enabled it — replaced with explicit `restore()` + context-manager.
2. A `meeting_id` filter mismatch that returned 0 hits (empty answers) for the transcript flow.
3. `pydantic-settings` rejecting unknown host env vars — fixed with `extra = "ignore"` in
   `config.py`.
