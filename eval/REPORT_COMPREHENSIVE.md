# Báo cáo benchmark RAG TOÀN DIỆN — `rag_base/rag_server`

> Sinh tự động từ `eval/results/comprehensive.json` bởi `eval/gen_report.py`.

## 0. Cấu hình

- **Model (answer + RAGAS judge):** `gemma-4-26b-a4b-it`
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dim 384)
- **Hybrid weights:** vector=0.7 / term=0.3
- **top_k:** docs=5, transcript=3
- **RAGAS context cap:** 4 contexts × 1000 chars
- **Metrics:** answer_relevancy, faithfulness, context_precision, context_recall
- **Qdrant:** in-memory (`:memory:`) — production code path (embedding/search/fuse).

## 1. Retrieval (deterministic, no LLM)

| Phase | Config | n | Hit@K | MRR | TokenRecall | TokenPrec | p50 (ms) |
|---|---|---|---|---|---|---|---|
| docs | pure_vector | 12 | 0.5833 | 0.3819 | 0.7410 | 0.2000 | 13.2 |
| docs | hybrid | 12 | 0.6667 | 0.3986 | 0.7925 | 0.2145 | 15.0 |
| transcript | pure_vector | 10 | 1.0000 | 0.8833 | 0.9653 | 0.4870 | 14.8 |
| transcript | hybrid | 10 | 1.0000 | 0.9333 | 1.0000 | 0.5261 | 11.8 |

### 1.1 Per-kind — docs (hybrid)

| Kind | n | Hit@K | MRR | TokenRecall | TokenPrec |
|---|---|---|---|---|---|
| factual | 7 | 0.4286 | 0.4286 | 0.6442 | 0.1728 |
| lookup_number | 3 | 1.0000 | 0.4444 | 1.0000 | 0.3118 |
| lookup_spec | 2 | 1.0000 | 0.2250 | 1.0000 | 0.2148 |

### 1.2 Per-kind — transcript (hybrid)

| Kind | n | Hit@K | MRR | TokenRecall | TokenPrec |
|---|---|---|---|---|---|
| factual | 4 | 1.0000 | 1.0000 | 1.0000 | 0.5416 |
| lookup_speaker | 3 | 1.0000 | 1.0000 | 1.0000 | 0.5807 |
| multi_hop | 2 | 1.0000 | 0.6666 | 1.0000 | 0.4231 |
| lookup_number | 1 | 1.0000 | 1.0000 | 1.0000 | 0.5060 |

## 2. RAGAS (LLM judge — full dataset)

### 2.1 docs

| Metric | pure_vector | hybrid | Δ (hybrid−pure) |
|---|---|---|---|
| answer_relevancy | 0.7610 | 0.7390 | -0.0220 |
| faithfulness | 1.0000 | 1.0000 | +0.0000 |
| context_precision | 0.9028 | 0.9167 | +0.0139 |
| context_recall | 1.0000 | 1.0000 | +0.0000 |

_n=12; thời gian pure=1007.1s, hybrid=1029.77s_

### 2.2 transcript

| Metric | pure_vector | hybrid | Δ (hybrid−pure) |
|---|---|---|---|
| answer_relevancy | 0.6614 | 0.6633 | +0.0019 |
| faithfulness | 0.9000 | 0.8667 | -0.0333 |
| context_precision | 0.7667 | 0.8167 | +0.0500 |
| context_recall | 0.7500 | 0.8500 | +0.1000 |

_n=10; thời gian pure=799.52s, hybrid=798.61s_

## 3. Ghi chú

- Metric **NaN** = RAGAS judge không tách được statement / vượt context / lỗi parse (xem `eval/comprehensive_run.log`). `mean` bỏ qua NaN.
- Retrieval dùng `_is_relevant` token-Jaccard ≥ 0.3 → có thể **under-estimate** với câu paraphrase nặng (xem REPORT.md §3.3).
- Raw answers lưu ở `eval/results/raw_answers_{config}.json` để backfill metric mà không cần gọi lại LLM.
