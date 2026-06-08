# Inventory — `rag_base/rag_server`

> Tài liệu mô tả **đầy đủ những gì đã có** trong thư mục `rag_base/rag_server` tính đến
> snapshot 2026-06-07. Phục vụ cho onboarding, code review, và làm baseline trước
> khi đánh giá chất lượng bằng RAGAS.

---

## 1. Tổng quan

`rag_base/rag_server` là một **FastAPI service** thực hiện hai luồng RAG tách biệt
nhưng dùng chung hạ tầng:

| Luồng | Namespace | Mục đích |
|-------|-----------|----------|
| **Phase 1 — Tài liệu** | `/embed/*`, `/query/` | Upload PDF/DOCX/TXT/MD, embed, semantic search. |
| **Phase 2 — Transcript cuộc họp** | `/transcript/*`, `/query/transcript` | Nhập từng câu hội thoại, build **context tích lũy** qua LLM, truy vấn kèm **window ±N câu**. |

Hạ tầng phụ thuộc:

- **Qdrant** (vector DB, cosine, dim=384) — `qdrant/qdrant:v1.10.0`.
- **Redis** (atomic sequence counter per collection, TTL 7 ngày, self-healing từ Qdrant) — `redis:7.2-alpine`.
- **LLM self-host** (Ollama / OpenAI-compatible / Gemini) — dùng để build `context` (Phase 2). Có thể tắt (`LLM_PROVIDER=none`).
- **Embedding model** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, đa ngôn ngữ, tiếng Việt).
- **Reranker (optional)** — `none | local (CrossEncoder) | http` (TEI/Infinity/Xinference), gated, mặc định TẮT.
- **Hybrid retrieval (optional)** — fusion BM25 + vector cosine (RAGFlow-style), gated, mặc định TẮT.

Phiên bản: `2.0.0` (xem `app/main.py:71`).

---

## 2. Cấu trúc thư mục

```
rag_base/
├── plan.md                          # Plan Phase 2 bản gốc (8 endpoint)
├── phase2plan_v2.md                 # Plan Phase 2 bản v2 (4 endpoint)  ← nguồn sự thật
├── STATE.md                         # Trạng thái hệ thống (cập nhật 2026-06-07)
├── API_USAGE.md                     # Hướng dẫn dùng API cho app thiết bị
├── API_USAGE.html                   # Bản HTML của API_USAGE.md
├── test_api_lmstudio.py             # Test gọi LM Studio (OpenAI-compatible) thủ công
├── venv/                            # Python 3.12 venv (Linux/WLS path)
└── rag_server/                      # ── Service chính ──
    ├── README.md                    # Tài liệu API + hướng dẫn vận hành
    ├── .env.example                 # Mẫu cấu hình
    ├── .env                         # Cấu hình thật (đã .gitignore)
    ├── Dockerfile                   # CPU-only torch, cài requirements, tải model
    ├── docker-compose.yml           # qdrant + redis (+ ollama optional) + rag_api
    ├── requirements.txt
    ├── qdrant_storage/              # Volume Qdrant
    ├── redis_data/                  # Volume Redis
    ├── .pytest_cache/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                  # FastAPI app + lifespan (Qdrant/Redis/LLM/Reranker/Worker)
    │   ├── config.py                # Pydantic Settings
    │   ├── dependencies.py          # DI singleton (TranscriptService, …)
    │   ├── routers/
    │   │   ├── embed.py             # /embed/* (Phase 1)
    │   │   ├── query.py             # /query/ (Phase 1) + /query/transcript (Phase 2)
    │   │   └── transcript.py        # /transcript/* (Phase 2)
    │   ├── schemas/
    │   │   ├── embed.py
    │   │   ├── query.py
    │   │   └── transcript.py
    │   ├── services/
    │   │   ├── embedding.py         # SentenceTransformer singleton
    │   │   ├── vector_store.py      # QdrantService (Phase 1, có create/list/delete collection)
    │   │   ├── transcript_store.py  # TranscriptStore (Phase 2, scroll + filter sequence_id range)
    │   │   ├── sequence_manager.py  # Redis INCR + lazy-init + rebuild từ Qdrant
    │   │   ├── context_builder.py   # Build context tích lũy bằng LLM (D3)
    │   │   ├── transcript_service.py# Orchestrator: embed, query, window, context
    │   │   ├── llm_client.py        # LLMClient: ollama/gemini/openai/none
    │   │   ├── retrieval.py         # Hybrid fusion (BM25 + vector cosine) — OPTIONAL
    │   │   ├── reranker.py          # Reranker: none/local/http — OPTIONAL
    │   │   └── document_parser.py   # PDF/DOCX/TXT/MD parser
    │   ├── workers/
    │   │   └── context_worker.py    # asyncio.Queue + 1 worker FIFO (D9)
    │   └── utils/
    │       ├── chunking.py          # Tách chunk (paragraph → sentence, 512 chars, 64 overlap)
    │       └── redis_client.py      # Singleton async Redis
    └── tests/
        ├── test_api.py              # Pytest Phase 1: collection/embed/query/delete/edge/conc
        ├── test_transcript_api.py   # Pytest Phase 2: embed→sequence→context→query(window)
        └── test_sequence_manager.py # Unit: atomic INCR, lazy-init, rebuild từ Qdrant
```

---

## 3. Endpoint đã wire (xem chi tiết `rag_server/README.md`)

### 3.1. System

| Method | Path | Mô tả |
|--------|------|-------|
| `GET` | `/health` | Health check Qdrant + Redis + báo LLM provider. Trả 503 nếu Qdrant/Redis chết (`app/main.py:90`). |

### 3.2. Phase 1 — Tài liệu (`/embed/*`, `/query/`)

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/embed/file` | Upload file (PDF/DOCX/TXT/MD, ≤50MB). Background embed. |
| `POST` | `/embed/text` | Embed text thường. Background embed. |
| `GET`  | `/embed/collections` | List collections. |
| `POST` | `/embed/collections` | Tạo collection (form `name`). |
| `DELETE` | `/embed/collections` | Xóa collection (form `name`). |
| `GET`  | `/embed/info` / `/embed/info/{collection}` | Thông tin collection (vector_size, points_count, status, distance). **Có check tồn tại trước, không tạo nhầm collection rỗng.** |
| `GET`  | `/embed/{collection}/documents` | Liệt kê tài liệu (group theo `doc_id`). |
| `DELETE` | `/embed/{collection}/source/{source}` | Xóa theo tên file. |
| `DELETE` | `/embed/{collection}/doc/{doc_id}` | Xóa theo UUID. |
| `POST` | `/query/` | Truy vấn ngữ nghĩa. Hỗ trợ `collection`, `top_k`, `source_filter`, `score_threshold`. **Có áp hybrid/rerank gated.** |

### 3.3. Phase 2 — Transcript (`/transcript/*`, `/query/transcript`)

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/transcript/{collection}/embed` | Lưu 1 câu, gán `sequence_id`, **enqueue** build context. Trả `202` ngay. **Yêu cầu prefix `meeting-`.** |
| `POST` | `/query/transcript` | Truy vấn, trả kết quả kèm `window.before/after` + `context`. |
| `GET`  | `/transcript/{collection}/context` | Lấy context mới nhất. `?sequence_id=N` lấy tại câu N. |
| `GET`  | `/transcript/{collection}/segments` | Liệt kê theo khoảng `sequence_id` (debug/biên bản). |

**Collection naming (quy ước Phase 2 v2):**

| Loại | Prefix | Endpoint dùng |
|------|--------|---------------|
| Transcript | `meeting-{uuid}` | `/transcript/*`, `/query/transcript` |
| Tài liệu | `docs-{uuid}` | `/embed/*`, `/query/` |

→ `meeting_id` **được suy ra từ tên collection** (`collection.removeprefix("meeting-")`),
client không gửi. Sai prefix → `400 Bad Request`.

---

## 4. Schema & payload quan trọng

### 4.1. Vector `documents` / `docs-*` (Phase 1)

```jsonc
{
  "text": "...",            // chunk gốc
  "source": "report.pdf",   // tên file nguồn
  "doc_id": "uuid",         // UUID của document
  "chunk_index": 0,
  "chunk_total": 24,
  "file_size": 12345,
  "mime_type": "application/pdf",
  "...": "..."              // extra_metadata tùy ý
}
```

### 4.2. Vector `meeting-*` (Phase 2)

```jsonc
{
  "meeting_id":        "c7cfdf57-...",   // suy ra từ collection
  "sequence_id":       42,               // server gán, atomic, liên tục
  "speaker":           "Đoàn Sỹ Nguyên",
  "speaker_id":        "user_017",        // optional
  "text":              "Chúng ta cần xem lại ngân sách Q4...",
  "timestamp":         "2026-05-11T19:52:27Z",
  "context":           "Tóm tắt hội thoại tính tới câu 42...",  // ← LLM sinh
  "context_status":    "ready",          // pending | processing | ready | failed | disabled
  "context_seq_base":  41,               // sequence_id mà context dựa trên
  "lang":              "vi",             // optional
  "created_at":        "2026-05-11T19:52:27Z"
}
```

### 4.3. State trên Redis

```
Key: rag:seq:{collection}   → INTEGER   # atomic counter, TTL 7 ngày, self-healing
```

Có thể thêm cache `meeting:{meeting_id}:latest_ctx` (JSON `{sequence_id, context, context_status}`)
trong `context_builder.py` — không bắt buộc.

---

## 5. Luồng xử lý Phase 2 (key flows)

### 5.1. Ingest transcript (đồng bộ + enqueue)

```
POST /transcript/{col}/embed
 ├ validate (prefix meeting-, text không rỗng)
 ├ TranscriptStore.ensure_collection()  (lazy)
 ├ SequenceManager.next(col)            → Redis INCR (rebuild từ Qdrant nếu miss)
 ├ EmbeddingService.encode(text)        → vector[384]
 ├ TranscriptStore.upsert_point(..., context_status="pending"|"disabled")
 ├ ContextWorker.enqueue(col, meeting_id, seq)   ← FIFO, đảm bảo thứ tự (D9)
 └ 202 { meeting_id, sequence_id, point_id, context_status }
```

### 5.2. Build context (background FIFO worker)

```
ContextWorker._drain() lấy job (col, mid, seq):
 ├ LLM_PROVIDER=none → set "disabled", return
 ├ seq == 1 (câu đầu) → context="", status="ready", không gọi LLM
 ├ Đọc payload câu (seq-1) từ Qdrant → lấy context[seq-1] và text[seq-1]
 ├ Set point[seq].context_status = "processing"
 ├ LLMClient.summarize(prev_context, new_utterance)   (retry CONTEXT_MAX_RETRY)
 ├ Set payload point[seq] = (context, "ready", context_seq_base=seq-1)
 ├ Cache latest_ctx vào Redis
 └ Lỗi → fallback context=prev_context, status="failed"
```

### 5.3. Query transcript (kèm window)

```
POST /query/transcript
 ├ validate prefix meeting-
 ├ EmbeddingService.encode(query)
 ├ TranscriptStore.search(vector, top_k=3, meeting_id, speaker_filter, threshold)
 ├ Hybrid + Rerank (nếu bật) — gated
 ├ Với mỗi kết quả (seq S): scroll sequence_id ∈ [S-w, S+w], tách before/after
 └ 200 { results: [{ sequence_id, text, score, context, context_status, window }] }
```

### 5.4. Hybrid retrieval (chỉ chạy khi `HYBRID_ENABLED=true`)

`app/services/retrieval.py`:
1. Vector search trả `top_k × N` (mặc định N=3) ứng viên.
2. Min-max normalize điểm vector.
3. Tính BM25 trên **chính tập ứng viên** (không re-index).
4. `score = vector_w * vec_norm + term_w * term_norm` (mặc định 0.7 / 0.3).
5. Sort giảm dần → cắt `top_k`.

Tokenizer tối giản (`\w+` Unicode). Có fallback token-overlap nếu thiếu `rank_bm25`.

### 5.5. Reranker (chỉ chạy khi `RERANK_PROVIDER != "none"`)

`app/services/reranker.py`:
- `none`  — no-op.
- `local` — `sentence_transformers.CrossEncoder` (mặc định `BAAI/bge-reranker-v2-m3`).
- `http`  — gọi `{RERANK_BASE_URL}/rerank` (TEI/Infinity/Xinference).

Điểm rerank chuẩn hóa min-max về `[0,1]`.

---

## 6. Quyết định thiết kế (Design Decisions, từ `phase2plan_v2.md` §4)

| # | Quyết định |
|---|------------|
| D1 | Server gán `sequence_id` (Redis INCR per collection, self-healing). |
| D2 | Query trả `window ±N` câu (mặc định 2, clamp 5). |
| D3 | Build context bằng LLM tóm tắt: `context[N] = LLM_summarize(context[N-1] + transcript[N-1])`. |
| D4 | Mỗi câu transcript = đúng 1 vector (không chunking). |
| D5 | LLM build context self-host **trên chính server VectorDB** (không phải QCS8550). |
| D6 | Context là field trong payload vector (không Redis). |
| D7 | Dùng lại embedding `paraphrase-multilingual-MiniLM-L12-v2`. |
| D8 | `meeting_id` suy từ tên collection, không yêu cầu client gửi. |
| D9 | Build context tuần tự theo collection (asyncio.Queue + 1 worker FIFO). |

Bổ sung Phase 2+ (ghi trong `STATE.md`, đợt 2026-06-07):
- **ContextWorker** thay `BackgroundTasks` để đảm bảo FIFO.
- **Hybrid retrieval + Reranker** (RAGFlow-style), mặc định TẮT, không phá response shape.
- **`/health` kiểm tra dependency**, trả 503 nếu Qdrant/Redis chết.
- **Logging** thay `print()`.
- **Wire `/embed/info` + `/embed/info/{collection}`**.

---

## 7. Cấu hình (`.env` chính, xem `STATE.md`)

```ini
QDRANT_HOST=localhost
REDIS_HOST=localhost
LLM_PROVIDER=openai                  # đang dùng LM Studio (OpenAI-compatible)
LLM_MODEL=gemma-4-26b-a4b-it
LLM_BASE_URL=http://192.168.240.1:1234/v1
LLM_API_KEY=sk-lm-***                # ⚠️ plaintext — cần rotate (xem §10)
TRANSCRIPT_WINDOW_SIZE=2
TRANSCRIPT_MAX_WINDOW_SIZE=5
CONTEXT_MAX_TOKENS=800
CONTEXT_MAX_RETRY=2
CONTEXT_TIMEOUT_SECONDS=30
```

Các biến hybrid/rerank (mặc định TẮT):
```ini
HYBRID_ENABLED=false
HYBRID_VECTOR_WEIGHT=0.7
HYBRID_TERM_WEIGHT=0.3
HYBRID_FETCH_MULTIPLIER=3
RERANK_PROVIDER=none
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

---

## 8. System prompt (LLM build context)

File: `app/services/llm_client.py:21-50`. Cấu trúc:

- **Vai trò:** trợ lý tóm tắt cuộc họp.
- **Input:** dòng 1 = context hiện tại; dòng 3 (sau `---`) = câu mới.
- **Output:** CHỈ một đoạn văn tóm tắt cập nhật, không markdown/bullet/giải thích.
- **6 mức ưu tiên giữ nội dung:** quyết định → con số → tên riêng → chủ đề → action item → quan điểm.
- **4 tình huống tích hợp:** mới / lặp / mâu thuẫn / thứ tự thời gian.
- **Edge cases:** context rỗng, câu dài (cắt theo `CONTEXT_MAX_TOKENS × 1.3` words).

---

## 9. Tests đã có (pytest)

| File | Loại | Yêu cầu |
|------|------|---------|
| `tests/test_api.py` | Integration Phase 1 (12 class) | Server + Qdrant chạy |
| `tests/test_transcript_api.py` | Integration Phase 2 (7 class) | Server + Qdrant + Redis chạy |
| `tests/test_sequence_manager.py` | Unit (5 test) | Redis chạy (+ Qdrant cho 1 test rebuild) |

Coverage:
- **Phase 1:** health, CRUD collection, embed text/file, query (filter/threshold/source), delete (source/doc_id), list docs, info, edge cases (unicode/large), performance, concurrent, error handling.
- **Phase 2:** health, embed (sequence liên tục, empty/whitespace, wrong prefix), segments (range, partial), context (latest + by sequence_id), query (window, clamp, speaker_filter, empty, wrong prefix), Phase 1 backward compat.
- **SequenceManager:** lazy init starts at `seq_start`, `current` reflect `next`, concurrent 100 unique values, rebuild từ Qdrant khi Redis miss.

---

## 10. Điểm cần lưu ý / tồn đọng

> Xem chi tiết trong `STATE.md` mục "Tiếp theo cần làm".

| # | Mức | Vấn đề |
|---|-----|--------|
| 1 | 🔴 | `LLM_API_KEY` đang plaintext trong `.env` — cần rotate + secret injection. |
| 2 | 🟡 | `pytest` trên Windows console (cp1252) in tiếng Việt bị lỗi → chạy với `PYTHONUTF8=1`. |
| 3 | 🟢 | 26 collection cũ (chủ yếu test) trong Qdrant cần dọn. |
| 4 | 🟢 | **Chưa có đánh giá chất lượng retrieval/context bằng RAGAS** — mục tiêu của tài liệu này + bước tiếp theo. |

---

## 11. Phụ thuộc (`requirements.txt`)

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
sentence-transformers==3.0.1
qdrant-client==1.10.1
pypdf==5.1.0
python-docx==1.1.2
openpyxl==3.1.5
pydantic>=2.10.0
pydantic-settings>=2.7.0
python-dotenv==1.0.1
numpy>=1.26.0
httpx==0.27.0
redis==5.0.8
rank-bm25==0.2.2
pytest==8.3.0
pytest-asyncio==0.23.8
```

---

## 12. Tóm tắt 1 câu

> Service đã **wire đầy đủ Phase 1 (tài liệu) + Phase 2 (transcript)**: 13 endpoint
> FastAPI, vector store Qdrant, sequence counter Redis self-healing, context builder
> LLM self-host chạy nền với hàng đợi FIFO, lazy create collection, hybrid retrieval
> + cross-encoder rerank gated (RAGFlow-style), health check dependency, log chuẩn,
> có test pytest Phase 1/Phase 2/unit cho SequenceManager.

---

## 13. Đánh giá chất lượng (RAGAS + retrieval eval)

Có **2 lần đánh giá**:
- `eval/REPORT.md` — lần đầu (2026-06-07), model judge 8K (`gemma-4-e4b`), subset 4 câu, faithfulness/context_* bị NaN do vượt context.
- **`eval/REPORT_COMPREHENSIVE.md` — lần TOÀN DIỆN** (model `gemma-4-26b-a4b-it`,
  **full dataset 12 docs + 10 transcript, đủ 4 RAGAS metrics × 2 config**). Sinh
  bởi `eval/run_comprehensive.py` + `eval/gen_report.py`, dữ liệu
  `eval/results/comprehensive.json`. Tóm tắt:

- **Hybrid BM25+vector > pure vector ở MỌI retrieval metric** (Hit@5 docs 0.58→0.67,
  MRR transcript 0.88→0.93, TokenRecall transcript 0.97→1.00).
- **RAGAS (đủ 4 metric, lần này faithfulness CHẠY được nhờ model 26b context lớn):**
  - **Docs xuất sắc**: faithfulness **1.00**, context_recall **1.00**,
    context_precision ~0.90–0.92, answer_relevancy ~0.74–0.76.
  - **Transcript tốt nhưng thấp hơn**: faithfulness 0.87–0.90, context_recall
    0.75→**0.85** (hybrid), context_precision 0.77→**0.82** (hybrid),
    answer_relevancy ~0.66 (metric thấp nhất — cần xem lại prompt answer-gen).
  - **Hybrid nâng rõ context_recall/precision của transcript** (+0.10 / +0.05),
    nhất quán với retrieval; answer_relevancy & faithfulness gần như đi ngang
    (phụ thuộc LLM sinh hơn là thứ tự retrieval).

Trong quá trình đánh giá đã phát hiện và sửa 3 bug trong `rag_base`:

1. **Bug retrieval hybrid** — `RetrievalEvaluator.__del__` của instance cũ bị
   GC gọi **SAU** khi instance mới đã set `settings.hybrid_enabled=True`,
   reset về `False` ngay trước khi queries chạy. Sửa: bỏ `__del__`, thêm
   `restore()` + `__enter__/__exit__` (xem `eval/retrieval_eval.py:101-145`).
2. **Bug meeting_id filter** — `run_ragas.py` dùng `MEETING_ID="bt2-eval"`
   nhưng gold ingest với `"meeting-bt2-eval"`. Filter trả 0 hits → câu trả
   lời rỗng. Sửa: `MEETING_ID="meeting-bt2-eval"` (xem
   `eval/run_ragas.py:74-78`).
3. **Bug pydantic-settings strict** — `Settings.Config` thiếu `extra="ignore"`,
   crash với `OPENROUTER_API_KEY` có sẵn trong env. Sửa:
   `rag_server/app/config.py:61`.
