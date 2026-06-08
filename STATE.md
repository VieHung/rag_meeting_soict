# Trạng thái hệ thống RAG Phase 2

> File này lưu trạng thái hiện tại để agent tiếp theo có thể tiếp tục làm việc.
> Cập nhật lần cuối: 2026-05-20

## Docker Services

| Service | Image | Port | Status |
|---------|-------|------|--------|
| qdrant | qdrant/qdrant:v1.10.0 | 6333, 6334 | ✅ Running |
| redis | redis:7.2-alpine | 6379 | ✅ Running |
| rag_api | rag_server-rag_api:latest | 8000 | ✅ Running |

Services đang chạy ổn định, không cần restart.

## API Endpoints

### Phase 2 — Transcript (v2, đã rút gọn)

| # | Method | Endpoint | Mô tả | Status |
|---|--------|----------|-------|--------|
| 1 | POST | `/transcript/{collection}/embed` | Lưu 1 câu transcript, server gán sequence_id, build context nền | ✅ Verified |
| 2 | POST | `/query/transcript` | Truy vấn transcript kèm window ±N câu + context | ✅ Verified |
| 3 | GET | `/transcript/{collection}/context` | Lấy context (mới nhất hoặc ?sequence_id=N) | ✅ Verified |
| 4 | GET | `/transcript/{collection}/segments` | Liệt kê transcript theo khoảng sequence_id | ✅ Verified |

### Phase 1 — Documents (giữ nguyên)

| Method | Endpoint | Status |
|--------|----------|--------|
| GET | `/embed/collections` | ✅ OK |
| POST | `/embed/{collection}` | ✅ OK |
| DELETE | `/embed/collections/{collection}` | ✅ OK |
| POST | `/query/` | ✅ OK |

### System

| Method | Endpoint | Status |
|--------|----------|--------|
| GET | `/health` | ✅ OK |

## Endpoint cũ đã xoá (theo phase2plan_v2.md)

- `POST /transcript/{collection}/meeting/init` — ❌ đã gỡ (lazy-init tự động)
- `DELETE /transcript/{collection}/meeting/{meeting_id}` — ❌ đã gỡ (dùng `DELETE /embed/collections`)
- `PATCH /transcript/{collection}/context/{sequence_id}` — ❌ đã gỡ (ContextBuilder in-process)
- `GET /transcript/{collection}/context/latest` — ❌ đã gỘP vào `/context` với `?sequence_id=`

## Cấu hình hiện tại (.env)

```ini
QDRANT_HOST=localhost
REDIS_HOST=localhost
LLM_PROVIDER=openai           # LM Studio (OpenAI-compatible)
LLM_MODEL=gemma-4-26b-a4b-it
LLM_BASE_URL=http://192.168.240.1:1234/v1
LLM_API_KEY=sk-lm-***         # ⚠️ Đang để plaintext — CẦN XỬ LÝ
TRANSCRIPT_WINDOW_SIZE=2
TRANSCRIPT_MAX_WINDOW_SIZE=5
CONTEXT_MAX_TOKENS=800
CONTEXT_MAX_RETRY=2
CONTEXT_TIMEOUT_SECONDS=30
```

## System Prompt (LLM Build Context)

File: `app/services/llm_client.py` (cập nhật 2026-05-20)

Prompt hiện tại gồm: vai trò trợ lý tóm tắt, cấu trúc input/output, 6 mức ưu tiên giữ nội dung, 4 tình huống tích hợp câu mới, edge cases.

Provider đang dùng: **openai** → LM Studio endpoint → model **gemma-4-26b-a4b-it**

## Kết quả test end-to-end (2026-05-20)

```bash
# ---- Embed 3 câu liên tiếp ----
curl -X POST http://localhost:8000/transcript/meeting-bt2/embed \
  -d '{"speaker":"Đoàn Sỹ Nguyên","text":"Chúng ta cần xem lại ngân sách Q4..."}'
# → 202 { sequence_id: 1, context_status: "pending" }

# Sau 5s, context được build bởi LLM
curl http://localhost:8000/transcript/meeting-bt2/context
# → { sequence_id: 3, context: "Đề xuất cắt giảm 15% chi phí vận hành...", context_status: "ready" }

# ---- Query với window ----
curl -X POST http://localhost:8000/query/transcript \
  -d '{"collection":"meeting-bt2","query":"ngân sách Q4","top_k":2,"window_size":1}'
# → Trả kết quả kèm context + window.before/after

# ---- Validate prefix ----
curl -X POST http://localhost:8000/transcript/docs-bt2/embed ...
# → 400 Bad Request

# ---- Text rỗng ----
curl -X POST http://localhost:8000/transcript/meeting-bt2/embed \
# → 422 Unprocessable Entity

# ---- Phase 1 backward compat ----
curl http://localhost:8000/embed/collections
# → 200 OK, list 26 collections
```

## Kiến trúc

```
Tầng thiết bị (QCS8550) ──HTTP──► FastAPI App
                                    ├── /embed/* (Phase 1)
                                    ├── /query/ (Phase 1)
                                    ├── /transcript/* (Phase 2)
                                    ├── /query/transcript (Phase 2)
                                    ├── SequenceManager ──► Redis
                                    ├── ContextBuilder ──► LLM (LM Studio/Ollama)
                                    └── QdrantService ──► Qdrant
```

- **ContextBuilder**: build context tuần tự, gọi LLM tóm tắt (logic không đổi)
- **context_worker.py** (asyncio.Queue + 1 worker FIFO): ✅ đã implement — `POST /transcript/{collection}/embed` enqueue job thay cho `BackgroundTasks` → đảm bảo thứ tự build (D9)
- **SequenceManager**: Redis atomic INCR, TTL 7 ngày, self-healing từ Qdrant

## Files quan trọng

| File | Vai trò |
|------|---------|
| `phase2plan_v2.md` | Design doc — bản v2 rút gọn (nguồn sự thật) |
| `rag_server/README.md` | Tài liệu hướng dẫn sử dụng (đã cập nhật v2) |
| `rag_server/.env` | Cấu hình hiện tại |
| `rag_server/docker-compose.yml` | Stack: qdrant + redis + rag_api (+ ollama optional) |
| `rag_server/app/services/llm_client.py` | LLM client + system prompt |
| `rag_server/app/services/context_builder.py` | ContextBuilder: build context tuần tự theo collection |
| `rag_server/app/services/sequence_manager.py` | SequenceManager: Redis atomic sequence |
| `rag_server/app/services/transcript_service.py` | TranscriptService: orchestrate transcript flow |
| `rag_server/app/services/transcript_store.py` | TranscriptStore: Qdrant operations cho transcript |
| `rag_server/app/schemas/transcript.py` | Pydantic schemas Phase 2 |
| `rag_server/app/routers/transcript.py` | 4 endpoint transcript |
| `rag_server/app/routers/query.py` | Query router (có /query/transcript) |
| `rag_server/app/config.py` | Pydantic Settings |
| `rag_server/app/main.py` | FastAPI app + lifespan |
| `rag_server/app/utils/redis_client.py` | Singleton async Redis client |
| `rag_server/app/dependencies.py` | DI wiring |

## Đã hoàn thiện (đợt 2026-06-07 — lấy cảm hứng RAGFlow, giữ tinh thần v2)

- ✅ **context_worker.py (D9)**: `app/workers/context_worker.py` — asyncio.Queue + 1 worker FIFO; router enqueue thay `BackgroundTasks`; start/stop trong lifespan (drain khi shutdown).
- ✅ **/embed/info + /embed/info/{collection}**: wire endpoint đã ghi trong README (dùng `QdrantService.collection_info()` sẵn có); kiểm tra tồn tại trước để không tạo nhầm collection rỗng.
- ✅ **/health kiểm tra dependency**: ping Redis + check Qdrant + báo LLM provider; trả 503 nếu Qdrant/Redis chết (vẫn giữ `status: ok` khi khỏe).
- ✅ **Logging**: thay `print()` bằng `logging.basicConfig` trong `main.py`.
- ✅ **Tests khớp v2**: rewrite `test_sequence_manager.py` theo API lazy-init (`next(collection)`, rebuild từ Qdrant); `test_api.py` đã hợp lệ nhờ wire `/embed/info`.
- ✅ **Hybrid retrieval + Reranker (OPTIONAL, mặc định TẮT)**: `app/services/retrieval.py` (BM25+vector fusion) + `app/services/reranker.py` (`none|local|http`); dùng chung cho `/query/` và `/query/transcript`; **không thêm endpoint, response không đổi hình dạng**.

## Tiếp theo cần làm

### 1. 🔴 Rotate API key đang lộ
`.env` (đã được `.gitignore`, KHÔNG track trong git) vẫn chứa `LLM_API_KEY` plaintext trên đĩa. Nên **rotate key** và dùng env injection / Docker secret cho production.

### 2. 🟡 pytest encoding trên Windows
File test KHÔNG có BOM và compile sạch — lỗi encoding cũ là do console Windows (cp1252) khi in tiếng Việt. Khắc phục: chạy với `PYTHONUTF8=1` (hoặc `set PYTHONIOENCODING=utf-8`).

### 3. 🟢 Cleanup collection test cũ
26 collections đang tồn tại, phần lớn là test. Nên xoá qua `DELETE /embed/collections` (form field `name`).

### 4. 🟢 Đánh giá chất lượng context + retrieval
Kiểm tra chất lượng tóm tắt với dữ liệu thực tế; benchmark `HYBRID_ENABLED=true` và `RERANK_PROVIDER=local` xem cải thiện thứ hạng cho truy vấn nhiều tên riêng/con số.
