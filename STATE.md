# Trạng thái hệ thống RAG Phase 2

> File này lưu trạng thái hiện tại để agent tiếp theo có thể tiếp tục làm việc.
> Cập nhật lần cuối: **2026-06-07**

## Docker Services

| Service | Container | Image | Port (host→container) | Status |
|---------|-----------|-------|-----------------------|--------|
| qdrant | `qdrant_server` | qdrant/qdrant:v1.10.0 | 6333→6333, 6334→6334 | ✅ Running |
| redis | `redis_server` | redis:7.2-alpine | 6379→6379 | ✅ Running |
| rag_api | `rag_api` | rag_server-rag_api | **18000→1904** | ⚠️ Exited (0) — cần khởi động lại |

> ⚠️ **rag_api hiện đang `Exited`** (dừng từ ~45h trước). Khởi động lại bằng:
> ```bash
> cd /home/ai/hungtv/rag_meeting_soict/rag_server
> docker compose up -d rag_api
> curl http://localhost:18000/health     # kỳ vọng {"status":"ok"}
> ```
> Backend embedding `qaic` cần SDK + QPC + device `/dev/accel/*` (đã cấu hình sẵn trong compose).
>
> **Lưu ý:** các container `meeting-redis`, `meeting_db` (postgres), … thuộc dự án KHÁC trên cùng
> máy — **không đụng vào**. Cũng không đụng container `elegant_darwin` (phiên dev Qualcomm).

## Embedding (qaic / NPU)

| Thông số | Giá trị |
|----------|---------|
| Backend | `qaic` (NPU **Qualcomm AI080**) — không phải CPU/GPU |
| Model | `intfloat/multilingual-e5-base` (đa ngữ Việt + Anh) |
| Số chiều | 768, distance Cosine |
| Prefix E5 | passage: `"passage: "` · query: `"query: "` |
| QPC | `/home/ai/hungtv/qpc/e5-base`, `EMBEDDING_MAX_SEQ_LEN=128`, batch tĩnh = 1 |

Đây là **encoder embedding**, KHÔNG phải LLM sinh văn bản. `cos(NPU fp16, CPU sentence-transformers) = 1.00000`.

## LLM build-context: ĐANG TẮT

`LLM_PROVIDER=none` ⇒ ContextBuilder **không chạy** (`context_builder.py` return sớm), mọi câu
transcript có `context_status = "disabled"`, trường `context` rỗng. Query transcript vẫn hoạt động
(trả `window` + `text`), chỉ thiếu phần tóm tắt bối cảnh.

Muốn bật lại tính năng context: chạy Ollama (`docker compose --profile ollama up -d`) và đổi
`LLM_PROVIDER=ollama` (model `qwen2.5:7b`, `LLM_BASE_URL=http://ollama:11434`).

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
| POST · DELETE | `/embed/collections` (Form `name=`) | ✅ OK |
| POST | `/embed/file` · `/embed/text` | ✅ OK |
| POST | `/query/` | ✅ OK |
| DELETE | `/embed/{collection}/source/{source}` · `/embed/{collection}/doc/{doc_id}` | ✅ OK |

### System

| Method | Endpoint | Status |
|--------|----------|--------|
| GET | `/health` | ✅ OK |

## Endpoint cũ đã xoá (theo phase2plan_v2.md)

- `POST /transcript/{collection}/meeting/init` — ❌ đã gỡ (lazy-init tự động)
- `DELETE /transcript/{collection}/meeting/{meeting_id}` — ❌ đã gỡ (dùng `DELETE /embed/collections`)
- `PATCH /transcript/{collection}/context/{sequence_id}` — ❌ đã gỡ (ContextBuilder in-process)
- `GET /transcript/{collection}/context/latest` — ❌ đã gộp vào `/context` với `?sequence_id=`

## Cấu hình hiện tại (.env)

```ini
# --- Qdrant / Redis ---
QDRANT_HOST=localhost            # trong Docker: qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents
REDIS_HOST=localhost             # trong Docker: redis
REDIS_PORT=6379

# --- Embedding (NPU AI080) ---
EMBEDDING_BACKEND=qaic
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DIM=768
EMBEDDING_QPC_PATH=/home/ai/hungtv/qpc/e5-base
EMBEDDING_MAX_SEQ_LEN=128
EMBEDDING_QUERY_PREFIX="query: "
EMBEDDING_PASSAGE_PREFIX="passage: "

# --- Transcript ---
TRANSCRIPT_SEQ_START=1
TRANSCRIPT_WINDOW_SIZE=2
TRANSCRIPT_MAX_WINDOW_SIZE=5

# --- LLM build-context: ĐANG TẮT ---
LLM_PROVIDER=none                # none | ollama | gemini | openai
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://ollama:11434
CONTEXT_MAX_TOKENS=800
CONTEXT_MAX_RETRY=2
CONTEXT_TIMEOUT_SECONDS=30
```

## Kiến trúc

```
Tầng thiết bị (QCS8550) ──HTTP──► FastAPI App (rag_api, host :18000)
                                    ├── /embed/* (Phase 1)
                                    ├── /query/ (Phase 1)
                                    ├── /transcript/* (Phase 2)
                                    ├── /query/transcript (Phase 2)
                                    ├── SequenceManager ──► Redis (atomic INCR, TTL 7 ngày)
                                    ├── ContextBuilder ──► LLM  [TẮT khi LLM_PROVIDER=none]
                                    └── Embedding (qaic) ──► NPU AI080
                                         QdrantService / TranscriptStore ──► Qdrant
```

- **Embedding**: backend `qaic`, chạy transformer trên NPU; tokenize + mean-pool + L2-norm ở CPU.
- **ContextBuilder**: background worker in-process (FastAPI BackgroundTasks) — chỉ chạy khi bật LLM.
- **SequenceManager**: Redis atomic INCR, self-healing rebuild từ Qdrant nếu Redis mất.
- **1 cuộc họp = 1 collection** `meeting-{uuid}` (transcript) + `docs-{uuid}` (tài liệu).

## Sự cố đã biết & cách xử lý

### 🔴 Qdrant "Too many open files" → `/transcript/.../embed` trả 500
- **Triệu chứng:** traceback `RocksDB open error: ... Too many open files` tại `create_collection`.
- **Nguyên nhân:** container `qdrant_server` chạy với `nofile` soft = **1024** (mặc định Docker).
  Mỗi cuộc họp tạo 1 collection mới (`meeting-{uuid}`), mỗi collection nhiều segment RocksDB
  (mở nhiều fd + mmap) → tải nặng nhiều collection song song vượt 1024 fd.
- **KHÔNG** liên quan model embedding / NPU. Đĩa còn trống (chỉ dùng ~13%).
- **Cách sửa (CHƯA áp dụng):** thêm `ulimits` vào service `qdrant` trong `docker-compose.yml` rồi
  **recreate** (recreate không mất dữ liệu — storage nằm ở volume `./qdrant_storage`):
  ```yaml
    qdrant:
      ...
      ulimits:
        nofile: { soft: 65536, hard: 65536 }
  ```
  ```bash
  docker compose up -d qdrant
  docker exec qdrant_server sh -c 'cat /proc/1/limits | grep "open files"'  # kỳ vọng 65536
  ```
- Giảm áp lực fd lâu dài (tùy chọn): `QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER=2`;
  định kỳ `DELETE /embed/collections` cho cuộc họp đã kết thúc.

### 🟡 Race khi tạo collection lần đầu (TOCTOU)
- `QdrantService._ensure_collection` (`vector_store.py`) — ✅ **đã sửa** idempotent + nuốt 409 Conflict.
- `TranscriptStore.ensure_collection` (`transcript_store.py:63`) — ⚠️ **CHƯA sửa**, vẫn check-then-create.
  Khi ~2 câu transcript đầu của một meeting mới tới gần như đồng thời, kẻ thua nhận 409 → 500 và
  **mất câu đó**. Nên sửa đồng bộ với `vector_store.py` (dùng `collection_exists()` + try/except).

## Kiểm thử

```bash
# Live edge-case suite (cần rag_api chạy ở :18000) — ~64 assertion
/home/ai/hungtv/venv/bin/python scripts/test_live_api.py

# Unit test (không cần hạ tầng)
pytest tests/test_embedding_prefix.py tests/test_sequence_manager.py -v
```

- Lần chạy gần nhất (khi rag_api còn up): **75/75 PASS**, không còn 409, cả 5 nguồn embed thành công
  sau khi vá idempotent `vector_store.py`.

## Files quan trọng

| File | Vai trò |
|------|---------|
| `phase2plan_v2.md` | Design doc — bản v2 rút gọn (nguồn sự thật thiết kế) |
| `API_USAGE.md` | Hợp đồng API cho đội app thiết bị |
| `rag_server/README.md` | Hướng dẫn cài đặt / vận hành (đã cập nhật v2 + qaic) |
| `rag_server/.env` | Cấu hình hiện tại (LLM tắt, backend qaic) |
| `rag_server/docker-compose.yml` | Stack: qdrant + redis + rag_api (+ ollama optional) |
| `rag_server/scripts/test_live_api.py` | Live edge-case suite (httpx) |
| `rag_server/scripts/build_qpc.sh` | Export ONNX + compile QPC cho AI080 |
| `rag_server/app/services/embedding_backends.py` | SentenceTransformerBackend \| QaicEmbeddingBackend |
| `rag_server/app/services/vector_store.py` | QdrantService (Phase 1) — đã vá idempotent collection |
| `rag_server/app/services/transcript_store.py` | TranscriptStore (Phase 2) — còn TOCTOU race |
| `rag_server/app/services/context_builder.py` | ContextBuilder (tắt khi LLM_PROVIDER=none) |
| `rag_server/app/services/sequence_manager.py` | SequenceManager: Redis atomic sequence |
| `rag_server/app/services/transcript_service.py` | Orchestrate transcript flow |
| `rag_server/app/routers/transcript.py` · `query.py` | Endpoints Phase 2 |

## Tiếp theo cần làm

### 1. 🔴 Sửa Qdrant fd limit (gốc của lỗi 500)
Thêm `ulimits.nofile` cho `qdrant` trong compose + `docker compose up -d qdrant`. Xem mục Sự cố ở trên.

### 2. 🟡 Idempotent `TranscriptStore.ensure_collection`
Sửa TOCTOU race ở `transcript_store.py:63` cho khớp `vector_store.py` để không mất câu transcript đầu.

### 3. 🟡 Khởi động lại rag_api
Container `rag_api` đang `Exited`. `docker compose up -d rag_api` rồi xác minh `/health`.

### 4. 🟢 Bật lại context build (nếu cần)
Hiện `LLM_PROVIDER=none`. Bật Ollama + đổi provider nếu sản phẩm cần trường `context`.

### 5. 🟢 Dọn collection test cũ
Qua `DELETE /embed/collections` cho các collection không còn dùng (giảm áp lực fd).
