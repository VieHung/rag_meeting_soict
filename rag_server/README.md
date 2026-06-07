# RAG Vector Store API — BKMEETING

API embedding tài liệu & truy vấn ngữ nghĩa cho **Phòng Họp Thông Minh (BKMEETING)**.
Xây trên **FastAPI + Qdrant + Redis**, dùng embedding **đa ngữ (Tiếng Việt + Tiếng Anh)**
`intfloat/multilingual-e5-base` (768 chiều) — chạy trên **CPU/GPU** lúc dev và trên
**NPU Qualcomm Cloud AI / AI080** lúc production.

- **Phase 1** — RAG tài liệu: upload PDF/DOCX/TXT/MD → chunk → embed → truy vấn ngữ nghĩa.
- **Phase 2** — RAG transcript cuộc họp: embed từng câu, build context bằng LLM, query kèm window + context.

> ✅ **Trạng thái:** đã verify chạy thật trên thiết bị **"Qualcomm Device a080"** — full stack
> Docker (qdrant + redis + rag_api) embed/query tiếng Việt trên NPU, `cos(NPU, CPU) = 1.00000`.

---

## Mục lục

- [Tính năng chính](#tính-năng-chính)
- [Embedding & Qualcomm AI080](#embedding--qualcomm-ai080)
- [Quick Start](#quick-start)
- [Deploy production trên AI080](#deploy-production-trên-ai080)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Phase 1 — RAG tài liệu](#phase-1--rag-tài-liệu)
- [Phase 2 — Transcript cuộc họp](#phase-2--transcript-cuộc-họp)
- [Environment Variables](#environment-variables)
- [Migrate dữ liệu khi đổi model](#migrate-dữ-liệu-khi-đổi-model)
- [Testing](#testing)

> Các ví dụ `curl` dùng `:8000` (dev native). Nếu chạy bằng Docker thì cổng host là **`:18000`**
> (xem [Deploy production](#deploy-production-trên-ai080)).

---

## Tính năng chính

| Nhóm | Khả năng |
|------|----------|
| Tài liệu | Upload PDF/DOCX/TXT/MD, chunk + overlap, embed nền (background), metadata tùy biến |
| Truy vấn | Tìm kiếm ngữ nghĩa cosine, lọc theo `source`, `score_threshold`, `top_k` |
| Collections | Tạo / xóa / liệt kê, nhiều collection song song |
| Transcript | Embed từng câu, `sequence_id` atomic (Redis), build context nền bằng LLM, query kèm window ±N câu |
| Embedding | Đa ngữ Việt+Anh; 2 backend: `sentence_transformers` (dev) và `qaic` (NPU AI080) |

---

## Embedding & Qualcomm AI080

Embedding tách thành **backend**, chọn qua biến `EMBEDDING_BACKEND`:

| Backend | Dùng khi | Thiết bị | Phụ thuộc |
|---------|----------|----------|-----------|
| `sentence_transformers` _(mặc định)_ | Dev / CI / máy thường | CPU / GPU | `sentence-transformers` |
| `qaic` | Production trên **Qualcomm AI080** | NPU Cloud AI | Qualcomm Apps SDK (`qaic-exec`, wheel `qaic`) + `transformers` |

**Vì sao tách backend?** `sentence-transformers` không chạy trực tiếp trên NPU QAIC. Backend
`qaic` chỉ chạy phần *transformer* trên NPU qua **QPC** (đã compile), còn **tokenize →
mean-pooling → L2-normalize** làm ở CPU:

```
embed("ngân sách quý 4")
  └─ thêm prefix E5 ("query: " / "passage: ")
       └─ tokenizer (CPU)  → input_ids/attention_mask [1,128]
            └─ QPC forward trên NPU AI080  → last_hidden_state [1,128,768]
                 └─ mean-pool theo mask + L2-normalize (CPU)  → vector[768]
```

API public của `EmbeddingService` (`embed_texts`, `embed_query`, `dim`) **giữ nguyên** nên
phần còn lại của app không phải đổi khi chuyển backend.

**Prefix kiểu E5 (quan trọng).** Model E5 yêu cầu gắn tiền tố:
- Đoạn lưu trữ (document chunk, câu transcript) → `passage: ` (`embed_texts`).
- Câu truy vấn → `query: ` (`embed_query`).

Cấu hình qua `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX`; đặt rỗng nếu dùng model
đối xứng (vd MiniLM cũ).

---

## Quick Start

Chạy nhanh ở chế độ **dev** (backend `sentence_transformers`, CPU — không cần phần cứng NPU):

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Khởi động Qdrant (+ Redis cho Phase 2)
docker compose up qdrant redis -d
curl http://localhost:6333/healthz

# 3. Cấu hình môi trường (đặt EMBEDDING_BACKEND=sentence_transformers cho dev)
cp .env.example .env

# 4. Chạy API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. Mở tài liệu API → http://localhost:8000/docs
```

Để chạy production trên NPU AI080: xem mục bên dưới.

---

## Deploy production trên AI080

Hai cách. **Cách A (Docker) là khuyến nghị và đã được verify end-to-end.**

### Bước chung — build QPC (một lần, trên host AI080)

Cần Qualcomm Apps SDK đã cài (`/opt/qti-aic`, có `qaic-exec`).

```bash
pip install -r requirements-build.txt    # optimum-onnx, onnxruntime, torch — chỉ cho bước export
bash scripts/build_qpc.sh                 # export ONNX → compile QPC → ./qpc/e5-base/programqpc.bin (~555MB)
```

> `qaic-exec` thường không sẵn trên PATH → script tự dò `/opt/qti-aic/exec/qaic-exec`.
> `seq_len` compile (mặc định 128) PHẢI khớp `EMBEDDING_MAX_SEQ_LEN`. Đổi batch/seq qua
> biến `EMBED_BATCH`, `EMBEDDING_MAX_SEQ_LEN`, `AIC_NUM_CORES` trước khi chạy script.

### Cách A — Full Docker (khuyến nghị)

`rag_api` container đã cấu hình sẵn backend `qaic`:

```bash
docker compose up --build -d        # qdrant + redis + rag_api
docker compose ps                   # kiểm tra trạng thái
docker compose logs -f rag_api      # theo dõi log (chờ "Embedding model ready")

# API:    http://localhost:18000/docs
# Qdrant: http://localhost:6333/dashboard
```

Container tự lo phần NPU (xem `docker-compose.yml`):
- Mount SDK `/opt/qti-aic:ro`, QPC (`/opt/qpc/e5-base`), HF cache (tokenizer offline).
- Passthrough device `/dev/accel/accel0..3` + `group_add: "999"` (gid `qaic`).
- `PYTHONPATH`/`LD_LIBRARY_PATH` trỏ native lib SDK (`qaicrt`/`qaiccc` + `apps/` chứa libtorch).
- `entrypoint.sh` cài wheel `qaic` từ SDK đã mount lúc khởi động.
- Base `ubuntu:22.04` (khớp glibc SDK) + apt `libpci3 libudev1 libzstd1 libcrypt1`.

Tất cả service `restart: unless-stopped` + Docker enable on-boot ⇒ chạy **fulltime** (tự dậy
sau crash/reboot). Đường dẫn QPC/HF host chỉnh qua biến `QPC_HOST_DIR` / `HF_HOST_DIR`; cổng
host đổi ở mục `ports` (mặc định 18000 vì 8000 đang bận trên máy hiện tại).

### Cách B — Native (systemd / chạy tay)

```bash
# 1. Cài qaic runtime (wheel do SDK cung cấp, không có trên PyPI)
pip install /opt/qti-aic/dev/lib/x86_64/qaic-*-py3-none-any.whl

# 2. Cấu hình .env
#   EMBEDDING_BACKEND=qaic
#   EMBEDDING_QPC_PATH=/duong/dan/qpc/e5-base    # thư mục QPC hoặc file .bin
#   EMBEDDING_MAX_SEQ_LEN=128

# 3. Đặt path SDK rồi chạy (user phải thuộc group qaic: `id | grep qaic`)
export PYTHONPATH=/opt/qti-aic/dev/lib/x86_64:/opt/qti-aic/dev/python
export LD_LIBRARY_PATH=/opt/qti-aic/dev/lib/x86_64:/opt/qti-aic/dev/lib/x86_64/apps:/opt/qti-aic/lib/x86_64
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> Nếu `~/.cache/huggingface` bị root sở hữu (lỗi `PermissionError .../token`) → đặt `HF_HOME`
> sang thư mục ghi được, hoặc `sudo chown -R $USER ~/.cache/huggingface`.

**Đặc tính NPU đã đo:** output fp16 trên AI080 khớp gần tuyệt đối với sentence-transformers
CPU (`cos = 1.00000`). QPC nạp **batch tĩnh = 1** (mỗi inference 1 câu) — đủ nhanh cho query
real-time. Cần throughput cao hơn thì build QPC batch lớn và nới logic batch trong
`embedding_backends.py`.

---

## Cấu trúc dự án

```
rag_server/
├── app/
│   ├── main.py                     # FastAPI app, startup/shutdown, mount routers
│   ├── config.py                   # Settings (pydantic-settings) đọc từ .env
│   ├── routers/                    # embed, query, transcript endpoints
│   ├── services/
│   │   ├── embedding.py            # EmbeddingService (singleton) + prefix E5
│   │   ├── embedding_backends.py   # SentenceTransformerBackend | QaicEmbeddingBackend
│   │   ├── vector_store.py         # QdrantService (Phase 1 — documents)
│   │   ├── transcript_store.py     # TranscriptStore (Phase 2 — meeting-*)
│   │   ├── transcript_service.py   # orchestrator: embed/query/window/context
│   │   ├── context_builder.py      # background worker build context
│   │   └── llm_client.py           # abstraction LLM (ollama/gemini/openai/none)
│   └── utils/                      # chunking, redis_client
├── scripts/
│   ├── build_qpc.sh                # export ONNX + compile QPC cho AI080
│   └── reembed.py                  # re-embed Qdrant sang model/dimension mới
├── docker/
│   └── entrypoint.sh               # cài qaic wheel + check native lib khi container start
├── tests/                          # test_embedding_prefix, test_sequence_manager, test_api, test_transcript_api
├── requirements.txt                # runtime deps (dev — gồm sentence-transformers)
├── requirements-runtime.txt        # runtime deps lean cho container qaic (không torch/ST)
├── requirements-build.txt          # deps chỉ cho bước build QPC (offline)
├── Dockerfile                      # image ubuntu:22.04 cho backend qaic
└── docker-compose.yml              # qdrant + redis + rag_api (+ ollama optional)
```

---

## Phase 1 — RAG tài liệu

### Quản lý Collections

```bash
curl http://localhost:8000/embed/collections                      # liệt kê
curl -X POST   http://localhost:8000/embed/collections -F "name=project_b"   # tạo
curl -X DELETE http://localhost:8000/embed/collections -F "name=project_b"   # xóa
curl http://localhost:8000/embed/project_b/documents              # liệt kê tài liệu
```

### Embed tài liệu

**Upload file** (`.pdf`, `.docx`, `.txt`, `.md`) — chỉ `file` bắt buộc:

```bash
curl -X POST http://localhost:8000/embed/file \
  -F "file=@document.pdf" \
  -F "collection=my_collection" \
  -F "doc_id=custom-uuid" \
  -F 'extra_metadata={"author": "Navis", "category": "tech"}'
```

**Embed plain text:**

```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nội dung tài liệu...",
    "source": "my_document",
    "collection": "my_collection",
    "metadata": {"author": "Navis", "version": "1.0"}
  }'
```

> Embedding chạy **nền** (background task): response trả về ngay, vector được ghi sau đó.

### Truy vấn

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
    "collection": "my_collection",
    "top_k": 3,
    "score_threshold": 0.3,
    "source_filter": "document.pdf"
  }'
```

Response:
```json
{
  "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
  "results": [
    {
      "text": "NLP (Natural Language Processing) là lĩnh vực...",
      "score": 0.8741,
      "source": "document.pdf",
      "doc_id": "550e8400-...",
      "chunk_index": 3,
      "chunk_total": 24
    }
  ],
  "total_found": 3
}
```

### Xóa tài liệu

```bash
curl -X DELETE "http://localhost:8000/embed/my_collection/source/document.pdf"   # theo source
curl -X DELETE "http://localhost:8000/embed/my_collection/doc/550e8400-..."      # theo doc_id
```

### Tất cả endpoint Phase 1

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Health check |
| `POST` | `/embed/file` | Upload & embed file |
| `POST` | `/embed/text` | Embed plain text |
| `POST` | `/query/` | Truy vấn ngữ nghĩa |
| `GET` | `/embed/collections` | Liệt kê collections |
| `POST` · `DELETE` | `/embed/collections` | Tạo / xóa collection |
| `GET` | `/embed/{collection}/documents` | Liệt kê tài liệu |
| `DELETE` | `/embed/{collection}/source/{source}` | Xóa theo source |
| `DELETE` | `/embed/{collection}/doc/{doc_id}` | Xóa theo doc_id |

---

## Phase 2 — Transcript cuộc họp

Luồng RAG riêng cho **transcript của cuộc họp** (chi tiết: `phase2plan_v2.md`).

### Kiến trúc 2 tầng

| Tầng | Phần cứng | Vai trò |
|------|-----------|---------|
| **Thiết bị (client)** | Qualcomm QCS8550 | AI on-device: live transcript, LLM nhỏ sinh câu trả lời |
| **Server (repo này)** | Server độc lập + **NPU AI080** | RAG API + Qdrant + Redis + LLM self-host build context |

### Quy ước Collection

Mỗi cuộc họp = một **cặp collection** (app tạo thủ công, phân biệt bằng tiền tố):

| Loại | Tiền tố | Ví dụ | Endpoint |
|------|---------|-------|----------|
| Transcript | `meeting-{uuid}` | `meeting-c7cfdf57-...` | `/transcript/*`, `/query/transcript` |
| Tài liệu | `docs-{uuid}` | `docs-c7cfdf57-...` | `/embed/*`, `/query/` |

`meeting_id` được **suy ra từ tên collection** (bỏ tiền tố `meeting-`) — client không cần gửi.

### Endpoints (4, rút gọn từ 8)

| # | Method | Endpoint | Bắt buộc | Mô tả |
|---|--------|----------|----------|-------|
| 1 | `POST` | `/transcript/{collection}/embed` | ✅ Core | Lưu 1 câu, server gán `sequence_id`, trigger build context nền |
| 2 | `POST` | `/query/transcript` | ✅ Core | Truy vấn ngữ nghĩa, trả kết quả kèm **window** + **context** |
| 3 | `GET` | `/transcript/{collection}/context` | ⬜ Optional | Context mới nhất; hỗ trợ `?sequence_id=` |
| 4 | `GET` | `/transcript/{collection}/segments` | ⬜ Optional | Liệt kê transcript theo khoảng `sequence_id` |

### Workflow

```bash
# 1. Embed từng câu (server tự sinh sequence_id, build context nền)
curl -X POST http://localhost:8000/transcript/meeting-abc123/embed \
  -H "Content-Type: application/json" \
  -d '{ "speaker": "Đoàn Sỹ Nguyên", "text": "Cần xem lại ngân sách Q4 trước khi chốt." }'
# → 202 { "sequence_id": 1, "point_id": "...", "context_status": "pending" }

# 2. Query (kèm window ±N câu + context)
curl -X POST http://localhost:8000/query/transcript \
  -H "Content-Type: application/json" \
  -d '{ "collection": "meeting-abc123", "query": "Quyết định về ngân sách Q4 là gì?", "top_k": 3, "window_size": 2 }'
# → results[]: text + score + context + window.before/after

# 3. Context mới nhất / tại một câu
curl "http://localhost:8000/transcript/meeting-abc123/context"
curl "http://localhost:8000/transcript/meeting-abc123/context?sequence_id=1"

# 4. Liệt kê transcript
curl "http://localhost:8000/transcript/meeting-abc123/segments?from_seq=1&to_seq=10"
```

### Payload mỗi vector (`meeting-*`)

```json
{
  "meeting_id": "abc123",
  "sequence_id": 42,
  "speaker": "Đoàn Sỹ Nguyên",
  "speaker_id": "user_017",
  "text": "Cần xem lại ngân sách Q4...",
  "timestamp": "2026-05-11T19:52:27Z",
  "context": "Tóm tắt hội thoại tính tới câu này...",
  "context_status": "ready",
  "context_seq_base": 41,
  "lang": "vi",
  "created_at": "2026-05-11T19:52:27Z"
}
```

`context_status` ∈ `pending` · `processing` · `ready` · `failed` · `disabled`.

### Context Building

`context[N] = LLM_summarize(context[N-1] + transcript[N-1])`

- Context tại câu N là **bối cảnh dẫn tới câu N** (chưa gồm chính câu N).
- LLM self-host trên server (Ollama / OpenAI-compatible); ContextBuilder là background worker in-process.
- Tắt bằng `LLM_PROVIDER=none` → `context_status = disabled`.
- System prompt: `app/services/llm_client.py`.

> **Embedding transcript**: câu transcript embed như **passage** (prefix `passage: `), câu query
> dùng prefix `query: ` — đúng quy ước E5.

---

## Environment Variables

### Embedding

| Variable | Default | Mô tả |
|----------|---------|-------|
| `EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` (dev/CPU/GPU) hoặc `qaic` (NPU AI080) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Model embedding đa ngữ (Việt + Anh) |
| `EMBEDDING_DIM` | `768` | Số chiều vector (phải khớp model) |
| `EMBEDDING_QUERY_PREFIX` | `query: ` | Prefix E5 cho câu truy vấn (rỗng nếu model đối xứng) |
| `EMBEDDING_PASSAGE_PREFIX` | `passage: ` | Prefix E5 cho đoạn lưu trữ |
| `EMBEDDING_QPC_PATH` | _(none)_ | Thư mục QPC (hoặc file `.bin`) — chỉ dùng khi backend `qaic` |
| `EMBEDDING_MAX_SEQ_LEN` | `128` | seq_len tĩnh; phải khớp lúc compile QPC |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Kích thước & overlap chunk (tokens) |
| `TOP_K_DEFAULT` | `5` | Số kết quả mặc định khi query |

### Qdrant

| Variable | Default | Mô tả |
|----------|---------|-------|
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Địa chỉ Qdrant (container: `qdrant`) |
| `QDRANT_COLLECTION_NAME` | `documents` | Collection mặc định Phase 1 |

### Phase 2 — Redis & Transcript & LLM

| Variable | Default | Mô tả |
|----------|---------|-------|
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `localhost` / `6379` / `0` | Redis (sequence counter; container: `redis`) |
| `TRANSCRIPT_SEQ_START` | `1` | `sequence_id` bắt đầu |
| `TRANSCRIPT_WINDOW_SIZE` | `2` | ±N câu lân cận mặc định khi query |
| `TRANSCRIPT_MAX_WINDOW_SIZE` | `5` | Giới hạn `window_size` client gửi |
| `TRANSCRIPT_DEFAULT_COLLECTION` | `meeting_transcripts` | Collection transcript mặc định |
| `LLM_PROVIDER` | `none` | `ollama` · `gemini` · `openai` · `none` |
| `LLM_MODEL` | `qwen2.5:7b` | Model build context |
| `LLM_BASE_URL` | `http://ollama:11434` | Endpoint LLM |
| `LLM_API_KEY` | _(rỗng)_ | Key cho gemini/openai |
| `CONTEXT_MAX_TOKENS` / `CONTEXT_MAX_RETRY` / `CONTEXT_TIMEOUT_SECONDS` | `800` / `2` / `30` | Tham số build context |

> Trong Docker, các biến `EMBEDDING_BACKEND`, `EMBEDDING_QPC_PATH`, `QDRANT_HOST`, `REDIS_HOST`,
> `HF_HOME`, `PYTHONPATH`, `LD_LIBRARY_PATH`… được đặt sẵn trong `docker-compose.yml` (ghi đè `.env`).
> Bật Ollama cho transcript: `docker compose --profile ollama up -d` + đổi `LLM_PROVIDER=ollama`.

---

## Migrate dữ liệu khi đổi model

Đổi model embedding làm thay đổi giá trị vector (và thường cả số chiều) ⇒ vector cũ không còn
dùng được. Script `scripts/reembed.py` scroll toàn bộ point, re-embed lại `payload["text"]`
bằng model hiện tại, **giữ nguyên payload + point_id** (quan trọng cho transcript:
`sequence_id`, `context`).

```bash
python -m scripts.reembed --collections all                       # → ghi sang <name>__v2 (dễ rollback)
python -m scripts.reembed --collections documents,meeting-abc123  # chỉ vài collection
python -m scripts.reembed --collections all --in-place            # khi số chiều KHÔNG đổi (vd e5-small 384)
```

Sau khi verify `points_count` khớp, đổi tên/đảo collection `__v2` về tên gốc thủ công.
Counter Redis (`rag:seq:{collection}`) không cần đụng — `sequence_id` nằm trong payload đã giữ.

---

## Testing

```bash
pip install pytest pytest-asyncio

# Unit test (không cần hạ tầng) — prefix E5, sequence manager
pytest tests/test_embedding_prefix.py tests/test_sequence_manager.py -v

# Integration test (cần stack chạy + LLM_PROVIDER=none cho ổn định)
docker compose up -d
pytest tests/test_api.py tests/test_transcript_api.py -v

# Live edge-case suite — bắn HTTP thật vào server đang chạy (mặc định :18000)
python scripts/test_live_api.py
```

| Test file | Phạm vi |
|-----------|---------|
| `tests/test_embedding_prefix.py` | Prefix E5 (`passage:` / `query:`), dim từ backend (mock, không tải model) |
| `tests/test_sequence_manager.py` | Atomic sequence counter, self-heal từ Qdrant |
| `tests/test_api.py` | Phase 1: collections, embed, query, delete, edge cases |
| `tests/test_transcript_api.py` | Phase 2: embed transcript, query + window + context |
| `scripts/test_live_api.py` | **Live** edge-case suite (httpx) bắn vào server thật: validation 400/422, ingest đa ngữ, query bounds, source/speaker filter, delete, collections CRUD, transcript prefix + sequence + window + context |

> **Lưu ý encoding**: file `*.py` phải là UTF-8 (không BOM / không UTF-16). File lưu nhầm
> encoding trên Windows sẽ làm pytest báo `source code string cannot contain null bytes`.

---

## Vận hành & sự cố thường gặp

### `/transcript/.../embed` trả `500` — Qdrant "Too many open files"

Traceback dừng tại `transcript_store.py` (`create_collection`) nhưng **nguyên nhân nằm ở Qdrant**:

```
RocksDB open error: IO error: While open a file for appending:
  ./storage/collections/meeting-.../segments/.../000000.dbtmp: Too many open files
```

Container `qdrant_server` mặc định có `nofile` soft = **1024**. Mỗi cuộc họp tạo một collection
`meeting-{uuid}`, mỗi collection nhiều segment RocksDB (mở rất nhiều fd + mmap) → khi tải nặng
nhiều collection song song, tổng fd vượt 1024. **Không** liên quan model embedding / NPU.

**Khắc phục** — nâng `nofile` cho service `qdrant` trong `docker-compose.yml` rồi **recreate**
(recreate KHÔNG mất dữ liệu — storage ở volume `./qdrant_storage`):

```yaml
  qdrant:
    image: qdrant/qdrant:v1.10.0
    # ...
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

```bash
docker compose up -d qdrant      # phải recreate; `docker restart` KHÔNG nạp ulimit mới
docker exec qdrant_server sh -c 'cat /proc/1/limits | grep "open files"'   # kỳ vọng 65536 65536
```

Giảm áp lực fd lâu dài (tùy chọn): đặt env `QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER=2`
(chỉ ảnh hưởng collection mới); định kỳ `DELETE /embed/collections` cho cuộc họp đã kết thúc.

### Race khi tạo collection lần đầu (TOCTOU) → mất câu / 409 Conflict

Khi nhiều câu đầu của một collection mới được embed gần như đồng thời, hai background task cùng
vượt qua check "chưa tồn tại" rồi cùng `create_collection` → kẻ thua nhận `409 Conflict`.
`QdrantService._ensure_collection` (`vector_store.py`) đã được vá idempotent (nuốt 409 nếu
collection cuối cùng đã tồn tại). `TranscriptStore.ensure_collection` nên áp dụng cùng cách.

### `rag_api` không phản hồi ở `:18000`

Kiểm tra container và khởi động lại (đừng đụng các container của dự án khác trên cùng máy):

```bash
docker compose ps
docker compose up -d rag_api
docker compose logs -f rag_api      # chờ "Embedding model ready"
```

---

## License

MIT
