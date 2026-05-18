# RAG Vector Store API

API embedding tài liệu và truy vấn ngữ nghĩa với Qdrant + MiniLM-L12-v2.

## Quick Start

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Khởi động Qdrant Server

```bash
# Sử dụng Docker
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.10.0

# Hoặc sử dụng docker-compose
docker compose up qdrant -d

# Kiểm tra Qdrant đã sẵn sàng
curl http://localhost:6333/healthz
```

### 3. Cấu hình Environment

```bash
cp .env.example .env
```

### 4. Chạy API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Truy cập API Documentation

```
http://localhost:8000/docs
```

---

## Quản lý Collections

### Liệt kê tất cả collections

```bash
curl -X GET http://localhost:8000/embed/collections
```

Response:
```json
{
  "collections": ["documents", "my_collection", "project_a"]
}
```

### Tạo collection mới

```bash
curl -X POST http://localhost:8000/embed/collections \
  -F "name=project_b"
```

Response:
```json
{
  "success": true,
  "message": "Created collection 'project_b'"
}
```

### Xóa collection

```bash
curl -X DELETE http://localhost:8000/embed/collections \
  -F "name=project_b"
```

Response:
```json
{
  "success": true,
  "message": "Deleted collection 'project_b'"
}
```

### Liệt kê tài liệu trong collection

```bash
# Collection mặc định
curl http://localhost:8000/embed/documents/documents

# Collection cụ thể
curl http://localhost:8000/embed/documents/my_collection
```

Response:
```json
{
  "documents": [
    {
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "source": "document.pdf",
      "chunks": 24,
      "chunk_index": 0
    },
    {
      "doc_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
      "source": "report.docx",
      "chunks": 12,
      "chunk_index": 0
    }
  ],
  "total": 2
}
```

### Thông tin collection

```bash
# Collection mặc định
curl http://localhost:8000/embed/info

# Collection cụ thể
curl http://localhost:8000/embed/info/project_b
```

Response:
```json
{
  "name": "documents",
  "vectors_count": 2448,
  "points_count": 2448,
  "status": "green",
  "vector_size": 384,
  "distance": "Cosine"
}
```

---

## Embed Tài liệu

### Upload file PDF/DOCX/TXT/MD

```bash
curl -X POST http://localhost:8000/embed/file \
  -F "file=@document.pdf"
```

**Upload vào collection cụ thể:**
```bash
curl -X POST http://localhost:8000/embed/file \
  -F "file=@document.pdf" \
  -F "collection=my_collection"
```

**Upload với doc_id tùy chỉnh:**
```bash
curl -X POST http://localhost:8000/embed/file \
  -F "file=@document.pdf" \
  -F "doc_id=custom-uuid-string"
```

**Upload với metadata tùy chỉnh:**
```bash
curl -X POST http://localhost:8000/embed/file \
  -F "file=@document.pdf" \
  -F 'extra_metadata={"author": "Navis", "category": "tech"}'
```

Response:
```json
{
  "success": true,
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "source": "document.pdf",
  "chunks_created": 24,
  "message": "Đã embed 24 chunks từ 'document.pdf'"
}
```

### Embed plain text

```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nội dung tài liệu...",
    "source": "my_document"
  }'
```

**Với collection cụ thể:**
```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nội dung tài liệu...",
    "source": "my_document",
    "collection": "my_collection"
  }'
```

**Với metadata:**
```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nội dung tài liệu...",
    "source": "my_document",
    "metadata": {"author": "Navis", "version": "1.0"}
  }'
```

---

## Query (Truy vấn)

### Truy vấn ngữ nghĩa

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
    "top_k": 3,
    "score_threshold": 0.3
  }'
```

**Từ collection cụ thể:**
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
    "collection": "my_collection",
    "top_k": 3
  }'
```

**Lọc theo file nguồn (exact match):**
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
    "source_filter": "document.pdf",
    "top_k": 3
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
      "doc_id": "550e8400-e29b-41d4-a716-446655440000",
      "chunk_index": 3,
      "chunk_total": 24
    }
  ],
  "total_found": 3
}
```

---

## Xóa Tài liệu

### Xóa theo source (tên file)

```bash
curl -X DELETE "http://localhost:8000/embed/documents/source/document.pdf"
```

**Trong collection cụ thể:**
```bash
curl -X DELETE "http://localhost:8000/embed/my_collection/source/document.pdf"
```

### Xóa theo doc_id (UUID)

```bash
curl -X DELETE "http://localhost:8000/embed/documents/doc/550e8400-e29b-41d4-a716-446655440000"
```

**Trong collection cụ thể:**
```bash
curl -X DELETE "http://localhost:8000/embed/my_collection/doc/550e8400-e29b-41d4-a716-446655440000"
```

Response:
```json
{
  "success": true,
  "message": "Đã xóa tài liệu có doc_id '550e8400-e29b-41d4-a716-446655440000'"
}
```

---

## Các Endpoint Khác

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| `GET` | `/health` | Health check |
| `GET` | `/embed/info` | Thông tin collection mặc định |
| `GET` | `/embed/info/{collection}` | Thông tin collection cụ thể |
| `GET` | `/embed/collections` | Liệt kê tất cả collections |
| `POST` | `/embed/collections` | Tạo collection mới |
| `DELETE` | `/embed/collections` | Xóa collection |
| `GET` | `/embed/{collection}/documents` | Liệt kê tài liệu trong collection |

---

## Supported File Types

- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF documents
- `.docx` - Word documents

---

## Environment Variables

### Phase 1 (tài liệu)

| Variable | Default | Mô Tả |
|----------|---------|-------|
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | documents | Tên collection mặc định |
| `EMBEDDING_MODEL` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Model embedding |
| `EMBEDDING_DIM` | 384 | Vector dimension |
| `CHUNK_SIZE` | 512 | Kích thước chunk (tokens) |
| `CHUNK_OVERLAP` | 64 | Overlap giữa các chunk |
| `TOP_K_DEFAULT` | 5 | Số kết quả mặc định |

### Phase 2 (transcript & cuộc họp)

| Variable | Default | Mô Tả |
|----------|---------|-------|
| `REDIS_HOST` | localhost | Redis host (sequence + context cache) |
| `REDIS_PORT` | 6379 | Redis port |
| `REDIS_DB` | 0 | Redis db |
| `TRANSCRIPT_SEQ_START` | 1 | sequence_id bắt đầu từ |
| `TRANSCRIPT_WINDOW_SIZE` | 2 | ±N câu lân cận mặc định khi query |
| `TRANSCRIPT_MAX_WINDOW_SIZE` | 5 | Giới hạn window_size client gửi |
| `TRANSCRIPT_DEFAULT_COLLECTION` | meeting_transcripts | Collection transcript mặc định |
| `LLM_PROVIDER` | ollama | `ollama` \| `gemini` \| `openai` \| `none` |
| `LLM_MODEL` | qwen2.5:7b | Model dùng để build context |
| `LLM_BASE_URL` | http://ollama:11434 | Endpoint LLM |
| `LLM_API_KEY` | (rỗng) | Key cho gemini/openai |
| `CONTEXT_MAX_TOKENS` | 800 | Độ dài tối đa context tóm tắt |
| `CONTEXT_MAX_RETRY` | 2 | Số lần retry khi LLM lỗi |
| `CONTEXT_TIMEOUT_SECONDS` | 30 | Timeout mỗi lần gọi LLM |

---

## Docker Deployment

### Build và chạy toàn bộ stack

```bash
docker compose up --build -d
```

> Mặc định stack gồm `qdrant`, `redis`, `rag_api`. Service `ollama` là **optional**:
> ```bash
> docker compose --profile ollama up -d
> ```
> Khi không bật ollama, đặt `LLM_PROVIDER=none` (mặc định trong `.env`) để tắt build context.

### Xem logs

```bash
docker compose logs -f rag_api
```

### Truy cập Qdrant Dashboard

```
http://localhost:6333/dashboard
```

---

## Integration với LLM

```python
import httpx

async def rag_answer(user_question: str, collection: str = None) -> str:
    async with httpx.AsyncClient() as client:
        payload = {"query": user_question, "top_k": 5, "score_threshold": 0.4}
        if collection:
            payload["collection"] = collection

        search_response = await client.post(
            "http://localhost:8000/query/",
            json=payload
        )

    results = search_response.json()["results"]
    context = "\n\n---\n\n".join([r["text"] for r in results])

    prompt = f"""Dựa vào các đoạn tài liệu sau:

{context}

Hãy trả lời câu hỏi: {user_question}

Nếu không tìm thấy thông tin liên quan trong tài liệu, hãy nói rõ điều đó."""

    return prompt
```

---

## Ví dụ Workflow đầy đủ

```bash
# 1. Tạo collection mới
curl -X POST http://localhost:8000/embed/collections -F "name=project_x"

# 2. Upload file vào collection
curl -X POST http://localhost:8000/embed/file \
  -F "file=@docs/report.pdf" \
  -F "collection=project_x"

# 3. Liệt kê tài liệu đã embed
curl http://localhost:8000/embed/project_x/documents

# 4. Query từ collection đó
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "tổng kết doanh thu 2024",
    "collection": "project_x"
  }'

# 5. Xóa document sau khi dùng xong
curl -X DELETE "http://localhost:8000/embed/project_x/doc/DOC_ID_TU_FILE_TREN"
```

---

## Testing

### Chạy pytest

```bash
# Cài đặt pytest
pip install pytest

# Chạy tất cả tests
pytest tests/test_api.py -v

# Chạy tests với detailed output
pytest tests/test_api.py -v --tb=short

# Chạy tests theo class
pytest tests/test_api.py::TestCollectionManagement -v

# Chạy tests cụ thể
pytest tests/test_api.py::TestQuery::test_query_basic -v
```

### Coverage Tests

| Test Class | Mô tả |
|------------|-------|
| `TestHealthCheck` | Health endpoint |
| `TestCollectionManagement` | CRUD collections |
| `TestEmbedText` | Embed text với các edge cases |
| `TestEmbedFile` | Upload file |
| `TestQuery` | Query với filters, thresholds |
| `TestDeleteDocuments` | Xóa theo source/doc_id |
| `TestListDocuments` | Liệt kê documents |
| `TestCollectionInfo` | Thông tin collection |
| `TestEdgeCases` | Unicode, special chars, large text |
| `TestPerformance` | Multiple queries, batch embeds |
| `TestConcurrent` | Concurrent requests |
| `TestErrorHandling` | Invalid inputs, errors |

---

## Phase 2 — Transcript cuộc họp (BKMEETING)

Phase 2 bổ sung luồng RAG riêng cho **transcript của cuộc họp**. Tách hoàn toàn với luồng tài liệu Phase 1: hai collection Qdrant khác nhau, hai endpoint query khác nhau. Xem `plan.md` ở repo root để biết design rationale.

### Khái niệm

- Mỗi câu transcript = **đúng 1 vector** (không chunking).
- Server tự gán `sequence_id` (atomic INCR trên Redis) → đảm bảo liên tục, không trùng.
- Context (tóm tắt cuộc họp tính tới câu N) được build **nền** bằng một LLM (Ollama / Gemini / OpenAI). Có thể tắt bằng `LLM_PROVIDER=none`.
- Khi query, kết quả trả kèm **window ±N câu lân cận** + context.

### Hạ tầng cần thêm

- **Redis** — sequence counter + context cache. Đã thêm sẵn vào `docker-compose.yml`.
- **LLM** — optional. Có thể dùng Ollama local (profile `ollama` trong compose) hoặc API ngoài (Gemini/OpenAI).

### Endpoints mới

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/transcript/{collection}/meeting/init` | Khởi tạo cuộc họp (reset counter) |
| `DELETE` | `/transcript/{collection}/meeting/{meeting_id}` | Xoá toàn bộ transcript + state Redis |
| `POST` | `/transcript/{collection}/embed` | Lưu 1 câu transcript, trả `sequence_id` ngay (`202 Accepted`) |
| `GET` | `/transcript/{collection}/context/latest?meeting_id=...` | Context mới nhất |
| `GET` | `/transcript/{collection}/context/{sequence_id}?meeting_id=...` | Context tại 1 thời điểm |
| `PATCH` | `/transcript/{collection}/context/{sequence_id}` | LLM nội bộ update context |
| `GET` | `/transcript/{collection}/meeting/{meeting_id}/segments` | Liệt kê transcript theo range |
| `POST` | `/query/transcript` | Query transcript (kèm window) |

### Workflow ví dụ

```bash
COL=meeting_transcripts
MID=meeting_2026_soict_001

# 1. Init cuộc họp
curl -X POST http://localhost:8000/transcript/$COL/meeting/init \
  -H "Content-Type: application/json" \
  -d "{\"meeting_id\": \"$MID\"}"

# 2. Embed từng câu (server gán sequence_id)
curl -X POST http://localhost:8000/transcript/$COL/embed \
  -H "Content-Type: application/json" \
  -d "{
    \"meeting_id\": \"$MID\",
    \"speaker\": \"Đoàn Sỹ Nguyên\",
    \"text\": \"Chúng ta cần xem lại ngân sách Q4 trước khi chốt.\"
  }"
# → 202 { "sequence_id": 1, "point_id": "...", "context_status": "pending" }

# 3. Query có window
curl -X POST http://localhost:8000/query/transcript \
  -H "Content-Type: application/json" \
  -d "{
    \"collection\": \"$COL\",
    \"query\": \"Quyết định về ngân sách Q4 là gì?\",
    \"meeting_id\": \"$MID\",
    \"top_k\": 3,
    \"window_size\": 2
  }"

# 4. Lấy context mới nhất
curl "http://localhost:8000/transcript/$COL/context/latest?meeting_id=$MID"

# 5. Liệt kê toàn bộ transcript đã thu
curl "http://localhost:8000/transcript/$COL/meeting/$MID/segments?from_seq=1&limit=200"

# 6. Kết thúc — xoá meeting
curl -X DELETE "http://localhost:8000/transcript/$COL/meeting/$MID"
```

### Cấu trúc payload (Qdrant)

```json
{
  "meeting_id":       "meeting_2026_soict_001",
  "sequence_id":      42,
  "speaker":          "Đoàn Sỹ Nguyên",
  "speaker_id":       "user_017",
  "timestamp":        "2026-05-11T19:52:27Z",
  "text":             "Chúng ta cần xem lại ngân sách Q4...",
  "context":          "Tóm tắt hội thoại tính tới câu này...",
  "context_status":   "ready",
  "context_seq_base": 41,
  "lang":             "vi",
  "created_at":       "2026-05-11T19:52:27Z"
}
```

`context_status` ∈ `pending` | `processing` | `ready` | `failed`.

### Định nghĩa context

Theo plan.md: `context[N] = LLM_summarize(context[N-1] + transcript[N-1])`.
Tức context tại câu N là **bối cảnh dẫn tới câu N** (chưa bao gồm chính câu N). Khi query trả kết quả câu N, app workstation ghép `context` + `text` câu N + `window` để đưa cho LLM phía user.

### Tests Phase 2

```bash
# Unit test SequenceManager (cần Redis chạy)
pytest tests/test_sequence_manager.py -v

# Integration test (cần stack đang chạy + LLM_PROVIDER=none cho test ổn định)
pytest tests/test_transcript_api.py -v
```

---

## Testing

### Manual Test (đã xác nhận hoạt động)

```bash
COL=test_collection
MID=meeting_$(date +%s)

# 1. Init meeting
curl -X POST http://localhost:8000/transcript/$COL/meeting/init \
  -H "Content-Type: application/json" \
  -d "{\"meeting_id\": \"$MID\"}"

# 2. Embed transcripts (sequence_id tự động tăng)
curl -X POST http://localhost:8000/transcript/$COL/embed \
  -H "Content-Type: application/json" \
  -d "{\"meeting_id\": \"$MID\", \"speaker\": \"Nguyễn Văn A\", \"text\": \"Chào mọi người.\"}"

# 3. Query với window
curl -X POST http://localhost:8000/query/transcript \
  -H "Content-Type: application/json" \
  -d "{\"collection\": \"$COL\", \"query\": \"ngân sách\", \"meeting_id\": \"$MID\", \"top_k\": 3, \"window_size\": 1}"

# 4. Get segments
curl "http://localhost:8000/transcript/$COL/meeting/$MID/segments?from_seq=1&to_seq=3"

# 5. Get context
curl "http://localhost:8000/transcript/$COL/context/latest?meeting_id=$MID"
```

### Docker Test

```bash
# Chạy tests trong container
docker exec rag_api pytest tests/test_transcript_api.py -v --tb=short
```

> **Lưu ý**: Nếu gặp lỗi encoding, kiểm tra file test có UTF-8 BOM hay không.

---

## Known Issues

- Tests file (`test_transcript_api.py`) có thể có encoding issue → chạy manual test thay thế
- Ollama chưa bật → LLM_PROVIDER=none → context để trống (acceptable)

---

## License

MIT
