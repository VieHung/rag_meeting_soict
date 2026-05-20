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

Phase 2 bổ sung luồng RAG riêng cho **transcript của cuộc họp**, theo thiết kế v2 (xem `phase2plan_v2.md`).

### Kiến trúc 2 tầng

| Tầng | Phần cứng | Vai trò |
|------|-----------|---------|
| **Thiết bị (client)** | Qualcomm QCS8550 | AI on-device: live transcript, LLM nhỏ sinh câu trả lời |
| **Server (cái này)** | Server độc lập, mạnh | RAG API + Qdrant + Redis + LLM self-host build context |

### Quy ước Collection

Mỗi cuộc họp = một **cặp collection** (do app tạo thủ công, phân biệt bằng tiền tố):

| Loại | Tiền tố | Ví dụ | Endpoint |
|------|---------|-------|----------|
| Transcript | `meeting-{uuid}` | `meeting-c7cfdf57-...` | `/transcript/*`, `/query/transcript` |
| Tài liệu | `docs-{uuid}` | `docs-c7cfdf57-...` | `/embed/*`, `/query/` |

`meeting_id` được **suy ra từ tên collection** (bỏ tiền tố `meeting-`), client không cần gửi.

### Endpoints (4 endpoints, đã rút gọn từ 8)

| # | Method | Endpoint | Bắt buộc | Mô tả |
|---|--------|----------|----------|-------|
| 1 | `POST` | `/transcript/{collection}/embed` | ✅ Core | Lưu 1 câu transcript, server gán `sequence_id`, trigger build context nền |
| 2 | `POST` | `/query/transcript` | ✅ Core | Truy vấn ngữ nghĩa transcript, trả kết quả kèm **window** + **context** |
| 3 | `GET` | `/transcript/{collection}/context` | ⬜ Optional | Lấy context (tóm tắt) mới nhất; hỗ trợ `?sequence_id=` |
| 4 | `GET` | `/transcript/{collection}/segments` | ⬜ Optional | Liệt kê transcript theo khoảng `sequence_id` |

### Workflow

Collection naming: `meeting-{uuid}` (ví dụ: `meeting-abc123`).

```bash
# 1. Embed từng câu (server tự sinh sequence_id, build context nền)
curl -X POST http://localhost:8000/transcript/meeting-abc123/embed \
  -H "Content-Type: application/json" \
  -d '{
    "speaker": "Đoàn Sỹ Nguyên",
    "text": "Chúng ta cần xem lại ngân sách Q4 trước khi chốt."
  }'
# → 202 { "sequence_id": 1, "point_id": "...", "context_status": "pending" }

curl -X POST http://localhost:8000/transcript/meeting-abc123/embed \
  -H "Content-Type: application/json" \
  -d '{
    "speaker": "Mai Xuân Ngọc",
    "text": "Tôi đề xuất cắt 15% chi phí vận hành."
  }'
# → 202 { "sequence_id": 2, "context_status": "pending" }

# Đợi 5-10s cho context worker hoàn thành...

# 2. Query transcript (kèm window ±N câu + context)
curl -X POST http://localhost:8000/query/transcript \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "meeting-abc123",
    "query": "Quyết định về ngân sách Q4 là gì?",
    "top_k": 3,
    "window_size": 2
  }'
# → results[]: text + score + context + window.before/after

# 3. Lấy context mới nhất (hoặc tại sequence_id cụ thể)
curl "http://localhost:8000/transcript/meeting-abc123/context"
# → { "meeting_id": "abc123", "sequence_id": 2, "context": "...", "context_status": "ready" }

curl "http://localhost:8000/transcript/meeting-abc123/context?sequence_id=1"
# → context tại câu 1

# 4. Liệt kê transcript
curl "http://localhost:8000/transcript/meeting-abc123/segments?from_seq=1&to_seq=10"
```

### Cấu trúc payload (Qdrant — collection `meeting-*`)

```json
{
  "meeting_id":     "abc123",
  "sequence_id":    42,
  "speaker":        "Đoàn Sỹ Nguyên",
  "speaker_id":     "user_017",
  "text":           "Chúng ta cần xem lại ngân sách Q4...",
  "timestamp":      "2026-05-11T19:52:27Z",
  "context":        "Tóm tắt hội thoại tính tới câu này...",
  "context_status": "ready",
  "context_seq_base": 41,
  "lang":           "vi",
  "created_at":     "2026-05-11T19:52:27Z"
}
```

`context_status` ∈ `pending` | `processing` | `ready` | `failed` | `disabled`.

### Context Building

`context[N] = LLM_summarize(context[N-1] + transcript[N-1])`

- Context tại câu N là **bối cảnh dẫn tới câu N** (chưa gồm chính câu N).
- LLM self-host trên server (Ollama / LM Studio / OpenAI-compatible).
- ContextBuilder là background worker in-process.
- Có thể tắt bằng `LLM_PROVIDER=none` (context_status = `disabled`).

### System Prompt

File: `app/services/llm_client.py`. Prompt hiện tại:

- Vai trò: trợ lý tóm tắt cuộc họp
- 6 mức ưu tiên giữ nội dung: quyết định → con số → tên → chủ đề → action → quan điểm
- 4 tình huống tích hợp: mới / lặp / mâu thuẫn / thứ tự thời gian
- Edge cases: context rỗng, câu dài

### Tests

```bash
# Cần stack đang chạy (qdrant + redis + rag_api)
docker compose up -d

# SequenceManager unit test
pytest tests/test_sequence_manager.py -v

# Integration test transcript (cần LLM_PROVIDER=none cho ổn định)
pytest tests/test_transcript_api.py -v

# Manual test nhanh
curl http://localhost:8000/health
curl -X POST http://localhost:8000/transcript/meeting-test/embed \
  -d '{"speaker":"A","text":"test"}'
```

> **Lưu ý**: Nếu gặp lỗi encoding khi chạy pytest, kiểm tra UTF-8 BOM trong file test.

---

## License

MIT
