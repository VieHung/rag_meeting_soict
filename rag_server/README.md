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

---

## Docker Deployment

### Build và chạy toàn bộ stack

```bash
docker compose up --build -d
```

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

## License

MIT