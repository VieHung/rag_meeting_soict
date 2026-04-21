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
# Copy .env.example thành .env và điều chỉnh nếu cần
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

## API Endpoints

### Embed Documents

**POST /embed/file** - Upload và embed tài liệu

```bash
curl -X POST http://localhost:8000/embed/file \
  -F "file=@document.pdf"
```

Response:
```json
{
  "success": true,
  "doc_id": "uuid-string",
  "source": "document.pdf",
  "chunks_created": 24,
  "message": "Đã embed 24 chunks từ 'document.pdf'"
}
```

**POST /embed/text** - Embed plain text

```bash
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nội dung tài liệu...",
    "source": "my_document"
  }'
```

### Query Documents

**POST /query/** - Truy vấn ngữ nghĩa

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
    "top_k": 3,
    "score_threshold": 0.3
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
      "doc_id": "uuid-string",
      "chunk_index": 3,
      "chunk_total": 24
    }
  ],
  "total_found": 3
}
```

### Other Endpoints

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| `GET` | `/health` | Health check |
| `GET` | `/embed/info` | Thông tin collection |
| `DELETE` | `/embed/{source}` | Xóa tài liệu đã index |

## Supported File Types

- `.txt` - Plain text
- `.md` - Markdown
- `.pdf` - PDF documents
- `.docx` - Word documents

## Environment Variables

| Variable | Default | Mô Tả |
|----------|---------|-------|
| `QDRANT_HOST` | localhost | Qdrant server host |
| `QDRANT_PORT` | 6333 | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | documents | Tên collection |
| `EMBEDDING_MODEL` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Model embedding |
| `EMBEDDING_DIM` | 384 | Vector dimension |
| `CHUNK_SIZE` | 512 | Kích thước chunk (tokens) |
| `CHUNK_OVERLAP` | 64 | Overlap giữa các chunk |
| `TOP_K_DEFAULT` | 5 | Số kết quả mặc định |

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

## Integration với LLM

```python
import httpx

async def rag_answer(user_question: str) -> str:
    async with httpx.AsyncClient() as client:
        search_response = await client.post(
            "http://localhost:8000/query/",
            json={"query": user_question, "top_k": 5, "score_threshold": 0.4}
        )

    results = search_response.json()["results"]
    context = "\n\n---\n\n".join([r["text"] for r in results])

    prompt = f"""Dựa vào các đoạn tài liệu sau:

{context}

Hãy trả lời câu hỏi: {user_question}

Nếu không tìm thấy thông tin liên quan trong tài liệu, hãy nói rõ điều đó."""
    
    return prompt
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          CLIENT                                  │
│   POST /embed/file ───────────────► POST /query/                 │
└──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│  ┌───────────────┐     ┌────────────────────┐                   │
│  │ /embed router │     │ /query router      │                   │
│  │               │     │                    │                   │
│  │ 1. Parse file │     │ 1. Embed query     │                   │
│  │ 2. Chunk text │     │ 2. Search Qdrant   │                   │
│  │ 3. Embed      │     │ 3. Return results │                   │
│  │ 4. Upsert     │     │                   │                   │
│  └───────┬───────┘     └─────────┬──────────┘                     │
│          │                      │                                 │
│          ▼                      ▼                                 │
│  ┌──────────────────────────────────────────┐                    │
│  │      EmbeddingService (MiniLM-L12-v2)    │                    │
│  │           384-dim vectors                │                    │
│  └────────────────────┬───────────────────┘                    │
└───────────────────────┼─────────────────────────────────────────┘
                        │
                        ▼
             ┌─────────────────────────┐
             │    Qdrant Vector DB    │
             │   Collection: documents │
             └─────────────────────────┘
```

## Testing

```bash
# Test health check
curl http://localhost:8000/health

# Test embed text
curl -X POST http://localhost:8000/embed/text \
  -H "Content-Type: application/json" \
  -d '{"text": "RAG là gì?", "source": "test"}'

# Test query
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "RAG", "top_k": 3}'
```

## License

MIT