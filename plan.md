# 🗂️ Kế Hoạch Xây Dựng RAG Pipeline với Qdrant + MiniLM-L12-v2

> **Stack:** FastAPI · Qdrant · sentence-transformers (multilingual-MiniLM-L12-v2) · PyMuPDF · python-docx  
> **Mục tiêu:** Server vectorDB độc lập + 2 REST endpoint: `/embed` (ingest tài liệu) & `/query` (truy vấn ngữ nghĩa)

---

## 📐 Tổng Quan Kiến Trúc

```
┌──────────────────────────────────────────────────────────────────┐
│                          CLIENT                                  │
│   POST /embed  ──────────────────►  POST /query                  │
│   (files/text)                       (question string)           │
└────────────┬────────────────────────────────┬────────────────────┘
             │                                │
             ▼                                ▼
┌────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                       │
│                                                                │
│  ┌─────────────────────┐     ┌──────────────────────────────┐  │
│  │   /embed  endpoint  │     │      /query  endpoint        │  │
│  │                     │     │                              │  │
│  │ 1. Parse file       │     │ 1. Embed query string        │  │
│  │ 2. Chunk text       │     │ 2. Search Qdrant (top-k)     │  │
│  │ 3. Embed chunks     │     │ 3. Return ranked results     │  │
│  │ 4. Upsert Qdrant    │     │    + metadata + scores       │  │
│  └──────────┬──────────┘     └──────────────┬───────────────┘  │
│             │                               │                   │
│             └─────────────┬─────────────────┘                   │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │   EmbeddingService     │                         │
│              │  (MiniLM-L12-v2)       │                         │
│              │  384-dim vectors       │                         │
│              └────────────┬───────────┘                         │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │    Qdrant Vector DB      │
             │  (Docker / local server) │
             │  Collection: documents   │
             └──────────────────────────┘
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
rag-server/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Cấu hình env vars
│   ├── dependencies.py          # Dependency injection (embedding, qdrant client)
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── embed.py             # POST /embed endpoint
│   │   └── query.py             # POST /query endpoint
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py         # EmbeddingService (MiniLM wrapper)
│   │   ├── vector_store.py      # QdrantService (CRUD operations)
│   │   └── document_parser.py   # FileParser (pdf, docx, txt, ...)
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── embed.py             # Pydantic models cho /embed
│   │   └── query.py             # Pydantic models cho /query
│   │
│   └── utils/
│       ├── __init__.py
│       └── chunking.py          # Text chunking strategies
│
├── tests/
│   ├── test_embed.py
│   ├── test_query.py
│   └── test_parser.py
│
├── docker-compose.yml           # Qdrant + FastAPI services
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ PHASE 1: Chuẩn Bị Môi Trường

### 1.1 Cài đặt Dependencies

```txt
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9          # upload file
sentence-transformers==3.0.1     # MiniLM model
qdrant-client==1.10.1            # Qdrant Python SDK
pymupdf==1.24.5                  # Parse PDF (fitz)
python-docx==1.1.2               # Parse .docx
openpyxl==3.1.5                  # Parse .xlsx (optional)
pydantic==2.8.0
pydantic-settings==2.4.0
python-dotenv==1.0.1
numpy==1.26.4
tiktoken==0.7.0                  # Đếm token khi chunking
httpx==0.27.0                    # Async HTTP (test)
```

```bash
pip install -r requirements.txt
```

### 1.2 Khởi Động Qdrant Server bằng Docker

```yaml
# docker-compose.yml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:v1.10.0
    container_name: qdrant_server
    ports:
      - "6333:6333"    # REST API
      - "6334:6334"    # gRPC
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
    restart: unless-stopped

  rag_api:
    build: .
    container_name: rag_api
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
    env_file:
      - .env
    depends_on:
      - qdrant
    restart: unless-stopped
```

```bash
# Khởi động chỉ Qdrant trước (development)
docker compose up qdrant -d

# Kiểm tra Qdrant đã sẵn sàng
curl http://localhost:6333/healthz
# → {"title":"qdrant - vector search engine","version":"..."}
```

### 1.3 Cấu Hình Environment

```bash
# .env.example  →  copy thành .env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
CHUNK_SIZE=512          # số token mỗi chunk
CHUNK_OVERLAP=64        # overlap giữa các chunk
TOP_K_DEFAULT=5         # số kết quả trả về mặc định
```

---

## 🧠 PHASE 2: Xây Dựng Core Services

### 2.1 Config (`app/config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_default: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2.2 Embedding Service (`app/services/embedding.py`)

```python
"""
Wrapper cho sentence-transformers MiniLM-L12-v2.
Model được load 1 lần duy nhất khi khởi động app (singleton).
Hỗ trợ batch encoding để tăng throughput.
"""

from sentence_transformers import SentenceTransformer
from app.config import settings
import numpy as np
from typing import List

class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer(settings.embedding_model)
            cls._instance._dim = settings.embedding_dim
        return cls._instance

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Nhận list các string, trả về list vector 384-dim.
        Tự động normalize về unit sphere (cosine similarity).
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # cosine = dot product sau normalize
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Shortcut cho single query."""
        return self.embed_texts([query])[0]
```

**Lý do chọn MiniLM-L12-v2:**
- Model `paraphrase-multilingual-MiniLM-L12-v2` hỗ trợ **50+ ngôn ngữ** bao gồm Tiếng Việt
- Output dimension: **384** — cân bằng tốt giữa chất lượng và tốc độ
- Kích thước model: ~120MB — phù hợp chạy trên CPU server thông thường
- Inference latency: ~5-15ms/text trên CPU (batch)

### 2.3 Document Parser (`app/services/document_parser.py`)

```python
"""
Hỗ trợ parse các định dạng: .txt, .md, .pdf, .docx, .xlsx
Trả về plain text đã được làm sạch.
"""

import io
import fitz                          # PyMuPDF
from docx import Document
from pathlib import Path
from typing import Union

SUPPORTED_TYPES = {
    "text/plain": "_parse_txt",
    "text/markdown": "_parse_txt",
    "application/pdf": "_parse_pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "_parse_docx",
}

class DocumentParser:

    def parse(self, file_bytes: bytes, filename: str, mime_type: str) -> str:
        """Entry point: nhận bytes của file, trả về text thuần."""
        ext = Path(filename).suffix.lower()

        # Override mime_type bằng extension khi cần
        if ext == ".pdf":
            return self._parse_pdf(file_bytes)
        elif ext in (".docx", ".doc"):
            return self._parse_docx(file_bytes)
        elif ext in (".txt", ".md", ".rst"):
            return self._parse_txt(file_bytes)
        else:
            # Cố parse như txt nếu không nhận diện được
            try:
                return self._parse_txt(file_bytes)
            except Exception:
                raise ValueError(f"Unsupported file type: {ext}")

    def _parse_txt(self, data: bytes) -> str:
        """Decode UTF-8, fallback sang latin-1."""
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="replace")

    def _parse_pdf(self, data: bytes) -> str:
        """
        Dùng PyMuPDF để extract text từ PDF.
        Giữ nguyên thứ tự reading order, loại bỏ ký tự control.
        """
        text_parts = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text("text")   # plain text, giữ layout
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")
        return "\n\n".join(text_parts)

    def _parse_docx(self, data: bytes) -> str:
        """
        Extract text từ .docx, bảo toàn thứ tự paragraph.
        Bỏ qua header/footer, lấy body chính.
        """
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        
        # Lấy text trong table nếu có
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    table_texts.append(row_text)

        return "\n\n".join(paragraphs + table_texts)
```

### 2.4 Chunking (`app/utils/chunking.py`)

```python
"""
Token-aware chunking với overlap.
Dùng tiktoken để đếm token (tương thích GPT tokenizer, gần đúng cho MiniLM).
Overlap giúp không mất ngữ cảnh ở ranh giới chunk.
"""

import tiktoken
from typing import List

_enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """
    Chia text thành các chunk theo đơn vị token.
    
    Chiến lược:
    1. Ưu tiên split tại ranh giới câu/đoạn (\n\n, \n, '. ', '! ', '? ')
    2. Nếu đoạn quá dài → force split theo token
    3. Áp dụng overlap để duy trì ngữ cảnh

    Returns:
        List[str]: Các chunk text, mỗi chunk ≤ chunk_size tokens
    """
    # Split theo đoạn văn trước
    paragraphs = _split_by_separators(text)
    
    chunks = []
    current_tokens = []
    current_len = 0

    for para in paragraphs:
        para_tokens = _enc.encode(para)
        
        # Đoạn quá dài → chia nhỏ hơn
        if len(para_tokens) > chunk_size:
            # Flush current buffer trước
            if current_tokens:
                chunks.append(_enc.decode(current_tokens))
                # Giữ lại overlap
                current_tokens = current_tokens[-overlap:]
                current_len = len(current_tokens)
            
            # Chia đoạn dài thành sub-chunks
            for sub_chunk in _force_split(para_tokens, chunk_size, overlap):
                chunks.append(_enc.decode(sub_chunk))
            continue

        # Nếu thêm vào vượt chunk_size → flush và bắt đầu chunk mới
        if current_len + len(para_tokens) > chunk_size:
            if current_tokens:
                chunks.append(_enc.decode(current_tokens))
            # Bắt đầu chunk mới với overlap từ chunk cũ
            current_tokens = current_tokens[-overlap:] + para_tokens
            current_len = len(current_tokens)
        else:
            current_tokens.extend(para_tokens)
            current_len += len(para_tokens)

    # Flush chunk cuối
    if current_tokens:
        chunks.append(_enc.decode(current_tokens))

    return [c.strip() for c in chunks if c.strip()]


def _split_by_separators(text: str) -> List[str]:
    """Split text theo các separator tự nhiên."""
    import re
    # Ưu tiên: double newline > single newline > sentence end
    parts = re.split(r'\n\n+', text)
    result = []
    for part in parts:
        # Tiếp tục split nếu vẫn còn quá dài
        sentences = re.split(r'(?<=[.!?])\s+', part)
        result.extend(sentences)
    return [p.strip() for p in result if p.strip()]


def _force_split(tokens: List[int], size: int, overlap: int) -> List[List[int]]:
    """Force split token list khi không có separator tự nhiên."""
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(tokens[start:end])
        start += size - overlap
    return chunks
```

### 2.5 Vector Store Service (`app/services/vector_store.py`)

```python
"""
Wrapper cho Qdrant client.
Tự động tạo collection nếu chưa tồn tại.
Hỗ trợ upsert batch và similarity search.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, ScoredPoint,
)
from typing import List, Dict, Any, Optional
from app.config import settings
import uuid

class QdrantService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30,
            )
            cls._instance._collection = settings.qdrant_collection_name
            cls._instance._ensure_collection()
        return cls._instance

    def _ensure_collection(self):
        """Tạo collection nếu chưa tồn tại."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,   # 384
                    distance=Distance.COSINE,       # cosine similarity
                ),
            )
            # Tạo payload index để filter nhanh theo source
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="source",
                field_schema="keyword",
            )
            print(f"✅ Created Qdrant collection: '{self._collection}'")

    def upsert_chunks(
        self,
        chunks: List[str],
        vectors: List[List[float]],
        metadata: Dict[str, Any],
    ) -> int:
        """
        Insert/update các chunk vào Qdrant.
        metadata: {"source": filename, "doc_id": uuid, ...}
        
        Returns:
            Số lượng point đã upsert
        """
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "chunk_index": i,
                "chunk_total": len(chunks),
                **metadata,
            }
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        # Upsert theo batch 100 points
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i:i+batch_size],
            )

        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm top_k chunk gần nhất với query vector.
        
        Args:
            query_vector: Vector 384-dim của query
            top_k: Số kết quả trả về
            source_filter: Lọc theo tên file nguồn (optional)

        Returns:
            List dict với keys: text, score, source, chunk_index, doc_id, ...
        """
        qdrant_filter = None
        if source_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter),
                )]
            )

        results: List[ScoredPoint] = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            score_threshold=0.0,    # Trả về tất cả, để client tự filter
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "score": round(r.score, 4),
                "source": r.payload.get("source", ""),
                "doc_id": r.payload.get("doc_id", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "chunk_total": r.payload.get("chunk_total", 0),
            }
            for r in results
        ]

    def delete_by_source(self, source: str) -> int:
        """Xóa toàn bộ chunk của một tài liệu theo tên file."""
        result = self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )]
            ),
        )
        return result.operation_id

    def collection_info(self) -> Dict[str, Any]:
        """Trả về thông tin collection (số points, status, ...)."""
        info = self._client.get_collection(self._collection)
        return {
            "name": self._collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }
```

---

## 🔌 PHASE 3: Xây Dựng API Endpoints

### 3.1 Schemas (`app/schemas/`)

```python
# app/schemas/embed.py
from pydantic import BaseModel, Field
from typing import Optional

class EmbedTextRequest(BaseModel):
    """Schema cho embed plain text (không upload file)."""
    text: str = Field(..., min_length=1, description="Nội dung text cần embed")
    source: str = Field(..., description="Tên định danh nguồn tài liệu")
    doc_id: Optional[str] = Field(None, description="UUID của tài liệu (tự generate nếu bỏ trống)")
    metadata: Optional[dict] = Field(default_factory=dict, description="Metadata tùy chỉnh")

class EmbedResponse(BaseModel):
    success: bool
    doc_id: str
    source: str
    chunks_created: int
    message: str
```

```python
# app/schemas/query.py
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi / query của người dùng")
    top_k: int = Field(5, ge=1, le=50, description="Số kết quả trả về")
    source_filter: Optional[str] = Field(None, description="Lọc kết quả theo tên file nguồn")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Ngưỡng similarity tối thiểu")

class QueryResult(BaseModel):
    text: str
    score: float
    source: str
    doc_id: str
    chunk_index: int
    chunk_total: int

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_found: int
```

### 3.2 Embed Endpoint (`app/routers/embed.py`)

```python
"""
POST /embed/file   → Upload file (multipart/form-data)
POST /embed/text   → Gửi text trực tiếp (JSON)
DELETE /embed/{source} → Xóa tài liệu đã index
GET /embed/info    → Thông tin collection
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from app.schemas.embed import EmbedResponse, EmbedTextRequest
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService
from app.services.document_parser import DocumentParser
from app.utils.chunking import chunk_text
from app.config import settings

router = APIRouter(prefix="/embed", tags=["Embedding"])

# Dependency injection
def get_embedder() -> EmbeddingService:
    return EmbeddingService()

def get_qdrant() -> QdrantService:
    return QdrantService()

parser = DocumentParser()


@router.post("/file", response_model=EmbedResponse, summary="Upload và embed tài liệu")
async def embed_file(
    file: UploadFile = File(..., description="File tài liệu: .txt, .pdf, .docx, .md"),
    doc_id: Optional[str] = Form(None, description="UUID tùy chỉnh (optional)"),
    extra_metadata: Optional[str] = Form(None, description="JSON string metadata tùy chỉnh"),
    embedder: EmbeddingService = Depends(get_embedder),
    qdrant: QdrantService = Depends(get_qdrant),
):
    """
    **Workflow:**
    1. Nhận file upload
    2. Parse text từ file (pdf/docx/txt/...)
    3. Chunk text thành các đoạn nhỏ (≤512 token)
    4. Embed từng chunk bằng MiniLM-L12-v2
    5. Upsert vectors vào Qdrant kèm metadata
    """
    # Đọc file bytes
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File rỗng")

    # Giới hạn kích thước: 50MB
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File quá lớn (tối đa 50MB)")

    # Parse text từ file
    try:
        text = parser.parse(file_bytes, file.filename, file.content_type or "")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Không extract được text từ file")

    # Chunk text
    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="Không tạo được chunks từ text")

    # Tạo doc_id nếu chưa có
    _doc_id = doc_id or str(uuid.uuid4())

    # Parse extra metadata
    import json
    _extra = {}
    if extra_metadata:
        try:
            _extra = json.loads(extra_metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="extra_metadata không phải JSON hợp lệ")

    # Embed tất cả chunks (batch)
    vectors = embedder.embed_texts(chunks)

    # Upsert vào Qdrant
    metadata = {
        "source": file.filename,
        "doc_id": _doc_id,
        "file_size": len(file_bytes),
        "mime_type": file.content_type,
        **_extra,
    }
    count = qdrant.upsert_chunks(chunks, vectors, metadata)

    return EmbedResponse(
        success=True,
        doc_id=_doc_id,
        source=file.filename,
        chunks_created=count,
        message=f"Đã embed {count} chunks từ '{file.filename}'",
    )


@router.post("/text", response_model=EmbedResponse, summary="Embed plain text")
async def embed_text(
    request: EmbedTextRequest,
    embedder: EmbeddingService = Depends(get_embedder),
    qdrant: QdrantService = Depends(get_qdrant),
):
    """Embed trực tiếp từ text string, không cần upload file."""
    _doc_id = request.doc_id or str(uuid.uuid4())

    chunks = chunk_text(
        request.text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    vectors = embedder.embed_texts(chunks)
    metadata = {
        "source": request.source,
        "doc_id": _doc_id,
        **(request.metadata or {}),
    }
    count = qdrant.upsert_chunks(chunks, vectors, metadata)

    return EmbedResponse(
        success=True,
        doc_id=_doc_id,
        source=request.source,
        chunks_created=count,
        message=f"Đã embed {count} chunks từ source '{request.source}'",
    )


@router.delete("/{source}", summary="Xóa tài liệu đã index")
async def delete_document(
    source: str,
    qdrant: QdrantService = Depends(get_qdrant),
):
    """Xóa toàn bộ vector của một tài liệu dựa trên tên source."""
    qdrant.delete_by_source(source)
    return {"success": True, "message": f"Đã xóa tài liệu '{source}'"}


@router.get("/info", summary="Thông tin collection")
async def collection_info(qdrant: QdrantService = Depends(get_qdrant)):
    """Trả về metadata của Qdrant collection."""
    return qdrant.collection_info()
```

### 3.3 Query Endpoint (`app/routers/query.py`)

```python
"""
POST /query   → Tìm kiếm ngữ nghĩa trong vectorDB
"""

from fastapi import APIRouter, HTTPException, Depends
from app.schemas.query import QueryRequest, QueryResponse, QueryResult
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService

router = APIRouter(prefix="/query", tags=["Query"])

def get_embedder() -> EmbeddingService:
    return EmbeddingService()

def get_qdrant() -> QdrantService:
    return QdrantService()


@router.post("/", response_model=QueryResponse, summary="Truy vấn ngữ nghĩa")
async def query_documents(
    request: QueryRequest,
    embedder: EmbeddingService = Depends(get_embedder),
    qdrant: QdrantService = Depends(get_qdrant),
):
    """
    **Workflow:**
    1. Embed query string thành vector 384-dim
    2. Tìm kiếm top-k vector gần nhất trong Qdrant (cosine similarity)
    3. Filter theo score_threshold nếu có
    4. Trả về list chunk text kèm metadata và score

    **Tích hợp với LLM:**
    Kết quả trả về có thể dùng trực tiếp làm context cho LLM:
    ```
    context = "\n\n".join([r["text"] for r in response["results"]])
    prompt = f"Dựa vào context:\n{context}\n\nTrả lời: {query}"
    ```
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query không được rỗng")

    # Embed query
    query_vector = embedder.embed_query(request.query)

    # Search Qdrant
    raw_results = qdrant.search(
        query_vector=query_vector,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    # Apply score threshold
    filtered = [
        r for r in raw_results
        if r["score"] >= request.score_threshold
    ]

    results = [QueryResult(**r) for r in filtered]

    return QueryResponse(
        query=request.query,
        results=results,
        total_found=len(results),
    )
```

### 3.4 Main App (`app/main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import embed, query
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Khởi tạo các singleton service khi app start.
    Đảm bảo model MiniLM được load vào RAM trước khi nhận request.
    """
    print("🚀 Loading embedding model...")
    EmbeddingService()     # load model vào RAM
    print("✅ Embedding model ready")

    print("🔌 Connecting to Qdrant...")
    QdrantService()        # kiểm tra connection + tạo collection
    print("✅ Qdrant ready")

    yield   # app đang chạy

    print("🛑 Shutting down...")


app = FastAPI(
    title="RAG Vector Store API",
    description="API embedding tài liệu và truy vấn ngữ nghĩa với Qdrant + MiniLM-L12-v2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(embed.router)
app.include_router(query.router)


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "RAG Vector Store API"}
```

---

## 🐳 PHASE 4: Dockerize

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps cho PyMuPDF
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model khi build (không cần internet lúc runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```bash
# Build và run toàn bộ stack
docker compose up --build -d

# Xem logs
docker compose logs -f rag_api
```

---

## 🧪 PHASE 5: Testing

### 5.1 Test Embed File

```bash
# Upload PDF
curl -X POST http://localhost:8000/embed/file \
  -F "file=@my_document.pdf" \
  -F "doc_id=doc-001"

# Upload DOCX
curl -X POST http://localhost:8000/embed/file \
  -F "file=@report.docx"

# Response
{
  "success": true,
  "doc_id": "doc-001",
  "source": "my_document.pdf",
  "chunks_created": 24,
  "message": "Đã embed 24 chunks từ 'my_document.pdf'"
}
```

### 5.2 Test Query

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
    "top_k": 3,
    "score_threshold": 0.3
  }'

# Response
{
  "query": "Phương pháp xử lý ngôn ngữ tự nhiên",
  "results": [
    {
      "text": "NLP (Natural Language Processing) là lĩnh vực...",
      "score": 0.8741,
      "source": "my_document.pdf",
      "doc_id": "doc-001",
      "chunk_index": 3,
      "chunk_total": 24
    },
    ...
  ],
  "total_found": 3
}
```

### 5.3 Unit Tests (`tests/test_embed.py`)

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_embed_text():
    r = client.post("/embed/text", json={
        "text": "Đây là tài liệu test về trí tuệ nhân tạo.",
        "source": "test_source"
    })
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["chunks_created"] >= 1

def test_query():
    r = client.post("/query/", json={
        "query": "trí tuệ nhân tạo",
        "top_k": 3
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data

def test_embed_pdf():
    # Tạo PDF giả (bytes hợp lệ)
    pdf_content = b"%PDF-1.4 ..."   # minimal PDF
    r = client.post(
        "/embed/file",
        files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")},
    )
    # Ít nhất không crash
    assert r.status_code in (200, 422)
```

---

## 📊 PHASE 6: Tối Ưu & Mở Rộng

### 6.1 Caching Query Vector (Optional)

```python
# Nếu cùng query được gửi nhiều lần, cache embedding để tiết kiệm compute
from functools import lru_cache

@lru_cache(maxsize=512)
def cached_embed_query(query: str) -> tuple:
    """Cache 512 query gần nhất. Convert list → tuple vì lru_cache cần hashable."""
    return tuple(EmbeddingService().embed_query(query))
```

### 6.2 Thêm Re-ranking (Nâng cao)

```python
"""
Sau khi lấy top-k từ Qdrant, áp dụng cross-encoder re-ranking
để cải thiện độ chính xác kết quả.
Model gợi ý: cross-encoder/ms-marco-MiniLM-L-6-v2
"""
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, results: list, top_n: int = 5) -> list:
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]
```

### 6.3 Multi-Collection Support (Optional)

```python
# Cho phép mỗi user/project có collection riêng
@router.post("/{collection_name}/file")
async def embed_to_collection(collection_name: str, file: UploadFile, ...):
    ...
```

### 6.4 Monitoring với Qdrant Dashboard

```
Truy cập: http://localhost:6333/dashboard
→ Xem số lượng points, vector space, health check
```

---

## 📋 Tóm Tắt API Reference

| Method | Endpoint | Mô Tả |
|--------|----------|-------|
| `POST` | `/embed/file` | Upload & embed file (pdf/docx/txt/...) |
| `POST` | `/embed/text` | Embed plain text qua JSON |
| `DELETE` | `/embed/{source}` | Xóa tài liệu đã index |
| `GET` | `/embed/info` | Thông tin Qdrant collection |
| `POST` | `/query/` | Truy vấn ngữ nghĩa top-k |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI (tự động) |

---

## 🔗 Tích Hợp với LLM (Full RAG Flow)

```python
"""
Ví dụ tích hợp kết quả /query với OpenAI / Ollama để tạo câu trả lời hoàn chỉnh.
"""
import httpx

async def rag_answer(user_question: str) -> str:
    # 1. Retrieve relevant chunks
    async with httpx.AsyncClient() as client:
        search_response = await client.post(
            "http://localhost:8000/query/",
            json={"query": user_question, "top_k": 5, "score_threshold": 0.4}
        )
    
    results = search_response.json()["results"]
    context = "\n\n---\n\n".join([r["text"] for r in results])

    # 2. Augment prompt
    prompt = f"""Dựa vào các đoạn tài liệu sau:

{context}

Hãy trả lời câu hỏi: {user_question}

Nếu không tìm thấy thông tin liên quan trong tài liệu, hãy nói rõ điều đó."""

    # 3. Generate answer (với Ollama local)
    gen_response = await client.post(
        "http://localhost:11434/api/generate",
        json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False}
    )
    return gen_response.json()["response"]
```

---

*Kế hoạch này bao gồm đủ để build một RAG pipeline production-ready từ đầu đến cuối. Có thể mở rộng thêm authentication (API key), rate limiting, và async embedding queue (Celery/RQ) cho workload lớn.*