import uuid
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from app.schemas.embed import EmbedResponse, EmbedTextRequest
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService
from app.services.document_parser import DocumentParser
from app.utils.chunking import chunk_text
from app.config import settings


router = APIRouter(prefix="/embed", tags=["Embedding"])

parser = DocumentParser()


def get_embedder() -> EmbeddingService:
    return EmbeddingService()


def get_qdrant() -> QdrantService:
    return QdrantService()


@router.post("/file", response_model=EmbedResponse, summary="Upload và embed tài liệu")
async def embed_file(
    file: UploadFile = File(..., description="File tài liệu: .txt, .pdf, .docx, .md"),
    doc_id: Optional[str] = Form(None, description="UUID tùy chỉnh (optional)"),
    collection: Optional[str] = Form(None, description="Collection name (mặc định: documents)"),
    extra_metadata: Optional[str] = Form(None, description="JSON string metadata tùy chỉnh"),
    embedder: EmbeddingService = Depends(get_embedder),
):
    qdrant = QdrantService(collection)
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File rỗng")

    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File quá lớn (tối đa 50MB)")

    try:
        text = parser.parse(file_bytes, file.filename, file.content_type or "")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Không extract được text từ file")

    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="Không tạo được chunks từ text")

    _doc_id = doc_id or str(uuid.uuid4())

    _extra = {}
    if extra_metadata:
        try:
            _extra = json.loads(extra_metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="extra_metadata không phải JSON hợp lệ")

    vectors = embedder.embed_texts(chunks)

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
):
    qdrant = QdrantService(request.collection)
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


@router.delete("/{collection}/source/{source}", summary="Xóa tài liệu theo source")
async def delete_document(collection: str, source: str):
    qdrant = QdrantService(collection)
    qdrant.delete_by_source(source)
    return {"success": True, "message": f"Đã xóa tài liệu '{source}'"}


@router.delete("/{collection}/doc/{doc_id}", summary="Xóa tài liệu theo doc_id (UUID)")
async def delete_document_by_id(collection: str, doc_id: str):
    qdrant = QdrantService(collection)
    qdrant.delete_by_doc_id(doc_id)
    return {"success": True, "message": f"Đã xóa tài liệu có doc_id '{doc_id}'"}


@router.get("/{collection}/documents", summary="Liệt kê tài liệu trong collection")
async def list_documents(collection: str):
    qdrant = QdrantService(collection)
    docs = qdrant.list_documents()
    return {"documents": docs, "total": len(docs)}


@router.get("/collections", summary="Liệt kê tất cả collections")
async def list_collections():
    return {"collections": QdrantService.list_collections()}


@router.post("/collections", summary="Tạo collection mới")
async def create_collection(name: str = Form(..., description="Collection name")):
    return QdrantService.create_collection(name)


@router.delete("/collections", summary="Xóa collection")
async def delete_collection(name: str = Form(..., description="Collection name")):
    return QdrantService.delete_collection(name)