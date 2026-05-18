import uuid
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional
from app.schemas.embed import EmbedResponse, EmbedTextRequest
from app.services.embedding import EmbeddingService
from app.services.vector_store import QdrantService
from app.services.document_parser import DocumentParser
from app.utils.chunking import chunk_text
from app.config import settings


router = APIRouter(prefix="/embed", tags=["Embedding"])

parser = DocumentParser()


def _embed_file_background(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    collection: Optional[str],
    doc_id: str,
    extra_metadata: dict,
):
    try:
        qdrant = QdrantService(collection)
        text = parser.parse(file_bytes, filename, content_type or "")
        if not text.strip():
            print(f"Skip embedding: empty parsed text for file '{filename}'")
            return

        chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        if not chunks:
            print(f"Skip embedding: no chunks generated for file '{filename}'")
            return

        vectors = EmbeddingService().embed_texts(chunks)
        metadata = {
            "source": filename,
            "doc_id": doc_id,
            "file_size": len(file_bytes),
            "mime_type": content_type,
            **extra_metadata,
        }
        count = qdrant.upsert_chunks(chunks, vectors, metadata)
        print(f"Background embedded {count} chunks from file '{filename}'")
    except Exception as e:
        print(f"Background embedding failed for file '{filename}': {str(e)}")


def _embed_text_background(
    text: str,
    source: str,
    collection: Optional[str],
    doc_id: str,
    metadata: dict,
):
    try:
        qdrant = QdrantService(collection)
        chunks = chunk_text(
            text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if not chunks:
            print(f"Skip embedding: no chunks generated for source '{source}'")
            return

        vectors = EmbeddingService().embed_texts(chunks)
        payload_metadata = {
            "source": source,
            "doc_id": doc_id,
            **(metadata or {}),
        }
        count = qdrant.upsert_chunks(chunks, vectors, payload_metadata)
        print(f"Background embedded {count} chunks from source '{source}'")
    except Exception as e:
        print(f"Background embedding failed for source '{source}': {str(e)}")


@router.post("/file", response_model=EmbedResponse, summary="Upload và embed tài liệu")
async def embed_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="File tài liệu: .txt, .pdf, .docx, .md"),
    doc_id: Optional[str] = Form(None, description="UUID tùy chỉnh (optional)"),
    collection: Optional[str] = Form(None, description="Collection name (mặc định: documents)"),
    extra_metadata: Optional[str] = Form(None, description="JSON string metadata tùy chỉnh"),
):
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File rỗng")

    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File quá lớn (tối đa 50MB)")

    _doc_id = doc_id or str(uuid.uuid4())

    _extra = {}
    if extra_metadata:
        try:
            _extra = json.loads(extra_metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="extra_metadata không phải JSON hợp lệ")

    background_tasks.add_task(
        _embed_file_background,
        file_bytes=file_bytes,
        filename=file.filename,
        content_type=file.content_type,
        collection=collection,
        doc_id=_doc_id,
        extra_metadata=_extra,
    )

    return EmbedResponse(
        success=True,
        doc_id=_doc_id,
        source=file.filename,
        chunks_created=0,
        message=f"Đã nhận file '{file.filename}', embedding sẽ chạy nền ngay sau response",
    )


@router.post("/text", response_model=EmbedResponse, summary="Embed plain text")
async def embed_text(
    request: EmbedTextRequest,
    background_tasks: BackgroundTasks,
):
    _doc_id = request.doc_id or str(uuid.uuid4())

    background_tasks.add_task(
        _embed_text_background,
        text=request.text,
        source=request.source,
        collection=request.collection,
        doc_id=_doc_id,
        metadata=request.metadata or {},
    )

    return EmbedResponse(
        success=True,
        doc_id=_doc_id,
        source=request.source,
        chunks_created=0,
        message=f"Đã nhận source '{request.source}', embedding sẽ chạy nền ngay sau response",
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