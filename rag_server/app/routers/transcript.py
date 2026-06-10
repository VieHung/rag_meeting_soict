"""Router cho luồng transcript (Phase 2) - Bản v2.

Endpoints theo phase2plan_v2.md:
- POST /transcript/{collection}/embed (Core)
- POST /query/transcript (Core)
- GET  /transcript/{collection}/context (Optional)
- GET  /transcript/{collection}/segments (Optional)

Các endpoint cũ đã gỡ:
- POST /transcript/{collection}/meeting/init (gỡ - lazy init)
- DELETE /transcript/{collection}/meeting/{meeting_id} (gỡ - dùng /embed/collections)
- PATCH /transcript/{collection}/context/{sequence_id} (gỡ - LLM chạy in-process)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import Optional

from app.dependencies import get_transcript_service
from app.schemas.transcript import (
    ContextResponse,
    SegmentListResponse,
    TranscriptEmbedRequest,
    TranscriptEmbedResponse,
)
from app.services.transcript_service import (
    InvalidCollectionPrefix,
    TranscriptService,
)
from app.workers.context_worker import get_context_worker


router = APIRouter(prefix="/transcript", tags=["Transcript"])

MEETING_PREFIX = "meeting-"


def _validate_meeting_collection(collection: str) -> str:
    if not collection.startswith(MEETING_PREFIX):
        raise InvalidCollectionPrefix(
            f"Collection must start with '{MEETING_PREFIX}' for transcript endpoints"
        )
    return collection.removeprefix(MEETING_PREFIX)


# ---- ingest ------------------------------------------------------------------


@router.post(
    "/{collection}/embed",
    response_model=TranscriptEmbedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Nhận một câu transcript, gán sequence_id, lưu thành 1 vector, build context nền",
)
async def embed_transcript(
    collection: str,
    request: TranscriptEmbedRequest,
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        meeting_id = _validate_meeting_collection(collection)
    except InvalidCollectionPrefix as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        response, _builder, seq = await service.embed_transcript(
            collection=collection,
            meeting_id=meeting_id,
            request=request,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # D9: build context tuần tự qua hàng đợi FIFO (không dùng BackgroundTasks,
    # vốn không đảm bảo thứ tự giữa các request).
    await get_context_worker().enqueue(collection, meeting_id, seq)
    return response


# ---- context (merged into single endpoint) ----------------------------------


@router.get(
    "/{collection}/context",
    response_model=ContextResponse,
    summary="Lấy context (mặc định mới nhất, hoặc tại sequence_id cụ thể)",
)
async def get_context(
    collection: str,
    sequence_id: Optional[int] = Query(None, description="Lấy context tại sequence_id cụ thể"),
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        meeting_id = _validate_meeting_collection(collection)
    except InvalidCollectionPrefix as e:
        raise HTTPException(status_code=400, detail=str(e))

    response = await service.get_context(
        collection=collection,
        meeting_id=meeting_id,
        sequence_id=sequence_id,
    )
    if response is None:
        detail = f"Context not found for meeting={meeting_id}"
        if sequence_id is not None:
            detail += f" seq={sequence_id}"
        raise HTTPException(status_code=404, detail=detail)
    return response


# ---- segments ---------------------------------------------------------------


@router.get(
    "/{collection}/segments",
    response_model=SegmentListResponse,
    summary="Liệt kê transcript theo khoảng sequence_id",
)
async def list_segments(
    collection: str,
    from_seq: int = Query(1, ge=1, description="Bắt đầu từ sequence_id"),
    to_seq: Optional[int] = Query(None, ge=1, description="Kết thúc tại sequence_id"),
    limit: int = Query(100, ge=1, le=1000),
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        meeting_id = _validate_meeting_collection(collection)
    except InvalidCollectionPrefix as e:
        raise HTTPException(status_code=400, detail=str(e))

    return await service.list_segments(
        collection=collection,
        meeting_id=meeting_id,
        from_seq=from_seq,
        to_seq=to_seq,
        limit=limit,
    )
