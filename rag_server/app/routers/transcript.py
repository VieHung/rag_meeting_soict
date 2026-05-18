"""Router cho luồng transcript (Phase 2).

Endpoints (theo plan.md mục 7):
- POST /transcript/{collection}/meeting/init
- POST /transcript/{collection}/embed
- GET  /transcript/{collection}/context/latest
- GET  /transcript/{collection}/context/{sequence_id}
- PATCH /transcript/{collection}/context/{sequence_id}
- GET  /transcript/{collection}/meeting/{meeting_id}/segments
- DELETE /transcript/{collection}/meeting/{meeting_id}
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from typing import Optional

from app.dependencies import get_transcript_service
from app.schemas.transcript import (
    ContextResponse,
    ContextUpdateRequest,
    DeleteMeetingResponse,
    MeetingInitRequest,
    MeetingInitResponse,
    SegmentListResponse,
    TranscriptEmbedRequest,
    TranscriptEmbedResponse,
)
from app.services.transcript_service import (
    MeetingAlreadyExists,
    MeetingNotInitialized,
    TranscriptService,
)


router = APIRouter(prefix="/transcript", tags=["Transcript"])


# ---- meeting lifecycle -------------------------------------------------------


@router.post(
    "/{collection}/meeting/init",
    response_model=MeetingInitResponse,
    summary="Khởi tạo cuộc họp mới (reset counter, ghi metadata)",
)
async def init_meeting(
    collection: str,
    request: MeetingInitRequest,
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        return await service.init_meeting(
            collection=collection,
            meeting_id=request.meeting_id,
            force_reset=request.force_reset,
        )
    except MeetingAlreadyExists as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete(
    "/{collection}/meeting/{meeting_id}",
    response_model=DeleteMeetingResponse,
    summary="Xóa toàn bộ transcript + state Redis của một cuộc họp",
)
async def delete_meeting(
    collection: str,
    meeting_id: str,
    service: TranscriptService = Depends(get_transcript_service),
):
    return await service.delete_meeting(collection=collection, meeting_id=meeting_id)


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
    background_tasks: BackgroundTasks,
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        response, builder, seq = await service.embed_transcript(
            collection=collection, request=request
        )
    except MeetingNotInitialized as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Schedule background context build.
    background_tasks.add_task(builder.build, request.meeting_id, seq)
    return response


# ---- context -----------------------------------------------------------------


@router.get(
    "/{collection}/context/latest",
    response_model=ContextResponse,
    summary="Lấy context mới nhất của cuộc họp",
)
async def get_latest_context(
    collection: str,
    meeting_id: str = Query(..., description="ID cuộc họp"),
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        return await service.get_latest_context(
            collection=collection, meeting_id=meeting_id
        )
    except MeetingNotInitialized as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/{collection}/context/{sequence_id}",
    response_model=ContextResponse,
    summary="Lấy context tại một thời điểm cụ thể",
)
async def get_context_at(
    collection: str,
    sequence_id: int,
    meeting_id: str = Query(..., description="ID cuộc họp"),
    service: TranscriptService = Depends(get_transcript_service),
):
    response = await service.get_context_at(
        collection=collection, meeting_id=meeting_id, sequence_id=sequence_id
    )
    if response is None:
        raise HTTPException(
            status_code=404,
            detail=f"Context not found for meeting={meeting_id} seq={sequence_id}",
        )
    return response


@router.patch(
    "/{collection}/context/{sequence_id}",
    response_model=ContextResponse,
    summary="Cập nhật context của một câu (LLM nội bộ hoặc background task)",
)
async def update_context(
    collection: str,
    sequence_id: int,
    request: ContextUpdateRequest,
    service: TranscriptService = Depends(get_transcript_service),
):
    try:
        return await service.update_context(
            collection=collection,
            sequence_id=sequence_id,
            meeting_id=request.meeting_id,
            context=request.context,
            context_status=request.context_status,
            context_seq_base=request.context_seq_base,
        )
    except MeetingNotInitialized as e:
        raise HTTPException(status_code=404, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---- segments ---------------------------------------------------------------


@router.get(
    "/{collection}/meeting/{meeting_id}/segments",
    response_model=SegmentListResponse,
    summary="Liệt kê transcript theo khoảng sequence_id",
)
async def list_segments(
    collection: str,
    meeting_id: str,
    from_seq: int = Query(0, ge=0),
    to_seq: Optional[int] = Query(None, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    service: TranscriptService = Depends(get_transcript_service),
):
    return await service.list_segments(
        collection=collection,
        meeting_id=meeting_id,
        from_seq=from_seq,
        to_seq=to_seq,
        limit=limit,
    )
