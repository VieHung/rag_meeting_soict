"""Pydantic schemas for transcript flow (Phase 2).

Tách khỏi schemas/embed.py & schemas/query.py để không động luồng tài liệu cũ.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ----------------------------- Meeting init ----------------------------------

class MeetingInitRequest(BaseModel):
    meeting_id: str = Field(..., min_length=1, description="ID cuộc họp duy nhất")
    force_reset: bool = Field(
        False,
        description="Nếu True, reset counter & metadata khi meeting đã tồn tại",
    )


class MeetingInitResponse(BaseModel):
    meeting_id: str
    collection: str
    status: Literal["initialized", "reset"]
    seq_counter: int = Field(0, description="Giá trị counter sau khi init")


# ----------------------------- Transcript embed ------------------------------

class TranscriptEmbedRequest(BaseModel):
    meeting_id: str = Field(..., min_length=1)
    speaker: str = Field(..., min_length=1, description="Tên người nói")
    speaker_id: Optional[str] = Field(None, description="ID định danh (optional)")
    text: str = Field(..., min_length=1, description="Nội dung câu transcript")
    timestamp: Optional[datetime] = Field(
        None,
        description="Thời điểm nói (ISO 8601). Nếu thiếu, server tự gán now()",
    )
    lang: Optional[str] = Field(None, description="Mã ngôn ngữ phát hiện (vi, en...)")


ContextStatus = Literal["pending", "processing", "ready", "failed"]


class TranscriptEmbedResponse(BaseModel):
    meeting_id: str
    sequence_id: int
    point_id: str
    context_status: ContextStatus = "pending"


# ----------------------------- Query transcript ------------------------------

class TranscriptQueryRequest(BaseModel):
    collection: Optional[str] = Field(
        None,
        description="Collection transcript. Nếu null, dùng `transcript_default_collection` trong config",
    )
    query: str = Field(..., min_length=1)
    meeting_id: Optional[str] = Field(None, description="Lọc theo cuộc họp")
    top_k: int = Field(3, ge=1, le=50)
    window_size: int = Field(
        2, ge=0, description="±N câu lân cận (sẽ clamp theo TRANSCRIPT_MAX_WINDOW_SIZE)"
    )
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)
    speaker_filter: Optional[str] = None
    speaker_id_filter: Optional[str] = None
    include_context: bool = True


class WindowEntry(BaseModel):
    sequence_id: int
    speaker: str
    speaker_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    text: str


class WindowResult(BaseModel):
    before: List[WindowEntry] = Field(default_factory=list)
    after: List[WindowEntry] = Field(default_factory=list)


class TranscriptQueryResult(BaseModel):
    sequence_id: int
    speaker: str
    speaker_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    text: str
    score: float
    meeting_id: str
    context: Optional[str] = None
    context_status: Optional[ContextStatus] = None
    window: Optional[WindowResult] = None


class TranscriptQueryResponse(BaseModel):
    query: str
    results: List[TranscriptQueryResult]
    count: int


# ----------------------------- Context endpoints -----------------------------

class ContextResponse(BaseModel):
    meeting_id: str
    sequence_id: int
    context: str = ""
    context_status: ContextStatus
    context_seq_base: Optional[int] = None


class ContextUpdateRequest(BaseModel):
    meeting_id: str = Field(..., min_length=1)
    context: str = Field(..., description="Context đã tóm tắt")
    context_status: ContextStatus = "ready"
    context_seq_base: Optional[int] = None


# ----------------------------- Segments / list -------------------------------

class SegmentEntry(BaseModel):
    sequence_id: int
    speaker: str
    speaker_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    text: str
    context_status: Optional[ContextStatus] = None


class SegmentListResponse(BaseModel):
    meeting_id: str
    collection: str
    from_seq: int
    to_seq: int
    count: int
    segments: List[SegmentEntry]


# ----------------------------- Delete ----------------------------------------

class DeleteMeetingResponse(BaseModel):
    success: bool
    meeting_id: str
    deleted_points: int = 0
    message: str = ""
