from pydantic import BaseModel, Field
from typing import Optional


class EmbedTextRequest(BaseModel):
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