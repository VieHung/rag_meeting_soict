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