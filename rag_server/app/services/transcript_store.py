"""TranscriptStore — Qdrant operations cho luồng transcript (Phase 2).

Tách khỏi `QdrantService` (Phase 1) để giữ API cũ nguyên vẹn:
- Phase 1 collection (documents): payload indexes "source"
- Phase 2 collection (transcripts): payload indexes "meeting_id" (keyword),
  "sequence_id" (integer), "speaker" (keyword)

Vector size & distance dùng chung với Phase 1 (D6: cùng embedding model).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings


class TranscriptStore:
    """Wrap Qdrant client cho transcript points.

    Singleton client per process; instance thì gắn với 1 collection cụ thể.
    """

    _client: Optional[QdrantClient] = None
    _collections_ready: set = set()

    def __init__(self, collection_name: Optional[str] = None):
        self._collection = collection_name or settings.transcript_default_collection
        self._init_client()

    @classmethod
    def _init_client(cls) -> QdrantClient:
        if cls._client is None:
            cls._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30,
            )
        return cls._client

    @property
    def client(self) -> QdrantClient:
        return self._init_client()

    @property
    def collection(self) -> str:
        return self._collection

    # ---- collection management -------------------------------------------

    def ensure_collection(self) -> None:
        """Tạo collection + payload index nếu chưa có."""
        if self._collection in TranscriptStore._collections_ready:
            return

        existing = {c.name for c in self.client.get_collections().collections}
        if self._collection not in existing:
            self.client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        # Best-effort tạo index cho field thường dùng để filter / scroll.
        for field, schema in (
            ("meeting_id", "keyword"),
            ("sequence_id", "integer"),
            ("speaker", "keyword"),
            ("speaker_id", "keyword"),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                # Index có thể đã tồn tại → bỏ qua.
                pass
        TranscriptStore._collections_ready.add(self._collection)

    # ---- ingest -----------------------------------------------------------

    def upsert_point(
        self,
        *,
        vector: List[float],
        meeting_id: str,
        sequence_id: int,
        speaker: str,
        text: str,
        timestamp: Optional[datetime] = None,
        speaker_id: Optional[str] = None,
        lang: Optional[str] = None,
        context: str = "",
        context_status: str = "pending",
        context_seq_base: Optional[int] = None,
    ) -> str:
        """Upsert 1 transcript point. Trả point_id (UUID v4)."""
        self.ensure_collection()
        point_id = str(uuid.uuid4())
        ts = timestamp or datetime.now(timezone.utc)
        payload = {
            "meeting_id": meeting_id,
            "sequence_id": sequence_id,
            "speaker": speaker,
            "speaker_id": speaker_id,
            "text": text,
            "timestamp": ts.isoformat(),
            "lang": lang,
            "context": context,
            "context_status": context_status,
            "context_seq_base": context_seq_base,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    # ---- read -------------------------------------------------------------

    def find_by_seq(
        self, meeting_id: str, sequence_id: int
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Tìm 1 point theo (meeting_id, sequence_id). Trả (point_id, payload)."""
        flt = Filter(
            must=[
                FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id)),
                FieldCondition(key="sequence_id", match=MatchValue(value=sequence_id)),
            ]
        )
        try:
            points, _ = self.client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=1,
            )
        except Exception:
            return None
        if not points:
            return None
        return str(points[0].id), dict(points[0].payload or {})

    def scroll_window(
        self,
        meeting_id: str,
        seq_min: int,
        seq_max: int,
        limit: int = 256,
    ) -> List[Dict[str, Any]]:
        """Lấy các điểm có sequence_id ∈ [seq_min, seq_max] cùng meeting_id."""
        if seq_min > seq_max:
            return []
        flt = Filter(
            must=[
                FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id)),
                FieldCondition(
                    key="sequence_id",
                    range=Range(gte=seq_min, lte=seq_max),
                ),
            ]
        )
        results: List[Dict[str, Any]] = []
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=limit,
                offset=next_offset,
            )
            for p in points:
                results.append({"id": str(p.id), **(p.payload or {})})
            if not next_offset:
                break
        results.sort(key=lambda r: r.get("sequence_id", 0))
        return results

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        meeting_id: Optional[str] = None,
        speaker_filter: Optional[str] = None,
        speaker_id_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        self.ensure_collection()
        must: List[FieldCondition] = []
        if meeting_id:
            must.append(FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id)))
        if speaker_filter:
            must.append(FieldCondition(key="speaker", match=MatchValue(value=speaker_filter)))
        if speaker_id_filter:
            must.append(FieldCondition(key="speaker_id", match=MatchValue(value=speaker_id_filter)))
        flt = Filter(must=must) if must else None

        results = self.client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
            score_threshold=max(0.0, float(score_threshold)),
        )
        return [
            {
                "id": str(r.id),
                "score": round(float(r.score), 4),
                **(r.payload or {}),
            }
            for r in results
        ]

    # ---- update -----------------------------------------------------------

    def set_payload(self, point_id: str, payload_partial: Dict[str, Any]) -> None:
        """Cập nhật một số field payload (giữ nguyên các field khác)."""
        self.client.set_payload(
            collection_name=self._collection,
            payload=payload_partial,
            points=[point_id],
            wait=True,
        )

    def update_context(
        self,
        point_id: str,
        *,
        context: str,
        context_status: str,
        context_seq_base: Optional[int] = None,
    ) -> None:
        self.set_payload(
            point_id,
            {
                "context": context,
                "context_status": context_status,
                "context_seq_base": context_seq_base,
            },
        )

    # ---- delete -----------------------------------------------------------

    def delete_meeting(self, meeting_id: str) -> int:
        flt = Filter(
            must=[FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id))]
        )
        # Đếm trước khi xóa (best-effort).
        try:
            count_resp = self.client.count(
                collection_name=self._collection,
                count_filter=flt,
                exact=True,
            )
            count = int(getattr(count_resp, "count", 0))
        except Exception:
            count = 0
        try:
            self.client.delete(
                collection_name=self._collection,
                points_selector=flt,
                wait=True,
            )
        except Exception:
            return 0
        return count

    def collection_exists(self) -> bool:
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            return self._collection in existing
        except Exception:
            return False

    def get_max_sequence_id(self) -> Optional[int]:
        """Lấy sequence_id lớn nhất trong collection (dùng để rebuild counter)."""
        try:
            flt = Filter(
                must=[
                    FieldCondition(
                        key="sequence_id",
                        range=Range(gte=1),
                    ),
                ]
            )
            points, _ = self.client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=100,
            )
            if not points:
                return None
            max_seq = 0
            for p in points:
                seq = p.payload.get("sequence_id") if p.payload else None
                if seq is not None:
                    seq_int = int(seq)
                    if seq_int > max_seq:
                        max_seq = seq_int
            return max_seq if max_seq > 0 else None
        except Exception:
            return None
