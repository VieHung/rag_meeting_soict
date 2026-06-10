"""TranscriptStore — Qdrant operations cho luồng transcript (Phase 2 + Phase 3).

Phase 3 thay đổi:
- _physical(): tách logical collection (meeting-{uuid}) khỏi physical Qdrant collection.
  Layout shared → tất cả cuộc họp vào 1 collection dùng chung; per_meeting là tương thích ngược.
- point_id: uuid5(meeting_id, sequence_id) thay uuid4 → deterministic, migration idempotent.
- ensure_collection: thêm on_disk + optimizers (ít segment lớn → ít file descriptor).
- get_max_sequence_id: fix bug (chỉ quét 100 điểm, không filter meeting_id) → dùng
  scroll(order_by desc, limit=1, filter=meeting_id); đúng cho collection dùng chung.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OptimizersConfigDiff,
    OrderBy,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings


class TranscriptStore:
    """Wrap Qdrant client cho transcript points.

    Singleton client per process. Instance mang logical collection name (meeting-{uuid})
    và tự resolve ra physical collection qua _physical().
    """

    _client: Optional[QdrantClient] = None
    _collections_ready: set = set()

    def __init__(self, collection_name: Optional[str] = None, meeting_id: Optional[str] = None):
        self._collection = collection_name or settings.transcript_default_collection
        # Derive meeting_id từ tên collection nếu không truyền vào.
        if meeting_id is None and self._collection.startswith(settings.transcript_collection_prefix):
            meeting_id = self._collection.removeprefix(settings.transcript_collection_prefix)
        self._meeting_id = meeting_id
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

    # ---- layout resolution -----------------------------------------------

    def _physical(self) -> str:
        """Logical collection → physical Qdrant collection name theo TRANSCRIPT_STORAGE_LAYOUT."""
        mid = self._meeting_id
        if mid is None:
            return self._collection
        layout = settings.transcript_storage_layout
        if layout == "shared":
            return settings.transcript_shared_collection
        if layout == "sharded":
            # hashlib.md5 để tránh PYTHONHASHSEED salt (hash() built-in đổi sau restart).
            h = int(hashlib.md5(mid.encode()).hexdigest(), 16)
            return f"meeting_bucket_{h % settings.transcript_num_shards}"
        # per_meeting — tương thích ngược tuyệt đối với Phase 1–2.
        return f"{settings.transcript_collection_prefix}{mid}"

    # ---- collection management -------------------------------------------

    def ensure_collection(self) -> None:
        """Tạo physical collection + payload index nếu chưa có."""
        phys = self._physical()
        if phys in TranscriptStore._collections_ready:
            return

        existing = {c.name for c in self.client.get_collections().collections}
        if phys not in existing:
            self.client.create_collection(
                collection_name=phys,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=settings.qdrant_on_disk,
                ),
                on_disk_payload=settings.qdrant_on_disk_payload,
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    max_segment_size=512_000,
                    memmap_threshold=20_000,
                ),
            )
        for field, schema in (
            ("meeting_id", "keyword"),
            ("sequence_id", "integer"),
            ("speaker", "keyword"),
            ("speaker_id", "keyword"),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=phys,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass
        TranscriptStore._collections_ready.add(phys)

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
        """Upsert 1 transcript point. Trả point_id (UUID v5, deterministic)."""
        self.ensure_collection()
        # uuid5 → deterministic: re-ingest cùng câu = ghi đè thay vì tạo mới.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meeting_id}:{sequence_id}"))
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
            collection_name=self._physical(),
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
                collection_name=self._physical(),
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
                collection_name=self._physical(),
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
            collection_name=self._physical(),
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
            collection_name=self._physical(),
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
        try:
            count_resp = self.client.count(
                collection_name=self._physical(),
                count_filter=flt,
                exact=True,
            )
            count = int(getattr(count_resp, "count", 0))
        except Exception:
            count = 0
        try:
            self.client.delete(
                collection_name=self._physical(),
                points_selector=flt,
                wait=True,
            )
        except Exception:
            return 0
        return count

    def collection_exists(self) -> bool:
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            return self._physical() in existing
        except Exception:
            return False

    def get_max_sequence_id(self) -> Optional[int]:
        """Lấy sequence_id lớn nhất của self._meeting_id (dùng để rebuild Redis counter).

        Dùng scroll(order_by desc, limit=1, filter=meeting_id) — O(log N), không quét toàn bộ.
        Yêu cầu payload index integer trên sequence_id (tạo trong ensure_collection) và
        Qdrant >= v1.8 (server đang là v1.10.0).
        """
        mid = self._meeting_id
        if mid is None:
            return None
        flt = Filter(
            must=[
                FieldCondition(key="meeting_id", match=MatchValue(value=mid)),
                FieldCondition(key="sequence_id", range=Range(gte=1)),
            ]
        )
        try:
            points, _ = self.client.scroll(
                collection_name=self._physical(),
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=1,
                order_by=OrderBy(key="sequence_id", direction="desc"),
            )
            if not points:
                return None
            seq = (points[0].payload or {}).get("sequence_id")
            return int(seq) if seq is not None else None
        except Exception:
            return None
