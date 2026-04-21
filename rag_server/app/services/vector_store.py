from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
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
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="source",
                field_schema="keyword",
            )
            print(f"Created Qdrant collection: '{self._collection}'")

    def upsert_chunks(
        self,
        chunks: List[str],
        vectors: List[List[float]],
        metadata: Dict[str, Any],
    ) -> int:
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
        qdrant_filter = None
        if source_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="source",
                    match=MatchValue(value=source_filter),
                )]
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            score_threshold=0.0,
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
        info = self._client.get_collection(self._collection)
        return {
            "name": self._collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }