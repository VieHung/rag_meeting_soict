from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from typing import List, Dict, Any, Optional
from app.config import settings
import uuid


class QdrantService:
    _client = None
    _collections_created = set()

    def __init__(self, collection_name: str = None):
        if QdrantService._client is None:
            QdrantService._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30,
            )
        self._collection = collection_name or settings.qdrant_collection_name
        if self._collection not in QdrantService._collections_created:
            self._ensure_collection()

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
            QdrantService._collections_created.add(self._collection)
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

    def delete_by_doc_id(self, doc_id: str) -> int:
        result = self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )]
            ),
        )
        return result.operation_id

    def list_documents(self) -> List[Dict[str, Any]]:
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=None,
            with_payload=True,
            limit=1000,
        )
        docs = {}
        for point in points:
            doc_id = point.payload.get("doc_id", "")
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source": point.payload.get("source", ""),
                    "chunks": point.payload.get("chunk_total", 0),
                    "chunk_index": point.payload.get("chunk_index", 0),
                }
        return list(docs.values())

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

    @classmethod
    def list_collections(cls) -> List[str]:
        if cls._client is None:
            cls._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30,
            )
        return [c.name for c in cls._client.get_collections().collections]

    @classmethod
    def create_collection(cls, name: str) -> dict:
        try:
            if cls._client is None:
                cls._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    timeout=30,
                )
            existing = [c.name for c in cls._client.get_collections().collections]
            if name in existing:
                return {"success": False, "message": f"Collection '{name}' already exists"}
            cls._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            cls._client.create_payload_index(
                collection_name=name,
                field_name="source",
                field_schema="keyword",
            )
            cls._collections_created.add(name)
            return {"success": True, "message": f"Created collection '{name}'"}
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    @classmethod
    def delete_collection(cls, name: str) -> dict:
        try:
            if cls._client is None:
                cls._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    timeout=30,
                )
            existing = [c.name for c in cls._client.get_collections().collections]
            if name not in existing:
                return {"success": False, "message": f"Collection '{name}' not found"}
            cls._client.delete_collection(collection_name=name)
            cls._collections_created.discard(name)
            return {"success": True, "message": f"Deleted collection '{name}'"}
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}