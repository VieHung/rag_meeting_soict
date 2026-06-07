#!/usr/bin/env python3
"""reembed.py — Re-embed dữ liệu Qdrant sang embedding model/dimension mới.

Đổi model embedding (vd MiniLM 384 → multilingual-e5-base 768) ⇒ vector cũ không
còn tương thích. Script này scroll toàn bộ point của mỗi collection, re-embed lại
`payload["text"]` bằng EmbeddingService HIỆN TẠI (đọc theo .env mới), GIỮ NGUYÊN
payload + point_id, rồi ghi sang collection mới.

Quan trọng:
- Giữ point_id và toàn bộ payload (transcript phụ thuộc sequence_id/context/...).
- Câu lưu trữ là PASSAGE → dùng EmbeddingService.embed_texts (prefix "passage:").
- Counter Redis (rag:seq:{collection}) không cần đụng — payload sequence_id được giữ.

Chạy (từ thư mục rag_server, sau khi đã cập nhật .env sang model mới):
    python -m scripts.reembed --collections all
    python -m scripts.reembed --collections documents,meeting-<uuid>
    python -m scripts.reembed --collections all --in-place   # khi dimension KHÔNG đổi (e5-small 384)

Mặc định ghi sang collection mới tên `<name>__v2` để dễ rollback. Sau khi verify,
tự đổi tên thủ công (xoá cũ, tạo lại tên gốc, hoặc trỏ app sang __v2).
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.services.embedding import EmbeddingService

MEETING_PREFIX = settings.transcript_collection_prefix  # "meeting-"

# Payload index theo loại collection (giống vector_store.py / transcript_store.py).
TRANSCRIPT_INDEXES = {
    "meeting_id": "keyword",
    "sequence_id": "integer",
    "speaker": "keyword",
    "speaker_id": "keyword",
}
DOCUMENT_INDEXES = {"source": "keyword"}


def _client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)


def _indexes_for(name: str) -> Dict[str, str]:
    return TRANSCRIPT_INDEXES if name.startswith(MEETING_PREFIX) else DOCUMENT_INDEXES


def _ensure_target(client: QdrantClient, name: str, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(name)
        size = info.config.params.vectors.size
        if size != dim:
            raise SystemExit(
                f"Collection đích '{name}' đã tồn tại với size={size} (cần {dim}). "
                "Xoá nó trước hoặc đổi tên đích."
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    for field, schema in _indexes_for(name).items():
        try:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
        except Exception:
            pass


def reembed_collection(
    client: QdrantClient,
    embedder: EmbeddingService,
    source: str,
    target: str,
    batch: int,
) -> int:
    """Re-embed toàn bộ point của `source` → `target`. Trả số point đã ghi."""
    _ensure_target(client, target, embedder.dim)

    migrated = 0
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=source,
            with_payload=True,
            with_vectors=False,
            limit=batch,
            offset=next_offset,
        )
        if not points:
            break

        texts: List[str] = []
        keep: List = []
        for p in points:
            payload = dict(p.payload or {})
            text = payload.get("text")
            if not text or not str(text).strip():
                # Point không có text để re-embed → bỏ qua (log).
                print(f"  [skip] point {p.id} không có 'text'")
                continue
            texts.append(str(text))
            keep.append((p.id, payload))

        if texts:
            vectors = embedder.embed_texts(texts)  # passage prefix
            new_points = [
                PointStruct(id=pid, vector=vec, payload=payload)
                for (pid, payload), vec in zip(keep, vectors)
            ]
            client.upsert(collection_name=target, points=new_points, wait=True)
            migrated += len(new_points)
            print(f"  ...migrated {migrated} points")

        if not next_offset:
            break

    return migrated


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Re-embed Qdrant collections sang model mới")
    parser.add_argument(
        "--collections",
        required=True,
        help="Danh sách collection ngăn cách dấu phẩy, hoặc 'all'",
    )
    parser.add_argument("--suffix", default="__v2", help="Hậu tố collection đích (mặc định __v2)")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Ghi đè ngay collection nguồn (chỉ khi dimension KHÔNG đổi)",
    )
    parser.add_argument("--batch", type=int, default=128, help="Batch scroll/embed")
    args = parser.parse_args(argv)

    client = _client()
    embedder = EmbeddingService()

    all_names = [c.name for c in client.get_collections().collections]
    if args.collections == "all":
        names = [n for n in all_names if not n.endswith(args.suffix)]
    else:
        names = [n.strip() for n in args.collections.split(",") if n.strip()]

    if not names:
        print("Không có collection nào để migrate.")
        return 0

    print(f"Model: {settings.embedding_model} (dim={embedder.dim}) | backend={settings.embedding_backend}")
    print(f"Collections: {names} | in_place={args.in_place}\n")

    for name in names:
        if name not in all_names:
            print(f"[!] Bỏ qua '{name}' — không tồn tại")
            continue
        target = name if args.in_place else f"{name}{args.suffix}"
        print(f"==> {name} -> {target}")
        count = reembed_collection(client, embedder, name, target, args.batch)

        src_count = client.get_collection(name).points_count
        dst_count = client.get_collection(target).points_count
        ok = "OK" if dst_count >= count else "MISMATCH"
        print(f"    [{ok}] source={src_count} target={dst_count} migrated={count}\n")

    print("Hoàn tất. Nếu dùng __v2: verify rồi đổi tên/đảo collection thủ công.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
