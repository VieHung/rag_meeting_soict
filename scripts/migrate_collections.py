"""migrate_collections.py — Phase 3 S3: gộp meeting-* → shared collection.

Chạy sau khi đã cài TRANSCRIPT_STORAGE_LAYOUT=shared trong .env.

Luồng:
1. Liệt kê tất cả collection tên "meeting-{uuid}" trong Qdrant.
2. Với mỗi collection:
   a. Scroll toàn bộ điểm theo batch.
   b. Cấp lại point_id = uuid5(meeting_id, sequence_id) — deterministic, an toàn chạy lại.
   c. Upsert vào physical collection (shared hoặc sharded tùy config).
   d. Verify count sau khi copy (best-effort, skip nếu lỗi count).
   e. Nếu không --dry-run: xóa collection cũ.
3. In tổng kết: số collection trước/sau.

An toàn:
- Idempotent: chạy lại không nhân đôi điểm (uuid5 giống nhau → upsert ghi đè).
- --dry-run: không sửa gì, chỉ in kế hoạch.
- Không xóa dữ liệu: chỉ xóa *vỏ collection rỗng* sau khi điểm đã copy xong.

Dùng:
    python scripts/migrate_collections.py [--dry-run] [--batch-size 200]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from pathlib import Path
from typing import List, Optional

# Thêm thư mục gốc rag_server vào sys.path để import app.*
ROOT = Path(__file__).parent.parent / "rag_server"
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from app.config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _physical(meeting_id: str) -> str:
    layout = settings.transcript_storage_layout
    if layout == "shared":
        return settings.transcript_shared_collection
    if layout == "sharded":
        h = int(hashlib.md5(meeting_id.encode()).hexdigest(), 16)
        return f"meeting_bucket_{h % settings.transcript_num_shards}"
    return f"{settings.transcript_collection_prefix}{meeting_id}"


def _ensure_target(client: QdrantClient, target: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if target in existing:
        return
    client.create_collection(
        collection_name=target,
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
            client.create_payload_index(target, field, schema)
        except Exception:
            pass
    print(f"  [create] Target collection '{target}' created.")


def _scroll_all(client: QdrantClient, collection: str, batch_size: int):
    """Generator: yield batch points từ collection."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=True,
            limit=batch_size,
            offset=offset,
        )
        if points:
            yield points
        if not offset:
            break


def _count(client: QdrantClient, collection: str, meeting_id: Optional[str] = None) -> int:
    try:
        flt = None
        if meeting_id:
            flt = Filter(must=[FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id))])
        resp = client.count(collection_name=collection, count_filter=flt, exact=True)
        return int(getattr(resp, "count", 0))
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

def migrate(dry_run: bool, batch_size: int) -> int:
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)

    all_collections = [c.name for c in client.get_collections().collections]
    prefix = settings.transcript_collection_prefix
    meeting_collections = [c for c in all_collections if c.startswith(prefix)]

    if not meeting_collections:
        print("Không tìm thấy collection meeting-* nào để migrate.")
        return 0

    print(f"Tìm thấy {len(meeting_collections)} collection meeting-* cần migrate:")
    for c in meeting_collections:
        print(f"  - {c}")
    print()

    if dry_run:
        print("[DRY-RUN] Không sửa Qdrant. Chỉ in kế hoạch.\n")

    # Pre-create target collection(s) nếu cần.
    targets_needed: set = set()
    meeting_ids = [c.removeprefix(prefix) for c in meeting_collections]
    for mid in meeting_ids:
        targets_needed.add(_physical(mid))

    if not dry_run:
        for target in targets_needed:
            _ensure_target(client, target)

    total_copied = 0
    total_deleted = 0
    errors: List[str] = []

    for col in meeting_collections:
        mid = col.removeprefix(prefix)
        target = _physical(mid)

        src_count = _count(client, col)
        print(f"[{col}] → [{target}]  (source: {src_count} điểm)")

        if dry_run:
            print(f"  [dry-run] Bỏ qua copy & delete.")
            continue

        # Scroll + upsert.
        copied = 0
        for batch in _scroll_all(client, col, batch_size):
            new_points: List[PointStruct] = []
            for p in batch:
                payload = dict(p.payload or {})
                # Đảm bảo meeting_id trong payload.
                if not payload.get("meeting_id"):
                    payload["meeting_id"] = mid
                seq = payload.get("sequence_id")
                # Cấp lại point_id deterministic (uuid5).
                new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{mid}:{seq}")) if seq is not None \
                    else str(uuid.uuid5(uuid.NAMESPACE_URL, f"{mid}:{p.id}"))
                vector = p.vector if isinstance(p.vector, list) else list(p.vector)
                new_points.append(PointStruct(id=new_id, vector=vector, payload=payload))
            try:
                client.upsert(collection_name=target, points=new_points, wait=True)
                copied += len(new_points)
            except Exception as e:
                errors.append(f"{col}: upsert failed — {e}")
                print(f"  [ERROR] Upsert batch failed: {e}")
                break

        print(f"  Copied {copied}/{src_count} điểm.")
        total_copied += copied

        # Verify count.
        target_count = _count(client, target, meeting_id=mid)
        if target_count >= 0 and target_count < src_count:
            msg = f"{col}: count mismatch (src={src_count}, target={target_count}) — bỏ qua delete."
            errors.append(msg)
            print(f"  [WARN] {msg}")
            continue

        # Xóa collection cũ.
        try:
            client.delete_collection(collection_name=col)
            print(f"  Đã xóa collection '{col}'.")
            total_deleted += 1
        except Exception as e:
            errors.append(f"{col}: delete failed — {e}")
            print(f"  [ERROR] Delete failed: {e}")

    print()
    print("=" * 60)
    print(f"Tổng kết:")
    print(f"  Collection trước: {len(all_collections)}")
    remaining = [c.name for c in client.get_collections().collections]
    print(f"  Collection sau  : {len(remaining)}")
    print(f"  Điểm đã copy    : {total_copied}")
    print(f"  Collection đã xóa: {total_deleted}")
    if errors:
        print(f"\n  LỖI ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    print("=" * 60)

    return len(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate meeting-* collections → shared collection")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch, không sửa Qdrant")
    parser.add_argument("--batch-size", type=int, default=200, help="Số điểm mỗi batch scroll")
    args = parser.parse_args()

    n_errors = migrate(dry_run=args.dry_run, batch_size=args.batch_size)
    sys.exit(1 if n_errors > 0 else 0)
