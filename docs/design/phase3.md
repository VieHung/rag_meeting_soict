# Kế hoạch Giai đoạn 3 (bản rebuild) — Dứt điểm "too many open files", Độ bền worker, Tokenizer VN

> **Tài liệu cho Coding Agent.** Bản này **thay thế** Phase 3 trước đó. Thay đổi lớn nhất:
> nhóm **S (storage)** được viết lại thành một **gói 4 bước bắt buộc làm trọn** (không còn là
> "đổi cách lưu trữ" đơn lẻ), dựa trên xác minh thực tế cơ chế file-descriptor của Qdrant.
> Mọi tính năng khác vẫn **OPTIONAL, gated qua `.env`, mặc định TẮT** — tắt cờ ⇒ hành vi y hệt
> hệ thống hiện tại. Đọc mục 1 (chẩn đoán) + mục 2 (nguyên tắc) trước khi code.
>
> **Cập nhật 2026-06-10 — đã đối chiếu code:** các kiểm tra code-side của S0 đã có kết quả:
> `QdrantClient` **đã singleton** (class-level ở cả `QdrantService` & `TranscriptStore`);
> `point_id` hiện là **`uuid4`** (an toàn với gộp, nhưng không idempotent — xem Bẫy 4 đã viết
> lại); Qdrant server **v1.10.0** + client **1.10.1** (hỗ trợ `scroll order_by`, bỏ fallback);
> và phát hiện **bug có sẵn** trong `get_max_sequence_id()` (quét 1 trang 100 điểm, không
> phân trang, không filter `meeting_id` — xem S2d). S0 còn lại chỉ là **đo fd trên host**.

---

## 1. Bối cảnh & chẩn đoán cốt lõi

**BKMEETING** (SoICT / NAVIS Center, ĐHBK Hà Nội) — kiến trúc **2 tầng**:
- **Thiết bị (QCS8550)**: live transcript + LLM nhỏ **sinh câu trả lời cho người dùng**. *Ngoài phạm vi server.*
- **Server (repo này)**: **nguồn tri thức** — RAG API + Qdrant + Redis + LLM self-host build context.

**Đã có** (Phase 1 + 2 + đợt hoàn thiện): 2 luồng tài liệu/transcript; `context_worker.py` (FIFO, D9);
hybrid retrieval + reranker (gated, off); `/health`, `/embed/info`, logging; benchmark RAGAS.

### 1.1. Vấn đề #1 (blocker): Qdrant "too many open files"

Thiết kế hiện tại dùng **1 collection vật lý / 1 cuộc họp** (`meeting-{uuid}`). Đây là hệ quả vô tình
của D8 ("`meeting_id` suy từ tên collection") → **logical = physical**. RAGFlow đi hướng ngược lại
ngay từ đầu: **1 index / 1 tenant** + filter `kb_id` (logical ≠ physical). Hệ quả của ta:

```
fd ≈ N_meetings × segments_per_collection × files_per_segment   → tăng TUYẾN TÍNH theo số cuộc họp
```

Mỗi collection (kể cả gần rỗng) luôn có ≥ vài segment, mỗi segment giữ nhiều file (RocksDB `.sst`,
mmap) **ở trạng thái MỞ**. Qdrant nạp toàn bộ collection lúc khởi động và **không tự đóng fd của
collection nhàn rỗi**. Tích lũy đủ cuộc họp → vượt `ulimit -n` → Qdrant từ chối mở segment.

> Hướng giải đã được **Qdrant maintainer xác nhận**: Qdrant *không thể* giới hạn số file mở; cách
> đúng là **dồn về một collection** — càng nhiều collection càng tốn fd, và họ khuyến nghị **không**
> tạo nhiều collection nhỏ.

### 1.2. Phân biệt sống còn: **file-descriptor (fd) ≠ RAM** (trực giao)

Đây là điểm dễ hiểu nhầm và làm lệch cả hướng giải:

| | **fd (số file đang MỞ)** | **RAM (cái gì trong bộ nhớ)** |
|---|---|---|
| Bản chất lỗi "too many open files" | ✅ chính là đây | ❌ không liên quan |
| Collection nhàn rỗi (không ai query 1 tháng) | **vẫn giữ file mở** → vẫn tốn fd | vector vẫn chiếm RAM (config mặc định) |
| "Đẩy dữ liệu cũ xuống disk" giải quyết? | ❌ KHÔNG — file vẫn mở | ✅ CÓ — đỡ RAM |
| "Hợp nhất collection" giải quyết? | ✅ CÓ — số segment có trần nhờ merge optimizer | một phần |
| `on_disk=True` (memmap) giải quyết? | một phần | ✅ CÓ — OS page cache lo hot/cold tự động |

**Hệ quả thiết kế:**
- **Hợp nhất collection** chữa **fd** (số segment do merge optimizer quản, có trần theo dung lượng, không theo số cuộc).
- **`on_disk`/memmap** chữa **RAM** — và cho **đúng hiệu ứng "tier nóng/nguội"** mà không cần xây
  pipeline lifecycle nào: vector cuộc cũ không nạp RAM tới khi bị query; kernel page-cache quyết định
  nóng/nguội ở mức trang, mịn và tự điều chỉnh hơn mọi ngưỡng "X ngày không dùng".
- ⇒ **Không phải chọn giữa "xóa" và "đẩy xuống disk".** Migration **không xóa nội dung** cuộc họp
  (chỉ xóa *vỏ collection rỗng* sau khi copy điểm sang collection chung — dữ liệu vẫn query được).
  `on_disk` lo phần tiering. Cold-tier rời Qdrant là **YAGNI** (xem §4.5).

### 1.3. Công thức dứt điểm (làm CÙNG LÚC, không phải "chọn một")

```
HỢP NHẤT collection  +  ulimit nofile = 65535  +  on_disk vector/payload  +  ít segment lớn
```
Chỉ làm một trong số đó **chưa chắc khỏi** — vì một collection lớn vẫn có thể đụng lỗi (xem Bẫy 2, §4.4).

---

## 2. Nguyên tắc bất biến (giữ từ Phase 2)

1. **Không đổi đường dẫn endpoint**; không đổi hình dạng response (chỉ `score` phản ánh điểm cuối).
   Tính năng mới = cờ bật/tắt trong endpoint cũ, hoặc endpoint **mới tách biệt** (nếu thật sự cần).
2. **Gated + default TẮT** — tắt cờ ⇒ hành vi & dependency y hệt hiện tại.
3. **Tôn trọng 2 tầng** — server là *nguồn tri thức*; điều phối agent ưu tiên đặt ở **thiết bị**.
4. **Đo trước khi mở rộng** — tính năng nặng chỉ bật khi có dữ liệu chứng minh.
5. **Tương thích ngược** — có chế độ `per_meeting` để khôi phục hành vi cũ tuyệt đối.

---

## 3. Thứ tự ưu tiên (rebuild)

| Thứ tự | Nhóm | Việc | Vì sao ở đây |
|---|---|---|---|
| **0** | **S0** | **Chẩn đoán host** (lsof / `/proc/1/fd`, đếm `docs-*`) — kiểm code đã xong ✅ | Tránh vá nhầm tầng — 15 phút, rẻ |
| **1** | **S1** | **Vá nền tảng** (ulimit 65535 + on_disk + ít segment lớn) | Chặn cháy NGAY, mua thời gian; **bắt buộc, không phải mitigation** |
| **2** | **S2** | **Hợp nhất (code)** `_physical()` + filter + point_id duy nhất | Lõi giải pháp fd |
| **3** | **S3** | **Migration + XÓA collection cũ** (bắt buộc) | Không xóa cũ = chưa giải quyết gì |
| **4** | **B1** | Worker per-collection **+ recovery scan** (độ bền) | Song song hóa + vá lỗ hổng mất job |
| **5** | **A1** | Tokenizer tiếng Việt | "Free win", rủi ro thấp |
| 6 | **E′** | Rotate `LLM_API_KEY` (tách khỏi E, làm sớm) | An toàn cơ bản; gắn với S1 |
| sau | A2, C, D, E còn lại | Sparse / PageIndex / Agentic / auth-metrics | Hoãn — đo trước, gated |

---

## 4. Nhóm S — Dứt điểm "too many open files" (🔴🔴 BLOCKER, làm trọn gói)

### S0 — Chẩn đoán trước (≈ 15 phút, không code)

Các kiểm tra **code-side đã xong** (đối chiếu source 2026-06-10):
- ✅ `QdrantClient` **là singleton** class-level ở cả `QdrantService._client` lẫn
  `TranscriptStore._client` — Bẫy 3 loại trừ, hợp nhất collection sẽ có tác dụng.
- ✅ `point_id` = `uuid.uuid4()` (`transcript_store.py`, `vector_store.py`) — toàn cục duy nhất,
  **không** có nguy cơ ghi đè khi gộp; vấn đề còn lại là **idempotency** (Bẫy 4 đã viết lại).
- ✅ Qdrant server **v1.10.0** (`docker-compose.yml`), client **1.10.1** → `scroll(order_by)` OK.
- ⚠️ Phát hiện bug có sẵn: `get_max_sequence_id()` quét **đúng 1 trang 100 điểm**, không
  phân trang, không filter `meeting_id` → rebuild counter sai khi meeting > 100 câu (S2d).

Còn lại duy nhất việc **đo fd trên host triển khai** (xác nhận fd tập trung ở qdrant + đếm số
collection `docs-*` thực tế cho Bẫy 5):
```bash
# fd mỗi process đang mở + giới hạn hiện tại, trong từng container
docker exec <qdrant>  sh -c 'ls /proc/1/fd | wc -l; cat /proc/1/limits | grep "open files"'
docker exec <rag_api> sh -c 'ls /proc/1/fd | wc -l; cat /proc/1/limits | grep "open files"'
# đếm collection theo loại
curl -s localhost:6333/collections | python -c "import sys,json;ns=[c['name'] for c in json.load(sys.stdin)['result']['collections']];print('meeting-*:',sum(n.startswith('meeting-') for n in ns),'| khác:',[n for n in ns if not n.startswith('meeting-')])"
# nếu nghi rò ở host/docker-proxy:
sudo lsof | awk '{print $1}' | sort | uniq -c | sort -rn | head
```

### S1 — Vá nền tảng (rẻ, làm NGAY, song song; **bắt buộc**)

`docker-compose.yml` — nâng giới hạn fd cho **cả hai** service (Qdrant khuyến nghị 65535):
```yaml
  qdrant:
    ulimits:
      nofile: { soft: 65535, hard: 65535 }
  rag_api:
    ulimits:
      nofile: { soft: 65535, hard: 65535 }
```
Tạo collection với **vector on_disk + payload on_disk + optimizer gom ít segment lớn** (xem S2 code):
- `on_disk=True` (vector memmap) → không thường trú RAM, OS lo tiering.
- `on_disk_payload=true` → payload (context summary) xuống disk.
- `optimizers_config`: tăng `max_segment_size_kb`, đặt `default_segment_number` thấp, `memmap_threshold`
  → Qdrant gom thành **ít segment lớn** thay vì nhiều segment nhỏ ⇒ **ít file hơn**.

> S1 thường đã đủ chặn cháy để có thời gian làm S2–S3. Nhưng **không thay thế** hợp nhất —
> một collection lớn vẫn có thể đụng lỗi nếu segment quá nhiều (Bẫy 2).

### S2 — Hợp nhất (code): tách logical ↔ physical

Giữ **hợp đồng API y nguyên** (`/transcript/meeting-{uuid}/...`, `meeting_id` suy từ tên). Chỉ đổi
lớp lưu trữ bên trong `TranscriptStore`.

**(a) Lớp resolve physical** theo `TRANSCRIPT_STORAGE_LAYOUT`:
```python
import hashlib
def _physical(self, meeting_id: str) -> str:
    layout = settings.transcript_storage_layout
    if layout == "shared":
        return settings.transcript_shared_collection           # vd "meeting_transcripts"
    if layout == "sharded":
        # KHÔNG dùng hash() built-in (bị salt theo PYTHONHASHSEED → đổi sau restart!)
        h = int(hashlib.md5(meeting_id.encode()).hexdigest(), 16)
        return f"meeting_bucket_{h % settings.transcript_num_shards}"
    return f"meeting-{meeting_id}"                              # per_meeting (tương thích ngược)
```

**(b) point_id DETERMINISTIC** (Bẫy 4 — idempotency, không phải chống ghi đè):
```python
import uuid
point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meeting_id}:{sequence_id}"))
```
Hiện trạng: code **đã dùng `uuid4`** — toàn cục duy nhất nên gộp collection **không** ghi đè.
Nhưng `uuid4` không deterministic ⇒ (i) migration chạy lại sẽ **nhân đôi điểm** thay vì ghi đè
chính nó; (ii) client gửi lại cùng utterance tạo điểm trùng. `uuid5(meeting_id, sequence_id)`
cho cả hai đường idempotent. Vẫn giữ cảnh báo: **không bao giờ** dùng `sequence_id` trần làm
point_id (id=1 của meeting B sẽ đè id=1 của meeting A khi chung collection).

**(c) `ensure_collection` (idempotent) với on_disk + optimizers** — collection dùng chung tạo 1 lần:
```python
from qdrant_client.models import VectorParams, Distance, OptimizersConfigDiff
client.create_collection(
    collection_name=phys,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE, on_disk=True),
    on_disk_payload=True,
    optimizers_config=OptimizersConfigDiff(
        default_segment_number=2,        # ít segment
        max_segment_size_kb=512_000,     # cho phép segment lớn → ít file
        memmap_threshold_kb=20_000,
    ),
)
for field, schema in (("meeting_id","keyword"),("sequence_id","integer"),("speaker","keyword")):
    client.create_payload_index(phys, field, schema)   # best-effort, bắt buộc cho filter + order_by
```

**(d) Mọi thao tác filter theo `meeting_id`** — `search/scroll_window/find_by_seq` **đã** filter sẵn.
Sửa thêm:
- `upsert_point`, `delete_meeting`, mọi nơi → dùng `self._physical(meeting_id)` thay tên logic.
- **`get_max_sequence_id(meeting_id)` — đây là BUG CÓ SẴN, sửa bất kể layout**: bản hiện tại
  scroll **một trang `limit=100`, không phân trang, không filter `meeting_id`** rồi lấy max.
  Hậu quả ngay với `per_meeting` hôm nay: meeting > 100 câu + Redis key hết TTL (7 ngày) ⇒
  rebuild counter **thấp hơn thực tế** ⇒ cấp **trùng `sequence_id`**. Với `shared` còn sai thêm
  vì quét cả meeting khác. Fix: Qdrant **không có `MAX()`** → dùng
  `scroll(order_by=sequence_id desc, limit=1, filter=meeting_id)` — server đang chạy **v1.10.0**,
  client **1.10.1**, `order_by` hỗ trợ từ ~v1.8 nên **không cần fallback** (chỉ cần payload index
  integer trên `sequence_id`, đã có sẵn trong `ensure_collection`).

**(e) SequenceManager**: key Redis vẫn `rag:seq:{meeting_id}` (uuid duy nhất — không đụng nhau dù chung
collection). Rebuild gọi `get_max_sequence_id(meeting_id)` đã filter. Nhân tiện **xóa dead code
`_ensure_collection_exists`** — chưa nơi nào gọi, và nếu gọi sẽ crash (`QdrantService` không có
method instance `collection_exists` / `create_collection()` không tham số).

**(f) Phía docs (`QdrantService`) — hệ quả Bẫy 5**: số collection docs là **client-driven**
(form param `collection` trên `/embed/file|text` + `POST /embed/collections` tạo tùy ý, auto-create
khi dùng lần đầu). Không đổi API, nhưng **áp cùng cấu hình `on_disk` + optimizers** vào cả hai
đường tạo collection (`_ensure_collection` và classmethod `create_collection`) để collection docs
mới sinh ra không tái tạo áp lực fd. Việc *gộp* docs-\* (nếu host đang có nhiều) để sau khi đo S0
quyết — khác transcript, docs không có hợp đồng `meeting-` prefix nên gộp là việc riêng, không
chặn S3.

### S3 — Migration + XÓA collection cũ (**bắt buộc**, không phải bước phụ)

> ⚠️ Đổi config sang `shared` **KHÔNG tự xóa** các `meeting-*` cũ. Nếu chỉ đổi config: ghi mới vào
> `meeting_transcripts` **CỘNG** tất cả collection cũ vẫn còn → **nhiều file hơn trước**, lỗi không
> biến mất. Migration kèm `delete_collection` là **điều kiện sống còn**.

`scripts/migrate_collections.py` (idempotent, có `--dry-run`):
```
cho mỗi collection tên "meeting-*":
    meeting_id = tên.removeprefix("meeting-")
    scroll theo batch → re-upsert mọi điểm vào _physical(meeting_id)
         · point_id = uuid5(meeting_id, sequence_id)   (cấp lại, toàn cục duy nhất)
         · payload giữ nguyên (đã có meeting_id)
    sau khi copy xong & verify count khớp → client.delete_collection("meeting-"+meeting_id)
in ra: số collection TRƯỚC vs SAU (phải giảm về 1 với shared)
```
**Migration KHÔNG xóa dữ liệu** — chỉ xóa vỏ collection rỗng sau khi điểm đã sang collection chung.

### 4.4. Bảy cái bẫy (đưa thẳng vào DoD, mỗi bẫy 1 kiểm tra)

| # | Bẫy | Trạng thái / Kiểm tra bắt buộc |
|---|---|---|
| 1 | Đổi `shared` không xóa collection cũ → lỗi vẫn còn | DoD: `len(client.get_collections())` **thực sự giảm về 1** sau migration (không chỉ "tạo 100 meeting không tăng") |
| 2 | Một collection lớn vẫn có thể đụng lỗi (RocksDB `.sst`/segment) | Làm **đồng thời** ulimit 65535 + on_disk + ít segment lớn — không coi là "optional mitigation" |
| 3 | Chẩn nhầm tầng (fd rò ở rag_api/docker-proxy, không phải qdrant) | ✅ **Loại trừ phía code** — `QdrantClient` đã singleton (đối chiếu source). Còn lại: đo `lsof`/`/proc/1/fd` trên host để xác nhận fd tập trung ở qdrant |
| 4 | `point_id` không deterministic → migration chạy lại **nhân đôi điểm**; client gửi lại tạo trùng | ✅ Đã xác minh code dùng `uuid4` (an toàn ghi đè). Chuyển sang `uuid5(meeting_id, sequence_id)` để migration + re-ingest idempotent; **không bao giờ** dùng `sequence_id` trần làm id |
| 5 | Bỏ quên `docs-*`: client tự đặt `collection` + `POST /embed/collections` → fd tái phát từ phía tài liệu | ✅ Đã xác minh: số collection docs là **client-driven**. S2(f): áp on_disk + optimizers vào đường tạo của `QdrantService`; đếm số `docs-*` thực tế lúc S0, > 1 nhiều → cân nhắc gộp riêng |
| 6 | `sharded`: `hash()` built-in bị salt; đổi `NUM_SHARDS` phải reshard | Dùng `hashlib.md5`; mặc định `shared` (sharded là YAGNI tới khi đo được collection phình) |
| 7 | **Bug có sẵn** `get_max_sequence_id`: 1 trang 100 điểm, không filter → counter rebuild sai, **trùng `sequence_id`** | Sửa trong S2(d) bằng `scroll(order_by desc, limit=1, filter=meeting_id)`; test: meeting > 100 câu, xóa Redis key, embed tiếp → seq không trùng |

### 4.5. Vì sao KHÔNG xây cold-tier (YAGNI) + lối thoát nếu cần

Phép tính khăn giấy: 384-d × 4 byte ≈ 1.5 KB/vector; ~1.000 utterance/cuộc ≈ 1.5 MB vector thô,
kể cả HNSW + payload ≈ **~10 MB/cuộc**. **10.000 cuộc ≈ vài chục GB** — vặt vãnh với SSD, và với
`on_disk` thì RAM gần như không bị ảnh hưởng. ⇒ **gần như không bao giờ chạm ngưỡng cần archival.**
Xây hot/warm/cold lúc này đi ngược nguyên tắc 4 ("đo trước khi mở rộng").

Nếu sau này dữ liệu thật sự bùng nổ, cơ chế đúng là **Qdrant snapshot**: snapshot phần cũ ra MinIO/S3
→ xóa khỏi instance live (lúc này mới thực sự free fd + disk) → restore khi có người mở lại cuộc đó.
Đánh đổi: lần truy cập đầu sau archive chậm (phải restore) + thêm một tầng phức tạp. **Để dành** cho
khi có số liệu, **đừng làm bây giờ.**

### 4.6. Config nhóm S
```ini
TRANSCRIPT_STORAGE_LAYOUT=shared          # shared (mặc định) | sharded | per_meeting
TRANSCRIPT_SHARED_COLLECTION=meeting_transcripts
TRANSCRIPT_NUM_SHARDS=8                    # chỉ dùng khi layout=sharded
QDRANT_ON_DISK=true                        # vector memmap (RAM tiering miễn phí)
QDRANT_ON_DISK_PAYLOAD=true
```

### 4.7. DoD nhóm S (gắn 7 bẫy)
- [ ] **S0**: đã đo fd bằng `/proc/1/fd` trên host, xác nhận fd tập trung ở qdrant; đã đếm `docs-*`.
      *(Kiểm code: singleton ✅, point_id=uuid4 ✅, Qdrant v1.10 ✅ — xong 2026-06-10.)*
- [ ] **S1**: `ulimits.nofile=65535` cho cả 2 service; collection tạo với `on_disk` + optimizers ít segment
      (cả `TranscriptStore` **lẫn** `QdrantService` — S2f).
- [ ] **S2**: `_physical()` mặc định `shared`; `point_id=uuid5` deterministic; `get_max_sequence_id`
      filter `meeting_id` + `order_by desc limit 1` (Bẫy 7 — test meeting >100 câu, xóa Redis key,
      embed tiếp → seq không trùng); dead code `_ensure_collection_exists` đã xóa.
- [ ] **S3 (then chốt)**: sau migration, **số collection thực giảm về 1** (`get_collections()`); dữ liệu mọi cuộc cũ vẫn query đúng & cô lập; migration idempotent (chạy lại không nhân đôi điểm — nhờ uuid5) + `--dry-run`.
- [ ] Tạo 100+ meeting mới → số collection **không tăng**; embed/query/context/segments/delete từng cuộc đúng.
- [ ] `per_meeting` cho hành vi == baseline (tương thích ngược).
- [ ] (Bẫy 5) Đã chốt số collection `docs-*` thực tế trên host; nếu nhiều → lên kế hoạch gộp docs riêng (không chặn S3).

---

## 5. Nhóm B1 — Worker per-collection + Recovery scan (độ bền)

**Hai mục tiêu trong một lần refactor `context_worker.py`:**

**(a) Song song hóa nhiều cuộc họp (giữ FIFO trong từng cuộc — D9):**
- Thay 1 queue global bằng **dict `{physical_or_meeting: asyncio.Queue}`** + semaphore
  `CONTEXT_WORKER_CONCURRENCY`. Mỗi key có **một** consumer task (FIFO nội bộ); nhiều key chạy song
  song tới mức semaphore. Dọn queue idle quá `WORKER_IDLE_TTL`. `stop()` drain tất cả.

**(b) Recovery scan lúc startup (vá lỗ hổng mất job — RAGFlow có, ta đang thiếu):**
- `ContextWorker` dùng `asyncio.Queue` **in-memory** → server restart khi còn job pending ⇒ **job biến
  mất**, utterance kẹt mãi ở `context_status="pending"`. Vì `context_status` lưu trong payload Qdrant
  (D6) nên **khôi phục được**: trong `main.py` lifespan startup, **quét các điểm
  `context_status ∈ {pending, processing}`** trên collection transcript và **enqueue lại**. Đây là
  cách rẻ để có độ bền mà **không cần kéo Redis Stream** vào (giữ tinh thần tối giản).
- Nếu sau này cần độ bền thật (at-least-once, redeliver qua restart đa worker), bài học trực tiếp từ
  RAGFlow là **đổi `asyncio.Queue` → Redis Stream + consumer group** — để dành, không làm bây giờ.

**Config**: `CONTEXT_WORKER_CONCURRENCY=4`, `CONTEXT_RECOVERY_SCAN=true`, `WORKER_IDLE_TTL=600`.
**DoD**: 2 cuộc họp embed song song → context build đồng thời, FIFO mỗi cuộc đúng; **kill + restart
server giữa chừng → các điểm pending được enqueue lại và chuyển `ready`** (không kẹt vĩnh viễn).

---

## 6. Nhóm A1 — Tokenizer tiếng Việt (🔴 cao, rủi ro thấp, "free win")

**Vì sao**: `retrieval.tokenize()` cắt theo `\w+` (âm tiết) → BM25 mất khớp **từ ghép** ("ngân sách",
"vận hành"). Segment từ ghép → khớp cụm/tên riêng chính xác hơn.

**Cách làm**: trừu tượng hóa `tokenize()` theo `HYBRID_TOKENIZER`: `simple` (mặc định, zero-dep) |
`pyvi` (`ViTokenizer.tokenize`) | `underthesea` (`word_tokenize`, nặng hơn). Load lười + cache; thiếu
lib → log + fallback `simple`. `pyvi` đặt optional trong requirements.

**Config**: `HYBRID_TOKENIZER=simple`. **DoD**: đo lại recall sau A1 (quyết định có cần A2 không);
đổi/tắt tokenizer không lỗi.

---

## 7. Hoãn (gated, default off — chỉ làm khi có dữ liệu chứng minh)

- **A2 — Sparse hybrid native** (`HYBRID_MODE=qdrant_sparse`): named sparse vector + Qdrant Query API
  (prefetch dense+sparse, fusion RRF). Bắt được câu mà BM25-in-process bỏ lỡ (vì in-process chỉ
  re-rank ứng viên vector). **Nhưng** buộc đổi `vectors_config` sang named vectors + re-index → **đo
  recall sau A1 trước**; nếu A1 đủ tốt thì **bỏ A2**.
- **C — PageIndex cho `docs-*`** (`DOCS_RETRIEVAL_MODE=pageindex`): cây TOC lưu thành điểm
  `kind="toc_node"` trong cùng collection; LLM điều hướng cây. Chỉ đáng làm nếu tài liệu họp **dài &
  có cấu trúc** — phần lớn ngắn thì bỏ. Tắt → `/query/` vector như cũ.
- **D — Agentic RAG**: **đặt ở tầng thiết bị** (điều phối gọi `/query/` + `/query/transcript`).
  Server chỉ thêm `/query/agentic` (endpoint mới, `AGENTIC_ENABLED=false`) **nếu** team yêu cầu agent
  phía server. Dễ "rườm rà" & tốn LLM nhất — làm cuối.
- **E (phần còn lại) — Hardening**: auth nội bộ (`INTERNAL_API_KEY`), rate limit, metrics
  (`context_status=failed` rate, backlog worker), mở rộng `/health`. *(Riêng **rotate `LLM_API_KEY`**
  tách ra làm sớm cùng S1.)*

---

## 8. Quyết định Kiến trúc (tiếp nối D1–D9 của [phase2.md](phase2.md))

| # | Quyết định |
|---|---|
| **D16** | Chống vỡ fd: tách **logical (API) ≠ physical (collection)** như RAGFlow (`index/tenant` + filter `kb_id`). Nhiều cuộc → ít collection chung + filter `meeting_id`. fd tỉ lệ dung lượng (có trần), không theo số cuộc. |
| **D17** | Layout mặc định **`shared`** (1 collection). `sharded` (hash `meeting_id`) là **YAGNI** tới khi đo được collection phình. `per_meeting` để tương thích ngược. |
| **D18** | fd ≠ RAM: **hợp nhất** chữa fd; **`on_disk`/memmap** chữa RAM **và** cho tiering miễn phí (OS page cache). Không xây cold-tier (napkin math: 10k cuộc ≈ vài chục GB). |
| **D19** | Gói S **làm trọn 4 bước**: S0 chẩn đoán → S1 vá nền (ulimit+on_disk+segment) → S2 hợp nhất → S3 migrate + **xóa collection cũ**. Thiếu bước nào ⇒ vẫn kẹt. |
| **D20** | `point_id` **deterministic** (`uuid5(meeting_id, sequence_id)`) — `uuid4` hiện tại đã an toàn với gộp, nhưng uuid5 cho migration/re-ingest **idempotent**; cấm dùng `sequence_id` trần làm id. |
| **D21** | Độ bền worker bằng **recovery scan** lúc startup (quét `pending/processing` trong payload Qdrant → enqueue lại). Redis Stream để dành nếu cần at-least-once thật. |
| **D22** | `get_max_sequence_id` filter `meeting_id` + `scroll(order_by desc, limit 1)` — đồng thời là **bug fix** (bản cũ chỉ quét 1 trang 100 điểm, không filter). Server v1.10.0 hỗ trợ `order_by` → không cần fallback. |
| **D23** | Tokenizer VN cắm được (`simple`/`pyvi`/`underthesea`), default `simple`, fallback an toàn. |

---

## 9. File tạo / sửa

**Tạo**
- `scripts/migrate_collections.py` — gộp `meeting-*` → collection chung, cấp lại `uuid5` point_id, `--dry-run`, verify số collection (S3)

**Sửa**
- `app/services/transcript_store.py` — `_physical()`; `point_id=uuid5`; `ensure_collection` on_disk + optimizers + index; **fix bug** `get_max_sequence_id(meeting_id)` → `order_by desc, limit=1, filter` (S2)
- `app/services/transcript_service.py`, `sequence_manager.py` — gọi store theo `meeting_id`; xóa dead code `_ensure_collection_exists` (S2)
- `app/services/vector_store.py` — áp on_disk + optimizers vào `_ensure_collection` + `create_collection` (S2f / Bẫy 5)
- `docker-compose.yml` — `ulimits.nofile=65535` cho qdrant + rag_api (S1)
- `app/workers/context_worker.py` — per-collection + semaphore (B1a)
- `app/main.py` — **recovery scan** lúc startup (B1b)
- `app/services/retrieval.py` — tokenizer cắm được (A1)
- `app/routers/embed.py` — thay `print()` còn sót trong background task bằng logging (cleanup nhỏ, tiện tay)
- `app/config.py`, `.env.example` — cờ mục 4.6 + A1 + B1
- `docs/architecture.md`, `rag_server/README.md` — đồng bộ

---

## 10. DoD tổng

- [ ] **S (then chốt)**: sau migration số collection **giảm về 1** (`get_collections()`); 100+ meeting mới không tăng collection; dữ liệu cũ query đúng & cô lập; point_id deterministic (uuid5, migration chạy lại không nhân đôi); **bug `get_max_sequence_id` đã fix** (Bẫy 7); `per_meeting` == baseline.
- [ ] **S1**: ulimit 65535 + on_disk + ít segment lớn áp dụng đồng thời (đo fd trước/sau bằng `/proc/1/fd`).
- [ ] **B1**: nhiều cuộc build song song, FIFO mỗi cuộc; **restart server → job pending khôi phục, không kẹt**.
- [ ] **A1**: đổi tokenizer cải thiện khớp cụm tiếng Việt; thiếu lib không lỗi.
- [ ] Mọi cờ TẮT + `TRANSCRIPT_STORAGE_LAYOUT=per_meeting` ⇒ hành vi == baseline hiện tại.
- [ ] Phase 1/2 + 4 endpoint transcript + `/docs` nguyên vẹn; không phát sinh endpoint lạ.

---

## 11. Câu hỏi mở (cập nhật 2026-06-10 — 2 câu đã đóng nhờ đối chiếu code)

1. **Số cuộc họp tích lũy dự kiến?** → khẳng định `shared` đủ (gần như chắc chắn) hay cần `sharded`.
2. **`docs-*` hiện có bao nhiêu collection trên host triển khai?** (Bẫy 5) — code đã xác nhận
   client *có thể* tạo tùy ý; số thực tế phải đếm lúc S0. S2(f) áp on_disk cho mọi collection docs
   mới bất kể câu trả lời.
3. ~~Version Qdrant server + qdrant-client?~~ ✅ **Đã đóng**: server v1.10.0, client 1.10.1 →
   `scroll order_by` hỗ trợ, bỏ fallback.
4. ~~`QdrantClient` đang singleton chưa?~~ ✅ **Đã đóng**: singleton class-level ở cả hai store.
5. **A2 có cần không?** → đo recall sau A1 rồi quyết.
6. **Đa tenant?** → nếu nhiều tổ chức, cân nhắc thêm `tenant_id` vào tên collection vật lý (RAGFlow-style).

---

*Hết tài liệu Phase 3 (rebuild). Nhóm S là blocker — triển khai **trọn 4 bước S0→S3**, thiếu bước nào
cũng còn kẹt. Mọi tính năng khác giữ nguyên tắc bất biến mục 2: gated, default TẮT, tắt = baseline.
Mọi sai khác so với Design Decisions (mục 8, tiếp nối D1–D9 của phase2.md) phải xác nhận lại.*
