# Kế hoạch Giai đoạn 3 — Nâng cấp RAG: Tokenizer VN, Sparse Hybrid, PageIndex & Agentic RAG

> **Tài liệu cho Coding Agent.** Đây là spec định hướng cho Giai đoạn 3 của hệ thống RAG
> BKMEETING. Phần lớn tính năng ở đây là **OPTIONAL, gated qua `.env`, mặc định TẮT** —
> tiếp nối đúng tinh thần Giai đoạn 2: *không phá endpoint, không phá tinh thần tối giản,
> Phase 1/2 chạy nguyên vẹn khi tắt cờ.* Đọc mục 2 (nguyên tắc) trước khi code.

---

## 1. Bối cảnh & hiện trạng (sau Giai đoạn 2 + đợt hoàn thiện 2026-06-07)

Hệ thống **BKMEETING** (SOICT / NAVIS Center, ĐHBK Hà Nội) — kiến trúc 2 tầng:
- **Tầng thiết bị (QCS8550)**: live transcript, LLM nhỏ sinh câu trả lời cho người dùng. *Ngoài phạm vi server.*
- **Tầng server (cái này)**: RAG API + Qdrant + Redis + LLM self-host build context.

**Đã có:**
- Phase 1 (tài liệu): `/embed/*`, `/query/`.
- Phase 2 (transcript): `/transcript/{collection}/embed`, `/query/transcript`, `/transcript/{collection}/context`, `/transcript/{collection}/segments`.
- Đợt hoàn thiện: `context_worker.py` (FIFO build context, D9); `/embed/info`; `/health` kiểm tra dependency; logging.
- **Nền retrieval kiểu RAGFlow đã đặt sẵn (gated, mặc định TẮT):**
  - `app/services/retrieval.py` — hybrid fusion BM25 + vector (`HYBRID_ENABLED`).
  - `app/services/reranker.py` — cross-encoder rerank (`RERANK_PROVIDER=none|local|http`).

Giai đoạn 3 **xây tiếp trên nền này**, không làm lại từ đầu.

---

## 2. Nguyên tắc bất biến (giữ nguyên từ Phase 2)

1. **Không thêm/đổi đường dẫn endpoint hiện có**; không đổi hình dạng response (chỉ `score` phản ánh điểm cuối). Tính năng mới = chế độ bật/tắt bên trong endpoint cũ, hoặc endpoint **mới hoàn toàn tách biệt** nếu thật sự cần (giải trình rõ).
2. **Gated + default TẮT**: mỗi tính năng có cờ `.env`; tắt cờ → hành vi y hệt hiện tại, không thêm dependency nặng, không thêm chi phí.
3. **Tôn trọng kiến trúc 2 tầng**: server là *nguồn tri thức*. Logic *điều phối agent* ưu tiên đặt ở **tầng thiết bị**; server chỉ cung cấp công cụ truy vấn tốt.
4. **1 collection = 1 cuộc họp**; `meeting_id` suy từ tên collection; Phase 1 (`docs-*`) và Phase 2 (`meeting-*`) tách biệt.
5. **Đo trước khi mở rộng**: tính năng nặng (PageIndex, Agentic) chỉ bật khi có dữ liệu thực chứng minh giá trị.

---

## 3. Phạm vi

### 3.1. IN SCOPE (xếp theo ưu tiên — làm từ trên xuống)

| Nhóm | Tính năng | Ưu tiên | Tác động endpoint |
|---|---|---|---|
| **S** | **Hợp nhất storage — chống "too many open files"** (tách tên logic khỏi collection vật lý, filter theo `meeting_id`) | 🔴🔴 Blocker | Không — endpoint giữ nguyên, chỉ đổi lưu trữ bên trong |
| **A1** | Tokenizer tiếng Việt cho hybrid (pyvi/underthesea) | 🔴 Cao | Không — nâng chất lượng `fuse()` |
| **A2** | Sparse vector native trên Qdrant (BM25/SPLADE) + RRF fusion | 🟡 TB | Không — chế độ retrieval thay thế in-process BM25 |
| **B1** | Context worker **per-collection** (song song nhiều cuộc họp, vẫn FIFO trong từng cuộc) | 🟡 TB | Không |
| **B2** | Batch build context (gộp nhiều câu / 1 lần gọi LLM) | 🟢 Thấp | Không |
| **C** | **PageIndex** cho luồng tài liệu (`docs-*`) — cây mục lục, retrieval reasoning-based | 🟡 TB | `/query/` thêm chế độ; ingest dựng cây |
| **D** | **Agentic RAG** — điều phối multi-hop trên cả docs + transcript | 🟢 Thấp/Optional | Ưu tiên ở tầng thiết bị; server tùy chọn `/query/agentic` |
| **E** | Production hardening: auth nội bộ, metrics, rate limit, rotate key | 🟡 TB | Middleware, không đổi path |

### 3.2. OUT OF SCOPE
- Không xây UI / phần app workstation.
- Không thay embedding model nền (giữ MiniLM-384) trừ khi A2 yêu cầu thêm sparse model riêng.
- Không bê nguyên GraphRAG/community detection của RAGFlow (quá nặng cho on-device meeting).
- Không phá luồng Phase 1/2 khi mọi cờ TẮT.

---

## 4. Quyết định Kiến trúc (tiếp nối D1–D9 của phase2plan_v2)

| # | Vấn đề | Quyết định |
|---|---|---|
| **D16** | Chống vỡ giới hạn file mở của Qdrant | Học mô hình RAGFlow (`index_name(tenant)` + filter `kb_id`): **tách tên collection logic (API) khỏi collection vật lý**. Nhiều cuộc họp dồn vào **ít collection vật lý dùng chung**, cô lập bằng filter `meeting_id` (đã có sẵn payload index). Số file mở tỉ lệ với **số bucket cố định**, KHÔNG với số cuộc họp. |
| **D17** | Bố cục vật lý | Mặc định **`shared`**: tất cả transcript trong 1 collection `meeting_transcripts` (filter `meeting_id`). Tùy chọn **`sharded`**: `meeting_bucket_{h}`, `h = stable_hash(meeting_id) % TRANSCRIPT_NUM_SHARDS` (vd 8/16) khi 1 collection quá lớn. Tùy chọn **`per_meeting`**: giữ hành vi cũ (1 collection/cuộc) cho ai cần cô lập tuyệt đối — KHÔNG khuyến nghị ở quy mô lớn. |
| **D18** | Giữ hợp đồng API | Endpoint vẫn `/transcript/meeting-{uuid}/...`; `meeting_id` suy từ tên như cũ. Thêm lớp `resolve_physical(meeting_id) -> (collection, filter)` trong `TranscriptStore`/service. `get_max_sequence_id()` PHẢI filter theo `meeting_id` khi `shared/sharded`. Sequence key Redis vẫn theo `meeting_id` (đã duy nhất). Xóa cuộc họp = `delete_by_meeting_id` (đã có `TranscriptStore.delete_meeting`), KHÔNG drop collection. |
| **D19** | Doc-store trừu tượng (định hướng) | Cân nhắc tách `VectorBackend` interface (như `DocStoreConnection` của RAGFlow) để sau này đổi/đặt cạnh engine khác (Infinity/ES) mà không sửa business logic. Phase 3 chỉ chuẩn bị interface mỏng quanh Qdrant, chưa đổi engine. |
| **D10** | Tokenize tiếng Việt cho BM25 | Thêm lớp tokenizer cắm được: `simple` (hiện tại) \| `pyvi` \| `underthesea`. Cấu hình `HYBRID_TOKENIZER`. Mặc định `simple` (zero-dep). Load lười, fallback `simple` nếu thư viện thiếu. |
| **D11** | Sparse hybrid trên Qdrant | Khi `HYBRID_MODE=qdrant_sparse`: thêm **named sparse vector** vào điểm Qdrant lúc ingest, query dùng Qdrant Query API (prefetch dense + sparse → fusion RRF). Khi `HYBRID_MODE=inprocess` (mặc định) → giữ BM25 in-process hiện tại, **không re-index**. |
| **D12** | Worker per-collection | Thay 1 hàng đợi global bằng **dict hàng đợi theo collection** + worker pool có giới hạn; trong mỗi collection vẫn FIFO (giữ D9). Số worker đồng thời tối đa = `CONTEXT_WORKER_CONCURRENCY`. |
| **D13** | PageIndex là *thêm*, không *thay* | PageIndex là **chế độ retrieval phụ** cho `docs-*`, bật bằng `DOCS_RETRIEVAL_MODE=pageindex`. Cây mục lục lưu thành **điểm metadata đặc biệt** trong cùng collection (`kind="toc_node"`), không cần store mới. Tắt → `/query/` chạy vector như cũ. |
| **D14** | Agentic đặt ở đâu | **Ưu tiên tầng thiết bị** điều phối (gọi tuần tự/lặp `/query/` + `/query/transcript`). Server **chỉ** thêm endpoint `/query/agentic` khi team yêu cầu chạy agent phía server — mặc định **không** bật (`AGENTIC_ENABLED=false`). Giữ server là nguồn tri thức gọn. |
| **D15** | Tương thích ngược tuyệt đối | Mọi cờ TẮT ⇒ nhị phân hành vi == hệ thống sau đợt 2026-06-07. CI phải có 1 suite chạy với toàn bộ cờ tắt để khẳng định điều này. |

---

## 5. Đặc tả từng nhóm

### S — Hợp nhất storage chống "too many open files" (🔴🔴 BLOCKER — làm TRƯỚC)

**Vì sao (vấn đề thực)**: thiết kế v2 dùng **1 Qdrant collection / 1 cuộc họp** (`meeting-{uuid}`) + `docs-{uuid}`. Mỗi collection giữ segment + file mmap riêng → **số file mở (file descriptor) tỉ lệ với số cuộc họp**. Khi tích lũy nhiều cuộc họp, server vượt `ulimit -n` → lỗi *"too many open files"*, Qdrant từ chối mở segment.

**Bài học RAGFlow** (`rag/nlp/search.py:34`, `common/doc_store/`): RAGFlow tạo **1 index / 1 tenant** (`ragflow_{tenant_id}`), mọi knowledge base/doc nằm chung index, **phân tách bằng filter `kb_id`/`doc_id`** — số index tỉ lệ với số tenant (ít), không với số KB (nhiều). Ta áp đúng nguyên lý: **partition logic ≠ collection vật lý.**

**Cách làm** (giữ endpoint y nguyên — chỉ đổi lưu trữ bên trong):
1. **Lớp resolve logical→physical**: thêm `TranscriptStore._physical(meeting_id) -> str` theo `TRANSCRIPT_STORAGE_LAYOUT`:
   - `shared` (mặc định): trả `TRANSCRIPT_SHARED_COLLECTION` (vd `meeting_transcripts`).
   - `sharded`: trả `meeting_bucket_{stable_hash(meeting_id) % TRANSCRIPT_NUM_SHARDS}`.
   - `per_meeting`: trả `meeting-{meeting_id}` (hành vi cũ — tương thích ngược).
2. **Mọi thao tác filter theo `meeting_id`** — `TranscriptStore.search/scroll_window/find_by_seq` **đã** filter sẵn ✅. Chỉ cần:
   - `upsert_point`: dùng collection vật lý từ `_physical`.
   - `get_max_sequence_id(meeting_id)`: **thêm filter `meeting_id`** (hiện đang quét cả collection — sai khi dùng chung).
   - `ensure_collection`: tạo **collection vật lý dùng chung** 1 lần (idempotent), payload index `meeting_id`(keyword)/`sequence_id`(integer)/`speaker`.
   - `delete_meeting(meeting_id)`: đã có — xóa theo filter, **không** drop collection.
3. **SequenceManager**: key Redis vẫn `rag:seq:{meeting_id}` (uuid duy nhất → không đụng nhau dù chung collection). Rebuild gọi `get_max_sequence_id(meeting_id)`.
4. **Migration**: script gộp các collection `meeting-*` cũ vào collection dùng chung (scroll → re-upsert kèm `meeting_id`), rồi `delete_collection` cái cũ. Idempotent, chạy được nhiều lần. `per_meeting` cho ai chưa muốn migrate.
5. **(Tùy chọn) Phase 1 docs**: cùng nguyên lý — gộp `docs-*` vào ít collection + filter `source`/`doc_id`. Xâm lấn hơn (app tự đặt tên collection) → để **giai đoạn sau**, ngoài scope bắt buộc của nhóm S.

**Mitigation bổ sung (làm song song, rẻ)**: nâng `ulimit -n` (vd 65535) trong `docker-compose.yml` (`ulimits.nofile`); cân nhắc Qdrant `on_disk_payload`, giảm số segment. Đây là *giảm nhẹ*, không thay cho hợp nhất.

**Config**: `TRANSCRIPT_STORAGE_LAYOUT=shared` (default) | `sharded` | `per_meeting`; `TRANSCRIPT_SHARED_COLLECTION=meeting_transcripts`; `TRANSCRIPT_NUM_SHARDS=8`.

**Lưu ý hiệu năng**: collection dùng chung cần payload index `meeting_id` tốt (đã có). Qdrant xử lý hàng triệu điểm/collection tốt; filter `meeting_id` rẻ. `sharded` dùng khi 1 collection phình quá lớn hoặc muốn phân tán.

**DoD nhóm S**:
- [ ] Tạo 100+ "cuộc họp" (meeting_id khác nhau) → **số collection Qdrant không tăng theo** (1 với `shared`, ≤N với `sharded`).
- [ ] Embed/query/context/segments/delete của từng cuộc vẫn đúng & cô lập (không lẫn dữ liệu cuộc khác).
- [ ] `get_max_sequence_id` trả max **theo meeting_id**, không phải max toàn collection → sequence rebuild đúng.
- [ ] `per_meeting` cho hành vi y hệt v2 (tương thích ngược).
- [ ] Script migration gộp collection cũ chạy idempotent, dữ liệu không mất.

---

### A1 — Tokenizer tiếng Việt (🔴 ưu tiên cao nhất, rủi ro thấp)

**Vì sao**: `retrieval.tokenize()` hiện cắt theo âm tiết (`\w+`), nên BM25 không khớp **từ ghép** ("ngân sách", "vận hành", "tuyển dụng"). Segment từ ghép → khớp cụm chính xác hơn nhiều, đúng nhu cầu giữ tên riêng/thuật ngữ.

**Cách làm**:
- `app/services/retrieval.py`: trừu tượng hóa `tokenize()` theo `settings.hybrid_tokenizer`:
  - `simple`: như hiện tại (mặc định, zero-dep).
  - `pyvi`: `from pyvi import ViTokenizer` → `ViTokenizer.tokenize(text)` rồi split.
  - `underthesea`: `word_tokenize(text)` (nặng/chậm hơn, chính xác hơn).
- Load lười + cache; thiếu thư viện → log cảnh báo, fallback `simple`.
- `requirements.txt`: `pyvi` đặt **optional** (extras hoặc comment hướng dẫn cài khi cần).

**Config**: `HYBRID_TOKENIZER=simple` (default).
**DoD**: query "ngân sách quý 4" khớp câu chứa cụm "ngân sách" tốt hơn baseline; tắt/đổi tokenizer không lỗi.

---

### A2 — Sparse hybrid native trên Qdrant (🟡)

**Vì sao**: BM25 in-process chỉ re-rank trên ứng viên vector trả về → **không cứu được** câu mà vector search trượt ngay từ đầu. Sparse vector index toàn bộ collection → bắt được khớp từ khóa thuần.

**Cách làm**:
- Sinh sparse vector (BM25/SPLADE) lúc ingest (`/embed/*` và `/transcript/embed`), thêm **named vector** `"sparse"` vào `PointStruct`. Cần đổi `vectors_config` của collection sang dạng named (dense + sparse) — **chỉ áp cho collection tạo mới khi `HYBRID_MODE=qdrant_sparse`**; collection cũ vẫn dùng `inprocess`.
- Query: Qdrant **Query API** với `prefetch` (dense KNN + sparse) và `fusion=RRF`. Bọc trong `retrieval.py` để router không đổi.
- Giữ `inprocess` làm mặc định để không buộc re-index dữ liệu hiện có.

**Config**: `HYBRID_MODE=inprocess` (default) | `qdrant_sparse`.
**Lưu ý**: cần qdrant-client hỗ trợ sparse (đã có từ ≥1.7). Kiểm tra version trong `requirements.txt`.
**DoD**: với `qdrant_sparse`, truy vấn chỉ-từ-khóa (tên riêng hiếm) tìm thấy câu mà chế độ vector-only bỏ lỡ.

---

### B1 — Context worker per-collection (🟡)

**Vì sao**: worker hiện là 1 hàng đợi FIFO **global** → nhiều cuộc họp song song bị serialize chung, build context cuộc B phải chờ cuộc A. Khi triển khai nhiều phòng họp đồng thời, độ trễ context tăng.

**Cách làm** (`app/workers/context_worker.py`):
- Thay 1 queue bằng **dict `{collection: asyncio.Queue}`** + một semaphore `CONTEXT_WORKER_CONCURRENCY`.
- Mỗi collection có **một** consumer task (đảm bảo FIFO trong cuộc — giữ D9); nhiều collection chạy song song tới mức semaphore.
- Dọn queue/task của collection khi idle quá `WORKER_IDLE_TTL` để tránh rò rỉ.
- `stop()` drain tất cả queue.

**Config**: `CONTEXT_WORKER_CONCURRENCY=4`.
**DoD**: embed song song 2 cuộc họp → context cả hai build đồng thời; thứ tự trong từng cuộc vẫn đúng (1..N).

---

### B2 — Batch build context (🟢)

**Vì sao**: khi người ta nói nhanh, mỗi câu 1 lần gọi LLM gây dồn tải. Gộp `k` câu liên tiếp thành 1 lần tóm tắt giảm số lần gọi.

**Cách làm**: worker gom job cùng collection trong cửa sổ thời gian ngắn (`BATCH_WINDOW_MS`) hoặc tới `BATCH_MAX`; build context cho mốc mới nhất, các câu trung gian kế thừa. Giữ định nghĩa `context[N]` nhất quán (bối cảnh dẫn tới câu N).
**Config**: `CONTEXT_BATCH_ENABLED=false`, `CONTEXT_BATCH_WINDOW_MS=500`, `CONTEXT_BATCH_MAX=5`.
**DoD**: tốc độ nói cao → số lần gọi LLM giảm rõ, chất lượng context không tụt đáng kể.

---

### C — PageIndex cho luồng tài liệu (🟡, optional)

**Vì sao**: tài liệu họp dài/có cấu trúc (báo cáo, quy chế) bị chunking + vector làm "vỡ" mạch; vector hay trượt câu hỏi cần đọc theo mục. PageIndex dựng **cây mục lục** rồi để LLM điều hướng tới đúng nhánh (reasoning-based, vectorless ở bước định vị).

**Cách làm** (chỉ cho `docs-*`, gated):
- **Ingest** (`/embed/file` khi `DOCS_PAGEINDEX_ON_INGEST=true`): sau parse, dùng LLM gán cấp mục lục (tham khảo prompt `assign_toc_levels` của RAGFlow) → lưu các **node TOC** thành điểm metadata `kind="toc_node"` (title, level, parent, chunk refs) trong cùng collection.
- **Query** (`/query/` khi `DOCS_RETRIEVAL_MODE=pageindex`): LLM "lật" cây TOC chọn nhánh liên quan → lấy chunk con của nhánh đó (kết hợp/không vector). Trả **đúng schema `QueryResult` cũ**.
- Tắt cờ → `/query/` vector như cũ; node TOC bị bỏ qua khi search thường (filter `kind != "toc_node"`).

**Config**: `DOCS_RETRIEVAL_MODE=vector` (default) | `pageindex`; `DOCS_PAGEINDEX_ON_INGEST=false`.
**Reuse**: `LLMClient` sẵn có; mẫu prompt TOC của RAGFlow (`rag/prompts/assign_toc_levels.md`).
**DoD**: với tài liệu dài có mục lục, câu hỏi "mục X quy định gì" trả đúng đoạn theo cấu trúc, tốt hơn vector-only.

---

### D — Agentic RAG (🟢, optional, ưu tiên tầng thiết bị)

**Vì sao**: câu hỏi khó cần multi-hop hoặc ghép cả tài liệu lẫn transcript (vd "so quyết định ngân sách trong họp với đề xuất ban đầu trong tài liệu"). Agent tự lập kế hoạch → chọn luồng → lặp truy vấn → tổng hợp.

**Quyết định kiến trúc (D14)**:
- **Mặc định**: agent **chạy ở tầng thiết bị QCS8550**, điều phối gọi 2 endpoint server sẵn có (`/query/`, `/query/transcript`). Server **không cần thay đổi** → giữ gọn, đúng 2 tầng.
- **Optional server-side**: nếu team muốn agent phía server, thêm endpoint **mới, tách biệt** `POST /query/agentic` (gated `AGENTIC_ENABLED=true`):
  - Vòng lặp: phân rã câu hỏi → chọn tool (`search_docs` / `search_transcript`) → gọi nội bộ retrieval đã có → tự đánh giá đủ chưa (tối đa `AGENTIC_MAX_STEPS`) → tổng hợp + trích nguồn.
  - Tham khảo `DeepResearcher` + `agent/` của RAGFlow ở mức ý tưởng, **không** bê framework.
- Response của `/query/agentic` là endpoint mới (không đụng `/query/` cũ).

**Config**: `AGENTIC_ENABLED=false`, `AGENTIC_MAX_STEPS=4`, `AGENTIC_LLM_*` (có thể tách model khác build-context).
**Cảnh báo**: đây là tính năng dễ "rườm rà" & tốn LLM nhất — chỉ làm khi có nhu cầu thực và đã đo lợi ích.
**DoD**: câu hỏi multi-hop ghép docs+transcript được trả lời kèm trích nguồn; tắt cờ → endpoint không tồn tại, hệ thống như cũ.

---

### E — Production hardening (🟡)

- **Rotate `LLM_API_KEY`** đang lộ plaintext; chuyển sang Docker secret / env injection.
- **Auth nội bộ**: API key/header giữa thiết bị và server (hiện CORS `*`, không auth). Middleware gated `INTERNAL_API_KEY`.
- **Rate limit + giới hạn kích thước** cho `/embed/text`, `/transcript/embed`, `/query/*`.
- **Metrics/observability**: thời gian build context, độ trễ query, tỉ lệ `context_status=failed`, độ sâu agent. Xuất Prometheus hoặc log có cấu trúc.
- **/health** mở rộng: thêm trạng thái worker (số queue, backlog).

---

## 6. File dự kiến tạo / sửa

**Tạo mới**
- `scripts/migrate_collections.py` — gộp `meeting-*` cũ vào collection dùng chung (nhóm S, idempotent)
- `app/services/pageindex.py` — dựng cây TOC + retrieval reasoning-based (nhóm C)
- `app/services/agentic.py` — vòng lặp agent + tool calling (nhóm D, nếu bật server-side)
- `app/routers/query.py` — (nếu D server-side) thêm `POST /query/agentic`
- `app/middleware/auth.py`, `app/middleware/ratelimit.py` (nhóm E)
- `app/utils/metrics.py` (nhóm E)

**Sửa**
- `app/services/transcript_store.py` — lớp `_physical(meeting_id)`; filter `meeting_id` cho `get_max_sequence_id`; `ensure_collection` dùng chung (nhóm S)
- `app/services/transcript_service.py` / `sequence_manager.py` — gọi store theo `meeting_id` thay vì tên collection logic (nhóm S)
- `docker-compose.yml` — `ulimits.nofile` cho qdrant + rag_api (nhóm S, mitigation)
- `app/services/retrieval.py` — tokenizer cắm được (A1); chế độ `qdrant_sparse` (A2)
- `app/services/embedding.py` / `transcript_store.py` / `vector_store.py` — sinh + lưu sparse vector (A2); filter `kind` (C)
- `app/workers/context_worker.py` — per-collection + concurrency (B1); batch (B2)
- `app/config.py` — toàn bộ cờ mục 7
- `app/main.py` — middleware auth/ratelimit/metrics; mở rộng /health
- `.env.example`, `rag_server/README.md`, `STATE.md` — đồng bộ
- `requirements.txt` — `pyvi` (optional); kiểm tra qdrant-client hỗ trợ sparse

---

## 7. Cấu hình `.env` mới (tất cả mặc định = hành vi cũ)

```ini
# === S: Hợp nhất storage (chống too-many-open-files) ===
TRANSCRIPT_STORAGE_LAYOUT=shared       # shared | sharded | per_meeting
TRANSCRIPT_SHARED_COLLECTION=meeting_transcripts
TRANSCRIPT_NUM_SHARDS=8                 # chỉ dùng khi layout=sharded

# === A1: Tokenizer hybrid ===
HYBRID_TOKENIZER=simple            # simple | pyvi | underthesea

# === A2: Sparse hybrid ===
HYBRID_MODE=inprocess              # inprocess | qdrant_sparse

# === B1/B2: Context worker ===
CONTEXT_WORKER_CONCURRENCY=4
CONTEXT_BATCH_ENABLED=false
CONTEXT_BATCH_WINDOW_MS=500
CONTEXT_BATCH_MAX=5

# === C: PageIndex (docs) ===
DOCS_RETRIEVAL_MODE=vector         # vector | pageindex
DOCS_PAGEINDEX_ON_INGEST=false

# === D: Agentic RAG (server-side, optional) ===
AGENTIC_ENABLED=false
AGENTIC_MAX_STEPS=4

# === E: Hardening ===
INTERNAL_API_KEY=                  # rỗng = tắt auth (giữ tương thích)
RATE_LIMIT_ENABLED=false
METRICS_ENABLED=false
```

---

## 8. Danh sách công việc (theo thứ tự ưu tiên)

> Mỗi task 1 commit/PR riêng, kèm test cờ-tắt-bằng-baseline.

**Task 0 — S Hợp nhất storage (LÀM TRƯỚC, blocker)**: lớp `_physical(meeting_id)` trong `TranscriptStore`; filter `meeting_id` cho `get_max_sequence_id`; `ensure_collection` dùng chung; nâng `ulimit nofile` trong compose; script migration gộp `meeting-*`; test 100+ meeting → số collection không tăng. Mặc định `shared`; `per_meeting` cho tương thích ngược.
**Task 1 — A1 Tokenizer VN**: tokenizer cắm được trong `retrieval.py`; pyvi optional; unit test khớp cụm từ.
**Task 2 — B1 Worker per-collection**: refactor `context_worker.py`; test 2 cuộc họp song song giữ thứ tự.
**Task 3 — A2 Sparse hybrid**: sparse vector lúc ingest + Query API RRF; chỉ collection mới; test khớp từ khóa hiếm.
**Task 4 — C PageIndex**: dựng cây TOC khi ingest + retrieval theo cây; gated; test tài liệu dài.
**Task 5 — E Hardening**: auth + rate limit + metrics; rotate key; mở rộng /health.
**Task 6 — B2 Batch context** (nếu cần sau khi đo tải): gộp job.
**Task 7 — D Agentic** (chỉ khi team yêu cầu): `/query/agentic` server-side; hoặc tài liệu hướng dẫn agent ở tầng thiết bị.
**Task 8 — Tài liệu & CI**: cập nhật README/STATE; suite "all-flags-off == baseline".

---

## 9. Tiêu chí hoàn thành tổng (DoD)

- [ ] Mọi cờ mục 7 TẮT ⇒ hành vi == hệ thống sau đợt 2026-06-07 (suite baseline pass). *(Ngoại lệ: `TRANSCRIPT_STORAGE_LAYOUT` mặc định `shared` đổi cách lưu vật lý — đặt `per_meeting` để bằng baseline tuyệt đối.)*
- [ ] **S**: tạo 100+ meeting → số collection Qdrant KHÔNG tăng theo (1 với `shared`, ≤N với `sharded`); dữ liệu từng cuộc cô lập; `get_max_sequence_id` filter đúng `meeting_id`; migration idempotent không mất dữ liệu; `per_meeting` == baseline.
- [ ] A1: đổi `HYBRID_TOKENIZER` cải thiện khớp cụm tiếng Việt, không lỗi khi thiếu thư viện.
- [ ] B1: nhiều cuộc họp build context song song; FIFO trong từng cuộc giữ nguyên (D9).
- [ ] A2: `qdrant_sparse` bắt được câu khớp-từ-khóa mà vector-only trượt; collection cũ không bị buộc re-index.
- [ ] C: `pageindex` trả đúng đoạn theo cấu trúc cho tài liệu dài; tắt → `/query/` vector như cũ; node TOC không lẫn vào kết quả thường.
- [ ] D: (nếu bật) `/query/agentic` trả lời multi-hop kèm trích nguồn; tắt → endpoint không tồn tại.
- [ ] E: auth/rate-limit/metrics bật được mà không phá client cũ khi tắt.
- [ ] Phase 1/2 + 4 endpoint transcript + `/docs` không phát sinh thay đổi ngoài ý muốn.

---

## 10. Câu hỏi mở (chốt với team trước khi code từng nhóm)

0. **Bố cục storage (nhóm S)**: chọn `shared` (1 collection, đơn giản nhất) hay `sharded` (N bucket, phân tán)? Ước lượng số cuộc họp/tháng & tổng tích lũy để chọn N. Có cần migrate dữ liệu `meeting-*` hiện có hay chấp nhận `per_meeting` cho dữ liệu cũ + `shared` cho cuộc mới?
0b. **Đa tenant**: một server có phục vụ nhiều tổ chức không? Nếu có, cân nhắc thêm `tenant_id` vào tên collection vật lý (giống `ragflow_{tenant}`) để cô lập theo tenant.
0c. **Docs `docs-*`**: có gộp luôn không, hay để giai đoạn sau? (xâm lấn hơn vì app tự đặt tên collection).
1. **A2 vs A1**: nếu A1 (tokenizer) đã đủ tốt cho nhu cầu thực, có cần A2 (sparse native, tốn re-index) không?
2. **PageIndex**: tài liệu họp thực tế có dài & có mục lục rõ không? Nếu phần lớn ngắn → bỏ nhóm C.
3. **Agentic ở đâu**: team muốn agent ở thiết bị (giữ server gọn) hay phía server (`/query/agentic`)? → quyết định có làm `app/services/agentic.py` không.
4. **LLM cho agent/PageIndex**: dùng chung model build-context hay tách model mạnh hơn? Ảnh hưởng VRAM server.
5. **Auth**: mức độ bảo mật giữa thiết bị–server (chỉ API key, hay mTLS)?

---

*Hết tài liệu Giai đoạn 3. Triển khai theo thứ tự ưu tiên mục 8; mọi sai khác so với Design Decisions (mục 4, tiếp nối D1–D9 của phase2plan_v2) phải xác nhận lại với người yêu cầu. Mọi tính năng phải giữ nguyên tắc bất biến mục 2: gated, default TẮT, tắt = baseline.*
