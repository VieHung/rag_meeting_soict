# Kế hoạch Giai đoạn 2 — RAG Transcript (BẢN v2 — RÚT GỌN)

> **Tài liệu cho Coding Agent.** Bản v2 này **thay thế** bản plan trước đó. Nếu hệ thống đã được code theo bản cũ (8 endpoint), đây là **plan refactor** — đọc kỹ mục 3 (endpoint cần GỠ/GỘP) trước khi sửa.
>
> **4 điểm làm rõ từ người yêu cầu — đã đưa vào v2:**
> 1. Việc tạo 2 collection là **thủ công** (do app làm). Phân biệt bằng **tiền tố tên collection**.
> 2. **Không thêm endpoint rườm rà** — cắt từ 8 xuống còn **2 core + 2 optional**.
> 3. Context sinh bởi **LLM chạy local trên chính server chứa VectorDB** → build trong tiến trình (in-process), không cần endpoint HTTP để cập nhật.
> 4. Context là **một trường metadata của vector** trong collection meeting.

---

## 0. Trạng thái triển khai (cập nhật 2026-06-07)

> Phần này phản ánh **hiện trạng vận hành**; phần thiết kế bên dưới (mục 1→14) giữ nguyên làm nguồn sự thật.

- ✅ **Đã xong:** 4 endpoint v2; SequenceManager (Redis atomic, self-heal); embedding chạy trên
  **NPU Qualcomm AI080** (backend `qaic`, `intfloat/multilingual-e5-base` 768d, prefix E5).
  Deploy bằng Docker, `rag_api` host **`:18000`** → container `1904`.
- ⚙️ **LLM build-context ĐANG TẮT:** `LLM_PROVIDER=none` → ContextBuilder bỏ qua, mọi câu có
  `context_status="disabled"`, trường `context` rỗng. (Để bật: Ollama + `LLM_PROVIDER=ollama`,
  `qwen2.5:7b`.) Vì thế các mục về luồng LLM (8.2) hiện chưa hoạt động trên môi trường thật.
- ⚠️ **Sự cố hạ tầng đã biết:** `/transcript/.../embed` có thể trả **500** do Qdrant
  "Too many open files" — container Qdrant chạy `nofile` soft = 1024 trong khi mỗi cuộc họp tạo
  một collection riêng. Khắc phục: nâng `ulimits.nofile` cho service `qdrant` + recreate
  (chi tiết: `rag_server/README.md` → "Vận hành & sự cố thường gặp"). **Chưa áp dụng.**
- ⚠️ **Race tạo collection (TOCTOU):** `QdrantService` (Phase 1) đã vá idempotent; còn
  `TranscriptStore.ensure_collection` chưa vá → câu transcript đầu của meeting mới có thể rớt khi
  gửi đồng thời. Cần sửa.
- 🧪 **Kiểm thử:** `rag_server/scripts/test_live_api.py` — bộ edge-case bắn HTTP thật vào `:18000`.

---

## 1. Bối cảnh

Hệ thống **BKMEETING — Phòng Họp Thông Minh kết hợp Tổ Thư Ký Ảo** (SOICT / NAVIS Center, ĐHBK Hà Nội).

### Kiến trúc 2 tầng (làm rõ)

Hệ thống gồm **hai tầng độc lập, chạy trên phần cứng khác nhau** — đừng nhầm lẫn:

| Tầng | Phần cứng | Vai trò |
|---|---|---|
| **Tầng thiết bị (client)** | **Qualcomm QCS8550** — chính là thiết bị trong poster (màn hình tương tác + biển tên). Mỗi người dự họp một workstation. | Chạy AI on-device: live transcript, nhận diện khuôn mặt, dịch, **LLM nhỏ để sinh câu trả lời cho người dùng**. Ưu tiên độ trễ thấp, bảo mật. |
| **Tầng server (cái plan này xây)** | **Server riêng, độc lập, mạnh hơn nhiều** — KHÔNG phải QCS8550. | Chạy **RAG API + Qdrant VectorDB + một LLM thật self-host** để build context. |

**Hệ quả quan trọng cho plan này:**

- Phần `LLM build context` của plan chạy trên **server mạnh độc lập**, **không** bị giới hạn phần cứng của QCS8550. Có thể self-host một **LLM thực thụ** (model lớn, không phải model tí hon cho thiết bị nhúng).
- Hai LLM khác nhau, hai mục đích khác nhau:
  - **LLM trên server (plan này lo)** — tóm tắt/build context cho từng câu transcript. Chạy phía VectorDB server.
  - **LLM trên thiết bị QCS8550 (ngoài phạm vi plan)** — nhận `context + window + text` do RAG trả về, sinh câu trả lời cuối cho người dùng.
- Vì server build-context mạnh và đặt cạnh VectorDB, ContextBuilder là **background worker in-process** gọi thẳng LLM self-host — không cần endpoint HTTP trung gian.

### Vai trò RAG trong sản phẩm

RAG API Server (tầng server) là **nguồn tri thức** cho chatbot cuộc họp. Thiết bị QCS8550 gửi câu hỏi tới server, server trả về nội dung liên quan, LLM trên thiết bị sinh câu trả lời. Hai luồng truy vấn:
1. **Truy vấn tài liệu** — tài liệu upload trước (Giai đoạn 1, giữ nguyên).
2. **Truy vấn transcript** — nội dung hội thoại đã transcript (**Giai đoạn 2**).

Người dùng **tự chọn** luồng nào ở phía app.

---

## 2. Quy ước Collection (ĐIỂM CỐT LÕI v2)

Mỗi cuộc họp có **một cặp collection**, do app **tạo thủ công** (qua endpoint có sẵn `POST /embed/collections`). Phân biệt bằng **tiền tố**:

| Loại | Mẫu tên collection | Ví dụ | Endpoint xử lý |
|---|---|---|---|
| Transcript cuộc họp | `meeting-{uuid}` | `meeting-c7cfdf57-fbdc-48fe-a882-22af4ee817ad` | `/transcript/*`, `/query/transcript` |
| Tài liệu cuộc họp | `docs-{uuid}` | `docs-c7cfdf57-fbdc-48fe-a882-22af4ee817ad` | `/embed/*`, `/query/` (giữ nguyên) |

**Hệ quả thiết kế (quan trọng):**

- `{uuid}` chính là **ID cuộc họp** (`meeting_id`). Server **suy ra** `meeting_id` từ tên collection (`collection.removeprefix("meeting-")`) — **client không cần gửi `meeting_id`** trong body.
- **Một collection = đúng một cuộc họp.** Vì vậy **không cần lọc theo `meeting_id`** khi query — bản thân collection đã cô lập dữ liệu.
- Sequence counter và context được quản lý **theo từng collection**.
- Endpoint transcript phải **kiểm tra tiền tố** `meeting-`; nếu nhận collection sai tiền tố → trả `400`.
- Cặp `meeting-{uuid}` / `docs-{uuid}` cùng chia sẻ một `{uuid}` → app dễ ghép đôi.

> App chịu trách nhiệm tạo collection và đặt tên đúng quy ước. Server **không** sinh endpoint tạo cặp collection.

---

## 3. Bộ Endpoint v2 — đã rút gọn

### 3.1. Endpoint MỚI (chỉ 4, trong đó 2 bắt buộc)

| # | Method | Endpoint | Bắt buộc? | Mô tả |
|---|---|---|---|---|
| 1 | `POST` | `/transcript/{collection}/embed` | ✅ Core | Lưu 1 câu transcript thành 1 vector, server gán `sequence_id`, trigger build context nền |
| 2 | `POST` | `/query/transcript` | ✅ Core | Truy vấn ngữ nghĩa transcript, trả kết quả kèm **window** + **context** |
| 3 | `GET` | `/transcript/{collection}/context` | ⬜ Optional | Lấy context (tóm tắt cuộc họp) mới nhất; hỗ trợ `?sequence_id=` để lấy tại 1 thời điểm |
| 4 | `GET` | `/transcript/{collection}/segments` | ⬜ Optional | Liệt kê transcript theo khoảng `sequence_id` (dựng lại biên bản / debug) |

### 3.2. Endpoint cần GỠ / GỘP so với bản cũ

| Endpoint cũ | Xử lý ở v2 | Lý do |
|---|---|---|
| `POST /transcript/{collection}/meeting/init` | **GỠ** | Counter Redis **lazy-init** (tự khởi tạo ở lần `/embed` đầu, tự rebuild từ Qdrant nếu Redis mất). Collection do app tạo sẵn thủ công. |
| `DELETE /transcript/{collection}/meeting/{meeting_id}` | **GỠ** | Dùng endpoint có sẵn `DELETE /embed/collections`. Counter Redis có TTL → tự dọn. |
| `PATCH /transcript/{collection}/context/{sequence_id}` | **GỠ** | LLM chạy **local cùng server** → ContextBuilder là background worker **in-process**, gọi thẳng `QdrantService`. Không cần HTTP. |
| `GET .../context/latest` + `GET .../context/{sequence_id}` | **GỘP** thành 1 | Một endpoint `GET /transcript/{collection}/context`, tham số `?sequence_id=` optional. |

**Kết quả: 8 endpoint → 4 endpoint** (2 core + 2 optional). Luồng tài liệu Giai đoạn 1 **không đổi**.

> **Vì sao vẫn tách `/query/transcript` riêng** thay vì dùng lại `/query/`? Vì response transcript khác hẳn (có `window` + `context`). Nhồi vào `/query/` sẽ làm endpoint trả hai hình dạng tùy tiền tố collection — khó dùng, dễ vỡ. Một endpoint query mới là **rõ ràng, không rườm rà**.

---

## 4. Quyết định Kiến trúc (đã chốt — v2)

| # | Vấn đề | Quyết định |
|---|---|---|
| D1 | `sequence_id` do ai gán? | **Server**, atomic INCR trên Redis, **theo từng collection**. Counter là cache tự lành — rebuild từ Qdrant nếu Redis mất. |
| D2 | Gửi câu lân cận khi query? | **Có.** Window `±N` câu (mặc định N=2, client truyền được, clamp theo max). |
| D3 | Build context kiểu gì? | LLM **tóm tắt**: `context[N] = LLM_summarize(context[N-1] + transcript[N-1])`. Không cộng dồn text thô. |
| D4 | Chunking transcript? | **Không.** Mỗi câu transcript = đúng 1 vector. |
| D5 | LLM build context đặt ở đâu? | **Self-host trên chính server VectorDB** — server này độc lập và **mạnh** (không phải thiết bị QCS8550), đủ sức chạy một LLM thực thụ. ContextBuilder là background worker **in-process**, gọi thẳng LLM self-host. Mặc định Ollama; cấu hình `LLM_MODEL` theo VRAM/CPU server thực tế. |
| D6 | Context lưu ở đâu? | **Trường `context` trong payload (metadata) của vector** trong collection `meeting-*`. (Yêu cầu trực tiếp của người dùng.) |
| D7 | Embedding transcript dùng model nào? | `intfloat/multilingual-e5-base` (768 chiều, Việt + Anh), chạy trên NPU Qualcomm AI080 (backend `qaic`). Câu transcript embed như **passage** (prefix `passage: `), query dùng prefix `query: `. |
| D8 | `meeting_id` lấy từ đâu? | **Suy ra từ tên collection**, không yêu cầu client gửi. Không dùng để lọc khi query. |
| D9 | Thứ tự build context | Build **tuần tự theo collection** (hàng đợi FIFO + worker) → đảm bảo `context[N-1]` sẵn sàng trước khi build `context[N]`. |

---

## 5. Mô hình dữ liệu

### 5.1. Collection `meeting-{uuid}` trên Qdrant
- Vector size **768**, distance **Cosine**.
- Point ID: UUID v4 tự sinh.
- App tạo collection thủ công trước khi embed (hoặc endpoint `/embed` lazy-create nếu chưa có — idempotent).

### 5.2. Payload (metadata) mỗi vector transcript

```jsonc
{
  "meeting_id":     "c7cfdf57-fbdc-48fe-a882-22af4ee817ad", // suy ra từ tên collection
  "sequence_id":    42,                      // SERVER gán, atomic, liên tục trong collection
  "speaker":        "Đoàn Sỹ Nguyên",        // client gửi
  "speaker_id":     "user_017",              // optional
  "text":           "Chúng ta cần xem lại ngân sách Q4...", // transcript gốc
  "timestamp":      "2026-05-11T19:52:27Z",  // client gửi, server tự điền nếu thiếu
  "context":        "Tóm tắt hội thoại tính tới câu này...", // ← context, do LLM sinh
  "context_status": "ready"                  // pending | processing | ready | failed | disabled
}
```

- `sequence_id` bắt đầu từ **1**, liên tục, duy nhất trong một collection.
- `context_status`: `pending` (vừa lưu) → `processing` (LLM đang chạy) → `ready` | `failed` (lỗi, đã fallback) | `disabled` (`LLM_PROVIDER=none`).
- `context` của câu N là **bối cảnh dẫn tới câu N** (chưa gồm chính câu N) — xem mục 8.

### 5.3. State trên Redis (chỉ 1 nhiệm vụ)
```
Key: rag:seq:{collection}   → INTEGER   # atomic counter, TTL 7 ngày, tự rebuild từ Qdrant nếu miss
```
Context **không** lưu ở Redis — nguồn sự thật là payload Qdrant (D6). Có thể cache `context` mới nhất ở Redis để tiết kiệm 1 truy vấn khi build (tùy chọn, không bắt buộc).

---

## 6. Kiến trúc & file

```
   ┌─────────────────────────────────────────────────────────────┐
   │  TẦNG THIẾT BỊ — Qualcomm QCS8550 (mỗi người dự họp 1 máy)    │
   │  Live transcript · nhận diện khuôn mặt · LLM nhỏ sinh câu     │
   │  trả lời cho người dùng                  [NGOÀI PHẠM VI PLAN] │
   └───────────────────────────┬─────────────────────────────────┘
                               │  HTTP (embed transcript / query)
                               ▼
┌───────────────────────────────────────────────────────────────┐
│   TẦNG SERVER — máy độc lập, mạnh  (PHẠM VI PLAN NÀY)           │
│                                                                 │
│  ┌──────────────────── FastAPI Application ──────────────────┐ │
│  │ ┌────────┐ ┌────────────────────┐ ┌──────────┐ ┌───────┐ │ │
│  │ │ /embed │ │ /query (docs +     │ │/transcript│ │/health│ │ │
│  │ │ (docs) │ │  /query/transcript)│ │  (MỚI)    │ │       │ │ │
│  │ └────────┘ └────────────────────┘ └─────┬─────┘ └───────┘ │ │
│  │                                          │                │ │
│  │      ┌───────────────────────────────────┴──────────┐     │ │
│  │      │         TranscriptService (MỚI)              │     │ │
│  │      │     SequenceManager · WindowFetcher          │     │ │
│  │      └──────────────────────────────────────────────┘     │ │
│  │                                                            │ │
│  │  ┌── Background Worker (in-process, FIFO theo collection)── │ │
│  │  │   ContextBuilder ──► LLMClient ──► LLM self-host        │ │
│  │  │                                                         │ │
│  │  ┌──────────────────────────────────────────┐             │ │
│  │  │  Services có sẵn: EmbeddingService,       │             │ │
│  │  │                   QdrantService          │             │ │
│  │  └──────────────────────────────────────────┘             │ │
│  └────────┬──────────────────┬──────────────┬────────────────┘ │
│           ▼                  ▼              ▼                   │
│    ┌────────────┐    ┌──────────────┐  ┌──────────────────┐    │
│    │   Redis    │    │    Qdrant    │  │  LLM self-host   │    │
│    │ seq counter│    │ meeting-*    │  │  (Ollama, model  │    │
│    └────────────┘    │ docs-*       │  │  lớn — server đủ │    │
│                      └──────────────┘  │  mạnh để chạy)   │    │
│                                         └──────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 6.1. File mới
```
app/routers/transcript.py        # 4 endpoint mục 3.1
app/schemas/transcript.py        # Pydantic schemas
app/services/transcript_service.py  # orchestration: embed, query, window, context-read
app/services/sequence_manager.py    # atomic sequence per collection (self-healing)
app/services/context_builder.py     # background worker build context
app/services/llm_client.py          # abstraction LLM (ollama/gemini/openai/none)
app/utils/redis_client.py           # singleton Redis async
app/workers/context_worker.py       # asyncio.Queue + worker draining build jobs
```

### 6.2. File cần sửa
```
app/main.py          # mount router transcript; startup: Redis, LLMClient, khởi động context worker; shutdown: đóng kết nối
app/config.py        # thêm config mục 9
app/dependencies.py  # DI: TranscriptService, RedisClient, LLMClient
app/routers/query.py # thêm POST /query/transcript
docker-compose.yml   # thêm service redis; ollama đặt trong profile optional
.env.example         # thêm biến mục 9
requirements.txt     # thêm: redis, httpx
```

---

## 7. Đặc tả Endpoint

### 7.1. `POST /transcript/{collection}/embed`  — CORE

Lưu **một câu** transcript. Server gán `sequence_id`, build context chạy nền.

**Validate:** `collection` phải bắt đầu bằng `meeting-` (sai → `400`); `text` không rỗng (rỗng → `422`).

**Request body:**
```json
{
  "speaker":    "Đoàn Sỹ Nguyên",
  "speaker_id": "user_017",
  "text":       "Chúng ta cần xem lại ngân sách Q4 trước khi chốt.",
  "timestamp":  "2026-05-11T19:52:27Z"
}
```
`speaker_id`, `timestamp` optional. **Không có** `meeting_id` (suy ra từ collection).

**Xử lý (đồng bộ):**
1. Validate.
2. Lazy-create collection nếu chưa tồn tại (size 768, Cosine).
3. `SequenceManager.next(collection)` → `sequence_id`.
4. `EmbeddingService.embed_texts([text])` (passage) → vector 768.
5. `QdrantService.upsert` 1 point, payload `context_status="pending"` (hoặc `"disabled"` nếu `LLM_PROVIDER=none`).
6. Đẩy job `(collection, sequence_id)` vào hàng đợi context worker.
7. Trả response ngay.

**Response `202`:**
```json
{
  "meeting_id":     "c7cfdf57-fbdc-48fe-a882-22af4ee817ad",
  "sequence_id":    42,
  "point_id":       "a3f1...uuid",
  "context_status": "pending"
}
```

---

### 7.2. `POST /query/transcript`  — CORE

Truy vấn transcript, trả kết quả kèm **window** + **context**.

**Request body:**
```json
{
  "collection":      "meeting-c7cfdf57-fbdc-48fe-a882-22af4ee817ad",
  "query":           "Quyết định về ngân sách Q4 là gì?",
  "top_k":           3,
  "window_size":     2,
  "score_threshold": 0.5,
  "speaker_filter":  null,
  "include_context": true
}
```
`window_size=0` → tắt window. `window_size` vượt `TRANSCRIPT_MAX_WINDOW_SIZE` → clamp.

**Xử lý:**
1. Validate tiền tố `meeting-`.
2. Embed `query` → vector; Qdrant search Cosine, lọc `score_threshold` (+ `speaker_filter` nếu có).
3. Lấy `top_k` kết quả.
4. Mỗi kết quả (seq = S): `WindowFetcher` scroll Qdrant theo filter `sequence_id ∈ [S-w, S+w]`, tách `before`/`after`, sắp xếp tăng dần.

**Response `200`:**
```json
{
  "query": "Quyết định về ngân sách Q4 là gì?",
  "count": 1,
  "results": [
    {
      "sequence_id": 42,
      "speaker":     "Đoàn Sỹ Nguyên",
      "timestamp":   "2026-05-11T19:52:27Z",
      "text":        "Chúng ta cần xem lại ngân sách Q4 trước khi chốt.",
      "score":       0.87,
      "context":     "Cuộc họp đang bàn kế hoạch tài chính Q4; các bên đã đồng ý cắt 15% chi phí vận hành.",
      "context_status": "ready",
      "window": {
        "before": [
          { "sequence_id": 40, "speaker": "Mai Xuân Ngọc", "text": "..." },
          { "sequence_id": 41, "speaker": "Đoàn Sỹ Nguyên", "text": "..." }
        ],
        "after": [
          { "sequence_id": 43, "speaker": "Mai Xuân Ngọc", "text": "..." },
          { "sequence_id": 44, "speaker": "Đoàn Sỹ Nguyên", "text": "..." }
        ]
      }
    }
  ]
}
```

App ghép `context` + `window` + `text` → prompt cho LLM phía workstation.

---

### 7.3. `GET /transcript/{collection}/context`  — OPTIONAL

Lấy context (tóm tắt cuộc họp). Mặc định trả context **mới nhất**; có `?sequence_id=N` thì trả context tại câu N.

**Response `200`:**
```json
{
  "meeting_id":     "c7cfdf57-fbdc-48fe-a882-22af4ee817ad",
  "sequence_id":    41,
  "context":        "Tóm tắt hội thoại tính tới câu 41...",
  "context_status": "ready"
}
```

---

### 7.4. `GET /transcript/{collection}/segments`  — OPTIONAL

Liệt kê transcript theo khoảng. Query params: `from_seq` (default 1), `to_seq` (default mới nhất), `limit` (default 100). Trả danh sách sắp xếp theo `sequence_id`.

---

## 8. Luồng xử lý

### 8.1. Ingest transcript
```
App gửi câu transcript
   └► POST /transcript/{collection}/embed
        ├ validate (tiền tố meeting-, text)
        ├ lazy-create collection nếu thiếu
        ├ SequenceManager.next(collection) ──INCR Redis──► sequence_id = N
        ├ EmbeddingService.embed_texts([text]) ──► vector[768]
        ├ QdrantService.upsert(point, context_status="pending")
        ├ enqueue job (collection, N) vào context worker
        └► 202 { sequence_id: N, context_status: "pending" }
```

### 8.2. Build context (background worker, in-process)
```
Context Worker lấy job (collection, N) từ hàng đợi FIFO
   │
   ├ Nếu LLM_PROVIDER=none → set point[N].context_status="disabled", bỏ qua
   │
   ├ Nếu N == 1 → context[1] = ""  → point[1].context_status="ready" (không gọi LLM)
   │
   ├ Đọc point[N-1] từ Qdrant → lấy context[N-1] và text[N-1]
   ├ Set point[N].context_status = "processing"
   ├ LLMClient.summarize(prev_context=context[N-1], new_utterance=text[N-1]) → context[N]
   ├ QdrantService set payload point[N]: context=context[N], context_status="ready"
   │
   └ Nếu LLM lỗi → retry tối đa CONTEXT_MAX_RETRY lần
         vẫn lỗi → fallback context[N] = context[N-1], status="failed"
```
Hàng đợi FIFO + worker đảm bảo `context[N-1]` đã `ready`/`failed` trước khi build `context[N]` (D9).

**Định nghĩa context (làm rõ):** `context[N] = LLM_summarize(context[N-1] + transcript[N-1])`. Context tại câu N là **bối cảnh DẪN TỚI câu N**, chưa gồm chính câu N. Khi query trả câu N, app dùng `context` (bối cảnh trước) + `text` câu N + `window` để đưa cho LLM.

### 8.3. Query transcript
```
App gửi câu hỏi → POST /query/transcript
   ├ embed(query) → vector
   ├ Qdrant search trong collection (score_threshold, speaker_filter) → top_k
   ├ mỗi match (seq S): WindowFetcher scroll sequence_id ∈ [S-w, S+w]
   └► results[] kèm context + window
        └► App ghép prompt → LLM workstation → câu trả lời
```

---

## 9. Cấu hình `.env`

```ini
# === Redis (sequence counter) ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
SEQ_KEY_TTL_SECONDS=604800        # TTL counter (7 ngày), refresh mỗi lần embed

# === Transcript ===
TRANSCRIPT_COLLECTION_PREFIX=meeting-   # tiền tố collection transcript
DOCS_COLLECTION_PREFIX=docs-            # tiền tố collection tài liệu
TRANSCRIPT_SEQ_START=1
TRANSCRIPT_WINDOW_SIZE=2                # ±N câu lân cận mặc định
TRANSCRIPT_MAX_WINDOW_SIZE=5            # giới hạn window_size client gửi

# === Context Builder LLM (self-host trên server VectorDB — server mạnh, độc lập) ===
LLM_PROVIDER=ollama               # ollama | gemini | openai | none
LLM_MODEL=qwen2.5:7b              # chọn theo VRAM/CPU server thực tế — có thể dùng model lớn hơn
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=                      # để trống nếu dùng ollama
CONTEXT_MAX_TOKENS=800            # độ dài tối đa context sau tóm tắt
CONTEXT_MAX_RETRY=2
CONTEXT_TIMEOUT_SECONDS=30
```

---

## 10. Prompt mẫu cho ContextBuilder

**System:**
```
Bạn là trợ lý tóm tắt hội thoại cuộc họp. Khi có một câu nói mới, hãy cập nhật
bản tóm tắt bối cảnh. Bản tóm tắt phải: ngắn gọn (tối đa ~{CONTEXT_MAX_TOKENS} tokens);
giữ các quyết định, con số, tên riêng, chủ đề đang bàn; viết bằng tiếng Việt,
trung lập, không bịa. Chỉ trả về bản tóm tắt, không thêm lời dẫn.
```

**User:**
```
[Bối cảnh hiện tại]
{previous_context}

[Câu nói mới cần tích hợp]
{new_utterance}

Hãy cập nhật bản tóm tắt bối cảnh.
```

---

## 11. Danh sách công việc (refactor + xây mới)

> Làm tuần tự. Nếu hệ thống đã code theo bản cũ, Task 0 thực hiện trước.

**Task 0 — Gỡ thiết kế cũ:** xóa endpoint `meeting/init`, `DELETE meeting`, `PATCH context`; gộp 2 endpoint context. Cập nhật router/test tương ứng.

**Task 1 — Redis & config:** service `redis` trong compose; `redis_client.py` (async singleton); mở rộng `config.py`; cập nhật `requirements.txt`.

**Task 2 — Schemas:** `transcript.py` — `TranscriptEmbedRequest/Response`, `TranscriptQueryRequest`, `TranscriptQueryResult`, `WindowResult`, `ContextResponse`, `SegmentListResponse`.

**Task 3 — SequenceManager:** `next/current/rebuild` theo collection; lazy-init + rebuild từ Qdrant (scroll `order_by sequence_id desc limit 1`) khi Redis miss; đặt TTL. Unit test: 100 lần `next` đồng thời → 100 giá trị duy nhất liên tục; xóa key Redis rồi `next` → rebuild đúng.

**Task 4 — LLMClient:** interface `summarize(prev_context, new_utterance) -> str`; provider `ollama` (gọi `LLM_BASE_URL`), `gemini`, `openai`, `none`; timeout + retry + cắt độ dài.

**Task 5 — ContextBuilder + Worker:** `asyncio.Queue` + worker FIFO khởi động lúc startup; `build()` theo luồng 8.2; xử lý câu đầu, fallback khi lỗi; ghi payload Qdrant.

**Task 6 — WindowFetcher:** scroll Qdrant theo filter range `sequence_id`; tách before/after; clamp `window_size`.

**Task 7 — TranscriptService:** gom `embed_transcript`, `query_transcript`, `get_context`, `list_segments`; suy `meeting_id` từ tên collection; validate tiền tố.

**Task 8 — Router transcript:** hiện thực 4 endpoint mục 3.1; enqueue job context trong `/embed`.

**Task 9 — Query transcript:** thêm `POST /query/transcript` (mục 7.2).

**Task 10 — Wiring:** DI trong `dependencies.py`; `main.py` startup/shutdown (Redis, LLMClient, context worker).

**Task 11 — Deployment:** `docker-compose.yml` (redis; ollama trong profile optional); `.env.example`.

**Task 12 — Test & tài liệu:** cập nhật README đồng bộ v2; integration test: embed 5 câu → kiểm tra sequence liên tục + context build → query có window.

---

## 12. Xử lý lỗi & trường hợp biên

| Tình huống | Xử lý |
|---|---|
| Collection sai tiền tố (`docs-...` gọi vào `/transcript`) | `400 Bad Request` |
| `text` rỗng / chỉ khoảng trắng | `422`, không tạo vector |
| Câu đầu tiên (seq = 1) | `context = ""`, không gọi LLM, status `ready` |
| `LLM_PROVIDER=none` | `context = ""`, status `disabled` |
| LLM build context lỗi/timeout | Retry `CONTEXT_MAX_RETRY`; vẫn lỗi → fallback `context[N]=context[N-1]`, status `failed` |
| Query khi context chưa `ready` | Vẫn trả kết quả; `context` lấy giá trị hiện có kèm `context_status` để app tự quyết |
| `window_size` > max | Clamp về `TRANSCRIPT_MAX_WINDOW_SIZE` |
| Câu ở đầu/cuối cuộc họp | Window thiếu before/after → trả mảng rỗng |
| Redis mất kết nối khi `/embed` | `503`; **không** để client tự gán sequence (giữ D1) |
| Redis bị flush giữa cuộc họp | `SequenceManager` rebuild counter từ Qdrant (max `sequence_id`) → không trùng |
| Xóa cuộc họp | Dùng `DELETE /embed/collections`; counter Redis tự hết hạn theo TTL |

---

## 13. Tiêu chí hoàn thành (DoD)

- [ ] Endpoint cũ (`meeting/init`, `DELETE meeting`, `PATCH context`) đã được gỡ; còn đúng 4 endpoint mới.
- [ ] Embed tuần tự ≥ 10 câu vào `meeting-{uuid}` → `sequence_id` liên tục 1..10, không trùng; mỗi câu = đúng 1 vector.
- [ ] Sau vài giây, `context_status` chuyển `ready`; trường `context` nằm trong payload vector và phản ánh nội dung câu 1..N-1.
- [ ] `POST /query/transcript` trả `top_k` kết quả kèm `context` + `window` (±N câu) đúng thứ tự.
- [ ] Gọi `/transcript/*` với collection `docs-*` → `400`.
- [ ] Luồng tài liệu Giai đoạn 1 (`/embed/*`, `/query/`) hoạt động nguyên vẹn.
- [ ] Xóa key Redis giữa chừng → embed tiếp vẫn cho `sequence_id` đúng (rebuild từ Qdrant).
- [ ] `docker compose up --build` chạy full stack: qdrant + redis + rag-api.
- [ ] `/docs` hiển thị đủ 4 endpoint mới.

---

## 14. Câu hỏi mở

1. **Model LLM self-host trên server build-context:** server VectorDB là máy mạnh độc lập (không phải QCS8550) → có thể chạy model lớn. Cần chốt `LLM_MODEL` mặc định theo cấu hình server thực tế (VRAM, có GPU không) và yêu cầu chất lượng tóm tắt tiếng Việt.
2. **Context câu đầu:** để rỗng, hay seed bằng tóm tắt tài liệu trong `docs-{uuid}` tương ứng (nếu có)? → hiện plan để rỗng.
3. **Độ trễ build context:** với tốc độ nói thực tế, một câu mất ~? giây để build xong — có cần giới hạn tần suất gọi LLM (gộp batch nhiều câu) không? Server mạnh nên độ trễ thấp, nhưng vẫn cần đo thực tế.
4. **TTL counter 7 ngày:** đủ cho cuộc họp dài nhất chưa? Nếu họp gián đoạn > 7 ngày, cơ chế rebuild-từ-Qdrant vẫn xử lý đúng.
5. **Triển khai LLM server:** Ollama cài trực tiếp trên server hay chạy trong container cùng stack? → ảnh hưởng `LLM_BASE_URL` và `docker-compose.yml`.

---

*Hết tài liệu v2. Coding agent triển khai theo Task 0 → 12; mọi sai khác so với mục 4 (Design Decisions) phải xác nhận lại với người yêu cầu.*