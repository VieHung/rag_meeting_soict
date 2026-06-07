# Hướng dẫn sử dụng API — RAG Vector Store

> Dành cho đội phát triển app phía thiết bị (QCS8550 / workstation).
> Base URL: `http://<server-ip>:8000` (dev native) hoặc **`http://<server-ip>:18000`** (Docker —
> host `18000` map vào container `1904`). Embedding chạy trên **NPU Qualcomm AI080** (backend `qaic`),
> model `intfloat/multilingual-e5-base` (768 chiều). Trường `context` chỉ có khi server bật LLM
> build-context — hiện `LLM_PROVIDER=none` nên `context_status = "disabled"` và `context` rỗng.

---

## Mục lục

1. [Quy ước chung](#1-quy-ước-chung)
2. [Phase 1 — Tài liệu](#2-phase-1--tài-liệu)
3. [Phase 2 — Transcript cuộc họp](#3-phase-2--transcript-cuộc-họp)
4. [Luồng nghiệp vụ điển hình](#4-luồng-nghiệp-vụ-điển-hình)
5. [Xử lý lỗi](#5-xử-lý-lỗi)

---

## 1. Quy ước chung

### Collection naming

Mỗi cuộc họp = một **cặp collection** trên Qdrant, do **app tạo thủ công**:

| Loại | Tiền tố | Ví dụ |
|------|---------|-------|
| Transcript | `meeting-{uuid}` | `meeting-c7cfdf57-...` |
| Tài liệu đính kèm | `docs-{uuid}` | `docs-c7cfdf57-...` |

`meeting_id` = `{uuid}` (giống nhau trong cặp), server suy ra từ tên collection.

**App phải tạo collection trước** qua `POST /embed/collections` trước khi gọi các endpoint.

### Meeting ID

Không gửi `meeting_id` trong body. Server tự suy từ collection name:
```
collection = "meeting-a1b2c3d4"  →  meeting_id = "a1b2c3d4"
```

### Content-Type

Tất cả request body đều là `application/json`.

### HTTP Status

| Code | Ý nghĩa |
|------|---------|
| `200` | Thành công |
| `202` | Đã nhận, đang xử lý nền |
| `400` | Lỗi validation (sai prefix, thiếu field) |
| `404` | Không tìm thấy |
| `422` | Unprocessable entity (text rỗng, query rỗng) |
| `500` | Lỗi server (vd Qdrant "Too many open files" — xem mục 5) |
| `503` | Service unavailable (Redis mất kết nối) |

---

## 2. Phase 1 — Tài liệu

> Giữ nguyên từ Giai đoạn 1. Dùng cho upload & truy vấn tài liệu cuộc họp.

### 2.1. Tạo collection mới

```bash
POST /embed/collections
Content-Type: multipart/form-data

name=meeting-a1b2c3d4
```

**Response `200`:**
```json
{
  "success": true,
  "message": "Created collection 'meeting-a1b2c3d4'"
}
```

### 2.2. Liệt kê tất cả collections

```bash
GET /embed/collections
```

**Response:**
```json
{
  "collections": ["documents", "meeting-a1b2c3d4", "docs-a1b2c3d4"]
}
```

### 2.3. Xoá collection

```bash
DELETE /embed/collections
Content-Type: multipart/form-data

name=meeting-a1b2c3d4
```

Dùng để xoá toàn bộ dữ liệu một cuộc họp (cả transcript lẫn tài liệu).

### 2.4. Upload file tài liệu

```bash
POST /embed/file
Content-Type: multipart/form-data

file=@report.pdf
collection=docs-a1b2c3d4
doc_id=  (optional, UUID tự sinh nếu để trống)
extra_metadata={"author": "Navis"}  (optional, JSON string)
```

**Response `200` (xử lý nền):**
```json
{
  "success": true,
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "source": "report.pdf",
  "chunks_created": 0,
  "message": "Đã nhận file 'report.pdf', embedding sẽ chạy nền ngay sau response"
}
```

Hỗ trợ: `.pdf`, `.docx`, `.txt`, `.md`. Tối đa 50MB.

### 2.5. Embed text

```bash
POST /embed/text
Content-Type: application/json

{
  "text": "Nội dung tài liệu...",
  "source": "my_note",
  "collection": "docs-a1b2c3d4",
  "doc_id": "custom-uuid",     // optional
  "metadata": {"author": "AI"} // optional
}
```

### 2.6. Truy vấn tài liệu

```bash
POST /query/
Content-Type: application/json

{
  "query": "nội dung cần tìm",
  "collection": "docs-a1b2c3d4",
  "top_k": 5,
  "score_threshold": 0.3,
  "source_filter": "report.pdf"    // optional, lọc theo file
}
```

**Response:**
```json
{
  "query": "nội dung cần tìm",
  "results": [
    {
      "text": "Đoạn văn khớp nhất...",
      "score": 0.8741,
      "source": "report.pdf",
      "doc_id": "550e8400-...",
      "chunk_index": 3,
      "chunk_total": 24
    }
  ],
  "total_found": 3
}
```

### 2.7. Xoá tài liệu

```bash
# Xoá theo tên file
DELETE /embed/{collection}/source/{filename}

# Xoá theo doc_id
DELETE /embed/{collection}/doc/{doc_id}
```

### 2.8. Liệt kê tài liệu trong collection

```bash
GET /embed/{collection}/documents
```

---

## 3. Phase 2 — Transcript cuộc họp

> 4 endpoint, trong đó **2 core** (bắt buộc dùng) và **2 optional**.

### 3.1. Core: Nhập transcript

**Endpoint:** `POST /transcript/{collection}/embed`

Gửi **một câu** transcript mỗi lần. Server tự gán `sequence_id`, lưu vector, và chạy build context nền.

**Request:**
```json
{
  "speaker": "Đoàn Sỹ Nguyên",
  "speaker_id": "user_017",
  "text": "Chúng ta cần xem lại ngân sách Q4 trước khi chốt.",
  "timestamp": "2026-05-11T19:52:27Z",
  "lang": "vi"
}
```

| Field | Type | Required | Ghi chú |
|-------|------|----------|---------|
| `speaker` | string | ✅ | Tên người nói |
| `speaker_id` | string | ❌ | ID định danh |
| `text` | string | ✅ | Nội dung câu nói, không rỗng |
| `timestamp` | ISO 8601 | ❌ | Nếu thiếu, server tự gán `now()` |
| `lang` | string | ❌ | Mã ngôn ngữ (vi, en...) |

**Response `202` (xử lý nền):**
```json
{
  "meeting_id": "a1b2c3d4",
  "sequence_id": 42,
  "point_id": "f6197721-bf9d-4dd0-a537-7268fb1fe357",
  "context_status": "pending"
}
```

| Field | Type | Ghi chú |
|-------|------|---------|
| `meeting_id` | string | Suy ra từ collection name |
| `sequence_id` | int | Server gán, bắt đầu từ 1, liên tục, không trùng |
| `point_id` | UUID | ID của vector trong Qdrant |
| `context_status` | enum | `pending` / `processing` / `ready` / `failed` / `disabled` |

**Quan trọng:**
- Context build chạy nền (bất đồng bộ). `context_status = "pending"` ngay sau response.
- Sau ~3-10 giây (tuỳ tốc độ LLM), status chuyển `ready` và context có thể query được.
- `sequence_id` bắt đầu từ 1, tăng dần, không trùng ngay cả khi Redis bị xoá (tự rebuild từ Qdrant).

**Edge cases:**
- Collection sai prefix (`docs-*`) → `400 Bad Request`
- `text` rỗng → `422 Unprocessable Entity`
- Redis mất kết nối → `503 Service Unavailable`

---

### 3.2. Core: Truy vấn transcript

**Endpoint:** `POST /query/transcript`

Truy vấn ngữ nghĩa trên transcript, trả kết quả kèm **window** (câu lân cận) + **context** (tóm tắt bối cảnh).

**Request:**
```json
{
  "collection": "meeting-a1b2c3d4",
  "query": "Quyết định về ngân sách Q4 là gì?",
  "meeting_id": null,
  "top_k": 3,
  "window_size": 2,
  "score_threshold": 0.5,
  "speaker_filter": null,
  "speaker_id_filter": null,
  "include_context": true
}
```

| Field | Type | Default | Ghi chú |
|-------|------|---------|---------|
| `collection` | string | — | Bắt buộc, phải tiền tố `meeting-` |
| `query` | string | — | Câu hỏi, không rỗng |
| `meeting_id` | string | null | (KHÔNG DÙNG — để null, server tự suy) |
| `top_k` | int | 3 | Số kết quả (1–50) |
| `window_size` | int | 2 | ±N câu lân cận. `0` = tắt. Bị clamp theo `TRANSCRIPT_MAX_WINDOW_SIZE` |
| `score_threshold` | float | 0.0 | Chỉ trả kết quả có score >= ngưỡng |
| `speaker_filter` | string | null | Lọc theo tên speaker (exact match) |
| `speaker_id_filter` | string | null | Lọc theo speaker_id |
| `include_context` | bool | true | `false` để bỏ qua context (tiết kiệm bandwidth) |

**Response `200`:**
```json
{
  "query": "Quyết định về ngân sách Q4 là gì?",
  "results": [
    {
      "sequence_id": 42,
      "speaker": "Đoàn Sỹ Nguyên",
      "speaker_id": "user_017",
      "timestamp": "2026-05-11T19:52:27Z",
      "text": "Chúng ta cần xem lại ngân sách Q4 trước khi chốt.",
      "score": 0.87,
      "meeting_id": "a1b2c3d4",
      "context": "Đã thống nhất cắt 15% chi phí vận hành Q4, giao phòng tài chính lên kế hoạch.",
      "context_status": "ready",
      "window": {
        "before": [
          { "sequence_id": 40, "speaker": "Mai Xuân Ngọc", "text": "..." },
          { "sequence_id": 41, "speaker": "Đoàn Sỹ Nguyên", "text": "..." }
        ],
        "after": [
          { "sequence_id": 43, "speaker": "Mai Xuân Ngọc", "text": "..." }
        ]
      }
    }
  ],
  "count": 1
}
```

**Cách app ghép prompt cho LLM phía workstation:**

Mỗi result có sẵn:
- `context` — bối cảnh dẫn tới câu này (do LLM server tóm tắt)
- `text` — chính câu transcript khớp
- `window.before` — N câu trước
- `window.after` — N câu sau

Gợi ý prompt template:
```
Bối cảnh cuộc họp: {context}

Đoạn hội thoại liên quan:
{các câu trong window.before + text + window.after}

Câu hỏi: {query}
Trả lời:
```

---

### 3.3. Optional: Lấy context

**Endpoint:** `GET /transcript/{collection}/context`

Lấy context (tóm tắt bối cảnh) của cuộc họp.

```bash
# Context mới nhất
GET /transcript/meeting-a1b2c3d4/context

# Context tại thời điểm cụ thể
GET /transcript/meeting-a1b2c3d4/context?sequence_id=10
```

**Response `200`:**
```json
{
  "meeting_id": "a1b2c3d4",
  "sequence_id": 10,
  "context": "Đã thống nhất cắt 15% chi phí vận hành Q4, giao phòng tài chính lên kế hoạch.",
  "context_status": "ready",
  "context_seq_base": 9
}
```

| Field | Ghi chú |
|-------|---------|
| `context` | Bản tóm tắt bối cảnh tại câu `sequence_id`. Rỗng nếu là câu đầu tiên hoặc context chưa build xong |
| `context_seq_base` | `sequence_id` của câu cuối cùng được tính vào context này (thường = `sequence_id - 1`) |

**Công thức:** `context[N] = LLM_summarize(context[N-1] + text[N-1])`
→ Context tại câu N là **bối cảnh dẫn tới câu N**, chưa gồm chính câu N.

---

### 3.4. Optional: Liệt kê transcript

**Endpoint:** `GET /transcript/{collection}/segments`

Dùng để debug, dựng lại biên bản, hoặc đồng bộ state.

```
GET /transcript/meeting-a1b2c3d4/segments?from_seq=1&to_seq=50&limit=100
```

| Param | Type | Default | Ghi chú |
|-------|------|---------|---------|
| `from_seq` | int | 1 | Bắt đầu từ |
| `to_seq` | int | (mới nhất) | Kết thúc tại |
| `limit` | int | 100 | Tối đa (cap 1000) |

**Response:**
```json
{
  "meeting_id": "a1b2c3d4",
  "collection": "meeting-a1b2c3d4",
  "from_seq": 1,
  "to_seq": 3,
  "count": 3,
  "segments": [
    {
      "sequence_id": 1,
      "speaker": "Đoàn Sỹ Nguyên",
      "speaker_id": null,
      "timestamp": "2026-05-20T04:13:25Z",
      "text": "Chúng ta cần xem lại ngân sách Q4...",
      "context_status": "ready"
    },
    { "...": "..." }
  ]
}
```

---

## 4. Luồng nghiệp vụ điển hình

### 4.1. Khởi tạo cuộc họp mới

App sinh UUID, tạo 2 collection:

```bash
UUID="meeting-$(uuidgen)"  # ví dụ: meeting-c7cfdf57-...

# Tạo collection transcript
curl -X POST http://localhost:8000/embed/collections -F "name=$UUID"

# Tạo collection tài liệu (cùng UUID, tiền tố docs-)
curl -X POST http://localhost:8000/embed/collections -F "name=docs-${UUID#meeting-}"
```

(Không cần gọi `meeting/init` — sequence counter tự khởi tạo ở lần embed đầu tiên.)

### 4.2. Trong cuộc họp (real-time)

Mỗi khi có câu transcript hoàn chỉnh, app gửi ngay:

```bash
curl -X POST http://localhost:8000/transcript/meeting-c7cfdf57-.../embed \
  -H "Content-Type: application/json" \
  -d '{
    "speaker": "Đoàn Sỹ Nguyên",
    "text": "Tôi đề xuất cắt 15% chi phí vận hành."
  }'
# → 202 { "sequence_id": 42, "context_status": "pending" }
```

Có thể gửi liên tục, không cần chờ response trước.

### 4.3. User đặt câu hỏi (real-time)

```bash
curl -X POST http://localhost:8000/query/transcript \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "meeting-c7cfdf57-...",
    "query": "Quyết định về ngân sách Q4 là gì?",
    "top_k": 3,
    "window_size": 2
  }'
```

Kết quả trả về gồm `context` + `window` + `text`. App ghép thành prompt → LLM workstation → câu trả lời.

### 4.4. Kết thúc cuộc họp

```bash
# Xoá cả transcript và tài liệu
curl -X DELETE http://localhost:8000/embed/collections -F "name=meeting-c7cfdf57-..."
curl -X DELETE http://localhost:8000/embed/collections -F "name=docs-c7cfdf57-..."
```

Counter Redis tự hết hạn sau 7 ngày TTL.

---

## 5. Xử lý lỗi

### HTTP Status code mapping

| Code | Nguyên nhân | Cách xử lý |
|------|-------------|------------|
| `400` | Sai collection prefix (gọi `docs-*` vào `/transcript`) | Kiểm tra collection name |
| `422` | `text` / `query` rỗng | Validate trước khi gửi |
| `404` | Context/collection không tồn tại | Kiểm tra `sequence_id` hợp lệ |
| `500` | Qdrant "Too many open files" (fd `nofile` của container Qdrant cạn) | Lỗi phía hạ tầng, không phải client. Báo đội vận hành nâng `ulimits.nofile` cho container Qdrant rồi recreate; client retry sau |
| `503` | Redis không kết nối được | Thử lại sau, kiểm tra Redis service |

> **Lỗi `500` "Too many open files"** không liên quan dữ liệu client gửi mà do container Qdrant
> chạy với `nofile` soft = 1024 (mặc định Docker) trong khi mỗi cuộc họp tạo một collection riêng.
> Khắc phục ở phía server (xem `rag_server/README.md` → "Vận hành & sự cố thường gặp"). Câu transcript
> bị rớt do lỗi này KHÔNG được cấp `sequence_id` → client nên retry câu đó.

### Context status

Khi query, mỗi result có `context_status`:

| Status | Ý nghĩa | App xử lý |
|--------|---------|-----------|
| `pending` | Vừa embed, chưa build context | Dùng `context` hiện có (có thể rỗng) hoặc đợi |
| `processing` | LLM đang build | Đợi 3-5s rồi query lại |
| `ready` | Context đã sẵn sàng | Dùng được |
| `failed` | LLM lỗi sau retry | Fallback: context là bản tóm tắt của câu trước |
| `disabled` | `LLM_PROVIDER=none` (tắt build context) | Không có context, dùng window + text thuần |

### Lưu ý quan trọng

1. **Sequence ID:** Server đảm bảo liên tục và không trùng. App không can thiệp.
2. **Meeting ID:** Không gửi trong body. Server suy từ collection name.
3. **Context build chậm hơn embed:** Embed trả `202` ngay, context cần ~3-10 giây. Nếu query ngay sau embed, `context_status` có thể là `pending`. Nên đợi hoặc retry.
4. **Window ở đầu/cuối cuộc họp:** `window.before` hoặc `window.after` sẽ là mảng rỗng, không phải lỗi.
5. **Window size:** Server tự clamp về `TRANSCRIPT_MAX_WINDOW_SIZE` (mặc định 5) nếu client gửi quá lớn.
