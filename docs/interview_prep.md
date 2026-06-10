# BKMEETING RAG — Tài liệu logic hệ thống & 30 câu hỏi phỏng vấn

> Tài liệu ôn phỏng vấn vị trí **AI Engineer**. Phần A tóm tắt toàn bộ logic hệ thống RAG;
> Phần B là 30 câu hỏi (dễ → khó) kèm đáp án mẫu, bám sát chính repo `rag_base`.

---

# PHẦN A — TOÀN BỘ LOGIC HỆ THỐNG

## A.0. Bài toán & kiến trúc 2 tầng

**BKMEETING** = phòng họp thông minh + thư ký ảo (SoICT / NAVIS Center, ĐHBK Hà Nội). Hệ chạy trên **2 tầng phần cứng độc lập**:

| Tầng | Phần cứng | Trách nhiệm |
|------|-----------|-------------|
| **Thiết bị** | Qualcomm QCS8550 (mỗi người 1 máy) | Live transcript, face ID, dịch, **LLM nhỏ on-device sinh câu trả lời cuối** cho người dùng. *Ngoài phạm vi repo.* |
| **Server** | Host mạnh, độc lập | **Repo này**: RAG API + Qdrant + Redis + **LLM self-host** build context. |

**Triết lý cốt lõi:** server là **nguồn tri thức (knowledge source)**, *không* phải bộ sinh câu trả lời. Thiết bị gửi câu hỏi → server trả về phần liên quan nhất (`context + window + text`) → **LLM trên thiết bị** mới soạn câu trả lời cho người dùng. Việc tách vai trò này là quyết định kiến trúc nền tảng.

Có **2 luồng RAG độc lập**, app tự chọn dùng luồng nào, phân biệt bằng **prefix tên collection**:
1. **Document retrieval** (Phase 1) — tài liệu họp upload trước. Namespace `docs-*` / `/embed/*`, `/query/`.
2. **Transcript retrieval** (Phase 2) — hội thoại đã transcribe trong lúc họp. Namespace `meeting-*` / `/transcript/*`, `/query/transcript`.

## A.1. Stack & thành phần

```
FastAPI (app/main.py)
 ├─ Routers:  embed.py · query.py · transcript.py
 ├─ Services:
 │   ├─ EmbeddingService   — SentenceTransformer paraphrase-multilingual-MiniLM-L12-v2 (384-d, chuẩn hóa), singleton
 │   ├─ QdrantService      — Phase 1 (documents)
 │   ├─ TranscriptStore    — Phase 2 (meeting-* collections), payload index meeting_id/sequence_id/speaker
 │   ├─ TranscriptService  — orchestrator: embed / query+window / get_context / list_segments
 │   ├─ SequenceManager    — atomic sequence_id qua Redis INCR
 │   ├─ ContextBuilder     — rolling summary bằng LLM
 │   ├─ LLMClient          — ollama / openai-compatible / gemini / none
 │   ├─ retrieval.fuse()   — hybrid BM25 + vector (gated, off)
 │   └─ Reranker           — cross-encoder (none/local/http, gated, off)
 └─ Worker:
     └─ ContextWorker      — asyncio.Queue 1 consumer FIFO
Hạ tầng: Qdrant (vectors) · Redis (counter + cache) · LLM self-host (build context)
```

## A.2. Mô hình dữ liệu

**Document vector** (`docs-*`): `text`, `source`, `doc_id`, `chunk_index`, `chunk_total`, `file_size`, `mime_type`.

**Transcript vector** (`meeting-*`): mỗi câu nói = **đúng 1 vector** (không chunk):
```jsonc
{
  "meeting_id": "...",        // suy từ tên collection, client KHÔNG gửi
  "sequence_id": 42,          // server gán, atomic, liên tục
  "speaker": "...", "speaker_id": "user_017",
  "text": "...",
  "timestamp": "2026-05-11T19:52:27Z",
  "context": "Bản tóm tắt cuộn tới câu này...",   // LLM sinh
  "context_status": "ready",  // pending|processing|ready|failed|disabled
  "context_seq_base": 41
}
```
- Vector **384-d**, distance **Cosine**.
- **`context[N]` = bối cảnh DẪN TỚI câu N** (không gồm chính câu N).

**Redis state:**
```
rag:seq:{collection}            → INTEGER (counter atomic, TTL 7 ngày, rebuild được từ Qdrant)
meeting:{meeting_id}:latest_ctx → JSON (cache context mới nhất)
```

## A.3. Ba luồng xử lý chính

**(1) Ingest transcript** (`POST /transcript/{collection}/embed`) — *đồng bộ + enqueue*:
```
validate prefix meeting- (else 400), text non-empty (else 422)
→ ensure_collection (lazy create)
→ SequenceManager.next() = Redis INCR → sequence_id = N
→ EmbeddingService.embed_query(text) → vector[384]
→ upsert_point(context_status = "pending" | "disabled" nếu LLM=none)
→ ContextWorker.enqueue(collection, meeting_id, N)   (FIFO)
→ 202 { meeting_id, sequence_id, point_id, context_status }
```

**(2) Build context** (ContextWorker, nền, FIFO):
```
drain job (collection, meeting_id, N):
  LLM_PROVIDER=none → "disabled", dừng
  N == 1            → context="", "ready", KHÔNG gọi LLM
  đọc câu N-1 → (context[N-1], text[N-1])
  mark N "processing"
  new = LLM.summarize(context[N-1], text[N-1])   (retry CONTEXT_MAX_RETRY, backoff)
  thành công → context[N]=new, "ready", context_seq_base=N-1
  thất bại   → context[N]=context[N-1], "failed"
```
Single-consumer đảm bảo `context[N-1]` xong **trước** khi build `context[N]` (D9).

**(3) Query transcript** (`POST /query/transcript`) — *vector + window + hybrid/rerank tùy chọn*:
```
validate prefix → derive meeting_id
fetch_k = top_k × HYBRID_FETCH_MULTIPLIER (nếu hybrid HOẶC rerank bật; else top_k)
embed_query → TranscriptStore.search(vector, fetch_k, filter meeting_id/speaker, score_threshold)
nếu HYBRID_ENABLED → retrieval.fuse(query, candidates)
nếu RERANK bật     → Reranker.rerank(query, candidates)
trim → top_k
mỗi hit seq S: scroll sequence_id ∈ [S-w, S+w] → window.before/after
→ { results: [{ sequence_id, text, score, context, context_status, window }] }
```

## A.4. 9 Quyết định thiết kế (D1–D9)

| # | Quyết định | Lý do |
|---|------------|-------|
| **D1** | Server gán `sequence_id` qua Redis `INCR`/collection, self-healing từ Qdrant | Một nguồn chân lý duy nhất, atomic, không phụ thuộc client |
| **D2** | Query trả **window ±N câu** (mặc định 2, clamp tới max 5) | Một câu rời nghĩa khó hiểu; lân cận cho ngữ cảnh hội thoại |
| **D3** | Context = **bản tóm tắt LLM cuộn**, không phải nối thô: `context[N]=summarize(context[N-1]+text[N-1])` | Giữ độ dài giới hạn, chắt lọc quyết định/số liệu/tên riêng |
| **D4** | **Không chunk** transcript — 1 câu nói = 1 vector | Câu nói đã là đơn vị ngữ nghĩa tự nhiên |
| **D5** | LLM build context **self-host trên server**, gọi in-process | Dữ liệu họp nhạy cảm, kiểm soát chi phí/độ trễ |
| **D6** | Context lưu trong **payload vector**, không phải Redis | Bền vững; cho phép recovery scan từ payload |
| **D7** | Dùng chung model embedding MiniLM-L12-v2 (384-d) cho cả 2 luồng | Một model nạp RAM, đồng nhất không gian vector |
| **D8** | `meeting_id` **suy từ tên collection**, client không gửi | API gọn; nhưng → logical=physical (gốc của rủi ro scaling) |
| **D9** | Context build **đúng thứ tự** qua 1 worker FIFO single-consumer | `context[N]` phụ thuộc `context[N-1]`; BackgroundTasks không đảm bảo thứ tự |

## A.5. Tính năng RAGFlow-inspired (gated, mặc định TẮT)

**Nguyên tắc gating:** `HYBRID_ENABLED=false` + `RERANK_PROVIDER=none` (mặc định) ⇒ **không nạp thêm model**, hành vi y hệt pure-vector. Bật cờ mới phát sinh chi phí.

- **Hybrid (`retrieval.fuse`)**: BM25 chạy **trên chính tập ứng viên** vector trả về (bounded, rẻ, không re-index). `final = vec_w·norm(vec) + term_w·norm(term)`, min-max normalize, mặc định 0.7/0.3. Tinh thần "two-pass" RAGFlow rút gọn: vector lọc thô → BM25 tinh chỉnh thứ tự.
- **Reranker**: cross-encoder chấm lại cặp (query, doc) chính xác hơn bi-encoder. `local` = sentence-transformers CrossEncoder; `http` = TEI/Infinity/Xinference. Lỗi rerank → degrade an toàn (giữ thứ tự cũ).

## A.6. Đánh giá (RAGAS + retrieval metrics)

Benchmark trên gold dataset tiếng Việt, LLM judge Gemma-4-26B.

| | Documents | Transcript |
|---|---|---|
| RAGAS faithfulness | 1.00 | 0.90 |
| RAGAS context recall | 1.00 | 0.85 (hybrid) |
| RAGAS context precision | 0.92 | 0.82 (hybrid) |
| Retrieval Hit@K | 0.67 (Hit@5, hybrid) | 1.00 (Hit@3) |

- **Hybrid thắng pure-vector trên mọi metric retrieval** (Hit@K, MRR, TokenRecall).
- Điểm yếu nhất: **transcript answer_relevancy ~0.66** → ứng viên cải thiện prompt sinh câu trả lời.

## A.7. Giới hạn đã biết & roadmap (Phase 3)

- **"Too many open files"** (blocker): 1 collection vật lý / cuộc họp (hệ quả D8) → fd tăng tuyến tính theo số cuộc. **fd ≠ RAM** (trực giao). Gói 4 bước bắt buộc: S0 chẩn đoán → S1 vá nền (ulimit 65535 + `on_disk` + ít segment lớn) → S2 hợp nhất (logical≠physical, filter `meeting_id`, `point_id=uuid5`) → S3 migrate + **xóa collection cũ**.
- **Worker durability**: `asyncio.Queue` in-memory → restart mất job pending. Fix rẻ: **recovery scan** lúc startup (quét `context_status ∈ {pending,processing}` từ payload Qdrant, D6 → enqueue lại).
- **`LLM_API_KEY`** plaintext trong `.env` → rotate + secret injection.
- Roadmap khác: tokenizer tiếng Việt cho BM25 (`pyvi`/`underthesea`), sparse hybrid native (Qdrant Query API + RRF), PageIndex, Agentic RAG (đặt ở **tầng thiết bị**).

---

# PHẦN B — 30 CÂU HỎI PHỎNG VẤN (dễ → khó)

## Mức DỄ — nền tảng RAG & thuật ngữ (Q1–Q10)

**Q1. RAG là gì? Vì sao cần RAG thay vì hỏi thẳng LLM?**
RAG = Retrieval-Augmented Generation: trước khi LLM sinh câu trả lời, ta **truy hồi** đoạn tri thức liên quan từ kho ngoài rồi đưa vào prompt. Cần vì: LLM có **knowledge cutoff**, không biết dữ liệu riêng/mới (vd nội dung cuộc họp); RAG **giảm hallucination** (câu trả lời bám nguồn), **cập nhật được** mà không cần fine-tune, và **truy vết nguồn**. Trong dự án này, nội dung họp thay đổi liên tục → không thể nhồi vào trọng số model.

**Q2. Embedding là gì? Vì sao dùng cosine similarity?**
Embedding = ánh xạ văn bản → vector số chiều cố định (ở đây **384-d**) sao cho văn bản gần nghĩa thì gần nhau trong không gian. Cosine đo **góc** giữa 2 vector (bỏ qua độ lớn) — phù hợp vì ý nghĩa nằm ở **hướng**, không phải độ dài. Vì model đã chuẩn hóa (normalize) vector nên cosine ≈ dot product.

**Q3. Vector database (Qdrant) khác gì DB thường? ANN là gì?**
Vector DB đánh chỉ mục để tìm **láng giềng gần nhất theo độ tương đồng vector** thay vì khớp chính xác. Tìm chính xác (brute force) là O(N) — chậm khi triệu vector. Nên dùng **ANN** (Approximate Nearest Neighbor, vd HNSW): đánh đổi một chút độ chính xác (recall) lấy tốc độ rất cao. Qdrant còn hỗ trợ **payload + filter**, quan trọng cho dự án (filter `meeting_id`).

**Q4. Pipeline RAG cơ bản gồm những bước nào?**
(1) **Ingest/Index**: parse → chunk → embed → lưu vector + payload. (2) **Retrieve**: embed query → ANN search → (tùy chọn) hybrid + rerank → lấy top-k. (3) **Generate**: nhồi context vào prompt → LLM sinh câu trả lời. Dự án này tách (3) ra **tầng thiết bị**; server chỉ lo (1)+(2).

**Q5. Chunking là gì và tại sao quan trọng?**
Chia tài liệu dài thành đoạn nhỏ để (a) vừa context window, (b) embedding **một đoạn ngắn** biểu diễn ngữ nghĩa sắc nét hơn cả tài liệu. Dự án dùng chunker paragraph→sentence cho documents (**512 ký tự, overlap 64**). Overlap để không cắt ngang ý ở ranh giới chunk.

**Q6. top_k là gì? Đặt cao/thấp ảnh hưởng gì?**
top_k = số đoạn truy hồi đưa vào context. **Thấp**: precision cao, ít nhiễu, rẻ token, nhưng dễ sót (recall thấp). **Cao**: recall tốt hơn nhưng nhiễu nhiều, tốn token, có thể làm LLM lạc. Dự án: docs top_k=5, transcript top_k=3.

**Q7. Hệ thống này nhận đầu vào gì và trả ra gì?**
Đầu vào: câu nói transcript (`/transcript/.../embed`) hoặc câu hỏi (`/query/transcript`). Đầu ra query: danh sách kết quả `{sequence_id, text, score, context, window}`. Server **không trả câu trả lời ngôn ngữ tự nhiên** — đó là việc LLM trên thiết bị.

**Q8. Vì sao tách "server = nguồn tri thức" và "thiết bị = sinh câu trả lời"?**
Vì 2 tầng phần cứng khác nhau: thiết bị (QCS8550) có LLM nhỏ chạy gần người dùng (độ trễ thấp, cá nhân hóa câu trả lời cuối); server mạnh lo phần nặng (embedding, vector search, build context). Tách giúp **mở rộng độc lập**, server tái dùng cho mọi thiết bị, và LLM cuối tùy biến theo từng người dùng/thiết bị.

**Q9. `score_threshold` để làm gì?**
Ngưỡng lọc bỏ kết quả có điểm tương đồng quá thấp — tránh trả về "rác" khi không có gì thực sự liên quan. Tốt hơn trả về top-k cứng kể cả khi tất cả đều không liên quan.

**Q10. Hai luồng documents và transcript khác nhau ở đâu?**
Documents: tài liệu tĩnh, **có chunk**, không context cuộn. Transcript: câu nói động trong lúc họp, **không chunk** (1 câu=1 vector), có **sequence_id**, **window ±N câu**, và **context tóm tắt cuộn** bằng LLM. Phân biệt qua prefix collection (`docs-*` vs `meeting-*`). Cả hai **dùng chung** model embedding và pipeline hybrid/rerank.

## Mức TRUNG BÌNH — quyết định thiết kế của hệ thống (Q11–Q22)

**Q11. Vì sao không chunk transcript mà chunk documents (D4)?**
Một **câu nói** đã là đơn vị ngữ nghĩa tự nhiên và đủ ngắn — chunk thêm chỉ làm vỡ ngữ cảnh và phức tạp việc đánh `sequence_id`/window. Tài liệu thì dài, không có ranh giới tự nhiên nên phải chunk để embedding sắc nét và vừa context.

**Q12. `context[N] = summarize(context[N-1] + text[N-1])` — giải thích và đánh đổi (D3).**
Đây là **rolling summary**: context tại câu N là bản tóm tắt cuộn của toàn bộ hội thoại *trước* N. Ưu: độ dài context **bị chặn** dù họp dài bao nhiêu, chắt lọc quyết định/số liệu/tên riêng/action items (theo system prompt). Nhược: **lỗi tích lũy** (summary sai ở bước trước lan về sau), **phụ thuộc tuần tự** (phải build đúng thứ tự → D9), tốn 1 lời gọi LLM/câu. Thay thế: nối thô (đơn giản nhưng phình vô hạn) hoặc tóm tắt lại từ đầu mỗi lần (chính xác hơn nhưng O(N²) chi phí).

**Q13. Vì sao server tự gán `sequence_id` qua Redis INCR thay vì để client gửi (D1)?**
Để có **một nguồn chân lý atomic**: nhiều thiết bị có thể gửi câu nói đồng thời; nếu client tự đánh số sẽ trùng/nhảy/đua. `INCR` của Redis là atomic, đảm bảo số **liên tục, duy nhất, đơn điệu tăng** trên mỗi collection — nền tảng cho window và thứ tự build context.

**Q14. "Self-healing" của SequenceManager hoạt động thế nào và giải quyết vấn đề gì?**
Counter Redis có **TTL 7 ngày** (có thể bị mất khi hết hạn hoặc Redis restart). Khi `next()`/`current()` thấy key không tồn tại, nó **rebuild từ Qdrant**: quét `max(sequence_id)` các điểm của collection rồi set lại counter. Nhờ vậy Redis chỉ là **cache tăng tốc**, không phải nguồn chân lý duy nhất — mất Redis không mất tính đúng đắn.

**Q15. Vì sao context lưu trong payload Qdrant chứ không phải Redis (D6)? Lợi ích phụ?**
Payload Qdrant **bền vững** cùng vector và đi kèm kết quả query không cần truy vấn thêm. Lợi ích phụ quan trọng: vì `context_status` nằm trong payload, khi server restart ta có thể **recovery scan** các điểm `pending/processing` và enqueue lại — bền vững mà không cần message queue bền. Redis ở đây chỉ là **cache** `latest_ctx`.

**Q16. ContextWorker FIFO single-consumer giải quyết vấn đề gì? Vì sao không dùng BackgroundTasks (D9)?**
Vì `context[N]` **phụ thuộc** `context[N-1]` (rolling summary). FastAPI `BackgroundTasks` chạy song song, **không đảm bảo thứ tự** → có thể build N trước N-1, đọc context sai. Một `asyncio.Queue` 1 consumer drain job **đúng thứ tự nhập**, đảm bảo N-1 xong trước N. Đánh đổi: throughput thấp (tuần tự) — roadmap Phase 3 nâng thành **per-meeting queue** (song song giữa các cuộc, FIFO trong từng cuộc).

**Q17. Window ±N câu là gì và vì sao cần (D2)?**
Khi truy hồi trúng câu S, server trả thêm các câu `[S-w, S+w]` (`before`/`after`). Một câu nói đơn lẻ thường thiếu ngữ cảnh ("đồng ý" — đồng ý cái gì?); window cấp ngữ cảnh hội thoại quanh nó để LLM thiết bị hiểu đúng. Mặc định w=2, clamp tới `TRANSCRIPT_MAX_WINDOW_SIZE=5` để chặn lạm dụng.

**Q18. Hybrid retrieval ở đây hoạt động ra sao? Khác hybrid "đúng chuẩn" thế nào?**
`fuse()` chạy **BM25 trên chính tập ứng viên** mà vector search trả về (bounded → rẻ, không re-index, không sparse vector), rồi `final = 0.7·norm(vector) + 0.3·norm(BM25)`. Đây là "two-pass rút gọn": vector lọc thô → từ khóa tinh chỉnh thứ tự. **Hạn chế:** vì BM25 chỉ chấm lại ứng viên *đã qua vector*, câu mà vector bỏ sót hoàn toàn thì BM25 không cứu được. Hybrid "đúng chuẩn" (sparse vector native + Qdrant Query API + RRF) truy hồi sparse **độc lập** rồi fuse — bắt được câu đó, nhưng phải re-index. Đó là lý do nó nằm ở roadmap (đo recall trước khi làm).

**Q19. Bi-encoder vs cross-encoder — vì sao cần reranker?**
Embedding là **bi-encoder**: mã hóa query và doc **riêng biệt** thành vector rồi so cosine — nhanh, đánh chỉ mục trước được, nhưng kém tinh tế. **Cross-encoder** đưa **cặp (query, doc) cùng lúc** qua model → điểm liên quan chính xác hơn nhiều, nhưng phải chạy lúc query cho từng cặp (đắt). Chiến lược: bi-encoder lấy nhiều ứng viên thô (fetch_k = top_k × multiplier) → cross-encoder rerank → cắt top_k. Cân bằng tốc độ và độ chính xác.

**Q20. "Gated, default off" nghĩa là gì và vì sao đáng giá về mặt kỹ thuật?**
Mọi tính năng nâng cao (hybrid, rerank) ẩn sau cờ env, **mặc định tắt**. Tắt cờ ⇒ **không nạp model, không thêm dependency, hành vi y hệt baseline**. Lợi: triển khai an toàn (bật/tắt không đổi API/response, chỉ `score` phản ánh điểm cuối), dễ A/B, dễ rollback, và **đo trước khi mở rộng** — chỉ trả chi phí khi có bằng chứng cải thiện (benchmark cho thấy hybrid thật sự tốt hơn nên đáng bật).

**Q21. LLMClient trừu tượng hóa provider thế nào và để làm gì?**
Một interface `summarize(prev_context, new_utterance)` chung, 4 backend: `ollama`, `openai` (kể cả vLLM/LM Studio/LocalAI compatible), `gemini`, `none` (no-op). Đổi provider chỉ qua `.env`, không sửa code nghiệp vụ. `none` cho phép **tắt hẳn build context** (status `disabled`) để chạy hệ không cần LLM. Temperature=0 để tóm tắt **ổn định, không sáng tạo**.

**Q22. Mô tả vòng đời `context_status`. Mỗi trạng thái nghĩa gì?**
`pending` (vừa ingest, chờ build) → `processing` (worker đang gọi LLM) → `ready` (xong) hoặc `failed` (LLM lỗi sau hết retry → fallback context = context[N-1]). `disabled` = LLM_PROVIDER=none, không bao giờ build. Trạng thái nằm trong payload nên truy được sau restart và là cơ sở cho recovery scan.

## Mức KHÓ — scaling, trade-off, vận hành production (Q23–Q30)

**Q23. Lỗi "too many open files" của Qdrant: nguyên nhân gốc, và vì sao "đẩy dữ liệu cũ xuống disk" KHÔNG chữa được?**
Gốc: thiết kế **1 collection vật lý / 1 cuộc họp** (hệ quả vô tình của D8: meeting_id=tên collection → logical=physical). Mỗi collection dù gần rỗng vẫn giữ ≥ vài segment, mỗi segment mở nhiều file (RocksDB `.sst`, mmap), Qdrant **không tự đóng fd collection nhàn rỗi** → `fd ≈ N_cuộc × segment × file/segment`, tăng tuyến tính theo số cuộc → vượt `ulimit -n`. **Điểm mấu chốt: fd ≠ RAM (trực giao).** "Đẩy xuống disk" (`on_disk`) chỉ giải phóng **RAM**, file **vẫn mở** → fd không giảm. Chữa fd phải **hợp nhất collection** (merge optimizer chặn số segment theo dung lượng, không theo số cuộc).

**Q24. Trình bày gói 4 bước dứt điểm. Vì sao thiếu một bước là vẫn kẹt?**
S0 chẩn đoán (lsof / `/proc/1/fd`, xác nhận fd ở qdrant chứ không phải rag_api rò connection, kiểm `QdrantClient` singleton) → S1 vá nền (ulimit 65535 + `on_disk` vector/payload + optimizer gom **ít segment lớn**) → S2 hợp nhất code (`_physical()` map meeting→collection chung + filter `meeting_id` + `point_id=uuid5`) → S3 **migrate + xóa collection cũ**. Thiếu bước chết người: nếu chỉ đổi config sang `shared` mà **không xóa** `meeting-*` cũ thì ghi mới *cộng thêm* collection cũ ⇒ **nhiều file hơn trước**. Và một collection lớn vẫn có thể đụng lỗi nếu segment quá nhiều → ulimit + on_disk + ít segment phải làm **đồng thời**, không phải "mitigation tùy chọn".

**Q25. Vì sao `point_id` phải toàn cục duy nhất (`uuid5(meeting_id, sequence_id)`) khi hợp nhất? Hậu quả nếu sai?**
Khi gộp nhiều cuộc vào 1 collection, nếu `point_id = sequence_id` (cục bộ) thì `id=1` của cuộc B **ghi đè** `id=1` của cuộc A — **mất dữ liệu âm thầm** (upsert không báo lỗi). `uuid5(meeting_id, sequence_id)` cho id **xác định + duy nhất toàn cục**, và **idempotent** (chạy migration lại không tạo bản trùng). Đây là một trong "6 cái bẫy" đưa thẳng vào Definition of Done.

**Q26. RAGFlow tách logical/physical từ đầu (1 index/tenant + filter kb_id). Vì sao dự án này lại đi ngược, và bài học là gì?**
Dự án ưu tiên **API gọn** (D8: meeting_id suy từ tên collection) nên vô tình ghép logical=physical — hợp lý ở MVP/Phase 2 nhưng không lường scaling. Bài học: **mức cô lập logic (per-meeting) không nên trùng đơn vị lưu trữ vật lý**. Cách đúng: **ít collection vật lý + filter field** (`meeting_id`) cho cô lập logic — fd tỉ lệ dung lượng (có trần) thay vì số thực thể. Đây là D16/D17 của Phase 3.

**Q27. Worker durability: lỗ hổng là gì và vì sao recovery scan đủ rẻ mà không cần Redis Stream?**
`ContextWorker` dùng `asyncio.Queue` **in-memory** → restart khi còn job pending ⇒ job biến mất, câu kẹt mãi ở `pending`. Vì `context_status` lưu trong **payload Qdrant** (D6), lúc startup chỉ cần **quét điểm `pending/processing` → enqueue lại** là khôi phục — không cần message broker bền. Chỉ khi cần **at-least-once thật / nhiều worker / redeliver qua restart** mới nâng lên Redis Stream + consumer group (để dành, đo trước — nguyên tắc YAGNI).

**Q28. Benchmark cho thấy transcript answer_relevancy ~0.66 là yếu nhất. Cách bạn chẩn đoán và cải thiện?**
Trước hết **phân tách metric**: faithfulness 0.90 (bám nguồn tốt) nhưng relevancy thấp → vấn đề ở **bước sinh câu trả lời**, không phải truy hồi (retrieval Hit@3=1.0). Hướng: (1) cải thiện prompt sinh câu trả lời ở tầng thiết bị; (2) đưa thêm `window` + `context` vào prompt cho đủ ngữ cảnh; (3) kiểm gold dataset xem có phải artifact đánh giá (token-Jaccard ≥0.3 có thể under-estimate paraphrase nặng); (4) thử cross-encoder rerank để câu trúng nhất lên đầu. Quan trọng: **đo lại sau mỗi thay đổi**, không tối ưu mù.

**Q29. Đánh giá RAG bằng RAGAS: 4 metric đó đo gì, và vì sao cần cả deterministic retrieval metrics?**
**faithfulness** (câu trả lời có bám context không — đo hallucination), **answer_relevancy** (trả lời có đúng trọng tâm câu hỏi không), **context_precision** (context truy hồi có ít rác không), **context_recall** (có lấy đủ thông tin cần không). RAGAS dùng **LLM làm judge** → tốn kém, có nhiễu, đôi khi NaN. Nên cũng cần metric **xác định, không LLM** (Hit@K, MRR, TokenRecall) — rẻ, lặp lại được, tách bạch **chất lượng retrieval** khỏi **chất lượng generation**. Hai loại bổ trợ: nếu retrieval kém thì sửa retriever; nếu retrieval tốt mà answer kém thì sửa prompt/generation.

**Q30. Nếu phải nâng hệ này lên đa-tenant (nhiều tổ chức) và 100k cuộc họp/tháng, bạn thay đổi gì?**
(1) **Storage**: hoàn tất hợp nhất `shared`; nếu collection phình thì `sharded` theo `hashlib.md5(meeting_id) % NUM_SHARDS` (KHÔNG dùng `hash()` built-in vì bị salt theo PYTHONHASHSEED, đổi sau restart); thêm `tenant_id` vào tên collection vật lý (RAGFlow-style) để cô lập tổ chức. (2) **on_disk** + OS page cache lo tiering RAM (napkin math: ~10MB/cuộc → 100k cuộc ≈ ~1TB SSD, vặt vãnh; chưa cần cold-tier/snapshot tới khi đo được). (3) **Worker**: per-meeting queue + semaphore concurrency + recovery scan. (4) **Bảo mật/vận hành**: `INTERNAL_API_KEY`, rate limit, rotate `LLM_API_KEY` + secret injection, metrics (tỉ lệ `context_status=failed`, backlog worker), mở rộng `/health`. (5) **Nguyên tắc xuyên suốt**: mọi thứ **gated, đo trước khi mở rộng, tương thích ngược** (`per_meeting` khôi phục baseline tuyệt đối).

---

# PHẦN C — CÂU HỎI BỔ SUNG (đào sâu khái niệm & tình huống)

## C.1. Kiến trúc hồi quy, bộ nhớ & build context (Q31–Q36)

**Q31. Cách build context `context[N]=summarize(context[N-1]+text[N-1])` giống kiến trúc nào? Phân tích điểm tương đồng và khác biệt.**
Giống **RNN**: `context` là hidden state, mỗi câu nói là một input step, build tuần tự FIFO vì có **phụ thuộc hồi quy** (`h_t` cần `h_{t-1}`). Khác biệt nền tảng: (1) hidden state ở đây là **văn bản tiếng Việt đọc được**, không phải vector dày đục — trả thẳng cho LLM thiết bị, debug được; (2) mỗi "cell" là **một forward pass Transformer đầy đủ (LLM)**, không phải nhân ma trận + gate. Mô tả chuẩn: "orchestration hình RNN bọc quanh cell Transformer" — đệ quy ở tầng ứng dụng, attention ở bên trong.

**Q32. Kiến trúc hồi quy này thoái hóa thế nào khi họp kéo dài?**
Ba dạng, đều là họ hàng bệnh RNN: (1) **Information bottleneck/quên dần** — summary bị chặn độ dài (`CONTEXT_MAX_TOKENS`) nên câu rất cũ bị ép văng ra (giống vanishing gradient). (2) **Error accumulation/drift** — sai ở bước k lan mãi về sau vì các bước sau đọc lại state lỗi (giống exposure bias); không sửa được quá khứ. (3) **Relevance decay** — chủ đề cũ không còn liên quan vẫn bị cõng theo → pha loãng context hiện tại.

**Q33. Có nên đổi LLM sang LSTM (hoặc LSTM+Encoder-Decoder) để build context không?**
Không — thậm chí tệ hơn. (1) LSTM **vẫn quên dài hạn** (state cố định chiều; chính vì vậy ngành chuyển sang attention/Transformer) → đi lùi. (2) Mất lợi ích lớn nhất: state dạng **văn bản đọc được** — hidden state LSTM là vector đục, không trả cho LLM thiết bị được. (3) Tóm tắt trừu tượng tiếng Việt của LSTM seq2seq **yếu hơn hẳn** LLM và **cần tập huấn luyện** domain họp (đắt). Mấu chốt: hệ hiện tại đã "tốt hơn LSTM"; nguyên nhân quên **không phải do model mà do summary bị chặn độ dài** → đổi model không chữa gốc.

**Q34. Vậy cách khắc phục đúng cho trí nhớ dài hạn là gì?**
Tận dụng **bộ nhớ ngoài đã có sẵn**: mỗi câu nói vẫn lưu nguyên 1 vector trong Qdrant. Phân vai: `context` = **trí nhớ làm việc ngắn hạn**; Qdrant + `/query/transcript` + window = **trí nhớ episodic** truy hồi theo nhu cầu (câu cũ không mất, search ngữ nghĩa lôi ra khi cần) — đây là cách RAG giải bệnh quên của RNN (external memory thay vì nén tất cả vào hidden state, tinh thần Memory-Augmented NN). Bổ sung: **tóm tắt phân cấp** (per-chủ đề + cấp cao), **state có cấu trúc** (slot quyết định/action/số liệu dạng JSON), **re-summarize định kỳ từ thô** (chặn drift), **phân đoạn theo chủ đề + recency** (giảm relevance decay).

**Q35. ContextBuilder xử lý đồng bộ hay bất đồng bộ với client? Đánh đổi?**
**Bất đồng bộ.** Ingest (`/transcript/.../embed`) đồng bộ phần gán seq + embed + upsert (trả `202` ngay với `context_status="pending"`); việc gọi LLM build summary chạy **nền** qua `ContextWorker` sau khi client đã nhận response. Đánh đổi: **độ-trễ-context** (có khoảng trễ giữa lúc lưu câu nói và lúc context `ready` → client phải poll `/context` hoặc chấp nhận `pending`) để lấy **phản hồi ingest tức thì** — quan trọng khi nhiều thiết bị bắn câu nói liên tục lúc họp.

**Q36. Nếu yêu cầu context phải `ready` ngay khi query (không chấp nhận `pending`), bạn làm gì?**
Vài lựa chọn theo đánh đổi: (1) **Build đồng bộ** câu mới nhất khi cần (block, chậm — chỉ hợp tần suất thấp). (2) **Lazy build on read**: nếu query thấy `pending`, build ngay tại chỗ cho đúng điểm đó. (3) **Bỏ rolling summary, ghép on-the-fly**: lúc query lấy window + các câu liên quan từ Qdrant rồi để LLM thiết bị tự tổng hợp (đổi chi phí build nền lấy chi phí query). (4) Giữ async nhưng **cảnh báo trạng thái** rõ trong response để client xử lý graceful. Chọn cái nào tùy SLA độ trễ vs tần suất ingest.

## C.2. Embedding, mô hình & lựa chọn kỹ thuật (Q37–Q42)

**Q37. Vì sao chọn `paraphrase-multilingual-MiniLM-L12-v2` (384-d)? Đánh đổi khi chọn model embedding?**
Lý do: **đa ngôn ngữ** (hỗ trợ tiếng Việt), **nhẹ** (384-d → ít RAM, search nhanh, index nhỏ), self-host được. Đánh đổi tổng quát: chiều cao hơn (768/1024) thường chất lượng tốt hơn nhưng **tốn RAM/đĩa/độ trễ** hơn; model lớn (bge-m3, multilingual-e5-large) recall cao hơn nhưng nặng. Nguyên tắc: chọn theo **ngôn ngữ + ngân sách tài nguyên + đo benchmark thực tế**, không chọn theo bảng xếp hạng chung.

**Q38. Vì sao phải chuẩn hóa (normalize) embedding? Liên hệ cosine vs dot product.**
Chuẩn hóa về độ dài 1 khiến **cosine ≈ dot product** (nhanh hơn), và đảm bảo so sánh chỉ theo **hướng** (ngữ nghĩa) chứ không lẫn độ lớn vector. Nhất quán giữa index và query là bắt buộc — lệch chuẩn hóa giữa hai phía sẽ làm điểm số sai.

**Q39. Dùng chung 1 model embedding cho cả documents lẫn transcript (D7) — lợi và rủi ro?**
Lợi: **1 model nạp RAM**, cùng **không gian vector** (so sánh nhất quán), vận hành đơn giản. Rủi ro: model không tối ưu riêng cho từng domain (văn phong tài liệu vs khẩu ngữ hội thoại); nếu sau này một luồng cần model chuyên biệt thì phải tách. Hiện tại đánh đổi nghiêng về đơn giản — hợp lý ở quy mô này.

**Q40. Nếu đổi model embedding (vd lên bge-m3 768-d) thì phải làm gì trong hệ này?**
Đây là thay đổi **phá vỡ**: phải (1) đổi `EMBEDDING_DIM` + `vectors_config` (Qdrant cố định size/distance theo collection → phải **tạo collection mới**), (2) **re-embed + re-index toàn bộ** dữ liệu cũ (vector cũ không tương thích không gian mới), (3) chạy migration song song (blue-green) để không downtime. Không thể "đổi nóng".

**Q41. Vì sao temperature=0 khi build context summary?**
Tóm tắt cần **ổn định, tái lặp, không sáng tạo/bịa**. Temperature=0 (greedy) cho output xác định, giảm hallucination — đúng với system prompt "TUYỆT ĐỐI KHÔNG bịa thông tin". Ngược lại với tác vụ cần đa dạng (brainstorm) mới đặt temperature cao.

**Q42. `_truncate` cắt theo word count (~1.3 words/token) — hạn chế và cách làm đúng hơn?**
Đây là **heuristic** vì token thật phụ thuộc tokenizer riêng từng model. Hạn chế: ước lượng sai (tiếng Việt + token hóa subword khác nhau) → có thể cắt cụt hoặc chưa chặn đủ. Đúng hơn: dùng **tokenizer của chính model** (vd `tiktoken`/HF tokenizer) đếm token thật; hoặc giới hạn output bằng tham số API (`max_tokens`/`num_predict`) — vốn đã làm — và chỉ dùng truncate như lưới an toàn.

## C.3. Tình huống vận hành & debug (Q43–Q47)

**Q43. Một cuộc họp có nhiều câu kẹt ở `context_status="pending"` mãi không lên `ready`. Bạn chẩn đoán thế nào?**
Khoanh vùng theo luồng: (1) **Worker chết/không chạy?** kiểm log `ContextWorker`, có task nền không. (2) **LLM lỗi/timeout?** kiểm `LLM_PROVIDER`, endpoint sống không, log `LLM summarize failed` — nếu `failed` thì là LLM, nếu mãi `pending` thì job chưa được xử lý. (3) **Server từng restart?** asyncio.Queue in-memory → job pending mất khi restart (lỗ hổng durability) → cần recovery scan. (4) **Tắc đầu hàng đợi** vì 1 câu cũ build mãi không xong (FIFO chặn cả sau). Phân biệt `pending` (chưa xử lý) vs `failed` (đã thử, LLM lỗi) là chìa khóa.

**Q44. Query trả về kết quả không liên quan / điểm thấp. Quy trình debug?**
Tách lớp: (1) **Embedding/model?** cùng model + normalize cho index và query chưa. (2) **Dữ liệu có trong collection?** đúng `meeting_id`/prefix chưa; collection rỗng → trả []. (3) **score_threshold quá cao?** lọc mất hết. (4) **Retrieval vs ngôn ngữ:** câu hỏi paraphrase nặng → vector miss → thử bật **hybrid** (BM25 bắt từ khóa/tên riêng/số) hoặc **rerank**. (5) Đo bằng **gold dataset** (Hit@K, MRR) để biết là lỗi retrieval hay cảm tính. Sửa theo lớp hỏng, không sửa mù.

**Q45. Redis chết giữa lúc đang họp thì sao? Hệ có mất tính đúng đắn không?**
Redis ở đây là **cache tăng tốc**, không phải nguồn chân lý duy nhất. `sequence_id` **self-healing**: mất counter → rebuild từ `max(sequence_id)` trong Qdrant (D1). Cache `latest_ctx` mất → đọc lại context từ payload Qdrant (D6). Nên **không mất tính đúng đắn**, chỉ chậm hơn một nhịp (phải rebuild). Đây là lợi ích trực tiếp của việc đặt nguồn chân lý ở Qdrant.

**Q46. Hai thiết bị gửi câu nói gần như đồng thời. Có race condition về `sequence_id` không?**
Không, vì `SequenceManager.next()` dùng **Redis `INCR` — atomic**: hai request đồng thời nhận hai số khác nhau, liên tục, không trùng (D1). Đây chính là lý do **server gán seq thay vì client** — client tự đánh số sẽ đua/trùng. Build context vẫn đúng thứ tự nhờ FIFO worker.

**Q47. Làm sao biết bật hybrid/rerank có thực sự đáng không, thay vì bật theo cảm tính?**
**Đo trước khi mở rộng**: chạy benchmark trên gold dataset, so pure-vector vs hybrid trên Hit@K/MRR/RAGAS. Trong dự án, số liệu cho thấy hybrid **thắng mọi metric retrieval** (vd transcript Hit@3 giữ 1.0 nhưng MRR 0.88→0.93, TokenRecall 0.97→1.0) nên đáng bật; nếu A/B không cải thiện thì giữ tắt (tiết kiệm model + độ trễ). Đây là nguyên tắc "gated, default off, đo rồi mới bật".

## C.4. Đào sâu hệ thống & mở rộng (Q48–Q50)

**Q48. Vì sao hybrid ở đây không thể "cứu" câu mà vector search bỏ sót hoàn toàn? Khi nào cần sparse vector native?**
Vì `fuse()` chạy BM25 **trên chính tập ứng viên đã qua vector search** (fetch_k) — câu nào vector không lấy vào ứng viên thì BM25 không có cơ hội chấm. Nó chỉ **re-rank**, không **re-retrieve**. Khi cần bắt đúng những câu đó (vd khớp tên riêng/mã số hiếm mà vector miss), phải dùng **sparse vector native** (Qdrant Query API: prefetch dense + sparse độc lập, fuse bằng RRF) — nhưng phải đổi `vectors_config` + re-index, nên roadmap yêu cầu **đo recall sau khi cải thiện tokenizer trước** rồi mới quyết.

**Q49. Thiết kế "context" cho luồng documents (Phase 1) có cần rolling summary như transcript không?**
Không. Tài liệu là **tĩnh, không có dòng thời gian hội thoại** → không có khái niệm "bối cảnh cuộn dẫn tới câu N". Documents chỉ cần chunk + embed + retrieve. Rolling summary là đặc thù của transcript (luồng động, tuần tự). Áp nhầm cơ chế của luồng này sang luồng kia là một lỗi thiết kế thường gặp — phải hiểu **bản chất dữ liệu khác nhau**.

**Q50. Nếu phải thêm "trả lời câu hỏi xuyên nhiều cuộc họp" (cross-meeting), kiến trúc hiện tại cản trở gì và sửa thế nào?**
Cản trở: D8 (meeting_id = tên collection) + 1 collection/cuộc → query bị **cô lập trong 1 cuộc**, không search xuyên cuộc dễ dàng. Sửa: hoàn tất **hợp nhất shared collection + filter `meeting_id`** (Phase 3 S2) → khi đó có thể **bỏ filter** hoặc filter theo nhóm cuộc để search xuyên cuộc trong cùng một index; thêm `tenant_id`/`project_id` vào payload để giới hạn phạm vi. Đây là ví dụ logical≠physical mở ra khả năng mới mà thiết kế cũ chặn.

---

# PHẦN D — LÝ DO LỰA CHỌN CÔNG CỤ (tech stack rationale)

> Stack thật của repo: **FastAPI + Uvicorn** (web), **Qdrant** (vector DB), **Redis** (counter/cache), **sentence-transformers** (embedding), **httpx** (gọi LLM), **rank-bm25** (hybrid), **pydantic-settings** (config), **Docker Compose** (orchestrate), **pytest** (test). Nguyên tắc chung khi nói về chọn tool: **gắn với yêu cầu cụ thể của bài toán**, nêu **đánh đổi** và **giải pháp thay thế**, đừng nói "vì nó phổ biến".

## D.1. Web framework & runtime (Q51–Q53)

**Q51. Vì sao chọn FastAPI thay vì Flask / Django?**
Lý do gắn với bài toán: (1) **Async native** — hệ này I/O-bound nặng (gọi Qdrant, Redis, LLM qua mạng); `async def` + `await` cho phép xử lý nhiều request đồng thời mà không block, đúng kịch bản nhiều thiết bị bắn câu nói liên tục. (2) **Pydantic tích hợp** — validate request/response tự động (schemas `TranscriptEmbedRequest`...), type-safe, ít bug. (3) **OpenAPI/Swagger tự sinh** (`/docs`) — app/device dev tự đọc hợp đồng API. (4) Nhẹ hơn Django (không cần ORM/admin/template cho một API service). Đánh đổi: Flask đơn giản hơn nhưng async kém tự nhiên và phải tự ghép validation/docs; Django nặng và thừa cho microservice này.

**Q52. FastAPI khác Uvicorn thế nào? Vì sao cần cả hai?**
**FastAPI** là **framework** (định nghĩa route, validation, DI) — không tự chạy được. **Uvicorn** là **ASGI server** thực sự lắng nghe socket, parse HTTP, chạy event loop async và gọi app. Cần ASGI (không phải WSGI như Gunicorn thuần) vì FastAPI là async. `uvicorn[standard]` kèm `uvloop`/`httptools` cho hiệu năng cao hơn. Production thường chạy nhiều worker (Gunicorn + Uvicorn workers) để dùng nhiều core.

**Q53. Hệ dùng nhiều `async` + `asyncio.to_thread`. Vì sao, và bẫy nào cần tránh?**
I/O (Redis, httpx tới LLM) là async thật → không block event loop. Nhưng **embedding (sentence-transformers) và Qdrant client là đồng bộ, CPU/blocking** → nếu gọi thẳng trong `async def` sẽ **chặn cả event loop**, làm treo mọi request khác. Vì vậy code bọc `await asyncio.to_thread(self._embedder.embed_query, ...)` — đẩy việc blocking sang threadpool. Bẫy kinh điển phỏng vấn: "gọi hàm blocking trong async không bao bọc" → mất hết lợi ích async.

## D.2. Lưu trữ: Qdrant & Redis (Q54–Q58)

**Q54. Vì sao chọn Qdrant làm vector DB thay vì FAISS / pgvector / Pinecone / Milvus?**
(1) **FAISS** là thư viện ANN thuần (in-process), không có server/persistence/filter/CRUD payload — phải tự xây tầng quản lý; Qdrant cho sẵn server + REST/gRPC + lọc payload. (2) **Filter theo payload mạnh** (`meeting_id`, `sequence_id`, `speaker`) — cực quan trọng cho dự án (cô lập cuộc họp, scroll window theo range seq). (3) **Self-host** (Docker), không phụ thuộc SaaS như Pinecone (chi phí + dữ liệu rời on-prem — họp nhạy cảm). (4) Nhẹ và đơn giản hơn Milvus (Milvus cần nhiều thành phần: etcd, MinIO, Pulsar). (5) **pgvector** hợp khi đã có Postgres và dữ liệu nhỏ, nhưng kém về ANN chuyên dụng/optimizer ở quy mô lớn. Đánh đổi: Qdrant 1 collection/cuộc gây vấn đề fd (Phase 3) — nhưng đó là cách dùng, không phải lỗi của Qdrant.

**Q55. Vì sao dùng Redis cho `sequence_id` mà không dùng chính Qdrant hay một biến trong RAM?**
(1) **Biến RAM** chết khi restart và không chia sẻ giữa nhiều worker/process → không atomic toàn cục. (2) **Qdrant không có `INCR` atomic** rẻ — muốn lấy số kế tiếp phải đọc max rồi +1 (race condition khi đồng thời). (3) **Redis `INCR` atomic, cực nhanh, in-memory** — đúng nhu cầu đếm tuần tự dưới tải đồng thời (D1). Redis chỉ là **cache tăng tốc**: mất nó vẫn self-heal từ Qdrant nên không thành điểm chết đơn (SPOF) về tính đúng đắn.

**Q56. Redis trong docker-compose bật `appendonly yes` + `save 60 1`. Nghĩa là gì và vì sao?**
Đây là **persistence**: `appendonly yes` bật **AOF** (Append-Only File — ghi từng lệnh write, bền hơn), `save 60 1` bật **RDB snapshot** (lưu nếu ≥1 key đổi trong 60s). Mục đích: nếu Redis/host restart, counter `sequence_id` và cache không mất sạch → giảm tần suất phải rebuild từ Qdrant. Vì counter có ý nghĩa (đảm bảo seq liên tục), persistence giúp khôi phục nhanh thay vì luôn rebuild.

**Q57. Redis ngoài counter còn dùng làm gì trong hệ? Vì sao không lưu context vào Redis luôn?**
Redis còn cache `meeting:{id}:latest_ctx` (context mới nhất, tăng tốc đọc nối tiếp khi build). Nhưng **nguồn chân lý của context là payload Qdrant** (D6), không phải Redis — vì (1) bền vững cùng vector, (2) cho phép **recovery scan** `context_status` sau restart, (3) trả kèm kết quả query không cần truy vấn thêm. Redis chỉ là lớp cache phía trước.

**Q58. Vì sao cần cả Qdrant lẫn Redis — không gộp một store cho gọn?**
Vì hai nhu cầu khác bản chất: Qdrant tối ưu **tìm kiếm tương đồng vector + filter payload** (không giỏi atomic counter); Redis tối ưu **thao tác atomic in-memory + cache TTL** (không làm ANN). Mỗi cái làm đúng việc của nó — ép một store làm cả hai sẽ kém ở một mặt. Đây là ví dụ "đúng công cụ cho đúng việc".

## D.3. Đóng gói & vận hành: Docker (Q59–Q61)

**Q59. Vì sao dùng Docker Compose? Lợi ích cụ thể cho hệ này?**
Hệ có **nhiều dịch vụ** (rag_api + qdrant + redis + ollama tùy chọn) cần chạy cùng nhau. Compose: (1) **một lệnh `docker compose up`** dựng cả cụm, định nghĩa khai báo (declarative). (2) **Network nội bộ** — service gọi nhau bằng tên (`QDRANT_HOST=qdrant`, `REDIS_HOST=redis`) không cần biết IP. (3) **`depends_on`** kiểm soát thứ tự khởi động. (4) **Volumes** giữ dữ liệu (`qdrant_storage`, `redis_data`) qua restart. (5) **Tái lập môi trường** — "chạy trên máy tôi cũng chạy trên server", gỡ bệnh version lệch.

**Q60. `profiles: ["ollama"]` trong compose để làm gì? Liên hệ nguyên tắc thiết kế.**
Ollama chỉ chạy khi bật profile (`docker compose --profile ollama up`). Đây là **gating ở tầng hạ tầng** — nhất quán với triết lý "gated, default off": ai dùng LLM provider khác (gemini/openai) hoặc `none` thì **không tốn tài nguyên chạy Ollama**. Tách thành phần nặng/tùy chọn ra khỏi core.

**Q61. `restart: unless-stopped` và `volumes` giải quyết vấn đề gì trong production?**
`restart: unless-stopped` — container tự bật lại khi crash/host reboot (trừ khi cố ý stop) → **độ sẵn sàng (availability)**. `volumes` map dữ liệu ra **bên ngoài container** → container có thể xóa/build lại mà **không mất dữ liệu** (vector trong `qdrant_storage`, AOF/RDB trong `redis_data`). Không có volume thì rebuild image = mất sạch dữ liệu.

## D.4. Thư viện ML & tiện ích (Q62–Q65)

**Q62. Vì sao dùng `sentence-transformers` thay vì gọi API embedding (OpenAI...) hay tự viết với transformers thô?**
(1) **Self-host, không gửi dữ liệu họp ra ngoài** (nhạy cảm) và không tốn phí API/độ trễ mạng. (2) `sentence-transformers` cho sẵn pipeline embedding chuẩn (pooling + normalize) + kho model đa ngôn ngữ — không phải tự code mean-pooling/normalize từ `transformers` thô (dễ sai). (3) Singleton nạp 1 lần lúc startup, tái dùng cho mọi request. Đánh đổi: chất lượng có thể thua model API lớn, nhưng đổi lại kiểm soát + chi phí + on-prem.

**Q63. Vì sao gọi LLM bằng `httpx` (REST) thay vì SDK riêng của từng provider?**
Một interface REST chung cho cả ollama/openai/gemini → **`LLMClient` trừu tượng hóa provider** chỉ bằng đổi URL/payload, không kéo 3 SDK nặng + xung đột version. `httpx` hỗ trợ **async** (`AsyncClient`) hợp với FastAPI, có timeout/retry dễ kiểm soát. Đổi/ thêm provider = thêm một hàm `_call_*`, không đụng nghiệp vụ. Đánh đổi: phải tự map format response mỗi provider (đã làm trong `llm_client.py`).

**Q64. Vì sao `rank-bm25` cho hybrid mà không dùng Elasticsearch / sparse vector ngay?**
**YAGNI + tối giản**: `rank-bm25` là thư viện thuần Python, chạy BM25 **trên tập ứng viên nhỏ** vector đã trả (bounded → rẻ), **không cần dựng thêm Elasticsearch** (một service nặng nữa) hay re-index Qdrant. Đủ cho nhu cầu re-rank theo từ khóa hiện tại. Khi cần re-retrieve sparse độc lập mới nâng lên Qdrant sparse vector (roadmap, đo recall trước). Triết lý: không thêm hạ tầng nặng khi một thư viện nhẹ giải quyết được.

**Q65. Vì sao dùng `pydantic-settings` cho config thay vì đọc `os.environ` trực tiếp? Ý nghĩa `extra="ignore"`?**
`pydantic-settings`: (1) **validate + ép kiểu** biến môi trường (vd `EMBEDDING_DIM` thành int, sai kiểu báo lỗi sớm), (2) **giá trị mặc định** tập trung một chỗ (`config.py`), (3) nạp từ `.env` gọn. `extra="ignore"` cho phép **bỏ qua biến môi trường lạ của host** thay vì crash khi startup — quan trọng khi deploy trên máy có nhiều env vars không liên quan. So với `os.environ` thô: an toàn kiểu, dễ test, có default rõ ràng.

---

## Phụ lục — câu hỏi ngược đáng hỏi nhà tuyển dụng
- Quy mô dữ liệu thật: số cuộc họp/tháng, độ dài trung bình → quyết định `shared` vs `sharded`.
- SLA độ trễ query và build context? Có yêu cầu real-time build context không?
- Mức độ đa-tenant / yêu cầu cô lập dữ liệu giữa tổ chức?
- Model embedding/LLM có ràng buộc on-prem (dữ liệu nhạy cảm) không?
- Có cần truy vấn xuyên nhiều cuộc họp (cross-meeting) trong roadmap không?
