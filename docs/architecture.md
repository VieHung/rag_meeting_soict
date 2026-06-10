# System Architecture

This document describes the design of the **BKMEETING RAG Server** — its components, data
model, request flows, and the design decisions behind them. For the API contract see
[api.md](api.md); for the design specs see [design/phase2.md](design/phase2.md) and
[design/phase3.md](design/phase3.md).

---

## 1. Context

**BKMEETING** is a smart meeting-room product that combines a virtual secretary with
on-device AI. It runs on **two independent tiers** on different hardware:

| Tier | Hardware | Responsibility |
|------|----------|----------------|
| **Device** | Qualcomm QCS8550 (one workstation per participant) | Live transcription, face ID, translation, and a **small on-device LLM** that writes the final answer shown to the user. *Out of scope for this repo.* |
| **Server** | An independent, more powerful host | This repo: **RAG API + Qdrant + Redis + a self-hosted LLM** that builds the rolling meeting context. |

The server is a **knowledge source**, not an answer generator. The device sends a question,
the server returns the most relevant content (`context + window + text`), and the device-tier
LLM composes the user-facing answer. Two query flows exist and the **app chooses** which to use:

1. **Document retrieval** — meeting documents uploaded ahead of time (Phase 1).
2. **Transcript retrieval** — conversation already transcribed during the meeting (Phase 2).

---

## 2. High-level architecture

```
                         ┌──────────── FastAPI application (app/main.py) ───────────┐
                         │                                                          │
  HTTP ───────────────►  │  Routers                                                 │
                         │   ├─ embed.py      /embed/*           (documents)         │
                         │   ├─ query.py      /query/, /query/transcript            │
                         │   └─ transcript.py /transcript/*      (transcript)        │
                         │            │                                              │
                         │            ▼                                              │
                         │  Services                                                 │
                         │   ├─ EmbeddingService   (MiniLM-L12-v2, 384-d, singleton) │
                         │   ├─ QdrantService      (Phase 1 documents)               │
                         │   ├─ TranscriptStore    (Phase 2 meeting-* collections)   │
                         │   ├─ TranscriptService  (orchestrates embed/query/window) │
                         │   ├─ SequenceManager    (atomic sequence_id)              │
                         │   ├─ ContextBuilder     (LLM rolling summary)             │
                         │   ├─ LLMClient          (ollama/openai/gemini/none)       │
                         │   ├─ retrieval.fuse     (hybrid BM25+vector — optional)   │
                         │   └─ Reranker           (cross-encoder — optional)        │
                         │            │                                              │
                         │  Workers                                                  │
                         │   └─ ContextWorker      (asyncio.Queue, single FIFO task) │
                         └────────┬──────────────┬───────────────┬──────────────────┘
                                  ▼              ▼               ▼
                            ┌──────────┐   ┌──────────┐   ┌──────────────────┐
                            │  Qdrant  │   │  Redis   │   │  Self-hosted LLM │
                            │ vectors  │   │ seq ctr  │   │ (context build)  │
                            └──────────┘   └──────────┘   └──────────────────┘
```

---

## 3. Components

All paths are under `rag_server/app/`.

### Entry point & wiring
- **`main.py`** — FastAPI app, CORS, logging, and the `lifespan` that warms the embedding
  model, connects Qdrant/Redis, initializes the LLM client + reranker, and **starts/stops
  the ContextWorker**. Hosts `GET /health`, which checks Qdrant + Redis and reports the LLM
  provider (returns `503` if a core dependency is down).
- **`config.py`** — `pydantic-settings` model loading from `.env`. `extra = "ignore"` so stray
  host env vars do not break startup. See §7 for the variables.
- **`dependencies.py`** — FastAPI DI providers; `TranscriptService` is a process singleton.

### Routers (HTTP surface)
- **`routers/embed.py`** — document ingestion & collection management: `/embed/file`,
  `/embed/text`, `/embed/collections` (GET/POST/DELETE), `/embed/info[/{collection}]`,
  `/embed/{collection}/documents`, delete-by-source / delete-by-doc_id.
- **`routers/query.py`** — `/query/` (documents) and `/query/transcript` (transcript). Both
  route through the optional hybrid + rerank pipeline.
- **`routers/transcript.py`** — `/transcript/{collection}/embed`,
  `/transcript/{collection}/context`, `/transcript/{collection}/segments`. Validates the
  `meeting-` prefix and **enqueues** the context-build job.

### Services (business logic)
- **`services/embedding.py`** — `EmbeddingService`: a singleton `SentenceTransformer`
  (`paraphrase-multilingual-MiniLM-L12-v2`, 384-d, normalized). Used by both flows.
- **`services/vector_store.py`** — `QdrantService`: Phase 1 documents. Auto-creates a
  collection on first use, upsert/search/scroll, collection CRUD, `collection_info`.
- **`services/transcript_store.py`** — `TranscriptStore`: Phase 2. Tailored payload indexes
  (`meeting_id`, `sequence_id`, `speaker`); upsert, **filtered** search, range scroll for the
  window, context payload update, `get_max_sequence_id` for counter rebuild.
- **`services/transcript_service.py`** — `TranscriptService`: orchestrates `embed_transcript`,
  `query_transcript` (with window fetch), `get_context`, `list_segments`; derives `meeting_id`
  from the collection name; applies hybrid + rerank on query.
- **`services/sequence_manager.py`** — `SequenceManager`: atomic `sequence_id` per collection
  via Redis `INCR` with a 7-day TTL. **Self-healing**: if the Redis key is missing it rebuilds
  the counter from the max `sequence_id` in Qdrant.
- **`services/context_builder.py`** — `ContextBuilder`: builds the rolling summary
  `context[N] = LLM_summarize(context[N-1] + transcript[N-1])`, marks status
  `pending → processing → ready|failed`, retries with backoff, caches the latest context in
  Redis. **Logic is provider-agnostic** and called by the worker.
- **`services/llm_client.py`** — `LLMClient`: abstraction over `ollama`, `openai`
  (OpenAI-compatible incl. LM Studio/vLLM), `gemini`, and `none`. Houses the Vietnamese
  summarization system prompt.
- **`services/retrieval.py`** — `fuse()`: RAGFlow-style hybrid fusion. BM25 over the candidate
  set + vector score, min-max normalized: `score = vec_w·vec + term_w·term`. No re-indexing.
- **`services/reranker.py`** — `Reranker`: gated cross-encoder rerank (`none|local|http`).
  `local` uses `sentence-transformers` CrossEncoder; `http` calls an external TEI/Infinity
  endpoint.
- **`services/document_parser.py`** — PDF/DOCX/TXT/MD parsing for `/embed/file`.

### Workers
- **`workers/context_worker.py`** — `ContextWorker`: a single `asyncio.Queue` consumer task.
  `/transcript/{collection}/embed` enqueues `(collection, meeting_id, sequence_id)`; the worker
  drains jobs **strictly in order**, constructs a `ContextBuilder` per job, and builds the
  context. This guarantees the FIFO ordering that `BackgroundTasks` could not (see D9).

### Utilities
- **`utils/redis_client.py`** — singleton async Redis client.
- **`utils/chunking.py`** — paragraph→sentence chunker for documents (512 chars, 64 overlap).

---

## 4. Data model

### 4.1 Document vector — `docs-*` / `documents` (Phase 1)
```jsonc
{
  "text": "...",            // chunk text
  "source": "report.pdf",   // file name
  "doc_id": "uuid",
  "chunk_index": 0,
  "chunk_total": 24,
  "file_size": 12345,
  "mime_type": "application/pdf"
}
```

### 4.2 Transcript vector — `meeting-*` (Phase 2)
```jsonc
{
  "meeting_id":     "c7cfdf57-...",  // derived from collection name
  "sequence_id":    42,              // server-assigned, atomic, contiguous
  "speaker":        "Đoàn Sỹ Nguyên",
  "speaker_id":     "user_017",      // optional
  "text":           "Chúng ta cần xem lại ngân sách Q4...",
  "timestamp":      "2026-05-11T19:52:27Z",
  "context":        "Rolling summary up to this utterance...",  // LLM-generated
  "context_status": "ready",         // pending | processing | ready | failed | disabled
  "context_seq_base": 41
}
```
- One utterance = exactly **one vector** (no chunking — D4).
- `context[N]` is the *context leading up to* utterance N (it does **not** include N itself).
- Vector size **384**, distance **Cosine**.
- Point IDs are **random `uuid4`** (both flows) — globally unique, so points from different
  meetings can never collide; the trade-off is that re-ingesting the same utterance creates a
  duplicate point (no idempotency). Phase 3 switches transcript points to deterministic
  `uuid5(meeting_id, sequence_id)` for idempotent migration/re-ingest.

### 4.3 Redis state
```
rag:seq:{collection}          → INTEGER   # atomic counter, 7-day TTL, rebuildable from Qdrant
meeting:{meeting_id}:latest_ctx → JSON     # optional cache of the most recent context
```

---

## 5. Request flows

### 5.1 Ingest transcript (synchronous + enqueue)
```
POST /transcript/{collection}/embed
 ├ validate prefix `meeting-` (else 400) and non-empty text (else 422)
 ├ TranscriptStore.ensure_collection()                 (lazy create)
 ├ SequenceManager.next(collection)  → Redis INCR      → sequence_id = N
 ├ EmbeddingService.embed_query(text)                  → vector[384]
 ├ TranscriptStore.upsert_point(..., context_status="pending"|"disabled")
 ├ ContextWorker.enqueue(collection, meeting_id, N)    (FIFO — D9)
 └ 202 { meeting_id, sequence_id: N, point_id, context_status }
```

### 5.2 Build context (background FIFO worker)
```
ContextWorker drains job (collection, meeting_id, N):
 ├ LLM_PROVIDER=none → mark "disabled", stop
 ├ N == 1 (first utterance) → context="", status="ready", no LLM call
 ├ read utterance N-1 from Qdrant → context[N-1], text[N-1]
 ├ mark point N "processing"
 ├ LLMClient.summarize(context[N-1], text[N-1])  (retry CONTEXT_MAX_RETRY)
 ├ write point N: context=context[N], status="ready", context_seq_base=N-1
 └ on failure → context[N]=context[N-1], status="failed"
```
The single-consumer queue guarantees `context[N-1]` is finished before `context[N]` starts.

### 5.3 Query transcript (with window + optional hybrid/rerank)
```
POST /query/transcript
 ├ validate prefix `meeting-`
 ├ fetch_k = top_k × HYBRID_FETCH_MULTIPLIER  (if hybrid or rerank enabled, else top_k)
 ├ EmbeddingService.embed_query(query)
 ├ TranscriptStore.search(vector, fetch_k, meeting_id, speaker_filter, score_threshold)
 ├ retrieval.fuse(query, candidates)          (if HYBRID_ENABLED)
 ├ Reranker.rerank(query, candidates)         (if RERANK_PROVIDER != none)
 ├ trim to top_k
 ├ for each hit (seq S): scroll sequence_id ∈ [S-w, S+w] → window.before/after
 └ 200 { results: [{ sequence_id, text, score, context, context_status, window }] }
```

### 5.4 Document flow (Phase 1)
`/embed/file` and `/embed/text` parse → chunk → embed → upsert (in a background task).
`/query/` embeds the query, searches Qdrant, applies the same optional hybrid+rerank pipeline,
and filters by `score_threshold` on the original vector score before fusion.

---

## 6. Design decisions

From [design/phase2.md](design/phase2.md) §4, as implemented:

| # | Decision |
|---|----------|
| **D1** | Server assigns `sequence_id` (Redis `INCR` per collection, self-healing from Qdrant). |
| **D2** | Query returns a ±N-sentence window (default 2, clamped to `TRANSCRIPT_MAX_WINDOW_SIZE`). |
| **D3** | Context is an **LLM summary**, not raw concatenation: `context[N]=summarize(context[N-1]+text[N-1])`. |
| **D4** | No chunking of transcript — one utterance = one vector. |
| **D5** | The context-building LLM is **self-hosted on the server** and called in-process. |
| **D6** | Context is stored in the vector **payload**, not Redis. |
| **D7** | Reuse the `MiniLM-L12-v2` (384-d) embedding model for both flows. |
| **D8** | `meeting_id` is **derived from the collection name**; clients do not send it. |
| **D9** | Context is built **strictly in order** via a single-consumer FIFO worker. |

Additions delivered after the original spec (RAGFlow-inspired, gated, off by default):
- **Hybrid retrieval** (`retrieval.fuse`) and **cross-encoder reranking** (`reranker.py`),
  shared by both query flows; response shape is unchanged (only `score` reflects the final score).
- **`/health`** checks Qdrant + Redis; **`/embed/info[/{collection}]`** wired; structured logging
  in `main.py`/services (the background tasks in `routers/embed.py` still use `print()` — minor
  cleanup pending).

---

## 7. Configuration

All variables have safe defaults (see `rag_server/.env.example`). Highlights:

| Group | Key variables |
|-------|---------------|
| Qdrant | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION_NAME` |
| Embedding | `EMBEDDING_MODEL`, `EMBEDDING_DIM=384`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_DEFAULT` |
| Redis | `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `SEQ_KEY_TTL_SECONDS=604800` |
| Transcript | `TRANSCRIPT_SEQ_START=1`, `TRANSCRIPT_WINDOW_SIZE=2`, `TRANSCRIPT_MAX_WINDOW_SIZE=5` |
| Context LLM | `LLM_PROVIDER` (`ollama`/`openai`/`gemini`/`none`), `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `CONTEXT_MAX_TOKENS`, `CONTEXT_MAX_RETRY`, `CONTEXT_TIMEOUT_SECONDS` |
| Hybrid (optional) | `HYBRID_ENABLED=false`, `HYBRID_VECTOR_WEIGHT=0.7`, `HYBRID_TERM_WEIGHT=0.3`, `HYBRID_FETCH_MULTIPLIER=3` |
| Reranker (optional) | `RERANK_PROVIDER=none`, `RERANK_MODEL`, `RERANK_BASE_URL`, `RERANK_API_KEY` |

**Gating principle:** with `HYBRID_ENABLED=false` and `RERANK_PROVIDER=none` (the defaults),
no extra models are loaded and query behaviour is identical to pure-vector search.

---

## 8. Storage partition model & scaling

**Deployment snapshot** (`docker-compose.yml`, verified against code): Qdrant **v1.10.0**
(client `qdrant-client==1.10.1` — `scroll(order_by=…)` is supported), Redis 7.2 (AOF + save),
optional Ollama profile, and the API container. No `ulimits` are configured yet (Phase 3 S1).
The `QdrantClient` is a **process-wide singleton** (class-level in both `QdrantService` and
`TranscriptStore`), so the fd pressure described below comes from the Qdrant server side, not
from client connections.

A deliberate contrast with RAGFlow frames the current design and its main scaling risk.

- **RAGFlow** separates *logical* partitions from the *physical* store: one index **per tenant**
  (`ragflow_{tenant_id}`), with every knowledge base / document isolated by a **filter field**
  (`kb_id`, `doc_id`). Index count grows with tenants (few), not with documents (many).
- **This server** currently couples logical and physical: **one Qdrant collection per meeting**
  (`meeting-{uuid}`). This is a side effect of D8 (derive `meeting_id` from the collection name)
  and is the root of the scaling limitation below.
- The document side has the same shape: the `collection` form param on `/embed/file`,
  `/embed/text` and `POST /embed/collections` let clients create **arbitrarily many document
  collections** (auto-created on first use). The fd math below therefore applies to `docs-*`
  collections too, at whatever rate clients mint them.

### 8.1 The "too many open files" risk (file descriptors ≠ RAM)

Each Qdrant collection — even a near-empty, idle one — keeps its segment files (RocksDB `.sst`,
mmap) **open**; Qdrant loads all collections at startup and does not close idle-collection fds.
So:

```
file_descriptors ≈ N_meetings × segments_per_collection × files_per_segment   (grows linearly with meetings)
```

Two **orthogonal** concerns are easy to conflate:

| | File descriptors (open files) | RAM (resident data) |
|---|---|---|
| Cause of "too many open files" | ✅ this | ❌ unrelated |
| Idle collection (never queried) | still holds files open | vectors still resident (default) |
| Fixed by *consolidating* collections | ✅ (segment count is capped by the merge optimizer) | partly |
| Fixed by `on_disk`/memmap | partly | ✅ (OS page cache handles hot/cold) |

The complete fix is a **package, applied together** — consolidate into few physical collections
(`meeting_id` filter) **+** `ulimit nofile = 65535` **+** `on_disk` vectors/payload **+** an optimizer
tuned for few large segments. Consolidation alone is not guaranteed to suffice (a single large
collection can still exhaust fds via per-segment `.sst` files). `on_disk` gives the RAM-tiering
("push cold data to SSD") effect for free, so no hot/warm/cold lifecycle code is needed. Migration
**consolidates without deleting any meeting content** — it removes the empty collection shells after
copying their points into the shared collection. The full 4-step plan (diagnose → patch foundation →
consolidate → migrate + delete old) and its traps (e.g. globally-unique `point_id` to avoid silent
overwrite on merge) are specified in [design/phase3.md](design/phase3.md) as the top-priority item.

## 9. Other known limitations & roadmap

- **`get_max_sequence_id()` is wrong past 100 points** (`transcript_store.py`) — the self-healing
  counter rebuild scrolls a **single page of 100 points** (no pagination, no `meeting_id` filter)
  and takes the max. For a meeting with >100 utterances whose Redis key has expired (7-day TTL),
  the rebuilt counter can be **too low → duplicate `sequence_id`s** are then issued. This is a live
  correctness bug independent of the storage layout; the fix
  (`scroll(order_by=sequence_id desc, limit=1, filter=meeting_id)`, supported on Qdrant ≥ v1.8;
  deployed server is v1.10.0) is part of Phase 3 S2.
- **Worker durability** — `ContextWorker` uses an in-memory `asyncio.Queue`; a restart with pending
  jobs leaves utterances stuck at `context_status="pending"`. A cheap fix is a **startup recovery
  scan** that re-enqueues `pending`/`processing` points (their status lives in the Qdrant payload,
  D6); see [design/phase3.md](design/phase3.md) §B1.
- **Dead code** — `SequenceManager._ensure_collection_exists` is never called and would raise
  `AttributeError` if it were (`QdrantService` has no instance `collection_exists` /
  zero-arg `create_collection`); scheduled for removal in Phase 3.
- **`LLM_API_KEY`** lives in `.env` (gitignored, untracked) but in plaintext on disk — rotate it
  and use secret injection for production.
- **Transcript answer relevancy** is the weakest evaluated metric (~0.66) — candidate for a
  better answer-generation prompt (see [evaluation.md](evaluation.md)).
- Further roadmap (Vietnamese tokenizer, sparse hybrid, per-collection worker, PageIndex,
  Agentic RAG) is detailed in [design/phase3.md](design/phase3.md).
