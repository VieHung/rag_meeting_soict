# BKMEETING RAG Server

> Retrieval-Augmented Generation (RAG) knowledge server for the **BKMEETING — Smart Meeting Room + Virtual Secretary** product (SoICT / NAVIS Center, Hanoi University of Science and Technology).

A FastAPI service that acts as the **knowledge source** for the meeting chatbot. It exposes two independent RAG flows over a shared infrastructure:

| Flow | Namespace | Purpose |
|------|-----------|---------|
| **Documents** (Phase 1) | `/embed/*`, `/query/` | Upload PDF/DOCX/TXT/MD, embed, semantic search. |
| **Meeting transcript** (Phase 2) | `/transcript/*`, `/query/transcript` | Ingest live utterances, build a rolling LLM **context summary**, query with a ±N-sentence **window**. |

---

## Architecture at a glance

The product runs on **two independent tiers** on different hardware:

```
┌─────────────────────────────────────────────────────────────┐
│  DEVICE TIER — Qualcomm QCS8550 (one per participant)        │
│  Live transcript · face ID · small on-device LLM that writes │
│  the final answer for the user                [OUT OF SCOPE] │
└───────────────────────────┬─────────────────────────────────┘
                            │  HTTP (embed transcript / query)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  SERVER TIER — this repo (independent, more powerful host)   │
│                                                              │
│   FastAPI  ──►  EmbeddingService (MiniLM-L12-v2, 384-d)      │
│      │                                                       │
│      ├─ /embed,  /query          → Qdrant (documents)        │
│      ├─ /transcript, /query/transcript → Qdrant (meeting-*)  │
│      ├─ SequenceManager          → Redis (atomic seq + TTL)  │
│      ├─ ContextWorker (FIFO)     → self-hosted LLM summary   │
│      └─ Hybrid retrieval + Reranker (optional, RAGFlow-style)│
└─────────────────────────────────────────────────────────────┘
```

The server is the **knowledge source**; the device-tier LLM consumes `context + window + text` returned by the server to produce the user-facing answer. See **[docs/architecture.md](docs/architecture.md)** for the full design.

---

## Repository structure

```
rag_base/
├── README.md               ← you are here (project overview)
├── docs/
│   ├── architecture.md      ← detailed system architecture & design decisions
│   ├── api.md               ← API reference for device/app developers
│   └── design/
│       ├── phase2.md        ← Phase 2 design spec (transcript) — source of truth
│       └── phase3.md        ← Phase 3 roadmap (storage consolidation, RAGFlow features)
├── rag_server/             ← the FastAPI service
│   ├── app/                 ← application package (routers, services, workers, schemas)
│   ├── tests/               ← pytest suites
│   ├── Dockerfile · docker-compose.yml · requirements.txt · .env.example
│   └── README.md            ← how to run & develop the service
└── eval/                   ← RAGAS + retrieval benchmark harness
    ├── README.md            ← how to run the benchmark
    └── REPORT_COMPREHENSIVE.md  ← latest results
```

---

## Quickstart

```bash
cd rag_server
cp .env.example .env          # then edit LLM_* / REDIS_* / QDRANT_* as needed
docker compose up --build -d  # starts qdrant + redis + rag_api
curl http://localhost:8000/health
open http://localhost:8000/docs   # interactive OpenAPI docs
```

Minimal transcript round-trip:

```bash
# 1. ingest an utterance (server assigns sequence_id, builds context in background)
curl -X POST http://localhost:8000/transcript/meeting-demo/embed \
  -H 'Content-Type: application/json' \
  -d '{"speaker":"Alice","text":"We need to review the Q4 budget."}'

# 2. query with window + context
curl -X POST http://localhost:8000/query/transcript \
  -H 'Content-Type: application/json' \
  -d '{"collection":"meeting-demo","query":"Q4 budget","top_k":3,"window_size":2}'
```

Full API reference: **[docs/api.md](docs/api.md)**.

---

## Key capabilities

- **Two isolated RAG flows** distinguished by collection prefix (`docs-*` vs `meeting-*`); `meeting_id` is derived from the collection name.
- **Server-assigned sequence IDs** via atomic Redis `INCR`, self-healing from Qdrant on cache loss.
- **Rolling context summary** built by a self-hosted LLM in a strict **FIFO worker** (guarantees `context[N]` is built after `context[N-1]`).
- **Optional hybrid retrieval** (BM25 + vector fusion) and **cross-encoder reranking**, RAGFlow-inspired, gated by env vars and **off by default** — when off, behaviour is identical to pure-vector search.
- **Pluggable LLM provider** (`ollama` / `openai` / `gemini` / `none`).

---

## Evaluation

Benchmarked end-to-end with **RAGAS** (LLM judge: Gemma-4-26B) on a Vietnamese gold dataset, plus deterministic retrieval metrics.

| | Documents | Meeting transcript |
|---|---|---|
| RAGAS faithfulness | **1.00** | 0.90 |
| RAGAS context recall | **1.00** | 0.85 (hybrid) |
| RAGAS context precision | 0.92 | 0.82 (hybrid) |
| Retrieval Hit@K (hybrid) | 0.67 (Hit@5) | 1.00 (Hit@3) |

Hybrid retrieval beats pure-vector on every retrieval metric. Full report: **[eval/REPORT_COMPREHENSIVE.md](eval/REPORT_COMPREHENSIVE.md)** · methodology: **[docs/evaluation.md](docs/evaluation.md)**.

---

## Documentation

| Document | Audience |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Engineers — system design, components, data flow, design decisions |
| [docs/api.md](docs/api.md) | App/device developers — endpoint reference |
| [docs/design/phase2.md](docs/design/phase2.md) | Transcript flow design spec (source of truth) |
| [docs/design/phase3.md](docs/design/phase3.md) | Roadmap — storage consolidation, Vietnamese tokenizer, PageIndex, Agentic RAG |
| [docs/evaluation.md](docs/evaluation.md) | Benchmark methodology & results |
| [rag_server/README.md](rag_server/README.md) | How to run & develop the service |

---

## License

MIT
