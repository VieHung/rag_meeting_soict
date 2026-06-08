# rag_server

The FastAPI service for the BKMEETING RAG knowledge server. For the system overview and design
see the repo root [README](../README.md) and [docs/architecture.md](../docs/architecture.md);
for the endpoint reference see [docs/api.md](../docs/api.md).

## Stack

- **FastAPI** (async, OpenAPI docs at `/docs`)
- **Qdrant** vector DB (cosine, 384-d)
- **Redis** — atomic sequence counter (Phase 2)
- **Embedding**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Self-hosted LLM** for rolling context (pluggable: `ollama`/`openai`/`gemini`/`none`)
- Optional **hybrid retrieval** (BM25 + vector) and **cross-encoder reranking**

## Run with Docker (recommended)

```bash
cp .env.example .env          # edit LLM_* / QDRANT_* / REDIS_* as needed
docker compose up --build -d  # qdrant + redis + rag_api
curl http://localhost:8000/health
# OpenAPI docs: http://localhost:8000/docs
```

`docker compose` starts `qdrant`, `redis`, and `rag_api`. An optional `ollama` service is
available via `--profile ollama`; when not used, set `LLM_PROVIDER=none` to disable context building.

## Run locally (dev)

```bash
pip install -r requirements.txt
# requires a reachable Qdrant + Redis (e.g. docker compose up qdrant redis -d)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Layout

```
app/
├── main.py            # FastAPI app, lifespan, /health
├── config.py          # pydantic-settings (.env)
├── dependencies.py    # DI wiring
├── routers/           # embed.py, query.py, transcript.py
├── schemas/           # pydantic request/response models
├── services/          # embedding, vector_store, transcript_store, transcript_service,
│                      # sequence_manager, context_builder, llm_client, retrieval, reranker, document_parser
├── workers/           # context_worker.py (FIFO context build)
└── utils/             # redis_client.py, chunking.py
tests/                 # pytest suites
```

## Configuration

All variables have defaults; see [`.env.example`](.env.example) and
[docs/architecture.md §7](../docs/architecture.md#7-configuration). Hybrid retrieval and reranking
are **off by default** — enabling them does not change response shape.

## Tests

```bash
# Phase 1 / Phase 2 integration tests need the stack running (qdrant + redis + rag_api).
# On Windows consoles run with PYTHONUTF8=1 to avoid cp1252 errors on Vietnamese output.
pytest tests/test_sequence_manager.py -v        # unit (needs Redis; +Qdrant for rebuild test)
pytest tests/test_transcript_api.py -v          # integration (set LLM_PROVIDER=none for stability)
pytest tests/test_api.py -v                      # Phase 1 integration
```

## Endpoints (summary)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + Qdrant/Redis/LLM status |
| POST | `/embed/file`, `/embed/text` | Ingest documents |
| GET/POST/DELETE | `/embed/collections` | Collection management |
| GET | `/embed/info[/{collection}]`, `/embed/{collection}/documents` | Collection info / listing |
| POST | `/query/` | Document semantic search |
| POST | `/transcript/{collection}/embed` | Ingest one utterance (202) |
| POST | `/query/transcript` | Transcript search with window + context |
| GET | `/transcript/{collection}/context` | Latest context (or `?sequence_id=`) |
| GET | `/transcript/{collection}/segments` | List utterances by range |

Full request/response details: [docs/api.md](../docs/api.md).
