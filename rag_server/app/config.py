from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # === Qdrant ===
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # === Embedding ===
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_default: int = 5

    # === Redis (sequence + context cache) ===
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    seq_key_ttl_seconds: int = 604800  # 7 days

    # === Transcript ===
    transcript_seq_start: int = 1
    transcript_window_size: int = 2
    transcript_max_window_size: int = 5
    transcript_default_collection: str = "default"
    transcript_collection_prefix: str = "meeting-"
    docs_collection_prefix: str = "docs-"

    # === Transcript storage layout (Phase 3 S2) ===
    # shared     : tất cả cuộc họp → 1 collection dùng chung (default, khuyến nghị)
    # sharded    : hash(meeting_id) % NUM_SHARDS → nhiều bucket (YAGNI — đo trước khi bật)
    # per_meeting: 1 collection / 1 cuộc họp (tương thích ngược với Phase 1–2)
    transcript_storage_layout: str = "shared"
    transcript_shared_collection: str = "meeting_transcripts"
    transcript_num_shards: int = 8

    # === Qdrant storage options (Phase 3 S1/S2) ===
    # on_disk=True: vector memmap → không thường trú RAM; OS page cache lo hot/cold.
    # on_disk_payload=True: payload (context summary) xuống disk.
    qdrant_on_disk: bool = True
    qdrant_on_disk_payload: bool = True

    # === Context Builder LLM ===
    llm_provider: str = "ollama"          # ollama | gemini | openai | none
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://ollama:11434"
    llm_api_key: Optional[str] = None
    context_max_tokens: int = 800
    context_max_retry: int = 2
    context_timeout_seconds: float = 30.0

    # === Context Worker (Phase 3 B1) ===
    # concurrency: số cuộc họp build context đồng thời (FIFO vẫn đảm bảo trong mỗi cuộc).
    # recovery_scan: lúc startup quét lại điểm pending/processing → enqueue lại.
    # idle_ttl: worker của 1 cuộc họp tự xóa sau N giây không có job mới.
    context_worker_concurrency: int = 4
    context_recovery_scan: bool = True
    worker_idle_ttl: int = 600

    # === Hybrid retrieval (RAGFlow-style fusion: term + vector) — OPTIONAL ===
    hybrid_enabled: bool = False
    hybrid_vector_weight: float = 0.7
    hybrid_term_weight: float = 0.3
    hybrid_fetch_multiplier: int = 3
    # Tokenizer cho BM25 (Phase 3 A1): simple | pyvi | underthesea
    # simple: regex \w+ (zero-dep, mỗi âm tiết = 1 token)
    # pyvi  : ViTokenizer — segment từ ghép, cần `pip install pyvi`
    # underthesea: word_tokenize — nặng hơn, cần `pip install underthesea`
    hybrid_tokenizer: str = "simple"

    # === Reranker (cross-encoder) — OPTIONAL, gated như LLM ===
    rerank_provider: str = "none"         # none | local | http
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_base_url: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_timeout_seconds: float = 30.0

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
