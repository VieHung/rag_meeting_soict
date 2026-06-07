from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # === Qdrant ===
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # === Embedding ===
    # Backend: "sentence_transformers" (dev/CPU/GPU) | "qaic" (NPU Qualcomm AI080)
    embedding_backend: str = "sentence_transformers"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    # Prefix kiểu E5 — đặt rỗng nếu dùng model đối xứng (vd MiniLM).
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "
    # Backend qaic: đường dẫn QPC đã compile + seq_len tĩnh khi compile.
    embedding_qpc_path: Optional[str] = None
    embedding_max_seq_len: int = 128
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

    # === Context Builder LLM ===
    llm_provider: str = "ollama"          # ollama | gemini | openai | none
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://ollama:11434"
    llm_api_key: Optional[str] = None
    context_max_tokens: int = 800
    context_max_retry: int = 2
    context_timeout_seconds: float = 30.0

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
