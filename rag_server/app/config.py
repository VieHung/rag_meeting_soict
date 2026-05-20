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
