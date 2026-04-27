"""
Configuration management for the RAG system.

All settings are loaded from environment variables (with .env support).
Supports multiple LLM backends: Ollama (default, free) and Claude (premium).
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class LLMBackend(str, Enum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    CLAUDE = "claude"
    OPENAI = "openai"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Supports .env file via python-dotenv. All values have sensible defaults
    so the system works out-of-the-box with Ollama and no API keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Backend ───────────────────────────────────────────────────────────
    llm_backend: LLMBackend = Field(
        default=LLMBackend.OLLAMA,
        description="LLM backend: 'ollama' (free local) | 'claude' | 'openai'",
    )

    # ── Anthropic / Claude ────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key (required for claude backend)")
    claude_model: str = Field(default="claude-sonnet-4-6", description="Claude model ID")

    # ── OpenAI (optional) ─────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key (required for openai backend)")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model ID")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="llama3.2", description="Ollama model name (must be pulled first)")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model for embeddings (runs locally, free)",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for embedding inference: cpu | cuda | mps",
    )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="Directory for persistent ChromaDB storage",
    )
    default_collection: str = Field(default="default", description="Default knowledge base collection name")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=6, ge=1, le=50, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum cosine similarity to include a result")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0, description="MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity")
    use_reranker: bool = Field(default=True, description="Apply cross-encoder reranking after retrieval")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    use_hyde: bool = Field(default=False, description="Use HyDE (Hypothetical Document Embeddings) for query expansion")
    use_multi_query: bool = Field(default=False, description="Expand query into multiple sub-queries")
    use_hybrid_search: bool = Field(default=True, description="Combine dense + sparse (BM25) retrieval")
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for dense search in hybrid mode (1-alpha for BM25)")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=64, le=4096, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=64, ge=0, le=512, description="Overlap between consecutive chunks")
    use_semantic_chunking: bool = Field(default=False, description="Use semantic (sentence-boundary) chunking instead of recursive")

    # ── Generation ────────────────────────────────────────────────────────────
    max_tokens: int = Field(default=1024, ge=64, le=8192, description="Max tokens for LLM response")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM temperature (lower = more factual)")

    # ── Caching ───────────────────────────────────────────────────────────────
    enable_cache: bool = Field(default=True, description="Enable semantic query cache")
    cache_similarity_threshold: float = Field(default=0.95, description="Similarity threshold for cache hits")
    cache_max_size: int = Field(default=500, description="Max number of cached query-answer pairs")

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="FastAPI server host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="FastAPI server port")
    api_workers: int = Field(default=1, ge=1, description="Number of uvicorn workers")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")

    # ── Contextual Retrieval (Anthropic Nov 2024) ─────────────────────────────
    use_contextual_retrieval: bool = Field(
        default=False,
        description="Prepend LLM-generated context to each chunk before embedding (49% fewer retrieval failures)",
    )
    contextual_retrieval_use_haiku: bool = Field(
        default=True,
        description="Use claude-haiku for contextual retrieval (cheaper than full model)",
    )

    # ── Web Search (CRAG fallback) ────────────────────────────────────────────
    tavily_api_key: str = Field(default="", description="Tavily API key for web search fallback")
    web_search_fallback: bool = Field(
        default=False,
        description="Fall back to web search when CRAG confidence is low",
    )
    web_search_provider: str = Field(
        default="tavily",
        description="Web search provider: 'tavily' | 'duckduckgo'",
    )
    crag_web_fallback_threshold: float = Field(
        default=0.4,
        description="CRAG confidence threshold below which web search is triggered",
    )

    # ── Langfuse Observability ────────────────────────────────────────────────
    langfuse_public_key: str = Field(default="", description="Langfuse public key (pk-lf-...)")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key (sk-lf-...)")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse host (use https://cloud.langfuse.com or self-hosted URL)",
    )

    # ── Agentic RAG ───────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(
        default=8,
        description="Max tool-call iterations for the agentic RAG loop",
    )

    # ── SQL / Structured Data ─────────────────────────────────────────────────
    sql_database_url: str = Field(
        default="",
        description="SQLAlchemy database URL (e.g., sqlite:///./data/mydb.db or postgresql://...)",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    enable_pii_redaction: bool = Field(
        default=False,
        description="Redact PII from ingested documents before embedding",
    )
    enable_injection_detection: bool = Field(
        default=True,
        description="Detect and sanitize prompt injection patterns in retrieved chunks",
    )
    enable_audit_log: bool = Field(
        default=True,
        description="Log query audit entries (hashed queries, PII flags) to data/audit_log.jsonl",
    )
    enable_pii_presidio: bool = Field(
        default=False,
        description="Use Microsoft Presidio for ML-based PII detection (requires: pip install presidio-analyzer)",
    )

    # ── GraphRAG Community Detection (Microsoft style) ────────────────────────
    graphrag_community_detection: bool = Field(
        default=False,
        description="Build community summaries for global GraphRAG queries",
    )

    # ── Sufficient Context (Google ICLR 2025) ────────────────────────────────
    enable_sufficient_context: bool = Field(
        default=True,
        description="Score context sufficiency before generating — abstain or retrieve more if insufficient",
    )
    sufficient_context_threshold: float = Field(
        default=0.45,
        ge=0.0, le=1.0,
        description="Sufficiency score below which the system retrieves more or abstains [0-1]",
    )
    sufficient_context_self_rating: bool = Field(
        default=False,
        description="Ask LLM to self-rate context confidence (adds ~200ms latency)",
    )

    # ── CoT-RAG (EMNLP 2025) ─────────────────────────────────────────────────
    cot_max_steps: int = Field(
        default=4,
        ge=1, le=10,
        description="Maximum reasoning steps for CoT-RAG",
    )
    cot_top_k_per_step: int = Field(
        default=3,
        ge=1, le=20,
        description="Chunks to retrieve per CoT reasoning step",
    )

    # ── LightRAG (EMNLP 2025) ─────────────────────────────────────────────────
    lightrag_max_low_level_hops: int = Field(
        default=2,
        description="Max graph hops for LightRAG low-level retrieval",
    )
    lightrag_max_communities: int = Field(
        default=5,
        description="Max community summaries to include in LightRAG high-level retrieval",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    @field_validator("chroma_persist_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Resolve chroma persist dir to absolute path."""
        return Path(v).resolve()

    def validate_backend_credentials(self) -> None:
        """
        Raise ValueError if required credentials are missing for the selected backend.
        Called at startup, not at import time.
        """
        if self.llm_backend == LLMBackend.CLAUDE and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_BACKEND=claude. "
                "Set it in .env or switch to LLM_BACKEND=ollama for free local inference."
            )
        if self.llm_backend == LLMBackend.OPENAI and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_BACKEND=openai. "
                "Set it in .env or switch to LLM_BACKEND=ollama for free local inference."
            )


# Module-level singleton — import this everywhere
settings = Settings()

# Configure root logger once at import time
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
