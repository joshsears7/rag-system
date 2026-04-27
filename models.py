"""
Pydantic v2 data models for the RAG system.

These models are shared across ingestion, retrieval, generation, the REST API,
and the evaluation harness — single source of truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class DocumentType(str, Enum):
    """Supported document source types."""

    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MARKDOWN = "markdown"
    URL = "url"
    UNKNOWN = "unknown"


class QueryMode(str, Enum):
    """Retrieval strategy modes."""

    DENSE = "dense"          # pure vector search
    SPARSE = "sparse"        # pure BM25
    HYBRID = "hybrid"        # dense + BM25 with RRF fusion


# ── Chunk / Document Models ───────────────────────────────────────────────────


class ChunkMetadata(BaseModel):
    """Rich metadata attached to every stored chunk."""

    source_file: str = Field(..., description="Original filename or URL")
    doc_type: DocumentType = Field(default=DocumentType.UNKNOWN)
    page_number: int | None = Field(default=None, description="PDF page number (1-indexed)")
    chunk_index: int = Field(..., ge=0, description="Position of chunk within its source document")
    timestamp_ingested: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    word_count: int = Field(..., ge=0)
    char_count: int = Field(..., ge=0)
    content_hash: str = Field(..., description="SHA-256 of chunk text for deduplication")
    section_title: str | None = Field(default=None, description="Nearest heading above this chunk (if detectable)")


class DocumentChunk(BaseModel):
    """A single chunk ready for embedding and storage."""

    text: str = Field(..., min_length=1)
    metadata: ChunkMetadata
    embedding: list[float] | None = Field(default=None, exclude=True)

    @property
    def chunk_id(self) -> str:
        """Stable ID derived from content hash + chunk index."""
        return f"{self.metadata.content_hash[:16]}-{self.metadata.chunk_index}"


# ── Ingestion Models ──────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """REST API request body for /ingest."""

    file_path: str = Field(..., description="Absolute or relative path to file, or a URL")
    collection: str = Field(default="default", min_length=1, max_length=64)
    overwrite: bool = Field(default=False, description="Re-ingest even if chunk hash already exists")


class IngestResult(BaseModel):
    """Result of an ingestion operation."""

    collection: str
    source: str
    chunks_added: int = Field(ge=0)
    duplicates_skipped: int = Field(ge=0)
    total_chunks_processed: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def check_totals(self) -> "IngestResult":
        assert self.chunks_added + self.duplicates_skipped == self.total_chunks_processed
        return self


# ── Retrieval Models ──────────────────────────────────────────────────────────


class RetrievalResult(BaseModel):
    """A single retrieved chunk with scoring information."""

    chunk_text: str
    source: str = Field(..., description="Source filename or URL")
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    rerank_score: float | None = Field(default=None, description="Cross-encoder score (higher = more relevant)")
    chunk_index: int
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalContext(BaseModel):
    """All retrieved chunks for a single query, ready for prompt construction."""

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    query_mode: QueryMode = QueryMode.HYBRID
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expanded_queries: list[str] = Field(default_factory=list, description="Multi-query expansions used")
    hyde_hypothesis: str | None = Field(default=None, description="HyDE hypothetical document if used")

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0


# ── Query / Response Models ───────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """REST API request body for /query."""

    question: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field(default="default")
    top_k: int = Field(default=6, ge=1, le=50)
    mode: QueryMode = Field(default=QueryMode.HYBRID)
    use_hyde: bool = Field(default=False)
    use_multi_query: bool = Field(default=False)


class SourceCitation(BaseModel):
    """A source citation returned with every answer."""

    source: str
    chunk_index: int
    page_number: int | None = None
    similarity_score: float
    excerpt: str = Field(..., description="First 200 chars of the chunk")


class QueryResponse(BaseModel):
    """REST API response for /query — also used internally."""

    question: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    tokens_used: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    collection: str
    llm_backend: str
    model_used: str
    cache_hit: bool = Field(default=False)
    retrieval_context: RetrievalContext | None = Field(default=None, exclude=True)


# ── Collection Models ─────────────────────────────────────────────────────────


class CollectionInfo(BaseModel):
    """Metadata about a ChromaDB collection (knowledge base)."""

    name: str
    document_count: int = Field(ge=0)
    created_at: datetime | None = None
    embedding_model: str


class CollectionListResponse(BaseModel):
    """Response for GET /collections."""

    collections: list[CollectionInfo]
    total: int = Field(ge=0)

    @model_validator(mode="after")
    def set_total(self) -> "CollectionListResponse":
        self.total = len(self.collections)
        return self


class DeleteCollectionResponse(BaseModel):
    """Response for DELETE /collection/{name}."""

    name: str
    deleted: bool
    message: str


# ── Evaluation Models ─────────────────────────────────────────────────────────


class EvalSample(BaseModel):
    """A single (question, expected_answer, relevant_sources) test case."""

    question: str
    expected_answer: str
    relevant_sources: list[str] = Field(default_factory=list, description="Filenames that should appear in top-k")
    collection: str = Field(default="default")


class EvalResult(BaseModel):
    """Evaluation result for a single test case."""

    question: str
    generated_answer: str
    expected_answer: str
    sources_retrieved: list[str]
    relevant_sources: list[str]
    recall_at_k: float = Field(ge=0.0, le=1.0, description="Fraction of relevant sources found in top-k")
    faithfulness_score: float = Field(ge=1.0, le=5.0, description="LLM-judged faithfulness score (1-5)")
    answer_relevancy: float = Field(ge=0.0, le=1.0, description="Cosine similarity of answer embedding to question")
    latency_ms: float = Field(ge=0.0)


class EvalSummary(BaseModel):
    """Aggregate evaluation summary across all test cases."""

    total_samples: int = Field(ge=0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_faithfulness: float = Field(ge=1.0, le=5.0)
    mean_answer_relevancy: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)
    results: list[EvalResult] = Field(default_factory=list)


# ── Cache Models ──────────────────────────────────────────────────────────────


class CacheEntry(BaseModel):
    """A cached query-answer pair with embedding for similarity lookup."""

    question: str
    response: QueryResponse
    embedding: list[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hit_count: int = Field(default=0, ge=0)


# ── Error Models ──────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response for the REST API."""

    error: str
    detail: str | None = None
    code: int
