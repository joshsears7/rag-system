"""
Observability layer — Prometheus metrics + structured request logging.

Exposes a /metrics endpoint compatible with Prometheus scraping.
Tracks:
  - Request counts and latencies per endpoint
  - Retrieval quality scores
  - Cache hit rates
  - Token usage
  - LLM backend errors
  - Ingestion throughput

Usage in FastAPI:
    from monitoring import instrument_app
    instrument_app(app)

Requires: pip install prometheus-client
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Prometheus metrics (optional dependency) ──────────────────────────────────

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        make_asgi_app,
        multiprocess,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus-client not installed. Metrics endpoint disabled. pip install prometheus-client")


def _make_metrics():
    """Initialize Prometheus metrics (only if library available)."""
    if not PROMETHEUS_AVAILABLE:
        return None

    class Metrics:
        # Request tracking
        request_count = Counter(
            "rag_requests_total",
            "Total number of requests",
            ["endpoint", "method", "status_code"],
        )
        request_latency = Histogram(
            "rag_request_latency_seconds",
            "Request latency in seconds",
            ["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # RAG-specific
        retrieval_score = Histogram(
            "rag_retrieval_similarity_score",
            "Distribution of retrieval similarity scores",
            buckets=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        chunks_retrieved = Histogram(
            "rag_chunks_retrieved_total",
            "Number of chunks returned per query",
            buckets=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
        )
        tokens_used = Counter(
            "rag_tokens_used_total",
            "Total LLM tokens consumed",
            ["backend", "model"],
        )

        # Cache
        cache_hits = Counter("rag_cache_hits_total", "Semantic cache hits")
        cache_misses = Counter("rag_cache_misses_total", "Semantic cache misses")
        cache_size = Gauge("rag_cache_size", "Current number of cached entries")

        # Ingestion
        chunks_ingested = Counter(
            "rag_chunks_ingested_total",
            "Total chunks successfully ingested",
            ["collection"],
        )
        ingest_latency = Histogram(
            "rag_ingest_latency_seconds",
            "Ingestion latency per document",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Errors
        llm_errors = Counter(
            "rag_llm_errors_total",
            "LLM backend errors",
            ["backend", "error_type"],
        )
        retrieval_errors = Counter("rag_retrieval_errors_total", "Retrieval errors")

    return Metrics()


_metrics = _make_metrics()


# ── FastAPI middleware instrumentation ────────────────────────────────────────


def instrument_app(app: Any) -> None:
    """
    Add Prometheus metrics middleware and /metrics endpoint to a FastAPI app.

    Call this after creating the FastAPI app instance.
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("prometheus-client not installed — skipping metrics instrumentation")
        return

    from fastapi import Request, Response
    from fastapi.routing import APIRoute

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start

        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        if _metrics:
            _metrics.request_count.labels(endpoint=endpoint, method=method, status_code=status).inc()
            _metrics.request_latency.labels(endpoint=endpoint).observe(latency)

        return response

    # Add /metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    logger.info("Prometheus metrics endpoint mounted at /metrics")


# ── Metric recording helpers ──────────────────────────────────────────────────


def record_query(
    chunks_retrieved: int,
    avg_similarity: float,
    tokens: int,
    backend: str,
    model: str,
    cache_hit: bool,
) -> None:
    """Record metrics for a completed RAG query."""
    if not _metrics:
        return
    _metrics.chunks_retrieved.observe(chunks_retrieved)
    _metrics.retrieval_score.observe(avg_similarity)
    _metrics.tokens_used.labels(backend=backend, model=model).inc(tokens)
    if cache_hit:
        _metrics.cache_hits.inc()
    else:
        _metrics.cache_misses.inc()


def record_ingestion(collection: str, chunks_added: int, elapsed: float) -> None:
    """Record metrics for a completed ingestion."""
    if not _metrics:
        return
    _metrics.chunks_ingested.labels(collection=collection).inc(chunks_added)
    _metrics.ingest_latency.observe(elapsed)


def record_llm_error(backend: str, error_type: str) -> None:
    """Record an LLM backend error."""
    if not _metrics:
        return
    _metrics.llm_errors.labels(backend=backend, error_type=error_type).inc()


def update_cache_size(size: int) -> None:
    """Update the current cache size gauge."""
    if not _metrics:
        return
    _metrics.cache_size.set(size)


# ── Structured logging helpers ────────────────────────────────────────────────


def log_query_event(
    question: str,
    collection: str,
    chunks_retrieved: int,
    tokens_used: int,
    latency_ms: float,
    cache_hit: bool,
    backend: str,
) -> None:
    """Emit a structured log event for a completed query (JSON-friendly)."""
    logger.info(
        "QUERY | collection=%s | chunks=%d | tokens=%d | latency=%.0fms | cache=%s | backend=%s | q=%s",
        collection, chunks_retrieved, tokens_used, latency_ms,
        "HIT" if cache_hit else "MISS", backend,
        question[:80].replace("\n", " "),
    )


def log_ingest_event(source: str, collection: str, chunks_added: int, elapsed: float) -> None:
    """Emit a structured log event for a completed ingestion."""
    logger.info(
        "INGEST | source=%s | collection=%s | chunks_added=%d | elapsed=%.2fs",
        source, collection, chunks_added, elapsed,
    )
