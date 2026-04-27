"""
Langfuse Observability — semantic tracing for RAG pipelines.

Prometheus tracks system metrics (latency, throughput, errors).
Langfuse tracks semantic metrics — the *meaning* of what happened:
  - Which chunks were retrieved and their scores
  - What the LLM received as context (full prompt)
  - What it responded
  - Token costs per request
  - User feedback linkage (thumbs up/down → span score)
  - Retrieval quality score at each step

This is what separates "I deployed a RAG system" from "I can debug why
a specific query failed at 2am on Tuesday."

Setup:
  pip install langfuse
  LANGFUSE_PUBLIC_KEY=pk-lf-... in .env
  LANGFUSE_SECRET_KEY=sk-lf-... in .env
  LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted

All functions are no-ops if Langfuse is not configured — zero overhead.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Module-level Langfuse client (None if not configured)
_langfuse = None
_enabled = False


def _get_langfuse():
    """Lazy-load and cache the Langfuse client."""
    global _langfuse, _enabled
    if _langfuse is not None:
        return _langfuse

    try:
        from langfuse import Langfuse
        from config import settings
        pk = getattr(settings, "langfuse_public_key", "")
        sk = getattr(settings, "langfuse_secret_key", "")
        host = getattr(settings, "langfuse_host", "https://cloud.langfuse.com")

        if pk and sk:
            _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
            _enabled = True
            logger.info("Langfuse observability initialized (host: %s)", host)
        else:
            logger.debug("Langfuse not configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY missing). Tracing disabled.")
    except ImportError:
        logger.debug("langfuse not installed. pip install langfuse for semantic tracing.")
    except Exception as e:
        logger.warning("Langfuse initialization failed: %s. Tracing disabled.", e)

    return _langfuse


def is_enabled() -> bool:
    """Return True if Langfuse tracing is active."""
    _get_langfuse()
    return _enabled


# ── Trace context ─────────────────────────────────────────────────────────────


class RAGTrace:
    """
    A single RAG request trace. Wraps a Langfuse trace with RAG-specific helpers.

    Usage:
        trace = start_trace(question="What is X?", collection="my_kb")
        with trace.span("retrieval"):
            results = retrieve(...)
            trace.log_retrieval(results)
        with trace.span("generation"):
            answer = generate(...)
        trace.finish(answer=answer, tokens=123)
    """

    def __init__(self, question: str, collection: str, session_id: str | None = None):
        self.question = question
        self.collection = collection
        self.session_id = session_id
        self._trace = None
        self._start = time.perf_counter()
        self._spans: list = []

        lf = _get_langfuse()
        if lf:
            try:
                self._trace = lf.trace(
                    name="rag_query",
                    input={"question": question, "collection": collection},
                    session_id=session_id,
                    metadata={"collection": collection},
                )
            except Exception as e:
                logger.debug("Failed to create Langfuse trace: %s", e)

    @contextmanager
    def span(self, name: str, input_data: dict | None = None) -> Generator[Any, None, None]:
        """Context manager for a named span within this trace."""
        span = None
        if self._trace:
            try:
                span = self._trace.span(
                    name=name,
                    input=input_data or {},
                    start_time=time.perf_counter(),
                )
            except Exception:
                pass

        try:
            yield span
        finally:
            if span:
                try:
                    span.end()
                except Exception:
                    pass

    def log_retrieval(
        self,
        results: list,
        query_mode: str = "hybrid",
        web_fallback: bool = False,
    ) -> None:
        """Log retrieval results as a Langfuse generation/span."""
        if not self._trace:
            return
        try:
            retrieved_docs = [
                {
                    "source": getattr(r, "source", "?"),
                    "score": getattr(r, "similarity_score", 0),
                    "chunk_index": getattr(r, "chunk_index", 0),
                    "excerpt": getattr(r, "chunk_text", "")[:200],
                }
                for r in results[:10]
            ]
            self._trace.span(
                name="retrieval",
                input={"question": self.question, "mode": query_mode},
                output={
                    "num_results": len(results),
                    "top_score": results[0].similarity_score if results else 0,
                    "web_fallback": web_fallback,
                    "results": retrieved_docs,
                },
                metadata={"collection": self.collection, "mode": query_mode},
            ).end()
        except Exception as e:
            logger.debug("Langfuse retrieval log failed: %s", e)

    def log_generation(
        self,
        prompt: str,
        answer: str,
        model: str,
        tokens: int,
        latency_ms: float,
    ) -> None:
        """Log LLM generation as a Langfuse generation event."""
        if not self._trace:
            return
        try:
            lf = _get_langfuse()
            if lf:
                self._trace.generation(
                    name="llm_generation",
                    model=model,
                    input=prompt[:2000],  # truncate for UI readability
                    output=answer,
                    usage={"total_tokens": tokens},
                    metadata={"latency_ms": latency_ms},
                ).end()
        except Exception as e:
            logger.debug("Langfuse generation log failed: %s", e)

    def score(self, name: str, value: float, comment: str = "") -> None:
        """
        Attach a numeric score to this trace (e.g., user thumbs up/down).

        Scores appear in Langfuse analytics dashboards.
        Useful for connecting user feedback to specific traces.
        """
        if not self._trace:
            return
        try:
            self._trace.score(name=name, value=value, comment=comment)
        except Exception as e:
            logger.debug("Langfuse score failed: %s", e)

    def finish(
        self,
        answer: str = "",
        tokens: int = 0,
        latency_ms: float | None = None,
        cache_hit: bool = False,
    ) -> None:
        """Finalize the trace with output and timing."""
        if not self._trace:
            return
        try:
            elapsed = latency_ms or ((time.perf_counter() - self._start) * 1000)
            self._trace.update(
                output={"answer": answer[:1000], "tokens": tokens, "cache_hit": cache_hit},
                metadata={"latency_ms": round(elapsed, 1)},
            )
            # Flush immediately so the trace is visible in UI
            lf = _get_langfuse()
            if lf:
                lf.flush()
        except Exception as e:
            logger.debug("Langfuse trace finish failed: %s", e)

    @property
    def trace_id(self) -> str | None:
        """Return the Langfuse trace ID (for linking to UI)."""
        if self._trace:
            try:
                return self._trace.id
            except Exception:
                pass
        return None


def start_trace(
    question: str,
    collection: str,
    session_id: str | None = None,
) -> RAGTrace:
    """
    Start a new RAG trace. Returns a RAGTrace (no-op if Langfuse not configured).

    Usage:
        trace = start_trace(question, collection)
        # ... do RAG pipeline ...
        trace.finish(answer=answer)
    """
    return RAGTrace(question=question, collection=collection, session_id=session_id)


def score_trace(trace_id: str, score_value: float, name: str = "user_feedback") -> None:
    """
    Attach a score to an existing trace by ID (e.g., from a feedback webhook).

    Args:
        trace_id: Langfuse trace ID
        score_value: 1.0 = thumbs up, 0.0 = thumbs down
        name: score metric name
    """
    lf = _get_langfuse()
    if not lf:
        return
    try:
        lf.score(trace_id=trace_id, name=name, value=score_value)
        lf.flush()
    except Exception as e:
        logger.debug("Langfuse score_trace failed: %s", e)
