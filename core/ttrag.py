"""
TTRAG — Test-Time Compute Scaling for RAG (ICLR 2025)

Iteratively rewrites queries and re-retrieves until sufficient context is found,
instead of one-shot retrieval. Treats RAG as a dynamic search process rather
than a static pipeline.

This is the RAG equivalent of chain-of-thought scaling: allocating more
compute at inference time produces better answers on hard questions.

The loop:
  1. Retrieve with the current query
  2. Score context sufficiency (density + coverage ensemble)
  3. If sufficient  → generate and return
  4. If not         → ask LLM what's missing, rewrite the query
  5. Repeat up to max_iterations

Context accumulates across iterations — each rewrite adds to the retrieved
chunk pool, so later iterations build on earlier findings.

Paper: "Adaptive Test-Time Compute Scaling for RAG" (ICLR 2025)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from config import settings
from models import QueryMode, QueryRequest, RetrievalContext, RetrievalResult
from core.sufficient_context import SufficientContextChecker, SufficiencyResult

logger = logging.getLogger(__name__)

_checker = SufficientContextChecker()


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class TTRAGIteration:
    """Result of one retrieval iteration."""

    iteration: int
    query_used: str
    rewrite_reason: str        # why the previous query was rewritten
    retrieved: list[RetrievalResult]
    sufficiency: SufficiencyResult
    latency_ms: float


@dataclass
class TTRAGResult:
    """Final output of a TTRAG run."""

    question: str
    answer: str
    iterations: list[TTRAGIteration]
    final_query: str
    total_chunks_retrieved: int
    unique_chunks_used: int
    tokens_used: int
    latency_ms: float
    converged: bool            # True if sufficiency threshold was met
    final_sufficiency: float

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)


# ── Query rewriter ────────────────────────────────────────────────────────────


_REWRITE_PROMPT = """\
You are a search query optimizer for a RAG system.

Original question: {question}
Query used in last attempt: {query}
Sufficiency score: {score:.0%} (threshold: {threshold:.0%})

Retrieved content (first 600 chars):
{context_snippet}

The retrieved content is insufficient to answer the question.
Identify what information is missing and write a new search query that:
1. Uses different terminology or phrasing than the previous query
2. Targets the specific gap in the retrieved content
3. Is concise and specific (one sentence max)

Output ONLY the new query — no explanation, no quotes, no preamble."""


def _rewrite_query(
    question: str,
    current_query: str,
    context: RetrievalContext,
    sufficiency: SufficiencyResult,
    llm_fn: Callable[[str], str],
    threshold: float,
) -> tuple[str, str]:
    """
    Ask the LLM to rewrite the query given what it found so far.
    Returns (new_query, rewrite_reason).
    """
    snippet = " | ".join(
        r.chunk_text[:200] for r in context.results[:3]
    ) if context.results else "(nothing retrieved)"

    prompt = _REWRITE_PROMPT.format(
        question=question,
        query=current_query,
        score=sufficiency.overall_score,
        threshold=threshold,
        context_snippet=snippet[:600],
    )

    try:
        new_query = llm_fn(prompt).strip().strip('"').strip("'")
        # Fall back to original if LLM returns something obviously wrong
        if not new_query or len(new_query) > 300:
            new_query = question
        reason = sufficiency.explanation or f"score {sufficiency.overall_score:.2f} < {threshold:.2f}"
    except Exception as e:
        logger.warning("TTRAG query rewrite failed: %s — using original", e)
        new_query = question
        reason = f"rewrite failed: {e}"

    return new_query, reason


# ── Core TTRAG loop ───────────────────────────────────────────────────────────


def run_ttrag(
    question: str,
    collection: str,
    retrieve_fn: Callable[[QueryRequest, ...], RetrievalContext],
    llm_fn: Callable[[str], str],
    generate_fn: Callable[[str, RetrievalContext], tuple[str, int]],
    max_iterations: int = 4,
    top_k: int = 6,
    sufficiency_threshold: float = 0.55,
    mode: QueryMode = QueryMode.HYBRID,
) -> TTRAGResult:
    """
    Run the TTRAG loop.

    Args:
        question:             The user's question (unchanged throughout)
        collection:           ChromaDB collection to search
        retrieve_fn:          Retrieval function — takes a QueryRequest, returns RetrievalContext
        llm_fn:               Raw LLM completion function (for query rewriting)
        generate_fn:          Answer generation function — takes (question, context) → (answer, tokens)
        max_iterations:       Maximum retrieval iterations before giving up
        top_k:                Chunks to retrieve per iteration
        sufficiency_threshold: Score above which we stop iterating
        mode:                 Retrieval mode (hybrid/dense/sparse)
    """
    t_start = time.perf_counter()

    current_query = question
    rewrite_reason = "initial query"
    iterations: list[TTRAGIteration] = []

    # Accumulated unique chunks across all iterations (deduped by chunk_index+source)
    seen_chunk_ids: set[str] = set()
    all_chunks: list[RetrievalResult] = []

    for i in range(1, max_iterations + 1):
        t_iter = time.perf_counter()

        req = QueryRequest(
            question=current_query,
            collection=collection,
            top_k=top_k,
            mode=mode,
        )

        try:
            ctx = retrieve_fn(req)
        except Exception as e:
            logger.warning("TTRAG iteration %d retrieval failed: %s", i, e)
            break

        # Deduplicate and accumulate chunks
        new_chunks: list[RetrievalResult] = []
        for r in ctx.results:
            chunk_id = f"{r.source}:{r.chunk_index}"
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                new_chunks.append(r)
                all_chunks.append(r)

        # Score sufficiency against accumulated context (not just this iteration)
        accumulated_ctx = RetrievalContext(
            query=question,
            results=sorted(all_chunks, key=lambda r: r.similarity_score, reverse=True)[:top_k * 2],
            query_mode=mode,
        )
        sufficiency = _checker.score(
            question=question,
            context=accumulated_ctx,
        )

        iter_ms = (time.perf_counter() - t_iter) * 1000
        iterations.append(TTRAGIteration(
            iteration=i,
            query_used=current_query,
            rewrite_reason=rewrite_reason,
            retrieved=new_chunks,
            sufficiency=sufficiency,
            latency_ms=iter_ms,
        ))

        logger.info(
            "TTRAG iter %d: query=%r chunks=%d new=%d sufficiency=%.2f",
            i, current_query, len(all_chunks), len(new_chunks), sufficiency.overall_score,
        )

        if sufficiency.overall_score >= sufficiency_threshold:
            logger.info("TTRAG converged at iteration %d (score=%.2f)", i, sufficiency.overall_score)
            break

        if i < max_iterations:
            current_query, rewrite_reason = _rewrite_query(
                question=question,
                current_query=current_query,
                context=ctx,
                sufficiency=sufficiency,
                llm_fn=llm_fn,
                threshold=sufficiency_threshold,
            )

    # Generate final answer from accumulated context
    final_ctx = RetrievalContext(
        query=question,
        results=sorted(all_chunks, key=lambda r: r.similarity_score, reverse=True)[:top_k * 2],
        query_mode=mode,
    )

    try:
        answer, tokens = generate_fn(question, final_ctx)
    except Exception as e:
        logger.error("TTRAG generation failed: %s", e)
        answer = f"Generation failed: {e}"
        tokens = 0

    final_sufficiency = iterations[-1].sufficiency.overall_score if iterations else 0.0
    converged = final_sufficiency >= sufficiency_threshold

    return TTRAGResult(
        question=question,
        answer=answer,
        iterations=iterations,
        final_query=current_query,
        total_chunks_retrieved=len(all_chunks) + (top_k * (max_iterations - len(iterations))),
        unique_chunks_used=len(seen_chunk_ids),
        tokens_used=tokens,
        latency_ms=(time.perf_counter() - t_start) * 1000,
        converged=converged,
        final_sufficiency=final_sufficiency,
    )
