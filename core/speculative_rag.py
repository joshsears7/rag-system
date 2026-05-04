"""
Speculative RAG — Google Research (2024)

Instead of feeding all retrieved chunks to the main LLM at once, Speculative RAG:
  1. Retrieves a larger pool of documents
  2. Partitions them into N independent subsets (drafts)
  3. Generates a draft answer from each subset using a concise "specialist" prompt
  4. Scores each draft (LLM self-rating — how well does this answer the question?)
  5. Selects the highest-scoring draft as the final answer

Why this is faster: each draft uses only 1/N of the total context, so generation
is much cheaper. The scoring step is a short call. Net result: ~51% latency
reduction over feeding all docs at once, with accuracy gains because multiple
independent drafts reduce the chance any single document bias dominates.

Why it's more accurate: each subset gives the model a focused view of a few
documents. A draft from a highly relevant subset beats a diluted answer from
all chunks mixed together.

Paper: "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting"
       Google Research (2024)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from models import QueryMode, QueryRequest, RetrievalContext, RetrievalResult

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class SpeculativeDraft:
    """One speculative draft — generated from a subset of retrieved chunks."""

    draft_id: int
    chunks: list[RetrievalResult]
    answer: str
    confidence_score: float    # LLM self-rating normalised to [0, 1]
    tokens_used: int
    latency_ms: float


@dataclass
class SpeculativeRAGResult:
    """Final output of a Speculative RAG run."""

    question: str
    answer: str
    selected_draft_id: int
    all_drafts: list[SpeculativeDraft]
    num_drafts: int
    total_chunks_retrieved: int
    tokens_used: int
    latency_ms: float
    latency_reduction_pct: float   # estimated vs. single full-context call

    @property
    def selected_draft(self) -> SpeculativeDraft:
        return next(d for d in self.all_drafts if d.draft_id == self.selected_draft_id)


# ── Prompts ───────────────────────────────────────────────────────────────────

_DRAFT_SYSTEM = (
    "You are a specialist answering questions from a curated document excerpt. "
    "Be accurate, concise, and grounded only in the provided context. "
    "If the context does not contain the answer, say so briefly."
)

_DRAFT_USER = """\
Context documents:
{context}

Question: {question}

Provide a focused draft answer based only on the context above."""


_SCORE_PROMPT = """\
You are evaluating how well a draft answer addresses a question.

Question: {question}

Draft answer: {draft}

Rate this draft on a scale of 1 to 10:
  10 = completely accurate, fully answers the question from the context
   5 = partially correct or incomplete
   1 = wrong, irrelevant, or "I don't know"

Output ONLY a single integer (1-10), nothing else."""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _partition_chunks(
    chunks: list[RetrievalResult],
    num_drafts: int,
) -> list[list[RetrievalResult]]:
    """Split retrieved chunks into num_drafts roughly equal subsets."""
    if not chunks:
        return [[] for _ in range(num_drafts)]

    # Sort by similarity so each partition gets a mix of scores
    sorted_chunks = sorted(chunks, key=lambda r: r.similarity_score, reverse=True)

    # Round-robin assignment: chunk[i] → partition[i % num_drafts]
    # This ensures each draft gets at least one high-similarity chunk
    partitions: list[list[RetrievalResult]] = [[] for _ in range(num_drafts)]
    for i, chunk in enumerate(sorted_chunks):
        partitions[i % num_drafts].append(chunk)

    return [p for p in partitions if p]  # drop any empty partitions


def _build_draft_context(chunks: list[RetrievalResult]) -> str:
    parts = []
    for i, r in enumerate(chunks, 1):
        parts.append(f"[{i}] {r.source}\n{r.chunk_text[:600]}")
    return "\n\n".join(parts)


def _score_draft(
    question: str,
    draft_text: str,
    llm_fn: Callable[[str], str],
) -> float:
    """Ask LLM to rate the draft 1-10. Returns normalised [0, 1] score."""
    prompt = _SCORE_PROMPT.format(question=question, draft=draft_text[:800])
    try:
        raw = llm_fn(prompt).strip()
        # Extract first integer found
        for token in raw.split():
            cleaned = token.strip(".,;:()[]")
            if cleaned.isdigit():
                score = int(cleaned)
                return max(0.0, min(1.0, (score - 1) / 9))
        return 0.5  # fallback if no integer found
    except Exception as e:
        logger.warning("Draft scoring failed: %s", e)
        return 0.5


# ── Core function ─────────────────────────────────────────────────────────────


def run_speculative_rag(
    question: str,
    collection: str,
    retrieve_fn: Callable[[QueryRequest], RetrievalContext],
    llm_complete_fn: Callable[[str, str], tuple[str, int, str]],
    llm_raw_fn: Callable[[str], str],
    num_drafts: int = 3,
    top_k: int = 9,
    mode: QueryMode = QueryMode.HYBRID,
) -> SpeculativeRAGResult:
    """
    Run Speculative RAG.

    Args:
        question:          User question
        collection:        ChromaDB collection
        retrieve_fn:       Retrieval function → RetrievalContext
        llm_complete_fn:   Full (system, user) → (answer, tokens, model)
        llm_raw_fn:        Raw prompt → str (for scoring)
        num_drafts:        Number of parallel drafts to generate
        top_k:             Total chunks to retrieve (split across drafts)
        mode:              Retrieval mode
    """
    t_start = time.perf_counter()

    # ── 1. Retrieve a larger pool ────────────────────────────────────────────
    req = QueryRequest(
        question=question,
        collection=collection,
        top_k=top_k,
        mode=mode,
    )
    ctx = retrieve_fn(req)
    total_chunks = len(ctx.results)

    # ── 2. Partition into subsets ────────────────────────────────────────────
    partitions = _partition_chunks(ctx.results, num_drafts)
    actual_drafts = len(partitions)

    # ── 3. Generate drafts ───────────────────────────────────────────────────
    drafts: list[SpeculativeDraft] = []

    for i, chunks in enumerate(partitions, 1):
        t_draft = time.perf_counter()
        context_text = _build_draft_context(chunks)
        user_prompt = _DRAFT_USER.format(context=context_text, question=question)

        try:
            answer, tokens, _ = llm_complete_fn(_DRAFT_SYSTEM, user_prompt)
        except Exception as e:
            logger.warning("Draft %d generation failed: %s", i, e)
            answer = ""
            tokens = 0

        draft_ms = (time.perf_counter() - t_draft) * 1000
        drafts.append(SpeculativeDraft(
            draft_id=i,
            chunks=chunks,
            answer=answer,
            confidence_score=0.0,   # filled in step 4
            tokens_used=tokens,
            latency_ms=draft_ms,
        ))
        logger.info("Draft %d/%d generated in %.0fms", i, actual_drafts, draft_ms)

    # ── 4. Score drafts ──────────────────────────────────────────────────────
    for draft in drafts:
        if draft.answer:
            draft.confidence_score = _score_draft(question, draft.answer, llm_raw_fn)

    # ── 5. Select best draft ─────────────────────────────────────────────────
    best = max(drafts, key=lambda d: d.confidence_score)
    logger.info(
        "Selected draft %d (score=%.2f) from %d candidates",
        best.draft_id, best.confidence_score, actual_drafts,
    )

    total_tokens = sum(d.tokens_used for d in drafts)
    total_ms = (time.perf_counter() - t_start) * 1000

    # Estimated latency reduction: each draft used ~1/N context vs. full context
    # Scoring adds overhead; net reduction is approximately (N-1)/2N * 100%
    estimated_reduction = ((actual_drafts - 1) / (2 * actual_drafts)) * 100

    return SpeculativeRAGResult(
        question=question,
        answer=best.answer,
        selected_draft_id=best.draft_id,
        all_drafts=drafts,
        num_drafts=actual_drafts,
        total_chunks_retrieved=total_chunks,
        tokens_used=total_tokens,
        latency_ms=total_ms,
        latency_reduction_pct=estimated_reduction,
    )
