"""
Adaptive RAG — intelligently decides WHEN and HOW to retrieve.

Papers:
  - "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models
    through Question Complexity" (Jeong et al., 2024)
  - "Self-RAG: Learning to Retrieve, Generate, and Critique through
    Self-Reflection" (Asai et al., 2023)

The Core Insight:
  Not every question needs retrieval. "What is 2+2?" → answer directly.
  Not every question needs one retrieval step. "Compare X and Y across these docs"
  → iterative, multi-step retrieval is needed.

  Blindly retrieving for every query wastes compute and often adds noise.
  Adaptive RAG classifies query complexity FIRST, then chooses the strategy:

  Strategy A — No Retrieval: simple factual/math questions
  Strategy B — Single-Step RAG: standard lookup (most questions)
  Strategy C — Iterative RAG: complex multi-hop questions needing chained retrieval

Self-RAG adds trained REFLECTION tokens:
  [Retrieve]          → should I retrieve at all?
  [IsREL]             → is this retrieved doc relevant?
  [IsSUP]             → does the doc support my draft answer?
  [IsUSE]             → is my answer useful/complete?

We implement a practical approximation using LLM-as-judge for all reflection
decisions (doesn't require fine-tuning the base model).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


# ── Query complexity classification ──────────────────────────────────────────


class RetrievalStrategy(str, Enum):
    """The three retrieval strategies in Adaptive RAG."""
    NO_RETRIEVAL = "no_retrieval"      # answer from model knowledge alone
    SINGLE_STEP = "single_step"        # standard single-pass RAG
    ITERATIVE = "iterative"            # multi-hop chained retrieval


def classify_query_complexity(
    question: str,
    llm_fn: Callable[[str], str],
) -> RetrievalStrategy:
    """
    Classify a query into one of three retrieval strategies.

    Uses the LLM as a zero-shot classifier. In production, this would be
    replaced with a fine-tuned small classifier (e.g. DeBERTa) trained on
    labeled question complexity examples for speed.

    Strategy selection logic:
      NO_RETRIEVAL:  math, greetings, general knowledge the model certainly knows
      SINGLE_STEP:   specific factual lookups from documents
      ITERATIVE:     questions requiring synthesis across multiple documents,
                     comparison, causality chains, or temporal reasoning
    """
    prompt = (
        "Classify this question into one of three retrieval strategies:\n\n"
        "A) NO_RETRIEVAL - General knowledge, math, simple facts the model knows\n"
        "   Examples: 'What is Python?', 'What is 15% of 200?', 'What year did WWII end?'\n\n"
        "B) SINGLE_STEP - Needs lookup in provided documents, single search sufficient\n"
        "   Examples: 'What is the refund policy?', 'Who is the CEO mentioned in the report?'\n\n"
        "C) ITERATIVE - Complex, multi-hop, needs synthesis across multiple document sections\n"
        "   Examples: 'Compare the Q1 and Q3 results', 'What caused the issue and how was it resolved?'\n\n"
        f"Question: {question}\n\n"
        "Reply with ONLY the letter: A, B, or C"
    )
    try:
        raw = llm_fn(prompt).strip().upper()
        if "A" in raw[:3]:
            return RetrievalStrategy.NO_RETRIEVAL
        elif "C" in raw[:3]:
            return RetrievalStrategy.ITERATIVE
        else:
            return RetrievalStrategy.SINGLE_STEP
    except Exception as e:
        logger.warning("Query classification failed: %s. Defaulting to SINGLE_STEP.", e)
        return RetrievalStrategy.SINGLE_STEP


# ── Self-RAG reflection tokens ────────────────────────────────────────────────


def should_retrieve(question: str, llm_fn: Callable[[str], str]) -> bool:
    """
    Self-RAG: [Retrieve] token — should we retrieve at all?

    Returns True if retrieval is needed, False if model can answer directly.
    """
    prompt = (
        "Can you answer the following question accurately WITHOUT looking up any documents? "
        "This is only YES if you are highly confident the answer is in your training data.\n\n"
        f"Question: {question}\n\n"
        "Reply with ONLY 'YES' (no retrieval needed) or 'NO' (retrieval required):"
    )
    try:
        raw = llm_fn(prompt).strip().upper()
        return "NO" in raw  # NO = retrieval IS needed
    except Exception:
        return True  # default: retrieve


def is_relevant(
    question: str,
    chunk_text: str,
    llm_fn: Callable[[str], str],
) -> bool:
    """
    Self-RAG: [IsREL] — is this retrieved chunk relevant to the question?
    """
    prompt = (
        f"Is the following text relevant to answering the question?\n\n"
        f"Question: {question}\n\n"
        f"Text: {chunk_text[:600]}\n\n"
        "Reply ONLY 'relevant' or 'irrelevant':"
    )
    try:
        raw = llm_fn(prompt).strip().lower()
        return "relevant" in raw and "irrelevant" not in raw
    except Exception:
        return True


def is_supported(
    answer_draft: str,
    chunk_text: str,
    llm_fn: Callable[[str], str],
) -> bool:
    """
    Self-RAG: [IsSUP] — does the retrieved text support the draft answer?
    """
    prompt = (
        "Does the following retrieved text SUPPORT the draft answer, or contradict/not mention it?\n\n"
        f"Retrieved text: {chunk_text[:500]}\n\n"
        f"Draft answer: {answer_draft[:300]}\n\n"
        "Reply ONLY 'supported' or 'not supported':"
    )
    try:
        raw = llm_fn(prompt).strip().lower()
        return "supported" in raw
    except Exception:
        return True


def is_useful(answer: str, question: str, llm_fn: Callable[[str], str]) -> bool:
    """
    Self-RAG: [IsUSE] — is the final answer useful and complete?
    """
    prompt = (
        "Is the following answer useful and sufficiently complete for the question?\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer[:500]}\n\n"
        "Reply ONLY 'useful' or 'not useful':"
    )
    try:
        raw = llm_fn(prompt).strip().lower()
        return "not useful" not in raw and "useful" in raw
    except Exception:
        return True


# ── Iterative multi-hop retrieval ─────────────────────────────────────────────


def iterative_retrieve_and_generate(
    question: str,
    collection: str,
    llm_fn: Callable[[str], str],
    retrieve_fn: Callable[[str, str, int], list],
    max_hops: int = 3,
    top_k: int = 4,
) -> tuple[list, list[str]]:
    """
    Multi-hop iterative retrieval for complex questions.

    Algorithm:
      1. Retrieve for original question
      2. Generate a partial answer from retrieved context
      3. Identify what information is STILL MISSING
      4. Generate a follow-up sub-query for the missing information
      5. Retrieve for the sub-query
      6. Combine all retrieved context
      7. Repeat up to max_hops times

    This enables answering questions like:
      "What caused the CEO's decision in Q3, and what were its financial implications?"
    which requires retrieving from multiple sections of documents.

    Returns:
        (all_retrieved_chunks, list_of_queries_used)
    """
    all_chunks = []
    queries_used = [question]
    accumulated_context = ""

    for hop in range(max_hops):
        # Retrieve for current query
        current_query = queries_used[-1]
        chunks = retrieve_fn(current_query, collection, top_k)
        new_chunks = [c for c in chunks if c not in all_chunks]
        all_chunks.extend(new_chunks)

        if not new_chunks:
            logger.info("Iterative RAG: no new chunks at hop %d, stopping.", hop)
            break

        accumulated_context = "\n\n".join(c if isinstance(c, str) else c.chunk_text for c in all_chunks)[:3000]

        # Check if we have enough to answer
        sufficiency_prompt = (
            f"Given this context, can you now fully answer the question, or is important information still missing?\n\n"
            f"Question: {question}\n\n"
            f"Current context:\n{accumulated_context}\n\n"
            "Reply with:\n"
            "COMPLETE: [brief explanation of what you can now answer]\n"
            "MISSING: [specific information still needed]"
        )
        try:
            assessment = llm_fn(sufficiency_prompt).strip()
            if assessment.upper().startswith("COMPLETE"):
                logger.info("Iterative RAG: complete at hop %d", hop)
                break

            # Generate follow-up sub-query
            follow_up_prompt = (
                f"Original question: {question}\n"
                f"What I still need to find: {assessment[8:300]}\n\n"
                "Write a specific search query to find the missing information "
                "(one line, no explanation):"
            )
            sub_query = llm_fn(follow_up_prompt).strip()
            if sub_query:
                queries_used.append(sub_query)
                logger.info("Iterative RAG hop %d: sub-query = '%s'", hop + 1, sub_query[:60])

        except Exception as e:
            logger.warning("Iterative RAG sufficiency check failed: %s", e)
            break

    return all_chunks, queries_used


# ── Adaptive RAG orchestrator ─────────────────────────────────────────────────


class AdaptiveRAGResult:
    """Result from the Adaptive RAG pipeline."""

    def __init__(
        self,
        answer: str,
        strategy_used: RetrievalStrategy,
        chunks_retrieved: list,
        queries_used: list[str],
        hops: int,
        latency_ms: float,
        self_rag_flags: dict,
    ) -> None:
        self.answer = answer
        self.strategy_used = strategy_used
        self.chunks_retrieved = chunks_retrieved
        self.queries_used = queries_used
        self.hops = hops
        self.latency_ms = latency_ms
        self.self_rag_flags = self_rag_flags


def adaptive_answer(
    question: str,
    collection: str,
    llm_fn: Callable[[str], str],
    retrieve_fn: Callable[[str, str, int], list],
    generate_fn: Callable[[str, str], str],
    top_k: int = 6,
    use_self_rag: bool = True,
    use_iterative: bool = True,
    max_hops: int = 3,
) -> AdaptiveRAGResult:
    """
    Full Adaptive RAG pipeline with Self-RAG reflection.

    Combines:
      - Query complexity classification (No retrieval / Single / Iterative)
      - Self-RAG [Retrieve] token (should we even retrieve?)
      - Relevance filtering ([IsREL] per chunk)
      - Iterative multi-hop retrieval for complex questions
      - [IsUSE] check + retry on low-quality answers

    Args:
        question: user's question
        collection: ChromaDB collection to search
        llm_fn: simple text → text LLM callable
        retrieve_fn: (query, collection, k) → list of chunk texts
        generate_fn: (system_prompt, user_prompt) → answer text
        top_k: chunks per retrieval step
        use_self_rag: apply [Retrieve]/[IsREL]/[IsUSE] reflection
        use_iterative: allow multi-hop retrieval for complex questions
        max_hops: max iterative retrieval hops

    Returns:
        AdaptiveRAGResult with answer, strategy, and introspection data
    """
    start = time.perf_counter()
    self_rag_flags: dict = {}

    # ── Step 1: Should we retrieve at all? ────────────────────────────────────
    if use_self_rag:
        needs_retrieval = should_retrieve(question, llm_fn)
        self_rag_flags["should_retrieve"] = needs_retrieval
        if not needs_retrieval:
            # Answer directly from model knowledge
            answer = generate_fn(
                "You are a helpful assistant. Answer accurately and concisely.",
                question,
            )
            return AdaptiveRAGResult(
                answer=answer,
                strategy_used=RetrievalStrategy.NO_RETRIEVAL,
                chunks_retrieved=[],
                queries_used=[question],
                hops=0,
                latency_ms=(time.perf_counter() - start) * 1000,
                self_rag_flags=self_rag_flags,
            )

    # ── Step 2: Classify complexity ───────────────────────────────────────────
    strategy = classify_query_complexity(question, llm_fn)
    logger.info("Adaptive RAG: strategy=%s for '%s'", strategy.value, question[:60])

    # ── Step 3: Retrieve ──────────────────────────────────────────────────────
    if strategy == RetrievalStrategy.ITERATIVE and use_iterative:
        all_chunks, queries_used = iterative_retrieve_and_generate(
            question, collection, llm_fn, retrieve_fn, max_hops=max_hops, top_k=top_k,
        )
        hops = len(queries_used) - 1
    else:
        raw_chunks = retrieve_fn(question, collection, top_k)
        all_chunks = raw_chunks
        queries_used = [question]
        hops = 1

    # ── Step 4: Self-RAG relevance filtering ─────────────────────────────────
    if use_self_rag and all_chunks:
        filtered = []
        for chunk in all_chunks:
            chunk_text = chunk if isinstance(chunk, str) else getattr(chunk, "chunk_text", str(chunk))
            relevant = is_relevant(question, chunk_text, llm_fn)
            if relevant:
                filtered.append(chunk)
        self_rag_flags["chunks_before_filter"] = len(all_chunks)
        self_rag_flags["chunks_after_filter"] = len(filtered)
        if filtered:
            all_chunks = filtered
        # else: keep originals to avoid empty context

    # ── Step 5: Generate answer ───────────────────────────────────────────────
    context_texts = [
        (c if isinstance(c, str) else getattr(c, "chunk_text", str(c)))
        for c in all_chunks
    ]
    context = "\n\n---\n\n".join(context_texts[:top_k])

    system_prompt = (
        "You are a precise research assistant. Answer ONLY from the provided context. "
        "Cite sources as [Source: chunk N]. Say 'I don't have enough context' if the answer isn't there."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer with inline citations:"

    answer = generate_fn(system_prompt, user_prompt)

    # ── Step 6: Self-RAG usefulness check ────────────────────────────────────
    if use_self_rag:
        useful = is_useful(answer, question, llm_fn)
        self_rag_flags["is_useful"] = useful
        if not useful and strategy != RetrievalStrategy.ITERATIVE:
            # One retry with expanded context
            logger.info("Self-RAG: answer not useful, retrying with more context")
            more_chunks = retrieve_fn(question, collection, top_k * 2)
            extended_context = "\n\n---\n\n".join(
                (c if isinstance(c, str) else getattr(c, "chunk_text", str(c)))
                for c in more_chunks
            )
            user_prompt_retry = f"CONTEXT:\n{extended_context[:6000]}\n\nQUESTION: {question}\n\nAnswer with citations:"
            answer = generate_fn(system_prompt, user_prompt_retry)
            self_rag_flags["retried"] = True

    return AdaptiveRAGResult(
        answer=answer,
        strategy_used=strategy,
        chunks_retrieved=all_chunks,
        queries_used=queries_used,
        hops=hops,
        latency_ms=(time.perf_counter() - start) * 1000,
        self_rag_flags=self_rag_flags,
    )
