"""
CoT-RAG — Chain-of-Thought integrated Retrieval-Augmented Generation.

Based on "CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation
to Enhance Reasoning in Large Language Models" (EMNLP 2025 Findings).
ACL Anthology: aclanthology.org/2025.findings-emnlp.168

Standard RAG retrieves once and generates immediately. For complex, multi-hop
questions this fails because a single retrieval can't surface all necessary evidence.

CoT-RAG pipeline:
  1. Decompose the question into explicit reasoning steps via LLM
  2. For each step, retrieve targeted context chunks (step-specific query)
  3. Accumulate evidence across all steps
  4. Generate the final answer with full reasoning trace visible

This enables multi-hop reasoning where each step retrieves different evidence:
  Q: "How did Company X's 2022 acquisition affect its market share by 2024?"
  Step 1: "Company X 2022 acquisition" → retrieves acquisition details
  Step 2: "Company X market share 2023 2024" → retrieves market data
  Step 3: Synthesize with both sets of evidence

Benefits over standard RAG:
  - Reduces hallucination on multi-hop questions (each step is grounded separately)
  - Makes reasoning transparent (reasoning trace is returned for display)
  - Allows the demo UI to show the system "thinking" step by step
  - Better for questions requiring connecting facts from multiple document sections

Returns both final answer and the full CoT trace for UI visualization.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from models import QueryMode, QueryRequest, RetrievalResult

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ReasoningStep:
    """A single step in the chain-of-thought reasoning trace."""

    step_number: int
    thought:        str              # The reasoning step (what we're trying to find)
    sub_query:      str              # Query used to retrieve for this step
    retrieved:      list[str]        # Retrieved chunk excerpts
    sources:        list[str]        # Source filenames
    intermediate:   str              # Intermediate finding from this step
    latency_ms:     float = 0.0


@dataclass
class CoTResult:
    """
    Full result from a CoT-RAG query.

    Includes the final answer plus the complete reasoning trace — suitable
    for display in the demo UI to show the system's thinking process.
    """

    question:       str
    answer:         str
    reasoning_steps: list[ReasoningStep]
    all_sources:    list[str]
    total_chunks:   int
    tokens_used:    int
    latency_ms:     float
    num_steps:      int
    model_used:     str = "unknown"
    warnings:       list[str] = field(default_factory=list)

    @property
    def trace_as_markdown(self) -> str:
        """Format the full reasoning trace as markdown for display."""
        lines = [f"## Reasoning Trace for: {self.question}\n"]
        for step in self.reasoning_steps:
            lines.append(f"### Step {step.step_number}: {step.thought}")
            lines.append(f"**Sub-query:** `{step.sub_query}`")
            if step.retrieved:
                lines.append(f"**Retrieved {len(step.retrieved)} chunks from:** {', '.join(set(step.sources))}")
                lines.append(f"**Finding:** {step.intermediate}")
            else:
                lines.append("**Finding:** No relevant chunks found for this step.")
            lines.append("")
        lines.append(f"### Final Answer\n{self.answer}")
        return "\n".join(lines)


# ── Step decomposition ────────────────────────────────────────────────────────


def decompose_question(
    question: str,
    llm_fn: Callable[[str], str],
    max_steps: int = 4,
) -> list[str]:
    """
    Use LLM to break a question into explicit reasoning steps.

    Returns a list of step descriptions (strings). Each will become a
    targeted sub-query for its own retrieval pass.

    Args:
        question:   The user's original question
        llm_fn:     LLM completion function
        max_steps:  Maximum number of reasoning steps to generate

    Returns:
        List of reasoning step descriptions
    """
    prompt = (
        f"Break down the following question into {max_steps} or fewer reasoning steps. "
        f"Each step should identify a specific piece of information needed to answer the question.\n\n"
        f"QUESTION: {question}\n\n"
        f"Format your response as a numbered list:\n"
        f"1. [First piece of information needed]\n"
        f"2. [Second piece of information needed]\n"
        f"...\n\n"
        f"Be specific and concise. Each step should be a sub-question or fact to find.\n"
        f"If the question is simple, use fewer steps (even 1-2 is fine).\n\n"
        f"Reasoning steps:"
    )

    try:
        raw = llm_fn(prompt).strip()
    except Exception as e:
        logger.warning("CoT step decomposition failed: %s. Falling back to single step.", e)
        return [question]

    # Parse numbered list
    steps: list[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        # Match lines like "1. ...", "1) ...", "- ..."
        m = re.match(r"^[\d]+[.)]\s+(.+)$", line)
        if m:
            step = m.group(1).strip()
            if step:
                steps.append(step)
        elif line.startswith("- ") and len(line) > 3:
            steps.append(line[2:].strip())

    if not steps:
        # Fallback: treat the whole response as a single step
        steps = [raw[:200]] if raw else [question]

    steps = steps[:max_steps]
    logger.debug("CoT: decomposed into %d steps: %s", len(steps), steps)
    return steps


# ── Step-specific retrieval ───────────────────────────────────────────────────


def retrieve_for_step(
    step: str,
    collection: str,
    retrieve_fn: Callable,
    top_k: int = 4,
) -> tuple[list[str], list[str]]:
    """
    Run retrieval for a single reasoning step.

    Args:
        step:        The reasoning step description (used as retrieval query)
        collection:  ChromaDB collection to search
        retrieve_fn: retrieve() function from core.retrieval
        top_k:       Number of chunks to retrieve per step

    Returns:
        (chunk_texts, source_names)
    """
    from models import QueryRequest, QueryMode
    req = QueryRequest(
        question=step,
        collection=collection,
        top_k=top_k,
        mode=QueryMode.HYBRID,
    )
    try:
        context = retrieve_fn(req)
        texts   = [r.chunk_text for r in context.results]
        sources = [r.source for r in context.results]
        return texts, sources
    except Exception as e:
        logger.warning("CoT retrieval for step '%s' failed: %s", step[:60], e)
        return [], []


# ── Intermediate synthesis ────────────────────────────────────────────────────


def synthesize_step(
    step: str,
    chunks: list[str],
    llm_fn: Callable[[str], str],
) -> str:
    """
    Extract the key finding from retrieved chunks for one reasoning step.

    This is a lightweight "mini-generation" — just extracting the relevant fact,
    not a full answer. Used to build context for the final synthesis.
    """
    if not chunks:
        return "No relevant information found for this step."

    context = "\n\n".join(chunks[:3])[:1200]
    prompt = (
        f"Based on the following context, answer this specific question concisely (1-2 sentences):\n\n"
        f"QUESTION: {step}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"If the context doesn't contain the answer, say 'Not found in context.'\n\n"
        f"Answer:"
    )
    try:
        result = llm_fn(prompt).strip()
        return result[:500] if result else "Could not extract finding."
    except Exception as e:
        logger.warning("Step synthesis failed: %s", e)
        return "Synthesis failed for this step."


# ── Final synthesis ───────────────────────────────────────────────────────────


def synthesize_final(
    question: str,
    steps: list[ReasoningStep],
    llm_fn: Callable[[str], str],
    system_prompt: str | None = None,
) -> tuple[str, int]:
    """
    Generate the final answer from all accumulated step findings.

    Presents the full reasoning trace to the LLM so it can synthesize
    a coherent answer that draws on evidence from all steps.

    Args:
        question:       Original user question
        steps:          All reasoning steps with their findings
        llm_fn:         LLM completion function
        system_prompt:  Optional override for the system prompt

    Returns:
        (final_answer, estimated_tokens)
    """
    # Build the accumulated context from all steps
    step_context = []
    for s in steps:
        step_context.append(
            f"Step {s.step_number} — {s.thought}:\n"
            f"Finding: {s.intermediate}\n"
            f"Supporting evidence: {' | '.join(s.retrieved[:2])[:400] if s.retrieved else 'None found'}"
        )

    full_context = "\n\n".join(step_context)

    prompt = (
        f"You are answering a question using evidence collected through step-by-step reasoning.\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"EVIDENCE COLLECTED:\n{full_context}\n\n"
        f"Using only the evidence above, provide a comprehensive and accurate answer to the original question. "
        f"Cite which step's evidence supports each claim. "
        f"If a step found nothing, note the gap.\n\n"
        f"ANSWER:"
    )

    try:
        answer = llm_fn(prompt).strip()
        # Rough token estimate
        tokens = len(prompt.split()) + len(answer.split())
        return answer, tokens
    except Exception as e:
        logger.error("CoT final synthesis failed: %s", e)
        return f"Final synthesis failed: {e}", 0


# ── Main CoT-RAG orchestrator ─────────────────────────────────────────────────


def run_cot_rag(
    question: str,
    collection: str,
    retrieve_fn: Callable,
    llm_fn: Callable[[str], str],
    max_steps: int = 4,
    top_k_per_step: int = 4,
) -> CoTResult:
    """
    Full CoT-RAG pipeline: decompose → retrieve per step → synthesize.

    Args:
        question:       User question
        collection:     ChromaDB collection to search
        retrieve_fn:    core.retrieval.retrieve function
        llm_fn:         LLM completion function (complete_raw)
        max_steps:      Maximum reasoning steps
        top_k_per_step: Chunks to retrieve per reasoning step

    Returns:
        CoTResult with answer + full reasoning trace
    """
    pipeline_start = time.perf_counter()
    warnings: list[str] = []

    # ── 1. Decompose question into reasoning steps ────────────────────────────
    logger.info("CoT-RAG: decomposing question '%s'…", question[:60])
    step_descriptions = decompose_question(question, llm_fn, max_steps=max_steps)
    logger.info("CoT-RAG: %d reasoning steps", len(step_descriptions))

    # ── 2. Retrieve + synthesize each step ───────────────────────────────────
    reasoning_steps: list[ReasoningStep] = []
    all_sources: set[str] = set()
    total_chunks = 0

    for i, step_desc in enumerate(step_descriptions, start=1):
        step_start = time.perf_counter()
        logger.info("CoT-RAG step %d/%d: '%s'", i, len(step_descriptions), step_desc[:60])

        chunks, sources = retrieve_for_step(
            step=step_desc,
            collection=collection,
            retrieve_fn=retrieve_fn,
            top_k=top_k_per_step,
        )

        intermediate = synthesize_step(step_desc, chunks, llm_fn)
        all_sources.update(sources)
        total_chunks += len(chunks)

        step_ms = (time.perf_counter() - step_start) * 1000

        reasoning_steps.append(ReasoningStep(
            step_number=i,
            thought=step_desc,
            sub_query=step_desc,
            retrieved=[c[:300] for c in chunks],  # excerpt for display
            sources=sources,
            intermediate=intermediate,
            latency_ms=round(step_ms, 1),
        ))

    if total_chunks == 0:
        warnings.append("No context retrieved across any reasoning step — answers may be fabricated.")

    # ── 3. Final synthesis ────────────────────────────────────────────────────
    logger.info("CoT-RAG: synthesizing final answer from %d steps, %d chunks", len(reasoning_steps), total_chunks)
    final_answer, tokens = synthesize_final(question, reasoning_steps, llm_fn)

    total_ms = (time.perf_counter() - pipeline_start) * 1000

    result = CoTResult(
        question=question,
        answer=final_answer,
        reasoning_steps=reasoning_steps,
        all_sources=sorted(all_sources),
        total_chunks=total_chunks,
        tokens_used=tokens,
        latency_ms=round(total_ms, 2),
        num_steps=len(reasoning_steps),
        warnings=warnings,
    )

    logger.info(
        "CoT-RAG: done in %.0fms — %d steps, %d chunks, %d tokens",
        total_ms, len(reasoning_steps), total_chunks, tokens,
    )
    return result
