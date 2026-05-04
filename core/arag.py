"""
A-RAG — Hierarchical Retrieval Interfaces (Feb 2026)

Most RAG systems treat retrieval as a black box: one embedding search, always.
A-RAG exposes multiple retrieval interfaces and lets the agent decide which to
use at each step based on what it needs:

  keyword_search  — BM25 sparse retrieval. Best for exact terms, names,
                    product codes, acronyms, dates. Zero semantic drift.

  semantic_search — Dense embedding retrieval. Best for conceptual queries,
                    paraphrased questions, meaning-based lookup.

  hybrid_search   — Dense+BM25+RRF fusion. Best when uncertain; balances
                    lexical precision with semantic coverage.

  read_section    — Fetch chunks from a specific known source/section.
                    Best when a prior step identified a relevant document.

The loop:
  1. Show the agent the question + what's been retrieved so far
  2. Agent picks a tool + query (or decides to generate)
  3. Execute retrieval, accumulate context
  4. Repeat until the agent calls "generate" or max steps reached
  5. Generate final answer from accumulated context

This is the most cutting-edge agentic RAG pattern as of Feb 2026.
Unlike Adaptive RAG (picks one strategy upfront) and Agentic RAG (picks from
web/SQL/calculator tools), A-RAG specializes in retrieval-interface selection
and can mix strategies within a single resolution.

Paper: "A-RAG: Hierarchical Retrieval Interfaces for Agentic RAG" (Feb 2026)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable

from config import settings
from models import QueryMode, QueryRequest, RetrievalContext, RetrievalResult

logger = logging.getLogger(__name__)


# ── Tool registry ─────────────────────────────────────────────────────────────

RETRIEVAL_TOOLS = {
    "keyword_search": (
        "BM25 sparse keyword search. Use for: exact names, product codes, "
        "acronyms, dates, quoted phrases, proper nouns. High precision, zero semantic drift."
    ),
    "semantic_search": (
        "Dense embedding semantic search. Use for: conceptual questions, paraphrased "
        "queries, 'what does X mean', finding thematically related content."
    ),
    "hybrid_search": (
        "BM25 + dense + RRF fusion. Use when unsure which type is better, or "
        "when the query has both keyword-specific and conceptual components."
    ),
    "read_section": (
        "Retrieve chunks from a specific source file or section. Use when a "
        "prior retrieval step found a relevant document and you want more from it. "
        "Pass the source filename as the query."
    ),
    "generate": (
        "Stop retrieving and generate the final answer from accumulated context. "
        "Use when you have sufficient information."
    ),
}

_TOOL_DESCRIPTIONS = "\n".join(
    f"  {name}: {desc}" for name, desc in RETRIEVAL_TOOLS.items()
)


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class ARAGStep:
    """One step of the A-RAG agent loop."""

    step: int
    tool_chosen: str
    query: str
    reasoning: str
    retrieved: list[RetrievalResult]
    latency_ms: float


@dataclass
class ARAGResult:
    """Final output of an A-RAG run."""

    question: str
    answer: str
    steps: list[ARAGStep]
    unique_chunks: int
    tokens_used: int
    latency_ms: float
    tools_used: list[str]

    @property
    def num_steps(self) -> int:
        return len(self.steps)


# ── Agent prompts ─────────────────────────────────────────────────────────────

_AGENT_SYSTEM = """\
You are a retrieval-augmented reasoning agent with access to multiple retrieval interfaces.
At each step you decide which retrieval tool to use — or whether to generate a final answer.
Choose tools strategically based on what information is still needed.
Always output valid JSON."""

_AGENT_STEP_PROMPT = """\
Question: {question}

Retrieval tools available:
{tools}

Context retrieved so far ({num_chunks} chunks from {num_steps} steps):
{context_summary}

Decide your next action. Output ONLY valid JSON:

If retrieving:
{{"tool": "<tool_name>", "query": "<your search query>", "reasoning": "<why this tool for this query>"}}

If you have enough context to answer:
{{"tool": "generate", "query": "", "reasoning": "<why you have enough context now>"}}"""


# ── Tool execution ────────────────────────────────────────────────────────────


def _execute_tool(
    tool: str,
    query: str,
    collection: str,
    top_k: int,
    retrieve_fn: Callable[[QueryRequest], RetrievalContext],
    all_chunks: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Execute a retrieval tool and return new chunks."""

    mode_map = {
        "keyword_search": QueryMode.SPARSE,
        "semantic_search": QueryMode.DENSE,
        "hybrid_search":   QueryMode.HYBRID,
        "read_section":    QueryMode.HYBRID,   # filter by source in post-process
    }
    mode = mode_map.get(tool, QueryMode.HYBRID)

    req = QueryRequest(
        question=query,
        collection=collection,
        top_k=top_k,
        mode=mode,
    )
    ctx = retrieve_fn(req)
    results = ctx.results

    # For read_section, filter to chunks whose source matches the query
    if tool == "read_section" and query:
        query_lower = query.lower()
        filtered = [r for r in results if query_lower in r.source.lower()]
        if filtered:
            results = filtered

    # Deduplicate against already-seen chunks
    seen = {f"{r.source}:{r.chunk_index}" for r in all_chunks}
    return [r for r in results if f"{r.source}:{r.chunk_index}" not in seen]


def _build_context_summary(chunks: list[RetrievalResult], max_chars: int = 1200) -> str:
    if not chunks:
        return "(nothing retrieved yet)"
    parts = []
    chars = 0
    for r in sorted(chunks, key=lambda x: x.similarity_score, reverse=True):
        snippet = f"[{r.source}] {r.chunk_text[:200]}"
        if chars + len(snippet) > max_chars:
            break
        parts.append(snippet)
        chars += len(snippet)
    return "\n---\n".join(parts) or "(nothing retrieved yet)"


def _parse_agent_response(raw: str) -> tuple[str, str, str]:
    """Parse LLM JSON output → (tool, query, reasoning). Returns fallback on parse error."""
    raw = raw.strip()
    # Extract JSON block if wrapped in markdown
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)
    try:
        data = json.loads(raw)
        tool = data.get("tool", "generate").strip()
        query = data.get("query", "").strip()
        reasoning = data.get("reasoning", "").strip()
        if tool not in RETRIEVAL_TOOLS:
            tool = "hybrid_search"
        return tool, query, reasoning
    except (json.JSONDecodeError, AttributeError):
        logger.warning("A-RAG: could not parse agent JSON: %r", raw[:200])
        return "generate", "", "parse error — generating with current context"


# ── Core loop ─────────────────────────────────────────────────────────────────


def run_arag(
    question: str,
    collection: str,
    retrieve_fn: Callable[[QueryRequest], RetrievalContext],
    llm_raw_fn: Callable[[str], str],
    llm_complete_fn: Callable[[str, str], tuple[str, int, str]],
    max_steps: int = 5,
    top_k_per_step: int = 4,
) -> ARAGResult:
    """
    Run the A-RAG agent loop.

    Args:
        question:          User question
        collection:        ChromaDB collection
        retrieve_fn:       Retrieval function
        llm_raw_fn:        Raw LLM call for agent decisions (prompt → str)
        llm_complete_fn:   Full LLM call for final answer (system, user → str, tokens, model)
        max_steps:         Maximum retrieval steps before forced generation
        top_k_per_step:    Chunks per retrieval call
    """
    t_start = time.perf_counter()

    all_chunks: list[RetrievalResult] = []
    steps: list[ARAGStep] = []
    tools_used: list[str] = []
    total_tokens = 0

    for step_num in range(1, max_steps + 1):
        t_step = time.perf_counter()

        # Build agent decision prompt
        prompt = _AGENT_STEP_PROMPT.format(
            question=question,
            tools=_TOOL_DESCRIPTIONS,
            num_chunks=len(all_chunks),
            num_steps=step_num - 1,
            context_summary=_build_context_summary(all_chunks),
        )

        raw_decision = llm_raw_fn(prompt)
        tool, query, reasoning = _parse_agent_response(raw_decision)

        logger.info(
            "A-RAG step %d: tool=%s query=%r reasoning=%s",
            step_num, tool, query[:60], reasoning[:60],
        )

        if tool == "generate":
            logger.info("A-RAG agent chose to generate at step %d", step_num)
            break

        # Execute retrieval
        try:
            new_chunks = _execute_tool(
                tool=tool,
                query=query or question,
                collection=collection,
                top_k=top_k_per_step,
                retrieve_fn=retrieve_fn,
                all_chunks=all_chunks,
            )
        except Exception as e:
            logger.warning("A-RAG tool %s failed: %s", tool, e)
            new_chunks = []

        all_chunks.extend(new_chunks)
        tools_used.append(tool)

        step_ms = (time.perf_counter() - t_step) * 1000
        steps.append(ARAGStep(
            step=step_num,
            tool_chosen=tool,
            query=query,
            reasoning=reasoning,
            retrieved=new_chunks,
            latency_ms=step_ms,
        ))

    # Generate final answer
    _ANSWER_SYSTEM = (
        "You are a precise question-answering assistant. "
        "Answer using only the provided context. "
        "If the context is insufficient, say so clearly."
    )

    context_text = _build_context_summary(all_chunks, max_chars=3000)
    answer_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer comprehensively based on the context above."
    )

    try:
        answer, tokens, _ = llm_complete_fn(_ANSWER_SYSTEM, answer_prompt)
        total_tokens += tokens
    except Exception as e:
        logger.error("A-RAG generation failed: %s", e)
        answer = f"Generation failed: {e}"

    return ARAGResult(
        question=question,
        answer=answer,
        steps=steps,
        unique_chunks=len(all_chunks),
        tokens_used=total_tokens,
        latency_ms=(time.perf_counter() - t_start) * 1000,
        tools_used=tools_used,
    )
