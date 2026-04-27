"""
Agentic RAG — LLM decides which tools to call and in what order.

Instead of a fixed pipeline (retrieve → generate), the LLM orchestrates:
  1. Decide what information is needed
  2. Choose the right tool (docs, web, SQL, calculator, code)
  3. Inspect the result, decide if it's sufficient
  4. Repeat until confident → synthesize final answer

This is the 2025 production pattern. It handles questions that require:
  - Combining information from docs AND live web data
  - Running calculations on retrieved data
  - Querying structured databases alongside documents
  - Multi-step reasoning with intermediate lookups

Tools available:
  search_docs     — query the local ChromaDB vector store
  search_web      — Tavily web search (requires TAVILY_API_KEY)
  query_sql       — natural language → SQL on configured database
  calculate       — evaluate a math expression safely
  get_date        — current date/time (grounding)
  summarize_docs  — summarize a collection overview

Uses Claude's native tool_use API for clean structured tool dispatch.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Tool definitions for Claude's tool_use API ────────────────────────────────

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Search the local knowledge base for relevant document chunks. "
            "Use this when the question likely relates to ingested documents. "
            "Returns up to 6 relevant text passages with their sources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document passages",
                },
                "collection": {
                    "type": "string",
                    "description": "The collection to search (default: 'default')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 6)",
                    "default": 6,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for current information not in the local knowledge base. "
            "Use when documents don't cover the question, or for recent events/data. "
            "Returns top web results with titles, URLs, and content excerpts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default: 4)",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_sql",
        "description": (
            "Query a structured database using natural language. "
            "Use for precise numerical data, filtered lookups, or aggregations "
            "that would be vague in document search. "
            "Returns a table of results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language description of what data to retrieve",
                },
                "database": {
                    "type": "string",
                    "description": "Database name or path (optional, uses default if omitted)",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression precisely. "
            "Use when the question involves arithmetic, percentages, or unit conversions. "
            "Safer than asking the LLM to do math in its head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python math expression to evaluate (e.g., '2.3e9 * 1.15')",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_date",
        "description": "Get the current date and time. Use for temporal grounding.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "summarize_collection",
        "description": (
            "Get an overview of what documents are in a collection. "
            "Use this first if unsure which collection to search or "
            "what topics the knowledge base covers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name to summarize",
                },
            },
            "required": ["collection"],
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A single tool invocation and its result."""
    tool_name: str
    tool_input: dict
    result: str
    latency_ms: float


@dataclass
class AgentResult:
    """Final result from the agentic RAG pipeline."""
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    total_tokens: int = 0
    latency_ms: float = 0.0
    model_used: str = ""
    iterations: int = 0


def _execute_tool(
    tool_name: str,
    tool_input: dict,
    collection: str,
    retrieve_fn: Callable | None = None,
    sql_fn: Callable | None = None,
) -> str:
    """
    Dispatch a tool call and return the result as a string.

    Args:
        tool_name: name of the tool to call
        tool_input: tool arguments
        collection: default collection for search_docs
        retrieve_fn: callable for doc retrieval
        sql_fn: callable for SQL queries

    Returns:
        String result to pass back to the LLM
    """
    start = time.perf_counter()

    try:
        if tool_name == "search_docs":
            return _tool_search_docs(
                query=tool_input.get("query", ""),
                collection=tool_input.get("collection", collection),
                top_k=tool_input.get("top_k", 6),
                retrieve_fn=retrieve_fn,
            )

        elif tool_name == "search_web":
            return _tool_search_web(
                query=tool_input.get("query", ""),
                max_results=tool_input.get("max_results", 4),
            )

        elif tool_name == "query_sql":
            return _tool_query_sql(
                question=tool_input.get("question", ""),
                database=tool_input.get("database"),
                sql_fn=sql_fn,
            )

        elif tool_name == "calculate":
            return _tool_calculate(tool_input.get("expression", ""))

        elif tool_name == "get_date":
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            return f"Current UTC date/time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

        elif tool_name == "summarize_collection":
            return _tool_summarize_collection(tool_input.get("collection", collection))

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as e:
        logger.warning("Tool '%s' failed: %s", tool_name, e)
        return f"Tool error: {e}"


def _tool_search_docs(
    query: str,
    collection: str,
    top_k: int,
    retrieve_fn: Callable | None,
) -> str:
    """Execute document search."""
    if retrieve_fn:
        try:
            from models import QueryRequest, QueryMode
            request = QueryRequest(question=query, collection=collection, top_k=top_k, mode=QueryMode.HYBRID)
            ctx = retrieve_fn(request)
            if not ctx.results:
                return "No relevant documents found."
            parts = []
            for i, r in enumerate(ctx.results[:top_k], 1):
                parts.append(f"[{i}] Source: {r.source} (score: {r.similarity_score:.3f})\n{r.chunk_text[:500]}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            return f"Document search failed: {e}"

    # Fallback: direct ChromaDB query
    try:
        from core.ingestion import get_or_create_collection, embed_texts
        col = get_or_create_collection(collection)
        if col.count() == 0:
            return "Collection is empty."
        emb = embed_texts([query])[0]
        results = col.query(query_embeddings=[emb], n_results=min(top_k, col.count()), include=["documents", "metadatas"])
        docs = results.get("documents", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []
        if not docs:
            return "No relevant documents found."
        parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            src = meta.get("source_file", "unknown") if meta else "unknown"
            parts.append(f"[{i}] Source: {src}\n{doc[:500]}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Document search failed: {e}"


def _tool_search_web(query: str, max_results: int) -> str:
    """Execute web search."""
    from core.web_search import web_search
    results = web_search(query, max_results=max_results)
    if not results:
        return "Web search returned no results (check TAVILY_API_KEY or install duckduckgo-search)."
    parts = []
    for i, r in enumerate(results, 1):
        date_str = f" ({r.published_date})" if r.published_date else ""
        parts.append(f"[{i}] {r.title}{date_str}\nURL: {r.url}\n{r.content[:400]}")
    return "\n\n---\n\n".join(parts)


def _tool_query_sql(question: str, database: str | None, sql_fn: Callable | None) -> str:
    """Execute SQL query via text-to-SQL."""
    if sql_fn:
        try:
            return sql_fn(question, database)
        except Exception as e:
            return f"SQL query failed: {e}"

    try:
        from core.sql_retrieval import query_natural_language
        return query_natural_language(question, database)
    except Exception as e:
        return f"SQL retrieval failed: {e}"


def _tool_calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    # Whitelist safe names only
    safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_names.update({"abs": abs, "round": round, "int": int, "float": float, "min": min, "max": max})
    try:
        # Only allow simple expressions (no builtins that could be dangerous)
        result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def _tool_summarize_collection(collection: str) -> str:
    """Summarize what's in a collection."""
    try:
        from core.ingestion import list_collections
        cols = list_collections()
        for c in cols:
            if c["name"] == collection:
                return (
                    f"Collection '{collection}': {c['document_count']} chunks, "
                    f"embedding model: {c['embedding_model']}"
                )
        return f"Collection '{collection}' not found."
    except Exception as e:
        return f"Collection summary failed: {e}"


# ── Main agentic loop ─────────────────────────────────────────────────────────


def run_agent(
    question: str,
    collection: str = "default",
    retrieve_fn: Callable | None = None,
    sql_fn: Callable | None = None,
    max_iterations: int = 8,
    model: str | None = None,
) -> AgentResult:
    """
    Run the agentic RAG loop using Claude's native tool_use API.

    The agent receives the question and a set of tools. It decides which tools
    to call, inspects results, and keeps going until it has a complete answer.

    This is the ReAct (Reasoning + Acting) pattern implemented with Claude's
    structured tool_use rather than text-based action parsing.

    Args:
        question: user's question
        collection: default collection for document search
        retrieve_fn: optional callable for document retrieval
        sql_fn: optional callable for SQL queries
        max_iterations: max tool-call rounds before forcing synthesis
        model: Claude model to use (defaults to config claude_model)

    Returns:
        AgentResult with answer, tool call trace, and usage stats
    """
    try:
        import anthropic
        from config import settings
    except ImportError:
        return AgentResult(
            answer="Agentic RAG requires the anthropic SDK and ANTHROPIC_API_KEY.",
            iterations=0,
        )

    if not settings.anthropic_api_key:
        return AgentResult(
            answer="Agentic RAG requires ANTHROPIC_API_KEY. Set it in .env.",
            iterations=0,
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    model_id = model or settings.claude_model

    system_prompt = (
        "You are an expert research assistant with access to multiple tools. "
        "Use tools to gather information before answering. "
        "Always use search_docs first for questions about internal documents. "
        "Use search_web for current events or when documents don't cover the topic. "
        "Use calculate for any arithmetic to ensure precision. "
        "After gathering sufficient information, provide a comprehensive answer "
        "with inline citations (e.g., [Source: filename] or [Web: URL])."
    )

    messages: list[dict] = [{"role": "user", "content": question}]
    tool_calls: list[ToolCall] = []
    total_tokens = 0
    start = time.perf_counter()
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        logger.debug("Agent iteration %d: calling %s", iterations, model_id)

        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            logger.error("Agent LLM call failed: %s", e)
            return AgentResult(
                answer=f"Agent error: {e}",
                tool_calls=tool_calls,
                total_tokens=total_tokens,
                latency_ms=(time.perf_counter() - start) * 1000,
                iterations=iterations,
            )

        total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Extract text answer from final response
            answer_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    answer_text += block.text
            return AgentResult(
                answer=answer_text,
                tool_calls=tool_calls,
                total_tokens=total_tokens,
                latency_ms=(time.perf_counter() - start) * 1000,
                model_used=model_id,
                iterations=iterations,
            )

        # Process tool use blocks
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            # No tools called, no end_turn — extract whatever text is there
            answer_text = " ".join(b.text for b in response.content if hasattr(b, "text"))
            return AgentResult(
                answer=answer_text or "No answer generated.",
                tool_calls=tool_calls,
                total_tokens=total_tokens,
                latency_ms=(time.perf_counter() - start) * 1000,
                model_used=model_id,
                iterations=iterations,
            )

        # Add assistant message (with tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and collect results
        tool_results = []
        for block in tool_use_blocks:
            t_start = time.perf_counter()
            result_text = _execute_tool(
                tool_name=block.name,
                tool_input=block.input,
                collection=collection,
                retrieve_fn=retrieve_fn,
                sql_fn=sql_fn,
            )
            t_latency = (time.perf_counter() - t_start) * 1000

            logger.info("Agent tool '%s' → %d chars in %.0fms", block.name, len(result_text), t_latency)
            tool_calls.append(ToolCall(
                tool_name=block.name,
                tool_input=block.input,
                result=result_text[:500],  # truncate for storage
                latency_ms=t_latency,
            ))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text[:4000],  # cap at 4K per tool result
            })

        # Add tool results to conversation
        messages.append({"role": "user", "content": tool_results})

    # Max iterations reached — force synthesis
    logger.warning("Agent reached max iterations (%d). Forcing synthesis.", max_iterations)
    try:
        synthesis_response = client.messages.create(
            model=model_id,
            max_tokens=2048,
            system=system_prompt + "\n\nYou have gathered enough information. Synthesize a final answer now.",
            messages=messages + [{"role": "user", "content": "Please provide your final answer based on all the information gathered."}],
        )
        total_tokens += synthesis_response.usage.input_tokens + synthesis_response.usage.output_tokens
        answer_text = " ".join(b.text for b in synthesis_response.content if hasattr(b, "text"))
    except Exception:
        answer_text = "Agent reached maximum iterations without a complete answer."

    return AgentResult(
        answer=answer_text,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        latency_ms=(time.perf_counter() - start) * 1000,
        model_used=model_id,
        iterations=iterations,
    )
