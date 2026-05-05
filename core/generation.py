"""
Generation layer — model-agnostic LLM interface.

Supports three backends, switchable via LLM_BACKEND env var:
  - ollama  : free local inference (llama3.2, mistral, etc.) — DEFAULT
  - claude  : Anthropic claude-sonnet-4-6 — premium quality
  - openai  : OpenAI GPT models — alternative paid option

Architecture:
  - All backends implement the same `LLMBackendProtocol`
  - The `generate()` function dispatches to the active backend
  - Prompt construction is backend-agnostic (shared)
  - Semantic cache is checked before any LLM call
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import requests

from config import LLMBackend, settings
from models import QueryRequest, QueryResponse, RetrievalContext, SourceCitation
from core.retrieval import get_cache, retrieve

logger = logging.getLogger(__name__)


# ── LLM Backend Protocol ──────────────────────────────────────────────────────


class LLMBackendProtocol(Protocol):
    """All LLM backends must implement this interface."""

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, str]:
        """
        Generate a completion.

        Args:
            system_prompt: system/instruction context
            user_prompt: user message with injected context

        Returns:
            (answer_text, tokens_used, model_name)
        """
        ...

    def complete_raw(self, prompt: str) -> str:
        """
        Simple single-prompt completion without system/user split.
        Used for HyDE, multi-query, CRAG rewrites, and eval.
        """
        ...


# ── Ollama backend (free, local) ──────────────────────────────────────────────


class OllamaBackend:
    """
    Ollama local inference backend.

    Ollama runs open-source models (Llama 3.2, Mistral, Qwen, etc.) locally.
    Zero API cost. Pull a model first: `ollama pull llama3.2`
    """

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self._check_connection()

    def _check_connection(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            logger.info("Ollama connected at '%s' (model: %s)", self.base_url, self.model)
        except requests.RequestException as e:
            logger.warning(
                "Ollama not reachable at '%s': %s. "
                "Start Ollama and run: ollama pull %s",
                self.base_url, e, self.model,
            )

    def _chat(self, messages: list[dict], stream: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        data = self._chat(messages)
        text = data.get("message", {}).get("content", "")
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        return text, eval_count + prompt_eval_count, self.model

    def complete_raw(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        data = self._chat(messages)
        return data.get("message", {}).get("content", "")


# ── Claude backend (Anthropic, premium) ──────────────────────────────────────


class ClaudeBackend:
    """
    Anthropic Claude backend using the official SDK.

    Model: claude-sonnet-4-6 (configurable via CLAUDE_MODEL env var)
    Requires ANTHROPIC_API_KEY in .env
    """

    def __init__(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key.strip())
            self.model = settings.claude_model
            logger.info("Claude backend initialized (model: %s)", self.model)
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, str]:
        import anthropic
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = msg.content[0].text if msg.content else ""
            tokens = msg.usage.input_tokens + msg.usage.output_tokens
            return text, tokens, self.model
        except anthropic.APIError as e:
            raise RuntimeError(f"Claude API error: {e}") from e

    def complete_raw(self, prompt: str) -> str:
        answer, _, _ = self.complete(
            system_prompt="You are a helpful assistant. Be concise.",
            user_prompt=prompt,
        )
        return answer


# ── OpenAI backend ────────────────────────────────────────────────────────────


class OpenAIBackend:
    """
    OpenAI GPT backend.

    Requires OPENAI_API_KEY in .env and openai package: pip install openai
    """

    def __init__(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
            self.model = settings.openai_model
            logger.info("OpenAI backend initialized (model: %s)", self.model)
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, str]:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
            return text, tokens, self.model
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def complete_raw(self, prompt: str) -> str:
        text, _, _ = self.complete(
            system_prompt="You are a helpful assistant. Be concise.",
            user_prompt=prompt,
        )
        return text


# ── Backend factory ───────────────────────────────────────────────────────────

_active_backend: LLMBackendProtocol | None = None


def get_backend() -> LLMBackendProtocol:
    """Return the active LLM backend singleton. Initializes on first call."""
    global _active_backend
    if _active_backend is None:
        settings.validate_backend_credentials()
        if settings.llm_backend == LLMBackend.CLAUDE:
            _active_backend = ClaudeBackend()
        elif settings.llm_backend == LLMBackend.OPENAI:
            _active_backend = OpenAIBackend()
        else:
            _active_backend = OllamaBackend()
    return _active_backend


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Answer using ONLY the provided context. Rules:
1. Cite every claim: [Source: <filename>, chunk <N>]
2. If context lacks the answer: "I don't have enough context to answer this."
3. No outside knowledge. No hallucination. Be concise.
"""


def build_user_prompt(context: RetrievalContext) -> str:
    """
    Construct the user prompt by injecting all retrieved chunks with source labels.

    Each chunk is labeled with its source filename and chunk index so the model
    can cite them accurately.
    """
    if context.is_empty:
        return (
            f"No relevant context was found for the following question. "
            f"Please respond accordingly.\n\nQuestion: {context.query}"
        )

    context_blocks = []
    for i, result in enumerate(context.results, start=1):
        source_label = f"{result.source}, chunk {result.chunk_index}"
        if result.page_number:
            source_label += f", page {result.page_number}"
        block = (
            f"[Context {i} | Source: {source_label} | Similarity: {result.similarity_score:.3f}]\n"
            f"{result.chunk_text}"
        )
        context_blocks.append(block)

    context_str = "\n\n---\n\n".join(context_blocks)

    return (
        f"CONTEXT DOCUMENTS:\n\n{context_str}\n\n"
        f"{'=' * 60}\n\n"
        f"QUESTION: {context.query}\n\n"
        f"Answer based ONLY on the context above. Cite sources inline."
    )


# ── Source extraction ─────────────────────────────────────────────────────────


def extract_sources(context: RetrievalContext) -> list[SourceCitation]:
    """Build SourceCitation objects from the retrieval context."""
    return [
        SourceCitation(
            source=r.source,
            chunk_index=r.chunk_index,
            page_number=r.page_number,
            similarity_score=r.similarity_score,
            excerpt=r.chunk_text[:200],
        )
        for r in context.results
    ]


# ── CRAG evaluation function ──────────────────────────────────────────────────


def make_crag_evaluator(backend: LLMBackendProtocol) -> "Callable[[str, list[str]], float]":  # type: ignore[name-defined]  # noqa: F821
    """
    Return a CRAG evaluation callable that uses the active LLM backend.

    The evaluator asks the LLM to rate how relevant retrieved chunks are
    to the original question (0-1 scale).
    """
    def evaluate(question: str, chunk_texts: list[str]) -> float:
        context_preview = "\n\n".join(chunk_texts[:3])[:1500]
        prompt = (
            f"Rate how relevant the following context is to answering the question.\n"
            f"Reply with ONLY a decimal number from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_preview}\n\n"
            f"Relevance score (0.0-1.0):"
        )
        try:
            raw = backend.complete_raw(prompt).strip()
            score = float(raw.split()[0].rstrip(",.:"))
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            logger.warning("CRAG: could not parse relevance score from LLM response '%s', defaulting to 0.5", raw[:50])
            return 0.5

    return evaluate


# ── Main generation orchestrator ──────────────────────────────────────────────


def answer_question(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline: cache check → retrieve → generate → cache store.
    Includes Langfuse tracing and security audit logging.
    """
    start = time.perf_counter()
    backend = get_backend()

    # ── Langfuse trace ─────────────────────────────────────────────────────────
    from core.observability import start_trace
    session_id = getattr(request, "session_id", None)
    trace = start_trace(request.question, request.collection, session_id=session_id)

    # ── Semantic cache check ───────────────────────────────────────────────────
    cache = get_cache()
    if cache and settings.enable_cache:
        from core.ingestion import get_embedding_model
        model = get_embedding_model()
        q_emb = model.encode([request.question], normalize_embeddings=True)[0].tolist()
        cached = cache.get(request.question, q_emb)
        if cached:
            response = cached.response.model_copy(
                update={"cache_hit": True, "latency_ms": (time.perf_counter() - start) * 1000}
            )
            logger.info("Cache hit for question: '%s'", request.question[:60])
            trace.finish(answer=response.answer, tokens=response.tokens_used, cache_hit=True)
            return response

    # ── Retrieval ──────────────────────────────────────────────────────────────
    generate_fn = backend.complete_raw
    evaluate_fn = make_crag_evaluator(backend) if settings.use_hybrid_search else None

    context = retrieve(request, generate_fn=generate_fn, evaluate_fn=evaluate_fn)
    trace.log_retrieval(context.results, query_mode=request.mode.value if hasattr(request.mode, "value") else str(request.mode))

    # ── Token budget: deduplicate + trim to model context window ──────────────
    if not context.is_empty:
        try:
            from core.token_budget import optimize_context
            context, budget_info = optimize_context(
                context=context,
                question=request.question,
                system_prompt=SYSTEM_PROMPT,
                model_name=settings.claude_model if settings.llm_backend == LLMBackend.CLAUDE else
                           settings.openai_model if settings.llm_backend == LLMBackend.OPENAI else
                           settings.ollama_model,
            )
            if budget_info.truncated:
                logger.info(
                    "Token budget: %d→%d chunks, %.0f%% savings (~%d tokens)",
                    budget_info.included + budget_info.excluded,
                    budget_info.included,
                    budget_info.savings_pct,
                    budget_info.estimated_tokens,
                )
        except Exception as e:
            logger.warning("Token budget optimization failed (non-fatal): %s", e)

    # ── Sufficient Context check (Google ICLR 2025) ────────────────────────────
    sufficiency_score: float | None = None
    sufficiency_recommendation: str = "generate"
    if settings.enable_sufficient_context and not context.is_empty:
        try:
            from core.sufficient_context import get_checker
            checker = get_checker()
            # Re-create if threshold changed in settings
            if abs(checker.sufficiency_threshold - settings.sufficient_context_threshold) > 0.01:
                from core.sufficient_context import SufficientContextChecker
                import core.sufficient_context as _sc_module
                _sc_module._checker = SufficientContextChecker(
                    sufficiency_threshold=settings.sufficient_context_threshold,
                )
                checker = _sc_module._checker
            suf = checker.score(
                question=request.question,
                context=context,
                llm_fn=backend.complete_raw if settings.sufficient_context_self_rating else None,
                enable_self_rating=settings.sufficient_context_self_rating,
            )
            sufficiency_score = suf.overall_score
            sufficiency_recommendation = suf.recommendation
            logger.debug("Sufficiency: %s (score=%.2f)", suf.recommendation, suf.overall_score)

            # If abstaining, return immediately without calling LLM
            if suf.recommendation == "abstain":
                from core.sufficient_context import abstention_response
                abstention_answer = abstention_response(suf)
                latency_ms = (time.perf_counter() - start) * 1000
                trace.finish(answer=abstention_answer, tokens=0, latency_ms=latency_ms)
                return QueryResponse(
                    question=request.question,
                    answer=abstention_answer,
                    sources=[],
                    tokens_used=0,
                    latency_ms=round(latency_ms, 2),
                    collection=request.collection,
                    llm_backend=settings.llm_backend.value,
                    model_used="abstained",
                    cache_hit=False,
                    retrieval_context=context,
                )
        except Exception as e:
            logger.debug("Sufficient context check failed (non-fatal): %s", e)

    # ── Prompt construction ────────────────────────────────────────────────────
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(context)

    # ── LLM call ──────────────────────────────────────────────────────────────
    logger.info("Calling %s backend for: '%s'…", settings.llm_backend.value, request.question[:60])
    answer, tokens_used, model_name = backend.complete(system_prompt, user_prompt)

    latency_ms = (time.perf_counter() - start) * 1000
    sources = extract_sources(context)

    trace.log_generation(user_prompt, answer, model_name, tokens_used, latency_ms)

    response = QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        tokens_used=tokens_used,
        latency_ms=round(latency_ms, 2),
        collection=request.collection,
        llm_backend=settings.llm_backend.value,
        model_used=model_name,
        cache_hit=False,
        retrieval_context=context,
    )

    # ── Store in cache ─────────────────────────────────────────────────────────
    if cache and settings.enable_cache and not context.is_empty:
        from core.ingestion import get_embedding_model
        model = get_embedding_model()
        q_emb = model.encode([request.question], normalize_embeddings=True)[0].tolist()
        cache.put(request.question, response, q_emb)

    # ── Security audit log ────────────────────────────────────────────────────
    if settings.enable_audit_log:
        try:
            from core.security import audit_query
            audit_query(
                question=request.question,
                collection=request.collection,
                answer=answer,
                sources_returned=len(sources),
                session_id=str(session_id) if session_id else None,
            )
        except Exception as e:
            logger.debug("Audit log failed: %s", e)

    trace.finish(answer=answer, tokens=tokens_used, latency_ms=latency_ms)
    logger.info("Generated answer in %.0fms (%d tokens, %d sources)", latency_ms, tokens_used, len(sources))
    return response


# ── Structured output generation ──────────────────────────────────────────────


def answer_structured(
    request: QueryRequest,
    output_schema: dict,
) -> dict:
    """
    Generate a structured JSON response extracted from retrieved context.

    Instead of prose, returns a typed dict matching the provided JSON schema.
    Uses Claude's native structured output for guaranteed schema compliance.

    Example schema:
        {"type": "object", "properties": {
            "revenue": {"type": "number"},
            "unit": {"type": "string"},
            "period": {"type": "string"}
        }}

    Args:
        request: standard QueryRequest
        output_schema: JSON Schema dict describing the desired output structure

    Returns:
        Dict matching the schema, extracted from retrieved context
    """
    import json as _json
    backend = get_backend()

    context = retrieve(request, generate_fn=backend.complete_raw)
    if context.is_empty:
        return {"error": "No relevant context found", "question": request.question}

    # ── Token budget optimization ──────────────────────────────────────────────
    try:
        from core.token_budget import optimize_context
        context, _ = optimize_context(
            context=context,
            question=request.question,
            system_prompt="",
            model_name=settings.claude_model if settings.llm_backend == LLMBackend.CLAUDE else
                       settings.openai_model if settings.llm_backend == LLMBackend.OPENAI else
                       settings.ollama_model,
        )
    except Exception as e:
        logger.warning("Token budget optimization skipped for structured output: %s", e)

    user_prompt = (
        f"{build_user_prompt(context)}\n\n"
        f"Extract the answer as a JSON object matching this schema:\n"
        f"{_json.dumps(output_schema, indent=2)}\n\n"
        f"Return ONLY valid JSON, no explanation:"
    )

    system = (
        "You are a structured data extraction assistant. "
        "Extract information from the provided context and return it as valid JSON. "
        "Only include data explicitly present in the context. "
        "Use null for missing fields."
    )

    # Use Claude's structured output if available
    if settings.llm_backend.value == "claude" and settings.anthropic_api_key:
        try:
            import anthropic, json as _json2
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key.strip())
            msg = client.messages.create(
                model=settings.claude_model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = msg.content[0].text.strip()
            # Strip markdown fences if present
            import re
            raw = re.sub(r"^```(?:json)?\s*", "", raw).rstrip("` \n")
            return _json2.loads(raw)
        except Exception as e:
            logger.warning("Structured output via Claude failed: %s. Falling back.", e)

    # Fallback: any backend
    try:
        import json as _json3, re as _re
        raw = backend.complete_raw(f"{system}\n\n{user_prompt}").strip()
        raw = _re.sub(r"^```(?:json)?\s*", "", raw).rstrip("` \n")
        return _json3.loads(raw)
    except Exception as e:
        return {"error": f"Structured extraction failed: {e}", "question": request.question}
