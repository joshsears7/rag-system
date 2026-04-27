"""
Contextual Retrieval — Anthropic's Nov 2024 technique.

The Problem:
  Standard chunking loses context. A chunk saying "The revenue increased 15%"
  is meaningless without knowing *which company, which period, which metric*.
  Embedding this chunk in isolation produces a poor embedding.

The Solution:
  Before embedding each chunk, call the LLM to prepend a 1-2 sentence context
  that situates the chunk within the full document. The chunk + context is then
  embedded together, dramatically improving retrieval accuracy.

Anthropic's published results: 49% reduction in retrieval failures.
Cost: ~1 Haiku call per chunk at ingest time (cheap; one-time cost).

Reference:
  https://www.anthropic.com/news/contextual-retrieval (Nov 2024)
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# System prompt for context generation
_CONTEXT_SYSTEM = (
    "You are a document indexing assistant. Your job is to write a single concise "
    "sentence (max 30 words) that describes what a text excerpt is about and where "
    "it fits within the larger document. Be specific — mention the document section, "
    "topic, and any named entities. Do NOT repeat the excerpt itself."
)


def generate_chunk_context(
    chunk_text: str,
    document_text: str,
    llm_fn: Callable[[str], str],
    max_doc_chars: int = 3000,
) -> str:
    """
    Generate a contextual prefix for a single chunk.

    The LLM sees a condensed version of the full document so it can understand
    where the chunk fits. It returns a single sentence to prepend.

    Args:
        chunk_text: the chunk to contextualize
        document_text: the full document text (truncated for cost control)
        llm_fn: LLM callable (prompt -> text)
        max_doc_chars: max chars of document to pass as context

    Returns:
        A 1-2 sentence contextual description to prepend to the chunk
    """
    doc_preview = document_text[:max_doc_chars]
    prompt = (
        f"Document (truncated):\n{doc_preview}\n\n"
        f"---\n\nExcerpt from this document:\n{chunk_text[:800]}\n\n"
        f"---\n\nWrite ONE sentence (max 30 words) describing what this excerpt is about "
        f"and where it fits in the document. Be specific about topic, section, and entities. "
        f"Do NOT repeat the excerpt text.\n\nContext sentence:"
    )
    try:
        context = llm_fn(prompt).strip()
        # Truncate if the LLM goes verbose
        sentences = context.split(". ")
        context = sentences[0].rstrip(".") + "." if sentences else context
        return context
    except Exception as e:
        logger.warning("Context generation failed for chunk: %s", e)
        return ""


def add_context_to_chunk(
    chunk_text: str,
    context: str,
) -> str:
    """
    Prepend context to a chunk text for embedding.

    The resulting string is what gets embedded — the context ensures the
    embedding captures document-level semantics, not just local semantics.
    """
    if not context:
        return chunk_text
    return f"{context}\n\n{chunk_text}"


def contextualize_chunks(
    chunks: list[str],
    document_text: str,
    llm_fn: Callable[[str], str],
    use_cheap_model: bool = True,
) -> list[str]:
    """
    Add contextual prefixes to a list of chunks from a single document.

    Processes all chunks from the same document. Uses Haiku for cost efficiency
    when use_cheap_model=True and claude backend is active.

    Args:
        chunks: list of chunk texts
        document_text: the full document these chunks came from
        llm_fn: LLM callable (use Haiku for cost efficiency)
        use_cheap_model: if True, switch to claude-haiku for context generation

    Returns:
        List of contextualized chunk texts (same length as input)
    """
    if not chunks:
        return chunks

    contextualized = []

    # Use Haiku for cost efficiency if claude backend is active
    actual_llm_fn = llm_fn
    if use_cheap_model:
        try:
            from config import settings, LLMBackend
            if settings.llm_backend == LLMBackend.CLAUDE and settings.anthropic_api_key:
                import anthropic
                _haiku_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

                def haiku_fn(prompt: str) -> str:
                    resp = _haiku_client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=100,
                        system=_CONTEXT_SYSTEM,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp.content[0].text

                actual_llm_fn = haiku_fn
        except Exception:
            pass  # fall back to the provided llm_fn

    logger.info("Contextualizing %d chunks (Anthropic Nov 2024 technique)…", len(chunks))
    successes = 0

    for i, chunk in enumerate(chunks):
        context = generate_chunk_context(chunk, document_text, actual_llm_fn)
        if context:
            contextualized.append(add_context_to_chunk(chunk, context))
            successes += 1
        else:
            contextualized.append(chunk)

        if (i + 1) % 10 == 0:
            logger.debug("Contextualized %d/%d chunks…", i + 1, len(chunks))

    logger.info(
        "Contextual retrieval: %d/%d chunks enhanced with document context",
        successes, len(chunks),
    )
    return contextualized
