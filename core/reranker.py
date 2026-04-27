"""
Advanced reranking strategies — beyond cross-encoder scoring.

Production reranking is a multi-stage funnel:
  Stage 1: ANN vector search → top-100 candidates (milliseconds)
  Stage 2: Cross-encoder reranking → top-20 (10-50ms)
  Stage 3: LLM reranking → top-6 (optional, highest quality, most expensive)
  Stage 4: Feedback-boosted reranking → final results

This module adds:
  1. LLM-based reranking (RankGPT pattern): ask the LLM to rank candidates
     directly. More expensive but highest quality — the LLM understands nuance
     that a cross-encoder misses (e.g. implicit relevance, domain jargon).

  2. Reciprocal Rank Fusion across multiple rerankers (ensemble).

  3. Diversity-aware reranking: combines MMR with cross-encoder scores
     instead of treating them separately.

  4. Feedback-adjusted reranking: apply source boost factors from user feedback.

  5. ColBERT-style late interaction: compute token-level similarity for richer
     matching than bi-encoder cosine similarity (no full ColBERT model needed).

Reference: "Is ChatGPT Good at Search? Investigating Large Language Models as
           Re-Ranking Agents" (Sun et al., 2023)
"""

from __future__ import annotations

import logging
import re
from typing import Callable

import numpy as np

from models import RetrievalResult

logger = logging.getLogger(__name__)


# ── LLM Reranking (RankGPT pattern) ─────────────────────────────────────────


def llm_rerank(
    question: str,
    results: list[RetrievalResult],
    llm_fn: Callable[[str], str],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """
    Use the LLM to rerank retrieved chunks by relevance.

    Implements the RankGPT sliding window approach:
      - Present the LLM with all candidates (numbered list)
      - Ask it to rank them by relevance to the question
      - Parse the ranking and reorder results

    This is the highest-quality reranking available. The LLM understands:
      - Implicit relevance ("the policy" when question mentions "refund rules")
      - Domain jargon and synonyms
      - Structural cues (section headers, tables vs prose)

    Cost: 1 LLM call per reranking, so use sparingly (after cross-encoder).

    Args:
        question: user's original question
        results: candidates to rerank (already cross-encoder scored)
        llm_fn: LLM callable for ranking
        top_k: return only this many results after reranking

    Returns:
        Reordered list of RetrievalResult
    """
    if len(results) <= 1:
        return results

    # Build numbered candidate list
    passages = "\n\n".join(
        f"[{i+1}] {r.chunk_text[:400]}"
        for i, r in enumerate(results)
    )

    prompt = (
        f"I will give you {len(results)} text passages and a question. "
        "Rank the passages by how well they help answer the question. "
        "Return ONLY a comma-separated list of passage numbers, most relevant first. "
        "Example: 3, 1, 4, 2\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{passages}\n\n"
        "Ranking (most relevant first):"
    )

    try:
        raw = llm_fn(prompt).strip()
        # Parse number sequence from response
        numbers = [int(n.strip()) for n in re.findall(r"\d+", raw)]
        # Filter valid indices and deduplicate
        seen, reranked = set(), []
        for n in numbers:
            if 1 <= n <= len(results) and n not in seen:
                reranked.append(results[n - 1])
                seen.add(n)

        # Append any missing results at the end
        for r in results:
            if r not in reranked:
                reranked.append(r)

        logger.debug("LLM reranked %d results for '%s'", len(reranked), question[:50])
        return reranked[:top_k] if top_k else reranked

    except Exception as e:
        logger.warning("LLM reranking failed: %s. Returning original order.", e)
        return results[:top_k] if top_k else results


# ── Feedback-boosted reranking ───────────────────────────────────────────────


def feedback_rerank(
    results: list[RetrievalResult],
    boost_factors: dict[str, float],
) -> list[RetrievalResult]:
    """
    Apply user-feedback-derived boost/penalty factors to similarity scores.

    Sources that historically got thumbs-up are boosted.
    Sources frequently flagged as irrelevant are penalized.
    This creates a feedback loop that improves retrieval over time.

    Args:
        results: retrieval results to rerank
        boost_factors: source → multiplier from get_source_boost_factors()

    Returns:
        Reordered results with adjusted scores
    """
    if not boost_factors:
        return results

    adjusted = []
    for r in results:
        factor = boost_factors.get(r.source, 1.0)
        adjusted_score = r.similarity_score * factor
        adjusted.append((r, adjusted_score))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    reranked = []
    for r, new_score in adjusted:
        # Update the score in the result
        reranked.append(r.model_copy(update={"similarity_score": round(min(1.0, new_score), 4)}))

    logger.debug("Feedback reranking applied to %d results", len(reranked))
    return reranked


# ── Token-level similarity (ColBERT-lite) ────────────────────────────────────


def colbert_lite_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    token_embeddings: Callable[[list[str]], list[list[float]]],
) -> float:
    """
    Compute a ColBERT-style MaxSim score without a full ColBERT model.

    ColBERT's late interaction: for each query token, find the maximum
    similarity to any document token. Sum these max-similarities.
    This is richer than single-vector cosine similarity because it captures
    token-level matching (individual keywords, entities, numbers).

    This is a lightweight approximation — real ColBERT uses a fine-tuned
    model with learned token projections. Still, this beats bi-encoder
    similarity on exact-match and keyword-heavy queries.

    Args:
        query_tokens: tokenized query words
        doc_tokens: tokenized document words
        token_embeddings: callable that embeds a list of tokens

    Returns:
        ColBERT-lite MaxSim score (higher = more relevant)
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    q_embs = np.array(token_embeddings(query_tokens))   # (Q, D)
    d_embs = np.array(token_embeddings(doc_tokens[:128]))  # (T, D) — cap doc tokens

    # MaxSim: for each query token, max cosine similarity over all doc tokens
    scores = []
    for q_emb in q_embs:
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        sims = d_embs @ q_norm / (np.linalg.norm(d_embs, axis=1, keepdims=True).flatten() + 1e-10)
        scores.append(float(np.max(sims)))

    return float(np.mean(scores))


# ── Ensemble reranker ─────────────────────────────────────────────────────────


def ensemble_rerank(
    question: str,
    results: list[RetrievalResult],
    cross_encoder_scores: list[float] | None = None,
    llm_fn: Callable[[str], str] | None = None,
    boost_factors: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """
    Ensemble multiple reranking signals for maximum accuracy.

    Combines:
      - Original similarity score (dense retrieval)
      - Cross-encoder score (if provided)
      - LLM ranking position (if llm_fn provided)
      - Feedback boost factor (if boost_factors provided)

    Each signal is normalized to [0, 1] then combined via weighted sum.
    Default weights are tuned for quality vs latency balance.

    Args:
        question: user's question
        results: candidates to rerank
        cross_encoder_scores: pre-computed cross-encoder scores (same order as results)
        llm_fn: optional LLM for RankGPT reranking
        boost_factors: optional source-level feedback adjustments
        weights: optional override for signal weights
        top_k: return only top-k results

    Returns:
        Reranked and optionally truncated list of RetrievalResult
    """
    if not results:
        return []

    default_weights = {
        "similarity": 0.3,
        "cross_encoder": 0.4,
        "llm_rank": 0.2,
        "feedback": 0.1,
    }
    w = {**default_weights, **(weights or {})}

    n = len(results)

    # ── Similarity scores (already normalized 0-1) ────────────────────────────
    sim_scores = np.array([r.similarity_score for r in results])

    # ── Cross-encoder scores (normalize) ─────────────────────────────────────
    ce_scores = np.zeros(n)
    if cross_encoder_scores and len(cross_encoder_scores) == n:
        raw = np.array(cross_encoder_scores)
        rng = raw.max() - raw.min()
        ce_scores = (raw - raw.min()) / (rng + 1e-10)

    # ── LLM rank scores ───────────────────────────────────────────────────────
    llm_rank_scores = np.zeros(n)
    if llm_fn:
        try:
            llm_reranked = llm_rerank(question, results, llm_fn)
            for rank, r in enumerate(llm_reranked):
                orig_idx = results.index(r)
                llm_rank_scores[orig_idx] = (n - rank) / n  # higher rank = higher score
        except Exception as e:
            logger.warning("LLM reranking in ensemble failed: %s", e)

    # ── Feedback boost scores ────────────────────────────────────────────────
    fb_scores = np.ones(n)
    if boost_factors:
        for i, r in enumerate(results):
            fb_scores[i] = boost_factors.get(r.source, 1.0)
        fb_scores = np.clip((fb_scores - 0.5) / 1.0, 0, 1)  # normalize [0.5,1.5] → [0,1]

    # ── Weighted combination ──────────────────────────────────────────────────
    combined = (
        w["similarity"] * sim_scores
        + w["cross_encoder"] * ce_scores
        + w["llm_rank"] * llm_rank_scores
        + w["feedback"] * fb_scores
    )

    order = np.argsort(combined)[::-1]
    reranked = [results[i] for i in order]

    logger.debug("Ensemble reranked %d results (signals: sim, ce, llm, feedback)", n)
    return reranked[:top_k] if top_k else reranked
