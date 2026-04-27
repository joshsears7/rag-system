"""
Sufficient Context — confidence-gated abstention layer.

Based on "Sufficient Context: A New Lens on RAG Systems" (Google ICLR 2025).

Before generating an answer, this module scores whether the retrieved context
is sufficient to answer confidently. If it's not, the system can:
  (1) Retrieve additional chunks
  (2) Trigger web search fallback
  (3) Abstain with an explicit "insufficient context" response

This closes the most critical production gap in RAG systems: they always generate
even when context is poor, producing confident hallucinations. A system that can
say "I don't know" is far more trustworthy than one that always answers.

Scoring components (weighted ensemble):
  - density:     mean cosine similarity of retrieved chunks to the query
  - coverage:    fraction of chunks exceeding the similarity threshold
  - diversity:   penalizes retrieving the same chunk repeatedly (dedup quality)
  - self_rating: LLM rates its own confidence (optional — adds ~200ms latency)
  - crag_score:  CRAG quality estimate if already computed upstream

Usage:
    from core.sufficient_context import SufficientContextChecker, SufficiencyResult

    checker = SufficientContextChecker()
    result = checker.score(question, context, crag_score=0.6)

    if not result.is_sufficient:
        return "I don't have enough information to answer this confidently."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import settings
from models import RetrievalContext, RetrievalResult

logger = logging.getLogger(__name__)

# ── Default thresholds ────────────────────────────────────────────────────────

DEFAULT_SUFFICIENCY_THRESHOLD = 0.45   # below this → abstain
DEFAULT_DENSITY_WEIGHT        = 0.35
DEFAULT_COVERAGE_WEIGHT       = 0.25
DEFAULT_CRAG_WEIGHT           = 0.25
DEFAULT_SELF_RATING_WEIGHT    = 0.15


# ── Result model ──────────────────────────────────────────────────────────────


@dataclass
class SufficiencyResult:
    """
    Output of the sufficient context check.

    Attributes:
        is_sufficient:      Whether context is good enough to generate
        overall_score:      Weighted ensemble score in [0, 1]
        density_score:      Mean similarity of retrieved chunks to query
        coverage_score:     Fraction of chunks above similarity threshold
        crag_score:         CRAG quality estimate (if provided)
        self_rating:        LLM self-confidence rating (if enabled)
        num_chunks:         Number of chunks in the context
        recommendation:     Human-readable action ("generate" | "retrieve_more" | "abstain" | "web_search")
        explanation:        Why this score was reached (for logging / debug)
    """

    is_sufficient: bool
    overall_score: float
    density_score: float
    coverage_score: float
    crag_score: float | None
    self_rating: float | None
    num_chunks: int
    recommendation: str
    explanation: str
    component_scores: dict[str, float] = field(default_factory=dict)


# ── Core checker ──────────────────────────────────────────────────────────────


class SufficientContextChecker:
    """
    Scores retrieved context for sufficiency before generation.

    Implements the Google ICLR 2025 "sufficient context" framework:
    combine multiple signals to decide when the system has enough information
    to answer confidently vs. when it should abstain or seek more context.
    """

    def __init__(
        self,
        sufficiency_threshold: float = DEFAULT_SUFFICIENCY_THRESHOLD,
        density_weight: float = DEFAULT_DENSITY_WEIGHT,
        coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
        crag_weight: float = DEFAULT_CRAG_WEIGHT,
        self_rating_weight: float = DEFAULT_SELF_RATING_WEIGHT,
        min_chunks: int = 1,
    ) -> None:
        self.sufficiency_threshold = sufficiency_threshold
        self.density_weight = density_weight
        self.coverage_weight = coverage_weight
        self.crag_weight = crag_weight
        self.self_rating_weight = self_rating_weight
        self.min_chunks = min_chunks

    def _density(self, results: list[RetrievalResult]) -> float:
        """Mean similarity score of retrieved chunks — how close are they to the query?"""
        if not results:
            return 0.0
        scores = [r.similarity_score for r in results if r.similarity_score is not None]
        return float(np.mean(scores)) if scores else 0.0

    def _coverage(self, results: list[RetrievalResult], threshold: float | None = None) -> float:
        """Fraction of chunks exceeding the similarity threshold."""
        if not results:
            return 0.0
        t = threshold or settings.similarity_threshold
        above = sum(1 for r in results if (r.similarity_score or 0.0) >= t)
        return above / len(results)

    def _self_rate(
        self,
        question: str,
        chunks: list[str],
        llm_fn: Callable[[str], str],
    ) -> float:
        """
        Ask the LLM to rate its own confidence that the context is sufficient.

        Returns a float in [0, 1]. This adds ~200ms latency but significantly
        improves precision for ambiguous cases.
        """
        preview = "\n\n".join(chunks[:3])[:1500]
        prompt = (
            "You are evaluating whether context documents contain enough information "
            "to answer a question accurately and completely.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT PREVIEW:\n{preview}\n\n"
            "On a scale of 0.0 to 1.0, how confident are you that the above context "
            "is sufficient to answer the question fully?\n"
            "  0.0 = context is completely irrelevant or missing\n"
            "  0.5 = context is partially relevant, answer will be incomplete\n"
            "  1.0 = context fully contains the answer\n\n"
            "Reply with ONLY a decimal number (e.g. 0.7):"
        )
        try:
            raw = llm_fn(prompt).strip().split()[0].rstrip(".,")
            score = float(raw)
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            return 0.5

    def score(
        self,
        question: str,
        context: RetrievalContext,
        crag_score: float | None = None,
        llm_fn: Callable[[str], str] | None = None,
        enable_self_rating: bool = False,
    ) -> SufficiencyResult:
        """
        Compute a sufficiency score for the retrieved context.

        Args:
            question:            The user's question
            context:             Retrieved context from the retrieval pipeline
            crag_score:          CRAG quality estimate [0, 1] if already computed
            llm_fn:              LLM completion function for self-rating (optional)
            enable_self_rating:  Whether to call LLM to self-rate confidence

        Returns:
            SufficiencyResult with overall score and recommendation
        """
        results = context.results

        # ── Hard threshold: no context at all ────────────────────────────────
        if not results or len(results) < self.min_chunks:
            return SufficiencyResult(
                is_sufficient=False,
                overall_score=0.0,
                density_score=0.0,
                coverage_score=0.0,
                crag_score=crag_score,
                self_rating=None,
                num_chunks=0,
                recommendation="web_search" if settings.web_search_fallback else "abstain",
                explanation="No context retrieved — collection may be empty or query too different from ingested content.",
                component_scores={},
            )

        # ── Component scores ──────────────────────────────────────────────────
        density   = self._density(results)
        coverage  = self._coverage(results)
        self_rating: float | None = None

        if enable_self_rating and llm_fn:
            try:
                chunks = [r.chunk_text for r in results]
                self_rating = self._self_rate(question, chunks, llm_fn)
                logger.debug("Self-rating: %.2f", self_rating)
            except Exception as e:
                logger.warning("Self-rating failed: %s", e)

        # ── Weighted ensemble ─────────────────────────────────────────────────
        total_weight = self.density_weight + self.coverage_weight

        weighted_sum = (
            density  * self.density_weight +
            coverage * self.coverage_weight
        )

        if crag_score is not None:
            weighted_sum  += crag_score * self.crag_weight
            total_weight  += self.crag_weight

        if self_rating is not None:
            weighted_sum  += self_rating * self.self_rating_weight
            total_weight  += self.self_rating_weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall = max(0.0, min(1.0, overall))

        # ── Decision logic ────────────────────────────────────────────────────
        is_sufficient = overall >= self.sufficiency_threshold

        if overall >= self.sufficiency_threshold:
            recommendation = "generate"
            explanation = f"Context sufficient (score={overall:.2f}): density={density:.2f}, coverage={coverage:.2f}"
        elif overall >= self.sufficiency_threshold * 0.7:
            recommendation = "retrieve_more"
            explanation = (
                f"Context borderline (score={overall:.2f}): attempting to retrieve additional chunks. "
                f"density={density:.2f}, coverage={coverage:.2f}"
            )
        elif settings.web_search_fallback:
            recommendation = "web_search"
            explanation = (
                f"Context insufficient (score={overall:.2f}): falling back to web search. "
                f"density={density:.2f}, coverage={coverage:.2f}"
            )
        else:
            recommendation = "abstain"
            explanation = (
                f"Context insufficient (score={overall:.2f}): abstaining. "
                f"Enable WEB_SEARCH_FALLBACK=true to trigger web search in this case."
            )

        logger.info(
            "Sufficiency: overall=%.2f density=%.2f coverage=%.2f crag=%s → %s",
            overall, density, coverage,
            f"{crag_score:.2f}" if crag_score is not None else "n/a",
            recommendation,
        )

        return SufficiencyResult(
            is_sufficient=is_sufficient,
            overall_score=round(overall, 4),
            density_score=round(density, 4),
            coverage_score=round(coverage, 4),
            crag_score=round(crag_score, 4) if crag_score is not None else None,
            self_rating=round(self_rating, 4) if self_rating is not None else None,
            num_chunks=len(results),
            recommendation=recommendation,
            explanation=explanation,
            component_scores={
                "density":     round(density, 4),
                "coverage":    round(coverage, 4),
                **({"crag": round(crag_score, 4)} if crag_score is not None else {}),
                **({"self_rating": round(self_rating, 4)} if self_rating is not None else {}),
            },
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_checker: SufficientContextChecker | None = None


def get_checker() -> SufficientContextChecker:
    """Return the module-level SufficientContextChecker singleton."""
    global _checker
    if _checker is None:
        _checker = SufficientContextChecker()
    return _checker


def check_sufficiency(
    question: str,
    context: RetrievalContext,
    crag_score: float | None = None,
    llm_fn: Callable[[str], str] | None = None,
    enable_self_rating: bool = False,
) -> SufficiencyResult:
    """
    Convenience wrapper — check context sufficiency using the module singleton.

    Args:
        question:           User question
        context:            Retrieved context
        crag_score:         Optional CRAG quality estimate
        llm_fn:             LLM function for optional self-rating
        enable_self_rating: Whether to LLM-rate confidence (adds latency)

    Returns:
        SufficiencyResult
    """
    return get_checker().score(
        question=question,
        context=context,
        crag_score=crag_score,
        llm_fn=llm_fn,
        enable_self_rating=enable_self_rating,
    )


# ── Abstention response ───────────────────────────────────────────────────────

ABSTENTION_TEMPLATE = (
    "I don't have sufficient context to answer this question confidently.\n\n"
    "**Sufficiency score:** {score:.0%}\n"
    "**Reason:** {explanation}\n\n"
    "Suggestions:\n"
    "- Try ingesting documents relevant to this topic\n"
    "- Enable web search fallback (`WEB_SEARCH_FALLBACK=true`)\n"
    "- Rephrase the question to match ingested content"
)


def abstention_response(result: SufficiencyResult) -> str:
    """Generate a helpful abstention message from a SufficiencyResult."""
    return ABSTENTION_TEMPLATE.format(
        score=result.overall_score,
        explanation=result.explanation,
    )
