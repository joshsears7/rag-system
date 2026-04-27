"""
Tests for core/sufficient_context.py — Google ICLR 2025 abstention layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models import RetrievalContext, RetrievalResult, QueryMode


def _make_result(sim: float, text: str = "test chunk") -> RetrievalResult:
    return RetrievalResult(
        chunk_text=text,
        source="test.txt",
        similarity_score=sim,
        chunk_index=0,
    )


def _make_context(results: list[RetrievalResult]) -> RetrievalContext:
    return RetrievalContext(
        query="test question",
        results=results,
        query_mode=QueryMode.HYBRID,
    )


class TestSufficientContextChecker:
    def test_empty_context_is_not_sufficient(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker()
        ctx = _make_context([])
        result = checker.score("what is X?", ctx)
        assert result.is_sufficient is False
        assert result.overall_score == 0.0
        assert result.recommendation in ("abstain", "web_search")

    def test_high_similarity_context_is_sufficient(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker(sufficiency_threshold=0.4)
        ctx = _make_context([
            _make_result(0.85, "very relevant chunk about X"),
            _make_result(0.80, "another highly relevant chunk"),
            _make_result(0.75, "also relevant to the question"),
        ])
        result = checker.score("what is X?", ctx)
        assert result.is_sufficient is True
        assert result.overall_score > 0.4
        assert result.recommendation == "generate"

    def test_low_similarity_context_is_not_sufficient(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker(sufficiency_threshold=0.6)
        ctx = _make_context([_make_result(0.1)])
        result = checker.score("what is X?", ctx)
        assert result.is_sufficient is False

    def test_density_score_is_mean_similarity(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker()
        ctx = _make_context([_make_result(0.6), _make_result(0.8)])
        result = checker.score("test", ctx)
        assert abs(result.density_score - 0.7) < 0.01

    def test_crag_score_included_in_ensemble(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker(sufficiency_threshold=0.3)
        ctx = _make_context([_make_result(0.1)])  # low density

        # Without CRAG score
        without_crag = checker.score("test", ctx)

        # With high CRAG score — should raise overall
        with_crag = checker.score("test", ctx, crag_score=0.9)
        assert with_crag.overall_score > without_crag.overall_score

    def test_self_rating_called_when_enabled(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker()
        ctx = _make_context([_make_result(0.7, "relevant content")])

        mock_llm = MagicMock(return_value="0.8")
        result = checker.score("test", ctx, llm_fn=mock_llm, enable_self_rating=True)
        assert result.self_rating is not None
        assert abs(result.self_rating - 0.8) < 0.01
        mock_llm.assert_called_once()

    def test_self_rating_not_called_when_disabled(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker()
        ctx = _make_context([_make_result(0.7)])

        mock_llm = MagicMock(return_value="0.8")
        result = checker.score("test", ctx, llm_fn=mock_llm, enable_self_rating=False)
        assert result.self_rating is None
        mock_llm.assert_not_called()

    def test_recommendation_abstain_when_very_low(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        from config import settings
        # Save original
        original = settings.web_search_fallback
        settings.web_search_fallback = False
        try:
            checker = SufficientContextChecker(sufficiency_threshold=0.9)
            ctx = _make_context([_make_result(0.1)])
            result = checker.score("test", ctx)
            assert result.recommendation == "abstain"
        finally:
            settings.web_search_fallback = original

    def test_component_scores_present(self) -> None:
        from core.sufficient_context import SufficientContextChecker
        checker = SufficientContextChecker()
        ctx = _make_context([_make_result(0.7)])
        result = checker.score("test", ctx)
        assert "density" in result.component_scores
        assert "coverage" in result.component_scores


class TestCheckSufficiencyConvenience:
    def test_module_function_works(self) -> None:
        from core.sufficient_context import check_sufficiency
        ctx = _make_context([_make_result(0.7)])
        result = check_sufficiency("test question", ctx)
        assert isinstance(result.overall_score, float)
        assert 0.0 <= result.overall_score <= 1.0


class TestAbstentionResponse:
    def test_abstention_response_contains_score(self) -> None:
        from core.sufficient_context import abstention_response, SufficiencyResult
        result = SufficiencyResult(
            is_sufficient=False,
            overall_score=0.2,
            density_score=0.1,
            coverage_score=0.3,
            crag_score=None,
            self_rating=None,
            num_chunks=1,
            recommendation="abstain",
            explanation="Low context quality",
        )
        response = abstention_response(result)
        assert "20%" in response
        assert "Low context quality" in response
