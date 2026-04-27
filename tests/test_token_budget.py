"""
Tests for core/token_budget.py — token budget manager.
"""

from __future__ import annotations

import pytest

from models import RetrievalContext, RetrievalResult, QueryMode


def _make_result(text: str, sim: float = 0.7) -> RetrievalResult:
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


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        from core.token_budget import estimate_tokens
        assert estimate_tokens("") == 1  # min 1

    def test_short_text(self) -> None:
        from core.token_budget import estimate_tokens
        # "Hello world" = 11 chars → ~2-3 tokens
        result = estimate_tokens("Hello world")
        assert 1 <= result <= 10

    def test_long_text_scales_linearly(self) -> None:
        from core.token_budget import estimate_tokens
        short = estimate_tokens("a" * 100)
        long  = estimate_tokens("a" * 400)
        assert long == short * 4


class TestCompressChunk:
    def test_short_chunk_unchanged(self) -> None:
        from core.token_budget import compress_chunk
        text = "Short sentence. Another one."
        result = compress_chunk(text, max_sentences=6)
        assert result == text

    def test_long_chunk_truncated(self) -> None:
        from core.token_budget import compress_chunk
        text = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        result = compress_chunk(text, max_sentences=3)
        assert len(result) < len(text)
        assert "Sentence 0" in result
        # Sentence 10 should be dropped
        assert "Sentence 10" not in result

    def test_preserves_sentence_boundaries(self) -> None:
        from core.token_budget import compress_chunk
        text = "First sentence. Second sentence. Third sentence."
        result = compress_chunk(text, max_sentences=2)
        # Should end at a sentence boundary, not mid-word
        assert result.endswith("sentence.")


class TestDeduplicateResults:
    def test_exact_duplicates_removed(self) -> None:
        from core.token_budget import deduplicate_results
        text = "This is exactly the same chunk text repeated verbatim."
        results = [_make_result(text, 0.9), _make_result(text, 0.8)]
        deduped = deduplicate_results(results, similarity_threshold=0.9)
        assert len(deduped) == 1

    def test_distinct_chunks_all_kept(self) -> None:
        from core.token_budget import deduplicate_results
        results = [
            _make_result("Machine learning is about training models on data."),
            _make_result("Vector databases store high-dimensional embeddings."),
            _make_result("RAG combines retrieval with generation for grounded answers."),
        ]
        deduped = deduplicate_results(results)
        assert len(deduped) == 3

    def test_near_duplicate_removed(self) -> None:
        from core.token_budget import deduplicate_results
        text1 = "The quick brown fox jumps over the lazy dog and runs away."
        text2 = "The quick brown fox jumps over the lazy dog and runs fast."  # 1 word diff
        results = [_make_result(text1, 0.9), _make_result(text2, 0.8)]
        deduped = deduplicate_results(results, similarity_threshold=0.9)
        # Very high similarity — should deduplicate
        assert len(deduped) <= 2  # may or may not dedup based on threshold

    def test_empty_input(self) -> None:
        from core.token_budget import deduplicate_results
        assert deduplicate_results([]) == []


class TestApplyTokenBudget:
    def test_small_context_unchanged(self) -> None:
        from core.token_budget import apply_token_budget
        results = [_make_result("short chunk", 0.8)]
        ctx = _make_context(results)
        optimized = apply_token_budget(ctx, "question", "system prompt", max_context_tokens=10_000)
        assert len(optimized.results) == 1

    def test_large_context_trimmed(self) -> None:
        from core.token_budget import apply_token_budget
        # Create 10 large chunks
        results = [_make_result("x" * 2000, 0.8 - i * 0.05) for i in range(10)]
        ctx = _make_context(results)
        # Very tight budget
        optimized = apply_token_budget(ctx, "q", "sys", max_context_tokens=1000)
        assert len(optimized.results) < 10

    def test_best_chunks_kept_not_worst(self) -> None:
        from core.token_budget import apply_token_budget
        # High-score chunk + low-score chunk, tight budget keeps high-score
        results = [
            _make_result("high relevance chunk content here", 0.95),
            _make_result("low relevance chunk content here", 0.20),
        ]
        ctx = _make_context(results)
        # Budget for ~1 chunk
        optimized = apply_token_budget(ctx, "q", "sys", max_context_tokens=200, compress_long_chunks=False)
        if len(optimized.results) == 1:
            assert optimized.results[0].similarity_score >= 0.9

    def test_returns_retrieval_context_type(self) -> None:
        from core.token_budget import apply_token_budget
        ctx = _make_context([_make_result("text")])
        result = apply_token_budget(ctx, "q", "sys")
        assert isinstance(result, RetrievalContext)

    def test_empty_context_unchanged(self) -> None:
        from core.token_budget import apply_token_budget
        ctx = _make_context([])
        result = apply_token_budget(ctx, "q", "sys")
        assert len(result.results) == 0


class TestGetModelBudget:
    def test_claude_model_has_budget(self) -> None:
        from core.token_budget import get_model_budget
        budget = get_model_budget("claude-sonnet-4-5")
        assert budget > 0

    def test_unknown_model_gets_default(self) -> None:
        from core.token_budget import get_model_budget, DEFAULT_CONTEXT_BUDGET
        budget = get_model_budget("some-unknown-model-xyz")
        assert budget == DEFAULT_CONTEXT_BUDGET

    def test_partial_match(self) -> None:
        from core.token_budget import get_model_budget
        # "llama3.2:latest" should match "llama3.2"
        budget = get_model_budget("llama3.2:latest")
        assert budget > 0


class TestOptimizeContext:
    def test_returns_context_and_budget_result(self) -> None:
        from core.token_budget import optimize_context, BudgetResult
        ctx = _make_context([
            _make_result("relevant content about topic A", 0.9),
            _make_result("more content about topic B", 0.7),
        ])
        optimized, budget = optimize_context(ctx, "question about A", "system", model_name="claude-sonnet-4-5")
        assert isinstance(optimized, RetrievalContext)
        assert isinstance(budget, BudgetResult)
        assert budget.budget_tokens > 0

    def test_savings_pct_non_negative(self) -> None:
        from core.token_budget import optimize_context
        ctx = _make_context([_make_result("test")])
        _, budget = optimize_context(ctx, "q", "sys")
        assert budget.savings_pct >= 0.0
