"""
Tests for the retrieval pipeline.

Tests:
  - Results above similarity threshold
  - MMR reduces redundancy
  - RRF fusion correctly merges rankings
  - Semantic cache hits on similar queries
  - BM25 index queries
  - Multi-query expansion produces multiple queries
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models import QueryMode, QueryRequest, RetrievalResult


# ── RRF tests ─────────────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    def test_combines_two_rankings(self) -> None:
        from core.retrieval import reciprocal_rank_fusion

        ranking1 = [("doc_a", 0.9), ("doc_b", 0.8), ("doc_c", 0.5)]
        ranking2 = [("doc_b", 0.95), ("doc_a", 0.7), ("doc_d", 0.6)]

        fused = reciprocal_rank_fusion([ranking1, ranking2])
        doc_ids = [doc_id for doc_id, _ in fused]

        # Both doc_a and doc_b appear in both rankings, should rank high
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids

    def test_single_ranking_passthrough(self) -> None:
        from core.retrieval import reciprocal_rank_fusion

        ranking = [("doc_x", 0.9), ("doc_y", 0.5)]
        fused = reciprocal_rank_fusion([ranking])
        assert fused[0][0] == "doc_x"  # top result preserved

    def test_consistent_doc_in_multiple_lists_ranks_higher(self) -> None:
        from core.retrieval import reciprocal_rank_fusion

        # doc_shared appears in both; doc_unique only in one
        ranking1 = [("doc_unique", 0.99), ("doc_shared", 0.5)]
        ranking2 = [("doc_shared", 0.9), ("doc_other", 0.8)]

        fused = reciprocal_rank_fusion([ranking1, ranking2])
        scores = dict(fused)

        # doc_shared benefits from appearing in both lists
        assert scores["doc_shared"] > scores.get("doc_other", 0)

    def test_empty_rankings(self) -> None:
        from core.retrieval import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([])
        assert result == []


# ── MMR tests ─────────────────────────────────────────────────────────────────


class TestMMR:
    def _make_result(self, text: str, score: float) -> RetrievalResult:
        return RetrievalResult(
            chunk_text=text,
            source="test.txt",
            similarity_score=score,
            chunk_index=0,
        )

    def test_mmr_returns_top_k(self) -> None:
        from core.retrieval import apply_mmr

        results = [
            self._make_result(f"chunk text {i}", 0.9 - i * 0.05)
            for i in range(8)
        ]
        # Distinct embeddings
        embeddings = {r.chunk_text: np.random.rand(384).tolist() for r in results}
        query_emb = np.random.rand(384).tolist()

        selected = apply_mmr(query_emb, results, embeddings, top_k=4)
        assert len(selected) == 4

    def test_mmr_with_zero_lambda_prefers_diversity(self) -> None:
        """lambda=0 maximizes diversity — result set should be varied."""
        from core.retrieval import apply_mmr

        # Two identical embeddings (redundant) and two different ones
        identical_emb = np.ones(10).tolist()
        unique_emb1 = np.zeros(10).tolist()
        unique_emb2 = (np.arange(10) / 10).tolist()

        r1 = self._make_result("identical doc 1", 0.9)
        r2 = self._make_result("identical doc 2", 0.85)
        r3 = self._make_result("unique doc 3", 0.7)
        r4 = self._make_result("unique doc 4", 0.65)

        embeddings = {
            r1.chunk_text: identical_emb,
            r2.chunk_text: identical_emb,
            r3.chunk_text: unique_emb1,
            r4.chunk_text: unique_emb2,
        }

        selected = apply_mmr(np.ones(10).tolist(), [r1, r2, r3, r4], embeddings, top_k=2, lambda_mult=0.0)
        # With pure diversity, should not pick both identical docs
        selected_texts = [r.chunk_text for r in selected]
        assert not ("identical doc 1" in selected_texts and "identical doc 2" in selected_texts)

    def test_mmr_empty_results(self) -> None:
        from core.retrieval import apply_mmr

        result = apply_mmr([], [], {}, top_k=5)
        assert result == []


# ── BM25 tests ────────────────────────────────────────────────────────────────


class TestBM25Index:
    def test_bm25_keyword_match(self) -> None:
        from core.retrieval import BM25Index

        docs = [
            "Python programming language tutorial",
            "Machine learning with scikit-learn",
            "Python data science and pandas",
            "Deep learning neural networks",
        ]
        ids = ["doc1", "doc2", "doc3", "doc4"]
        index = BM25Index(docs, ids)

        results = index.query("Python programming", top_k=2)
        result_ids = [r[0] for r in results]

        # Python docs should rank first
        assert "doc1" in result_ids or "doc3" in result_ids

    def test_bm25_returns_normalized_scores(self) -> None:
        from core.retrieval import BM25Index

        docs = ["hello world", "foo bar baz", "hello foo world"]
        ids = ["a", "b", "c"]
        index = BM25Index(docs, ids)

        results = index.query("hello world", top_k=3)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_bm25_no_match_returns_empty(self) -> None:
        from core.retrieval import BM25Index

        docs = ["apples and oranges", "bananas and grapes"]
        ids = ["x", "y"]
        index = BM25Index(docs, ids)

        results = index.query("quantum physics", top_k=5)
        assert results == []


# ── Semantic cache tests ──────────────────────────────────────────────────────


def _make_query_response(question: str = "test?") -> "QueryResponse":
    from models import QueryResponse
    return QueryResponse(
        question=question,
        answer="test answer",
        sources=[],
        tokens_used=10,
        latency_ms=100.0,
        collection="default",
        llm_backend="ollama",
        model_used="llama3.2",
        cache_hit=False,
    )


class TestSemanticCache:
    def test_cache_hit_on_identical_query(self) -> None:
        from core.retrieval import SemanticCache

        cache = SemanticCache(max_size=10, threshold=0.95)

        q = "What is machine learning?"
        emb = np.random.rand(384).tolist()
        response = _make_query_response(q)
        cache.put(q, response, emb)

        hit = cache.get(q, emb)
        assert hit is not None
        assert hit.response.question == q

    def test_cache_miss_on_different_query(self) -> None:
        from core.retrieval import SemanticCache

        cache = SemanticCache(max_size=10, threshold=0.95)

        emb1 = np.ones(384).tolist()
        emb2 = (-np.ones(384)).tolist()  # orthogonal — cosine similarity is negative

        cache.put("first query", _make_query_response("first query"), emb1)

        hit = cache.get("completely different query", emb2)
        assert hit is None

    def test_cache_evicts_lru_at_capacity(self) -> None:
        from core.retrieval import SemanticCache

        cache = SemanticCache(max_size=3, threshold=0.95)

        for i in range(5):
            emb = np.random.rand(384).tolist()
            cache.put(f"query {i}", _make_query_response(f"query {i}"), emb)

        assert len(cache) <= 3

    def test_cache_hit_increments_count(self) -> None:
        from core.retrieval import SemanticCache

        cache = SemanticCache(max_size=10, threshold=0.9)
        emb = np.ones(384).tolist()
        cache.put("test query", _make_query_response("test query"), emb)

        cache.get("test query", emb)
        cache.get("test query", emb)

        assert cache._entries[0].hit_count == 2


# ── Query transformation tests ────────────────────────────────────────────────


class TestQueryTransformation:
    def test_multi_query_expansion(self) -> None:
        from core.retrieval import expand_query

        mock_llm = MagicMock(return_value="What is AI?\nHow does AI work?\nDefine artificial intelligence")
        queries = expand_query("Tell me about AI", generate_fn=mock_llm, n=3)

        assert len(queries) >= 2  # original + at least 1 expansion
        assert "Tell me about AI" in queries

    def test_hyde_generates_hypothesis(self) -> None:
        from core.retrieval import generate_hypothetical_document

        mock_llm = MagicMock(return_value="Machine learning is a subset of AI that enables systems to learn.")
        hypothesis = generate_hypothetical_document("What is machine learning?", generate_fn=mock_llm)

        assert isinstance(hypothesis, str)
        assert len(hypothesis) > 0
        mock_llm.assert_called_once()
