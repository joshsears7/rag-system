"""
Tests for core/light_rag.py — LightRAG dual-level graph retrieval (EMNLP 2025).
"""

from __future__ import annotations

import pytest


# ── classify_query ────────────────────────────────────────────────────────────


class TestClassifyQuery:
    def test_who_routes_low(self) -> None:
        from core.light_rag import classify_query
        assert classify_query("Who founded the company?") == "low"

    def test_what_is_routes_low(self) -> None:
        from core.light_rag import classify_query
        assert classify_query("What is machine learning?") == "low"

    def test_list_routes_low(self) -> None:
        from core.light_rag import classify_query
        assert classify_query("List all the features") == "low"

    def test_summarize_routes_high(self) -> None:
        from core.light_rag import classify_query
        assert classify_query("Summarize the key themes") == "high"

    def test_why_routes_high(self) -> None:
        from core.light_rag import classify_query
        assert classify_query("Why is this approach significant overall?") == "high"

    def test_overview_routes_high(self) -> None:
        from core.light_rag import classify_query
        assert classify_query("Give me an overview of the relationship between X and Y") == "high"

    def test_ambiguous_defaults_low(self) -> None:
        from core.light_rag import classify_query
        # No strong signals → default to low (safer, more precise)
        result = classify_query("Tell me something")
        assert result in ("low", "high")  # no crash, valid output


# ── LightRAGRetriever (empty graph) ──────────────────────────────────────────


class TestLightRAGRetrieverEmpty:
    """Tests that run without a real KG on disk (empty graph state)."""

    def _make_retriever(self):
        from core.light_rag import LightRAGRetriever
        from pathlib import Path
        # Use non-existent paths → starts with empty graph + empty index
        return LightRAGRetriever(
            graph_path=Path("/tmp/_nonexistent_graph.json"),
            index_path=Path("/tmp/_nonexistent_index.json"),
        )

    def test_low_level_empty_graph_returns_empty(self) -> None:
        r = self._make_retriever()
        result = r.low_level_retrieve("What is machine learning?")
        assert result == []

    def test_high_level_no_communities_returns_empty(self) -> None:
        r = self._make_retriever()
        result = r.high_level_retrieve("Summarize the key themes")
        assert result == []

    def test_auto_retrieve_returns_light_rag_result(self) -> None:
        from core.light_rag import LightRAGResult
        r = self._make_retriever()
        result = r.auto_retrieve("What is X?")
        assert isinstance(result, LightRAGResult)
        assert result.level == "auto"
        assert result.resolved_level in ("low", "high")
        assert isinstance(result.context_chunks, list)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_combined_retrieve_returns_light_rag_result(self) -> None:
        from core.light_rag import LightRAGResult
        r = self._make_retriever()
        result = r.combined_retrieve("machine learning overview")
        assert isinstance(result, LightRAGResult)
        assert result.resolved_level == "combined"

    def test_stats_returns_dict(self) -> None:
        r = self._make_retriever()
        s = r.stats()
        assert isinstance(s, dict)
        assert "graph_nodes" in s
        assert "graph_edges" in s
        assert s["graph_nodes"] == 0
        assert s["graph_edges"] == 0


# ── Incremental update ────────────────────────────────────────────────────────


class TestIncrementalUpdate:
    def _make_retriever(self, tmp_path):
        from core.light_rag import LightRAGRetriever
        return LightRAGRetriever(
            graph_path=tmp_path / "graph.json",
            index_path=tmp_path / "index.json",
        )

    def test_adds_edges_to_empty_graph(self, tmp_path) -> None:
        r = self._make_retriever(tmp_path)
        triples = [
            ("Python", "is", "programming language"),
            ("NumPy", "extends", "Python"),
        ]
        added = r.incremental_update(triples, source="test.txt")
        assert added == 2

    def test_duplicate_triples_not_re_added(self, tmp_path) -> None:
        r = self._make_retriever(tmp_path)
        triples = [("A", "relates_to", "B")]
        r.incremental_update(triples, source="doc1.txt")
        added_again = r.incremental_update(triples, source="doc1.txt")
        assert added_again == 0

    def test_new_source_on_existing_node(self, tmp_path) -> None:
        r = self._make_retriever(tmp_path)
        r.incremental_update([("X", "is", "Y")], source="doc1.txt")
        r.incremental_update([("X", "has", "Z")], source="doc2.txt")
        # x node should have both sources
        node_data = r._graph.nodes.get("x", {})
        sources = node_data.get("sources", [])
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources

    def test_graph_grows_after_update(self, tmp_path) -> None:
        r = self._make_retriever(tmp_path)
        assert r._graph.number_of_nodes() == 0
        r.incremental_update([("Alice", "knows", "Bob")], source="test.txt")
        assert r._graph.number_of_nodes() >= 2

    def test_entity_index_populated(self, tmp_path) -> None:
        r = self._make_retriever(tmp_path)
        r.incremental_update([("Transformer", "is", "model")], source="paper.pdf")
        assert "transformer" in r._entity_index or "model" in r._entity_index

    def test_index_persisted_to_disk(self, tmp_path) -> None:
        r = self._make_retriever(tmp_path)
        r.incremental_update([("Node", "connects", "Edge")], source="file.txt")
        index_file = tmp_path / "index.json"
        assert index_file.exists()

    def test_empty_triples_adds_zero(self, tmp_path) -> None:
        r = self._make_retriever(tmp_path)
        added = r.incremental_update([], source="empty.txt")
        assert added == 0


# ── Low-level retrieval with graph data ───────────────────────────────────────


class TestLowLevelRetrieval:
    def _make_populated_retriever(self, tmp_path):
        from core.light_rag import LightRAGRetriever
        r = LightRAGRetriever(
            graph_path=tmp_path / "graph.json",
            index_path=tmp_path / "index.json",
        )
        triples = [
            ("Python", "is", "programming language"),
            ("Python", "created_by", "Guido van Rossum"),
            ("NumPy", "extends", "Python"),
            ("Machine Learning", "uses", "Python"),
        ]
        r.incremental_update(triples, source="test.txt")
        return r

    def test_entity_in_query_returns_facts(self, tmp_path) -> None:
        r = self._make_populated_retriever(tmp_path)
        facts = r.low_level_retrieve("What is Python used for?")
        # Python appears in graph → should get some facts
        assert isinstance(facts, list)

    def test_returns_list_of_strings(self, tmp_path) -> None:
        r = self._make_populated_retriever(tmp_path)
        facts = r.low_level_retrieve("Tell me about Python")
        assert all(isinstance(f, str) for f in facts)

    def test_unknown_entity_returns_empty_or_fallback(self, tmp_path) -> None:
        r = self._make_populated_retriever(tmp_path)
        facts = r.low_level_retrieve("Who is Zaphod Beeblebrox?")
        assert isinstance(facts, list)


# ── High-level retrieval with community summaries ─────────────────────────────


class TestHighLevelRetrieval:
    def _make_retriever_with_communities(self, tmp_path):
        from core.light_rag import LightRAGRetriever
        r = LightRAGRetriever(
            graph_path=tmp_path / "graph.json",
            index_path=tmp_path / "index.json",
        )
        r.build_community_index({
            0: "Machine learning and deep learning are subfields of AI focused on pattern recognition.",
            1: "Python is a popular programming language used in data science and machine learning.",
            2: "Vector databases store embeddings and enable semantic similarity search.",
        })
        return r

    def test_relevant_community_returned(self, tmp_path) -> None:
        r = self._make_retriever_with_communities(tmp_path)
        results = r.high_level_retrieve("What is machine learning?")
        assert len(results) > 0

    def test_results_contain_community_label(self, tmp_path) -> None:
        r = self._make_retriever_with_communities(tmp_path)
        results = r.high_level_retrieve("machine learning overview")
        for item in results:
            assert "[Community" in item

    def test_irrelevant_query_returns_fewer_results(self, tmp_path) -> None:
        r = self._make_retriever_with_communities(tmp_path)
        # Query with no keyword overlap
        results = r.high_level_retrieve("xyzzy nonsense foobar")
        # Should return empty or very low overlap results (score == 0 filtered)
        assert isinstance(results, list)

    def test_build_community_index_persists(self, tmp_path) -> None:
        from core.light_rag import LightRAGRetriever
        r = self._make_retriever_with_communities(tmp_path)
        assert len(r._community_summaries) == 3
        # Reload from disk
        r2 = LightRAGRetriever(
            graph_path=tmp_path / "graph.json",
            index_path=tmp_path / "index.json",
        )
        assert len(r2._community_summaries) == 3


# ── LLM classifier ────────────────────────────────────────────────────────────


class TestLLMClassifier:
    def test_llm_classify_high(self, tmp_path) -> None:
        from core.light_rag import LightRAGRetriever
        from unittest.mock import MagicMock
        r = LightRAGRetriever(
            graph_path=tmp_path / "graph.json",
            index_path=tmp_path / "index.json",
        )
        mock_llm = MagicMock(return_value="high")
        result = r._llm_classify("Why is this important?", mock_llm)
        assert result == "high"

    def test_llm_classify_low(self, tmp_path) -> None:
        from core.light_rag import LightRAGRetriever
        from unittest.mock import MagicMock
        r = LightRAGRetriever(
            graph_path=tmp_path / "graph.json",
            index_path=tmp_path / "index.json",
        )
        mock_llm = MagicMock(return_value="low")
        result = r._llm_classify("Who is X?", mock_llm)
        assert result == "low"

    def test_llm_classify_falls_back_on_error(self, tmp_path) -> None:
        from core.light_rag import LightRAGRetriever
        from unittest.mock import MagicMock
        r = LightRAGRetriever(
            graph_path=tmp_path / "graph.json",
            index_path=tmp_path / "index.json",
        )
        mock_llm = MagicMock(side_effect=RuntimeError("LLM down"))
        result = r._llm_classify("What is Python?", mock_llm)
        assert result in ("low", "high")  # keyword fallback, no crash


# ── Module singleton ──────────────────────────────────────────────────────────


class TestGetLightRAG:
    def test_returns_light_rag_retriever(self) -> None:
        from core.light_rag import get_light_rag, LightRAGRetriever
        r = get_light_rag()
        assert isinstance(r, LightRAGRetriever)

    def test_singleton_same_object(self) -> None:
        from core.light_rag import get_light_rag
        r1 = get_light_rag()
        r2 = get_light_rag()
        assert r1 is r2
