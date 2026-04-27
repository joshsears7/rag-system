"""
Tests for the generation layer.

Tests:
  - Prompt construction injects context and question correctly
  - Answer contains citation format [Source: ...]
  - Source extraction from retrieval context
  - Multi-backend configuration (mocked)
  - Empty context triggers "I don't have enough context" response
  - Token counting passthrough
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from models import QueryMode, QueryRequest, RetrievalContext, RetrievalResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_retrieval_context(n_results: int = 3, empty: bool = False) -> RetrievalContext:
    results = []
    if not empty:
        for i in range(n_results):
            results.append(RetrievalResult(
                chunk_text=f"This is chunk {i} with important information about the topic.",
                source=f"document_{i}.pdf",
                similarity_score=0.85 - i * 0.05,
                chunk_index=i,
                page_number=i + 1,
            ))
    return RetrievalContext(
        query="What is the main topic?",
        results=results,
        query_mode=QueryMode.HYBRID,
    )


# ── Prompt construction tests ─────────────────────────────────────────────────


class TestPromptConstruction:
    def test_user_prompt_contains_question(self) -> None:
        from core.generation import build_user_prompt

        context = make_retrieval_context(n_results=2)
        prompt = build_user_prompt(context)

        assert "What is the main topic?" in prompt

    def test_user_prompt_contains_source_labels(self) -> None:
        from core.generation import build_user_prompt

        context = make_retrieval_context(n_results=3)
        prompt = build_user_prompt(context)

        for result in context.results:
            assert result.source in prompt

    def test_user_prompt_contains_chunk_text(self) -> None:
        from core.generation import build_user_prompt

        context = make_retrieval_context(n_results=2)
        prompt = build_user_prompt(context)

        for result in context.results:
            assert result.chunk_text in prompt

    def test_empty_context_prompt_indicates_no_context(self) -> None:
        from core.generation import build_user_prompt

        context = make_retrieval_context(empty=True)
        prompt = build_user_prompt(context)

        assert "No relevant context" in prompt or "no" in prompt.lower()

    def test_source_citation_format_in_prompt(self) -> None:
        """Prompt must include 'Source:' labels so model can cite correctly."""
        from core.generation import build_user_prompt

        context = make_retrieval_context(n_results=1)
        prompt = build_user_prompt(context)

        assert "Source:" in prompt

    def test_system_prompt_contains_citation_instruction(self) -> None:
        from core.generation import SYSTEM_PROMPT

        assert "[Source:" in SYSTEM_PROMPT or "cite" in SYSTEM_PROMPT.lower()
        assert "hallucinate" in SYSTEM_PROMPT.lower() or "outside knowledge" in SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_fallback_instruction(self) -> None:
        from core.generation import SYSTEM_PROMPT

        assert "don't have enough context" in SYSTEM_PROMPT.lower() or "I don't have" in SYSTEM_PROMPT


# ── Source extraction tests ───────────────────────────────────────────────────


class TestSourceExtraction:
    def test_extracts_correct_number_of_sources(self) -> None:
        from core.generation import extract_sources

        context = make_retrieval_context(n_results=4)
        sources = extract_sources(context)

        assert len(sources) == 4

    def test_source_fields_populated(self) -> None:
        from core.generation import extract_sources

        context = make_retrieval_context(n_results=2)
        sources = extract_sources(context)

        for src in sources:
            assert src.source.startswith("document_")
            assert src.chunk_index >= 0
            assert 0.0 <= src.similarity_score <= 1.0
            assert len(src.excerpt) > 0

    def test_excerpt_truncated_to_200_chars(self) -> None:
        from core.generation import extract_sources

        long_text = "X" * 500
        context = RetrievalContext(
            query="test",
            results=[RetrievalResult(
                chunk_text=long_text,
                source="doc.txt",
                similarity_score=0.9,
                chunk_index=0,
            )],
            query_mode=QueryMode.DENSE,
        )
        sources = extract_sources(context)
        assert len(sources[0].excerpt) == 200

    def test_empty_context_produces_no_sources(self) -> None:
        from core.generation import extract_sources

        context = make_retrieval_context(empty=True)
        sources = extract_sources(context)
        assert sources == []


# ── Backend tests (mocked) ────────────────────────────────────────────────────


class TestBackends:
    def test_ollama_backend_formats_request(self) -> None:
        """OllamaBackend should call /api/chat with correct payload structure."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            mock_get.return_value = MagicMock(status_code=200)
            mock_get.return_value.raise_for_status = MagicMock()

            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "message": {"content": "The answer is 42."},
                    "eval_count": 20,
                    "prompt_eval_count": 100,
                }),
            )
            mock_post.return_value.raise_for_status = MagicMock()

            from core.generation import OllamaBackend
            backend = OllamaBackend()
            text, tokens, model = backend.complete("You are helpful.", "What is 6 * 7?")

            assert text == "The answer is 42."
            assert tokens == 120
            mock_post.assert_called_once()

    def test_claude_backend_requires_api_key(self) -> None:
        """ClaudeBackend.complete should raise when the API call fails."""
        try:
            import anthropic
            from core.generation import ClaudeBackend
        except ImportError:
            pytest.skip("anthropic package not installed")

        # Patch the client so it raises an APIError on any message call
        with patch("core.generation.settings") as mock_settings:
            mock_settings.anthropic_api_key = "sk-ant-test-fake-key"
            mock_settings.claude_model = "claude-sonnet-4-5"
            mock_settings.temperature = 0.2
            mock_settings.max_tokens = 1024

            with patch("anthropic.Anthropic") as mock_anthropic_cls:
                mock_client = MagicMock()
                mock_client.messages.create.side_effect = anthropic.AuthenticationError(
                    message="invalid api key",
                    response=MagicMock(status_code=401),
                    body={},
                )
                mock_anthropic_cls.return_value = mock_client

                backend = ClaudeBackend()
                with pytest.raises((RuntimeError, anthropic.AuthenticationError, anthropic.APIError)):
                    backend.complete("sys", "user")

    def test_complete_raw_passthrough(self) -> None:
        """complete_raw should call complete and return just the text."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            mock_get.return_value = MagicMock(status_code=200)
            mock_get.return_value.raise_for_status = MagicMock()
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "message": {"content": "Short answer."},
                    "eval_count": 5,
                    "prompt_eval_count": 50,
                }),
            )
            mock_post.return_value.raise_for_status = MagicMock()

            from core.generation import OllamaBackend
            backend = OllamaBackend()
            result = backend.complete_raw("Say hello.")
            assert result == "Short answer."


# ── Full answer_question (integration, mocked LLM) ───────────────────────────


class TestAnswerQuestion:
    @patch("core.generation.get_backend")
    @patch("core.generation.retrieve")
    def test_answer_question_returns_response(
        self,
        mock_retrieve: MagicMock,
        mock_get_backend: MagicMock,
    ) -> None:
        from core.generation import answer_question

        # Mock retrieval context
        mock_retrieve.return_value = make_retrieval_context(n_results=3)

        # Mock LLM backend
        mock_backend = MagicMock()
        mock_backend.complete.return_value = (
            "The answer is documented in [Source: document_0.pdf, chunk 0].",
            150,
            "llama3.2",
        )
        mock_backend.complete_raw.return_value = "0.8"
        mock_get_backend.return_value = mock_backend

        request = QueryRequest(
            question="What is the main topic?",
            collection="test",
            top_k=3,
            mode=QueryMode.DENSE,
        )

        # Disable cache for clean test
        with patch("core.generation.settings") as mock_settings:
            mock_settings.enable_cache = False
            mock_settings.llm_backend.value = "ollama"
            mock_settings.use_hybrid_search = False

            response = answer_question(request)

        assert response.answer != ""
        assert response.tokens_used == 150
        assert response.latency_ms >= 0

    @patch("core.generation.get_backend")
    @patch("core.generation.retrieve")
    def test_citation_format_in_answer(
        self,
        mock_retrieve: MagicMock,
        mock_get_backend: MagicMock,
    ) -> None:
        """Verify the model response contains citation format strings."""
        from core.generation import answer_question

        mock_retrieve.return_value = make_retrieval_context(n_results=2)

        mock_backend = MagicMock()
        mock_backend.complete.return_value = (
            "According to the docs, X is true [Source: document_0.pdf, chunk 0] and Y is also noted [Source: document_1.pdf, chunk 1].",
            200,
            "llama3.2",
        )
        mock_backend.complete_raw.return_value = "0.9"
        mock_get_backend.return_value = mock_backend

        request = QueryRequest(
            question="Tell me about X and Y",
            collection="test",
            top_k=2,
            mode=QueryMode.DENSE,
        )

        with patch("core.generation.settings") as mock_settings:
            mock_settings.enable_cache = False
            mock_settings.llm_backend.value = "ollama"
            mock_settings.use_hybrid_search = False

            response = answer_question(request)

        assert "[Source:" in response.answer
