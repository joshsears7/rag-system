"""
Tests for core/cot_rag.py — Chain-of-Thought RAG (EMNLP 2025).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestDecomposeQuestion:
    def test_returns_list_of_strings(self) -> None:
        from core.cot_rag import decompose_question

        mock_llm = MagicMock(return_value="1. What is X?\n2. How does Y work?\n3. What is Z?")
        steps = decompose_question("Complex multi-hop question", mock_llm, max_steps=4)

        assert isinstance(steps, list)
        assert len(steps) >= 1
        assert all(isinstance(s, str) for s in steps)

    def test_respects_max_steps(self) -> None:
        from core.cot_rag import decompose_question

        mock_llm = MagicMock(return_value="1. Step one\n2. Step two\n3. Step three\n4. Step four\n5. Step five")
        steps = decompose_question("question", mock_llm, max_steps=3)
        assert len(steps) <= 3

    def test_fallback_on_llm_failure(self) -> None:
        from core.cot_rag import decompose_question

        mock_llm = MagicMock(side_effect=RuntimeError("LLM down"))
        steps = decompose_question("What is the capital of France?", mock_llm)
        assert len(steps) >= 1
        assert "France" in steps[0] or "capital" in steps[0].lower() or len(steps[0]) > 0

    def test_parses_numbered_list(self) -> None:
        from core.cot_rag import decompose_question

        mock_llm = MagicMock(return_value="1. Find the revenue figure\n2. Calculate the growth rate")
        steps = decompose_question("question", mock_llm)
        assert "Find the revenue figure" in steps
        assert "Calculate the growth rate" in steps

    def test_handles_empty_response(self) -> None:
        from core.cot_rag import decompose_question

        mock_llm = MagicMock(return_value="")
        steps = decompose_question("question", mock_llm)
        # Should return at least one step (fallback)
        assert len(steps) >= 1


class TestSynthesizeStep:
    def test_returns_string(self) -> None:
        from core.cot_rag import synthesize_step

        mock_llm = MagicMock(return_value="The answer is 42.")
        result = synthesize_step("What is the answer?", ["Context chunk 1"], mock_llm)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_chunks_returns_not_found(self) -> None:
        from core.cot_rag import synthesize_step

        mock_llm = MagicMock()
        result = synthesize_step("What is X?", [], mock_llm)
        assert "not found" in result.lower() or "no relevant" in result.lower()
        mock_llm.assert_not_called()

    def test_handles_llm_failure(self) -> None:
        from core.cot_rag import synthesize_step

        mock_llm = MagicMock(side_effect=RuntimeError("timeout"))
        result = synthesize_step("question", ["chunk"], mock_llm)
        assert isinstance(result, str)  # should not raise


class TestSynthesizeFinal:
    def test_returns_answer_and_token_count(self) -> None:
        from core.cot_rag import synthesize_final, ReasoningStep

        steps = [
            ReasoningStep(
                step_number=1,
                thought="Find revenue",
                sub_query="revenue 2023",
                retrieved=["Revenue was $2.3B"],
                sources=["report.pdf"],
                intermediate="Revenue was $2.3B in 2023",
            )
        ]

        mock_llm = MagicMock(return_value="The revenue grew to $2.3B.")
        answer, tokens = synthesize_final("What was the revenue?", steps, mock_llm)

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(tokens, int)
        assert tokens > 0


class TestRunCoTRAG:
    def test_returns_cot_result(self) -> None:
        from core.cot_rag import run_cot_rag
        from models import RetrievalContext, QueryMode

        mock_llm = MagicMock(side_effect=[
            "1. Find what X is\n2. Find how X works",  # decompose
            "X is a framework for RAG",                 # step 1 synthesis
            "X works by retrieving chunks",             # step 2 synthesis
            "X is a RAG framework that retrieves chunks.",  # final synthesis
        ])

        def mock_retrieve(req):
            return RetrievalContext(
                query=req.question,
                results=[],
                query_mode=QueryMode.HYBRID,
            )

        result = run_cot_rag(
            question="What is X and how does it work?",
            collection="default",
            retrieve_fn=mock_retrieve,
            llm_fn=mock_llm,
            max_steps=2,
        )

        assert result.question == "What is X and how does it work?"
        assert isinstance(result.answer, str)
        assert result.num_steps == 2
        assert isinstance(result.reasoning_steps, list)
        assert result.latency_ms > 0

    def test_warns_on_no_context(self) -> None:
        from core.cot_rag import run_cot_rag
        from models import RetrievalContext, QueryMode

        mock_llm = MagicMock(side_effect=[
            "1. Step one",
            "Not found in context.",
            "No answer available.",
        ])

        def mock_retrieve(req):
            return RetrievalContext(
                query=req.question,
                results=[],
                query_mode=QueryMode.HYBRID,
            )

        result = run_cot_rag(
            question="Unknown question",
            collection="empty",
            retrieve_fn=mock_retrieve,
            llm_fn=mock_llm,
            max_steps=1,
        )

        assert len(result.warnings) > 0

    def test_trace_markdown_property(self) -> None:
        from core.cot_rag import run_cot_rag, CoTResult, ReasoningStep

        result = CoTResult(
            question="test?",
            answer="answer",
            reasoning_steps=[
                ReasoningStep(
                    step_number=1,
                    thought="Find X",
                    sub_query="X",
                    retrieved=["chunk"],
                    sources=["doc.txt"],
                    intermediate="X is Y",
                )
            ],
            all_sources=["doc.txt"],
            total_chunks=1,
            tokens_used=100,
            latency_ms=500.0,
            num_steps=1,
        )

        md = result.trace_as_markdown
        assert "Step 1" in md
        assert "Find X" in md
        assert "answer" in md


class TestQueryClassifier:
    def test_who_what_routes_low(self) -> None:
        from core.cot_rag import decompose_question
        # Just testing the classify function doesn't error
        from core.light_rag import classify_query
        assert classify_query("Who is the CEO?") == "low"
        assert classify_query("What is machine learning?") == "low"

    def test_summarize_routes_high(self) -> None:
        from core.light_rag import classify_query
        assert classify_query("Summarize the key themes") == "high"
        assert classify_query("Why is this important overall?") == "high"
