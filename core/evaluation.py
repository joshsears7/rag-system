"""
RAG evaluation harness — RAGAS-inspired metrics, fully local (no paid API).

Metrics implemented:
  - Recall@K          : did the correct source appear in top-k retrieved chunks?
  - Faithfulness      : LLM-as-judge (1-5 scale): is the answer grounded in context?
  - Answer Relevancy  : cosine similarity between answer embedding and question embedding
  - Context Precision : fraction of retrieved chunks that are genuinely relevant
  - Latency           : end-to-end response time

All metrics run without external APIs — only the local LLM backend + sentence-transformers.
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import Callable

from rich.console import Console
from rich.table import Table

from config import settings
from core.generation import answer_question, get_backend
from core.ingestion import get_embedding_model
from models import EvalResult, EvalSample, EvalSummary, QueryMode, QueryRequest

logger = logging.getLogger(__name__)
console = Console()


# ── Metric: Recall@K ─────────────────────────────────────────────────────────


def recall_at_k(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """
    Fraction of relevant sources that were retrieved.

    recall@k = |relevant ∩ retrieved| / |relevant|

    Args:
        retrieved_sources: filenames/URLs of retrieved chunks
        relevant_sources: expected source filenames from the test case

    Returns:
        Float in [0, 1]. 1.0 = all relevant sources found.
    """
    if not relevant_sources:
        return 1.0  # no ground truth = can't penalise
    # Normalize to basename so full paths match bare filenames
    import os
    retrieved_set = {os.path.basename(s).lower() for s in retrieved_sources}
    relevant_set = {os.path.basename(s).lower() for s in relevant_sources}
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


# ── Metric: Faithfulness (LLM-as-judge) ──────────────────────────────────────


def faithfulness_score(
    question: str,
    answer: str,
    context_chunks: list[str],
    llm_fn: Callable[[str], str],
) -> float:
    """
    Ask the LLM to score whether the answer is faithful to the retrieved context.

    Faithfulness = is every claim in the answer directly supported by context?
    Score: 1 (not faithful) → 5 (perfectly faithful, no hallucinations)

    This is the RAGAS faithfulness metric implemented with a zero-shot LLM judge.
    """
    context_str = "\n\n".join(context_chunks[:5])[:2000]
    prompt = (
        "You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.\n\n"
        "Evaluate the following answer for FAITHFULNESS — whether every claim in the answer "
        "is directly supported by the provided context. Do not consider factual accuracy "
        "against world knowledge; only judge whether the answer stays within the context.\n\n"
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Score the faithfulness from 1 to 5:\n"
        "  1 = Answer contains significant hallucinations not in context\n"
        "  2 = Answer has some claims not in context\n"
        "  3 = Mostly faithful with minor stretches\n"
        "  4 = Almost entirely faithful to context\n"
        "  5 = Perfectly faithful — every claim is directly supported\n\n"
        "Reply with ONLY the integer score (1-5):"
    )
    try:
        raw = llm_fn(prompt).strip()
        score = float(raw.split()[0].rstrip(".,"))
        return max(1.0, min(5.0, score))
    except (ValueError, IndexError):
        logger.warning("Could not parse faithfulness score from: '%s'", raw if "raw" in dir() else "?")
        return 3.0


# ── Metric: Answer Relevancy ──────────────────────────────────────────────────


def answer_relevancy_score(question: str, answer: str) -> float:
    """
    Cosine similarity between question embedding and answer embedding.

    A good answer should be topically aligned with the question.
    Higher = more relevant. This mirrors the RAGAS answer relevancy metric.
    """
    model = get_embedding_model()
    embeddings = model.encode([question, answer], normalize_embeddings=True)
    q_emb, a_emb = embeddings[0], embeddings[1]
    import numpy as np
    return float(np.dot(q_emb, a_emb))


# ── Metric: Context Precision ─────────────────────────────────────────────────


def context_precision_score(
    question: str,
    context_chunks: list[str],
    llm_fn: Callable[[str], str],
) -> float:
    """
    Fraction of retrieved chunks that were actually useful for answering.

    For each chunk, ask LLM: "Is this relevant to answering the question?"
    precision = (useful chunks) / (total chunks)
    """
    if not context_chunks:
        return 0.0

    useful = 0
    for chunk in context_chunks:
        prompt = (
            f"Is the following text relevant to answering this question?\n\n"
            f"Question: {question}\n\n"
            f"Text: {chunk[:500]}\n\n"
            "Reply with ONLY 'yes' or 'no':"
        )
        try:
            answer = llm_fn(prompt).strip().lower()
            if "yes" in answer:
                useful += 1
        except Exception:
            useful += 1  # assume relevant on error

    return useful / len(context_chunks)


# ── Single-sample evaluator ───────────────────────────────────────────────────


def evaluate_sample(sample: EvalSample) -> EvalResult:
    """
    Run the full RAG pipeline on a single test case and compute all metrics.

    Args:
        sample: (question, expected_answer, relevant_sources, collection)

    Returns:
        EvalResult with all metric scores
    """
    start = time.perf_counter()
    backend = get_backend()

    request = QueryRequest(
        question=sample.question,
        collection=sample.collection,
        top_k=settings.top_k,
        mode=QueryMode.HYBRID,
    )

    try:
        response = answer_question(request)
    except Exception as e:
        logger.error("Generation failed for sample '%s': %s", sample.question[:60], e)
        return EvalResult(
            question=sample.question,
            generated_answer=f"ERROR: {e}",
            expected_answer=sample.expected_answer,
            sources_retrieved=[],
            relevant_sources=sample.relevant_sources,
            recall_at_k=0.0,
            faithfulness_score=1.0,
            answer_relevancy=0.0,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    retrieved_sources = [s.source for s in response.sources]
    context_chunks = [s.excerpt for s in response.sources]

    # Compute metrics
    r_at_k = recall_at_k(retrieved_sources, sample.relevant_sources)
    faith = faithfulness_score(
        question=sample.question,
        answer=response.answer,
        context_chunks=context_chunks,
        llm_fn=backend.complete_raw,
    )
    relevancy = answer_relevancy_score(sample.question, response.answer)

    return EvalResult(
        question=sample.question,
        generated_answer=response.answer,
        expected_answer=sample.expected_answer,
        sources_retrieved=retrieved_sources,
        relevant_sources=sample.relevant_sources,
        recall_at_k=round(r_at_k, 4),
        faithfulness_score=round(faith, 2),
        answer_relevancy=round(relevancy, 4),
        latency_ms=round(response.latency_ms, 2),
    )


# ── Full eval harness ─────────────────────────────────────────────────────────


def run_evaluation(samples: list[EvalSample]) -> EvalSummary:
    """
    Run the evaluation harness over all samples and return an aggregated summary.

    Args:
        samples: list of test cases

    Returns:
        EvalSummary with per-sample results and aggregate stats
    """
    results: list[EvalResult] = []

    console.print(f"\n[bold cyan]Running evaluation on {len(samples)} samples…[/bold cyan]\n")

    for i, sample in enumerate(samples, start=1):
        console.print(f"[dim]Sample {i}/{len(samples)}:[/dim] {sample.question[:70]}…")
        result = evaluate_sample(sample)
        results.append(result)
        console.print(
            f"  recall@k={result.recall_at_k:.2f}  "
            f"faithfulness={result.faithfulness_score:.1f}/5  "
            f"relevancy={result.answer_relevancy:.2f}  "
            f"latency={result.latency_ms:.0f}ms"
        )

    summary = EvalSummary(
        total_samples=len(results),
        mean_recall_at_k=round(statistics.mean(r.recall_at_k for r in results), 4) if results else 0.0,
        mean_faithfulness=round(statistics.mean(r.faithfulness_score for r in results), 2) if results else 1.0,
        mean_answer_relevancy=round(statistics.mean(r.answer_relevancy for r in results), 4) if results else 0.0,
        mean_latency_ms=round(statistics.mean(r.latency_ms for r in results), 2) if results else 0.0,
        results=results,
    )

    return summary


def print_eval_summary(summary: EvalSummary) -> None:
    """Render evaluation summary as a Rich table."""
    # Aggregate table
    agg_table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
    agg_table.add_column("Metric", style="cyan", no_wrap=True)
    agg_table.add_column("Score", justify="right")
    agg_table.add_column("Interpretation")

    agg_table.add_row("Recall@K", f"{summary.mean_recall_at_k:.3f}", "Fraction of relevant sources retrieved")
    agg_table.add_row("Faithfulness", f"{summary.mean_faithfulness:.2f}/5.0", "LLM-judged groundedness in context")
    agg_table.add_row("Answer Relevancy", f"{summary.mean_answer_relevancy:.3f}", "Semantic alignment with question")
    agg_table.add_row("Avg Latency", f"{summary.mean_latency_ms:.0f}ms", "End-to-end response time")
    agg_table.add_row("Samples", str(summary.total_samples), "Total evaluated")

    console.print("\n")
    console.print(agg_table)

    # Per-sample table
    detail_table = Table(title="Per-Sample Results", show_header=True, header_style="bold blue", show_lines=True)
    detail_table.add_column("#", style="dim", width=4)
    detail_table.add_column("Question", max_width=40)
    detail_table.add_column("Recall@K", justify="right", width=10)
    detail_table.add_column("Faith.", justify="right", width=8)
    detail_table.add_column("Relev.", justify="right", width=8)
    detail_table.add_column("Latency", justify="right", width=10)

    for i, r in enumerate(summary.results, start=1):
        faith_color = "green" if r.faithfulness_score >= 4 else ("yellow" if r.faithfulness_score >= 3 else "red")
        recall_color = "green" if r.recall_at_k >= 0.8 else ("yellow" if r.recall_at_k >= 0.5 else "red")
        detail_table.add_row(
            str(i),
            r.question[:40] + ("…" if len(r.question) > 40 else ""),
            f"[{recall_color}]{r.recall_at_k:.2f}[/{recall_color}]",
            f"[{faith_color}]{r.faithfulness_score:.1f}[/{faith_color}]",
            f"{r.answer_relevancy:.2f}",
            f"{r.latency_ms:.0f}ms",
        )

    console.print("\n")
    console.print(detail_table)
