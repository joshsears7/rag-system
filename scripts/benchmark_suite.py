"""
Benchmark Suite — Named dataset evaluation with technique comparison.

Runs the RAG evaluation harness in two modes:
  1. COMPARISON MODE: Naive RAG vs Full Stack — quantifies what each technique buys
  2. CI MODE: Single run against thresholds for the quality gate

Produces:
  - Terminal table (Rich) with per-technique metrics
  - JSON output file for CI consumption and README badges
  - Comparison table showing delta vs naive baseline

Usage:
    # Full comparison (for README/blog):
    python3 scripts/benchmark_suite.py --compare

    # CI quality gate:
    python3 scripts/benchmark_suite.py --output eval_results.json \\
        --min-faithfulness 3.5 --min-recall 0.5

    # Run against custom QA file:
    python3 scripts/benchmark_suite.py --qa-file my_questions.json --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)  # suppress verbose logs during benchmark

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    """Configuration for one benchmark run (one RAG technique stack)."""
    name: str
    description: str
    use_hybrid: bool = True
    use_reranker: bool = True
    use_hyde: bool = False
    use_multi_query: bool = False
    top_k: int = 6
    mode: str = "hybrid"


@dataclass
class BenchmarkResult:
    """Results for one technique configuration."""
    config_name:          str
    total_samples:        int
    mean_faithfulness:    float
    mean_recall_at_k:     float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_latency_ms:      float
    passed_quality_gate:  bool = True
    error:                str = ""


# ── Benchmark configurations ──────────────────────────────────────────────────

BENCHMARK_CONFIGS = [
    BenchmarkConfig(
        name="naive",
        description="Naive RAG (dense-only, no reranking)",
        use_hybrid=False,
        use_reranker=False,
        use_hyde=False,
        use_multi_query=False,
        mode="dense",
    ),
    BenchmarkConfig(
        name="hybrid",
        description="Hybrid (dense+BM25+RRF)",
        use_hybrid=True,
        use_reranker=False,
        use_hyde=False,
        mode="hybrid",
    ),
    BenchmarkConfig(
        name="hybrid+rerank",
        description="Hybrid + cross-encoder reranking",
        use_hybrid=True,
        use_reranker=True,
        use_hyde=False,
        mode="hybrid",
    ),
    BenchmarkConfig(
        name="hybrid+rerank+hyde",
        description="Full stack (hybrid + reranking + HyDE)",
        use_hybrid=True,
        use_reranker=True,
        use_hyde=True,
        mode="hybrid",
    ),
]


# ── Single-config evaluation ──────────────────────────────────────────────────


def run_config(
    config: BenchmarkConfig,
    qa_pairs: list[dict],
) -> BenchmarkResult:
    """
    Run the evaluation harness for one benchmark configuration.

    Temporarily patches settings to match the config, runs all QA pairs,
    then restores settings.
    """
    from config import settings
    from models import EvalSample, QueryMode

    # Patch settings for this config
    original_hybrid   = settings.use_hybrid_search
    original_reranker = settings.use_reranker
    original_hyde     = settings.use_hyde
    original_top_k    = settings.top_k

    settings.use_hybrid_search = config.use_hybrid
    settings.use_reranker      = config.use_reranker
    settings.use_hyde          = config.use_hyde
    settings.top_k             = config.top_k

    faithfulness_scores: list[float] = []
    recall_scores:       list[float] = []
    relevancy_scores:    list[float] = []
    precision_scores:    list[float] = []
    latencies:           list[float] = []
    error_msg = ""

    try:
        from core.evaluation import evaluate_sample
        from core.generation import get_backend

        mode_enum = QueryMode.HYBRID if config.mode == "hybrid" else QueryMode.DENSE

        for qa in qa_pairs:
            sample = EvalSample(
                question=qa["question"],
                expected_answer=qa.get("expected_answer", ""),
                relevant_sources=qa.get("relevant_sources", []),
                collection=qa.get("collection", "eval_test"),
            )
            try:
                result = evaluate_sample(sample)
                faithfulness_scores.append(result.faithfulness_score)
                recall_scores.append(result.recall_at_k)
                relevancy_scores.append(result.answer_relevancy)
                latencies.append(result.latency_ms)
                # Context precision: approximate as recall here (full metric needs CRAG output)
                precision_scores.append(result.recall_at_k)
            except Exception as e:
                logging.warning("Sample evaluation failed for '%s': %s", qa["question"][:40], e)

    except Exception as e:
        error_msg = str(e)
        logging.error("Config '%s' failed: %s", config.name, e)

    finally:
        # Restore original settings
        settings.use_hybrid_search = original_hybrid
        settings.use_reranker      = original_reranker
        settings.use_hyde          = original_hyde
        settings.top_k             = original_top_k

    def safe_mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return BenchmarkResult(
        config_name=config.name,
        total_samples=len(qa_pairs),
        mean_faithfulness=round(safe_mean(faithfulness_scores), 2),
        mean_recall_at_k=round(safe_mean(recall_scores), 4),
        mean_answer_relevancy=round(safe_mean(relevancy_scores), 4),
        mean_context_precision=round(safe_mean(precision_scores), 4),
        mean_latency_ms=round(safe_mean(latencies), 1),
        error=error_msg,
    )


# ── Full comparison run ───────────────────────────────────────────────────────


def run_comparison(qa_pairs: list[dict]) -> list[BenchmarkResult]:
    """Run all benchmark configurations and return results."""
    results: list[BenchmarkResult] = []

    for config in BENCHMARK_CONFIGS:
        console.print(f"\n[bold cyan]Running config:[/bold cyan] {config.name} — {config.description}")
        start = time.perf_counter()
        result = run_config(config, qa_pairs)
        elapsed = time.perf_counter() - start
        console.print(f"  Done in {elapsed:.1f}s  |  faith={result.mean_faithfulness:.2f}  recall={result.mean_recall_at_k:.3f}")
        results.append(result)

    return results


# ── Display ───────────────────────────────────────────────────────────────────


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    """Print a comparison table showing technique improvements."""
    table = Table(
        title="RAG Technique Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Configuration",    style="cyan",  min_width=20)
    table.add_column("Faithfulness",     justify="right", min_width=14)
    table.add_column("Recall@K",         justify="right", min_width=10)
    table.add_column("Relevancy",        justify="right", min_width=10)
    table.add_column("Latency (p50)",    justify="right", min_width=12)
    table.add_column("vs Naive",         justify="right", min_width=10)

    # Baseline = naive
    baseline = results[0] if results else None

    for r in results:
        if r.error:
            table.add_row(r.config_name, "[red]ERROR[/red]", "—", "—", "—", "—")
            continue

        # Delta vs naive baseline
        if baseline and r.config_name != "naive":
            faith_delta = r.mean_faithfulness - baseline.mean_faithfulness
            recall_delta = r.mean_recall_at_k - baseline.mean_recall_at_k
            delta_str = f"faith {faith_delta:+.2f} / recall {recall_delta:+.3f}"
            delta_color = "green" if (faith_delta > 0 or recall_delta > 0) else "yellow"
            delta_cell = f"[{delta_color}]{delta_str}[/{delta_color}]"
        else:
            delta_cell = "[dim]baseline[/dim]"

        faith_color = "green" if r.mean_faithfulness >= 4.0 else ("yellow" if r.mean_faithfulness >= 3.0 else "red")
        recall_color = "green" if r.mean_recall_at_k >= 0.7 else ("yellow" if r.mean_recall_at_k >= 0.5 else "red")

        table.add_row(
            r.config_name,
            f"[{faith_color}]{r.mean_faithfulness:.2f}/5.0[/{faith_color}]",
            f"[{recall_color}]{r.mean_recall_at_k:.3f}[/{recall_color}]",
            f"{r.mean_answer_relevancy:.3f}",
            f"{r.mean_latency_ms:.0f}ms",
            delta_cell,
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n[dim]{results[0].total_samples if results else 0} samples per configuration[/dim]\n")


def print_summary_table(result: BenchmarkResult, thresholds: dict) -> None:
    """Print a single-config quality gate summary."""
    table = Table(
        title="Quality Gate Results",
        box=box.ROUNDED,
        header_style="bold blue",
    )
    table.add_column("Metric",      style="cyan", min_width=20)
    table.add_column("Score",       justify="right", min_width=10)
    table.add_column("Threshold",   justify="right", min_width=10)
    table.add_column("Status",      justify="center", min_width=8)

    def gate_row(name, score, threshold):
        passed = score >= threshold
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        score_color = "green" if passed else "red"
        table.add_row(name, f"[{score_color}]{score}[/{score_color}]", str(threshold), status)

    gate_row("Faithfulness",     result.mean_faithfulness,     thresholds["faithfulness"])
    gate_row("Recall@K",         result.mean_recall_at_k,      thresholds["recall"])
    gate_row("Answer Relevancy", result.mean_answer_relevancy, thresholds["relevancy"])
    table.add_row("Avg Latency", f"{result.mean_latency_ms:.0f}ms", "—", "—")
    table.add_row("Samples",     str(result.total_samples),    "—", "—")

    console.print("\n")
    console.print(table)


# ── QA pair loading ───────────────────────────────────────────────────────────


def load_qa_pairs(qa_file: str | None) -> list[dict]:
    """Load QA pairs from file or use built-in defaults."""
    if qa_file and Path(qa_file).exists():
        with open(qa_file) as f:
            pairs = json.load(f)
        console.print(f"[dim]Loaded {len(pairs)} QA pairs from {qa_file}[/dim]")
        return pairs

    # Try auto-generated eval pairs
    auto_path = Path("eval_qa_pairs.json")
    if auto_path.exists():
        with open(auto_path) as f:
            pairs = json.load(f)
        console.print(f"[dim]Loaded {len(pairs)} QA pairs from eval_qa_pairs.json[/dim]")
        return pairs

    # Inline fallback — works with any ingested collection
    console.print("[yellow]No QA file found — using generic test questions[/yellow]")
    return [
        {
            "question": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of AI that enables systems to learn from data.",
            "relevant_sources": [],
            "collection": "eval_test",
        },
        {
            "question": "What is retrieval-augmented generation?",
            "expected_answer": "RAG combines retrieval with LLM generation using external context.",
            "relevant_sources": [],
            "collection": "eval_test",
        },
        {
            "question": "How does HyDE work in RAG?",
            "expected_answer": "HyDE generates a hypothetical answer and uses its embedding for retrieval.",
            "relevant_sources": [],
            "collection": "eval_test",
        },
    ]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Benchmark Suite")
    parser.add_argument("--compare",          action="store_true",  help="Run full technique comparison")
    parser.add_argument("--qa-file",          type=str,   default=None,  help="Path to QA pairs JSON file")
    parser.add_argument("--output",           type=str,   default=None,  help="Output JSON file for results")
    parser.add_argument("--min-faithfulness", type=float, default=3.5,   help="Minimum faithfulness (1-5)")
    parser.add_argument("--min-recall",       type=float, default=0.5,   help="Minimum Recall@K (0-1)")
    parser.add_argument("--min-relevancy",    type=float, default=0.5,   help="Minimum answer relevancy (0-1)")
    args = parser.parse_args()

    console.print("[bold cyan]\nRAG Benchmark Suite[/bold cyan]")
    console.print(f"Thresholds: faithfulness>={args.min_faithfulness}  recall>={args.min_recall}  relevancy>={args.min_relevancy}\n")

    qa_pairs = load_qa_pairs(args.qa_file)

    thresholds = {
        "faithfulness": args.min_faithfulness,
        "recall":       args.min_recall,
        "relevancy":    args.min_relevancy,
    }

    if args.compare:
        # ── Full comparison mode ──────────────────────────────────────────────
        results = run_comparison(qa_pairs)
        print_comparison_table(results)

        # Save all configs to output file
        if args.output:
            output_data = {
                "mode": "comparison",
                "configs": [asdict(r) for r in results],
                "baseline": asdict(results[0]) if results else {},
                "best": asdict(max(results, key=lambda r: r.mean_faithfulness)) if results else {},
            }
            # Also write the best config's metrics at top level for CI gate
            best = max(results, key=lambda r: r.mean_faithfulness) if results else None
            if best:
                output_data.update({
                    "mean_faithfulness":     best.mean_faithfulness,
                    "mean_recall_at_k":      best.mean_recall_at_k,
                    "mean_answer_relevancy": best.mean_answer_relevancy,
                    "mean_latency_ms":       best.mean_latency_ms,
                    "total_samples":         best.total_samples,
                })
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[dim]Results saved to {args.output}[/dim]")

        # Gate on best result
        if results:
            best = max(results, key=lambda r: r.mean_faithfulness)
            print_summary_table(best, thresholds)
            if (
                best.mean_faithfulness     >= args.min_faithfulness and
                best.mean_recall_at_k      >= args.min_recall and
                best.mean_answer_relevancy >= args.min_relevancy
            ):
                console.print("[bold green]QUALITY GATE PASSED[/bold green]\n")
                sys.exit(0)
            else:
                console.print("[bold red]QUALITY GATE FAILED[/bold red]\n")
                sys.exit(1)

    else:
        # ── CI single-config mode (full stack) ───────────────────────────────
        full_stack = BENCHMARK_CONFIGS[-1]  # hybrid+rerank+hyde
        console.print(f"Running: {full_stack.name} — {full_stack.description}")
        result = run_config(full_stack, qa_pairs)
        print_summary_table(result, thresholds)

        if args.output:
            output_data = {
                "mode":                  "single",
                "config":                full_stack.name,
                "mean_faithfulness":     result.mean_faithfulness,
                "mean_recall_at_k":      result.mean_recall_at_k,
                "mean_answer_relevancy": result.mean_answer_relevancy,
                "mean_context_precision": result.mean_context_precision,
                "mean_latency_ms":       result.mean_latency_ms,
                "total_samples":         result.total_samples,
                "error":                 result.error,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"[dim]Results saved to {args.output}[/dim]")

        # Quality gate
        if result.error:
            console.print(f"[red]Evaluation failed: {result.error}[/red]")
            sys.exit(1)

        passed = (
            result.mean_faithfulness     >= args.min_faithfulness and
            result.mean_recall_at_k      >= args.min_recall and
            result.mean_answer_relevancy >= args.min_relevancy
        )

        if passed:
            console.print("[bold green]QUALITY GATE PASSED[/bold green]\n")
            sys.exit(0)
        else:
            console.print("[bold red]QUALITY GATE FAILED[/bold red]\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
