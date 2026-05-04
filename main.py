#!/usr/bin/env python3
"""
RAG System CLI — production entrypoint.

Commands:
  ingest              Ingest files, directories, or URLs
  query               One-shot Q&A with citations
  chat                Interactive multi-turn conversation mode
  list-collections    Show all knowledge bases
  delete-collection   Remove a knowledge base
  eval                Run RAGAS-style evaluation harness
  graph               Knowledge graph commands (stats, entity lookup)
  route               Show which collection a query would route to
  benchmark           Performance benchmark (latency, throughput)
  serve               Start the FastAPI server

Advanced commands:
  adaptive            Adaptive RAG query (auto-selects retrieval strategy)
  raptor-ingest       RAPTOR recursive tree ingestion for long documents
  multimodal-ingest   Ingest PDFs with tables and figures (vision LLM)
  feedback            Feedback analytics and contrastive pair mining
  finetune            Fine-tune embedding model on domain-specific data
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from config import settings
from models import EvalSample, QueryMode, QueryRequest

app = typer.Typer(
    name="rag",
    help="[bold cyan]RAG System[/bold cyan] — production AI document intelligence",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _backend_badge() -> str:
    colors = {"ollama": "green", "claude": "blue", "openai": "cyan"}
    color = colors.get(settings.llm_backend.value, "white")
    return f"[{color} bold]{settings.llm_backend.value.upper()}[/{color} bold]"


def _print_header(subtitle: str = "") -> None:
    sub = f"  [dim]{subtitle}[/dim]" if subtitle else ""
    console.print(
        Panel.fit(
            f"[bold white]RAG System v2[/bold white]  ·  {_backend_badge()}{sub}  ·  "
            f"[yellow]{settings.embedding_model}[/yellow]",
            border_style="dim cyan",
        )
    )


def _print_ingest_result(result) -> None:
    added_color = "green" if result.chunks_added > 0 else "yellow"
    console.print(
        Panel(
            f"[{added_color}]✓[/{added_color}] Chunks added:    [bold]{result.chunks_added}[/bold]\n"
            f"[dim]⊘[/dim] Duplicates:       [bold]{result.duplicates_skipped}[/bold]\n"
            f"[dim]∑[/dim] Total processed:  [bold]{result.total_chunks_processed}[/bold]\n"
            f"[dim]⏱[/dim] Elapsed:          [bold]{result.elapsed_seconds:.2f}s[/bold]\n"
            f"Collection:          [yellow]{result.collection}[/yellow]",
            title="[bold green]Ingestion Complete[/bold green]",
            border_style="green",
        )
    )


def _print_answer(response, show_stats: bool = True) -> None:
    cache_label = " [dim](cached ⚡)[/dim]" if response.cache_hit else ""
    console.print(
        Panel(
            response.answer,
            title=f"[bold cyan]Answer[/bold cyan]{cache_label}",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if response.sources:
        src_table = Table(title="Sources", header_style="bold magenta", show_lines=True)
        src_table.add_column("#", width=4, style="dim")
        src_table.add_column("Source", style="cyan")
        src_table.add_column("Chunk", justify="right", width=7)
        src_table.add_column("Page", justify="right", width=6)
        src_table.add_column("Score", justify="right", width=8)
        src_table.add_column("Excerpt", max_width=50)

        for i, src in enumerate(response.sources, start=1):
            page_str = str(src.page_number) if src.page_number else "—"
            sim_color = "green" if src.similarity_score >= 0.7 else ("yellow" if src.similarity_score >= 0.5 else "red")
            name = Path(src.source).name if ("/" in src.source or "\\" in src.source) else src.source
            src_table.add_row(
                str(i), name, str(src.chunk_index), page_str,
                f"[{sim_color}]{src.similarity_score:.3f}[/{sim_color}]",
                src.excerpt[:50] + "…" if len(src.excerpt) > 50 else src.excerpt,
            )
        console.print(src_table)

    if show_stats:
        console.print(
            f"\n[dim]Backend:[/dim] {_backend_badge()}  "
            f"[dim]Model:[/dim] [dim]{response.model_used}[/dim]  "
            f"[dim]Tokens:[/dim] [dim]{response.tokens_used}[/dim]  "
            f"[dim]Latency:[/dim] [dim]{response.latency_ms:.0f}ms[/dim]  "
            f"[dim]Sources:[/dim] [dim]{len(response.sources)}[/dim]\n"
        )


# ── ingest ────────────────────────────────────────────────────────────────────


@app.command()
def ingest(
    path: Annotated[str, typer.Option("--path", "-p", help="File path, directory, or URL")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    strategy: Annotated[str, typer.Option("--strategy", "-s", help="recursive|semantic|hierarchical")] = "recursive",
    extract_graph: Annotated[bool, typer.Option("--graph", help="Extract entities into knowledge graph")] = False,
    summarize: Annotated[bool, typer.Option("--summarize", help="Generate LLM summary at ingest time")] = False,
) -> None:
    """[bold]Ingest[/bold] documents into a knowledge base (file, directory, or URL)."""
    from core.ingestion import ingest_document
    from core.document_processor import analyze_document
    from core.graph_rag import get_knowledge_graph, extract_triples

    _print_header("ingestion")
    console.print(f"\n[bold]Source:[/bold] [cyan]{path}[/cyan] → [yellow]{collection!r}[/yellow]\n")

    p = Path(path)
    is_url = path.startswith("http://") or path.startswith("https://")

    if is_url or (p.exists() and p.is_file()):
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("Processing…", total=None)
            try:
                result = ingest_document(source=path, collection_name=collection, overwrite=overwrite, chunking_strategy=strategy)
            except (ValueError, RuntimeError, OSError) as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1) from e

        _print_ingest_result(result)

        # Optional graph extraction
        if extract_graph and result.chunks_added > 0:
            console.print("[dim]Extracting knowledge graph triples…[/dim]")
            try:
                from core.generation import get_backend
                from core.ingestion import load_document
                pages, _ = load_document(path)
                graph = get_knowledge_graph()
                backend = get_backend()
                total_triples = 0
                for page_text, _ in pages[:5]:
                    triples = extract_triples(page_text, path, backend.complete_raw)
                    total_triples += graph.add_triples(triples)
                graph.save()
                console.print(f"[green]✓[/green] Graph: added [bold]{total_triples}[/bold] triples")
            except Exception as e:
                console.print(f"[yellow]Graph extraction failed: {e}[/yellow]")

    elif p.exists() and p.is_dir():
        supported = [".pdf", ".txt", ".docx", ".md", ".markdown"]
        files = [f for f in p.rglob("*") if f.suffix.lower() in supported and f.is_file()]
        if not files:
            console.print(f"[yellow]No supported files found in '{path}'[/yellow]")
            raise typer.Exit(0)

        console.print(f"Found [bold]{len(files)}[/bold] files.\n")
        total_added = total_skipped = errors = 0

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("Ingesting…", total=len(files))
            for file_path in files:
                prog.update(task, description=f"[dim]{file_path.name[:40]}[/dim]")
                try:
                    result = ingest_document(str(file_path), collection, overwrite, strategy)
                    total_added += result.chunks_added
                    total_skipped += result.duplicates_skipped
                except (ValueError, RuntimeError, OSError) as e:
                    console.print(f"\n[red]  ✗ {file_path.name}: {e}[/red]")
                    errors += 1
                finally:
                    prog.advance(task)

        console.print(Panel(
            f"[green]✓[/green] Added: [bold]{total_added}[/bold]  [dim]|[/dim]  "
            f"Skipped: [bold]{total_skipped}[/bold]  [dim]|[/dim]  "
            f"Errors: [{'red' if errors else 'dim'}]{errors}[/{'red' if errors else 'dim'}]\n"
            f"Collection: [yellow]{collection}[/yellow]",
            title="[bold green]Batch Ingestion Complete[/bold green]", border_style="green",
        ))
    else:
        console.print(f"[red]Path not found:[/red] '{path}'")
        raise typer.Exit(1)


# ── query ─────────────────────────────────────────────────────────────────────


@app.command()
def query(
    question: Annotated[str, typer.Option("--question", "-q")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = settings.top_k,
    mode: Annotated[str, typer.Option("--mode", "-m", help="dense|sparse|hybrid")] = "hybrid",
    hyde: Annotated[bool, typer.Option("--hyde")] = False,
    multi_query: Annotated[bool, typer.Option("--multi-query")] = False,
    use_graph: Annotated[bool, typer.Option("--graph", help="Augment with knowledge graph")] = False,
    auto_route: Annotated[bool, typer.Option("--auto-route", help="Auto-select best collection")] = False,
) -> None:
    """[bold]Query[/bold] a knowledge base for a grounded, cited answer."""
    from core.generation import answer_question, get_backend
    from core.graph_rag import get_knowledge_graph, retrieve_graph_context
    from core.router import get_router

    _print_header("query")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    try:
        mode_enum = QueryMode(mode)
    except ValueError:
        console.print(f"[red]Invalid mode '{mode}'. Use: dense | sparse | hybrid[/red]")
        raise typer.Exit(1)

    if auto_route:
        backend = get_backend()
        router = get_router()
        router.auto_register()
        collection = router.route_single(question, use_llm=True, llm_fn=backend.complete_raw)
        console.print(f"[dim]Auto-routed to collection:[/dim] [yellow]{collection}[/yellow]\n")

    request = QueryRequest(
        question=question, collection=collection, top_k=top_k,
        mode=mode_enum, use_hyde=hyde, use_multi_query=multi_query,
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Retrieving + generating…", total=None)
        try:
            response = answer_question(request)
        except (RuntimeError, ValueError) as e:
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1) from e
        prog.update(task, completed=True)

    _print_answer(response)

    # Graph augmentation
    if use_graph:
        try:
            graph = get_knowledge_graph()
            graph_ctx = retrieve_graph_context(question, graph)
            if graph_ctx.entities_found:
                g_table = Table(title=f"Knowledge Graph: {len(graph_ctx.triples)} triples", header_style="bold yellow")
                g_table.add_column("Subject", style="cyan")
                g_table.add_column("Relation")
                g_table.add_column("Object", style="green")
                g_table.add_column("Source", style="dim")
                for triple in graph_ctx.triples[:10]:
                    g_table.add_row(triple.subject, triple.predicate, triple.obj, Path(triple.source).name)
                console.print(g_table)
        except Exception as e:
            console.print(f"[dim]Graph lookup failed: {e}[/dim]")


# ── chat ──────────────────────────────────────────────────────────────────────


@app.command()
def chat(
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    session_id: Annotated[Optional[str], typer.Option("--session", help="Resume a session by ID")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = settings.top_k,
) -> None:
    """
    [bold]Chat[/bold] — interactive multi-turn conversation with your documents.

    References ("it", "that document", "the above") are automatically resolved.
    Type [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan] to end the session.
    Type [bold cyan]/clear[/bold cyan] to reset conversation history.
    Type [bold cyan]/history[/bold cyan] to show the current conversation.
    Type [bold cyan]/collection <name>[/bold cyan] to switch collections.
    """
    from core.generation import answer_question, get_backend
    from core.conversation import get_or_create_session, ConversationTurn

    _print_header("interactive chat")
    sid = session_id or f"cli-{uuid.uuid4().hex[:8]}"
    session = get_or_create_session(sid)
    backend = get_backend()
    current_collection = collection

    console.print(Panel(
        f"Session: [bold]{sid}[/bold]\nCollection: [yellow]{current_collection}[/yellow]\n"
        f"Commands: [cyan]/clear[/cyan]  [cyan]/history[/cyan]  [cyan]/collection <name>[/cyan]  [cyan]exit[/cyan]",
        title="[bold cyan]Chat Session Started[/bold cyan]",
        border_style="cyan",
    ))

    turn = len(session.turns) + 1

    while True:
        try:
            console.print(f"\n[bold cyan]You[/bold cyan] [dim](turn {turn})[/dim]", end=" ")
            question = Prompt.ask("")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Session ended.[/yellow]")
            break

        if not question.strip():
            continue
        if question.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Goodbye.[/yellow]")
            break
        if question.strip() == "/clear":
            session.clear()
            turn = 1
            console.print("[green]Conversation cleared.[/green]")
            continue
        if question.strip() == "/history":
            if not session.turns:
                console.print("[dim]No history yet.[/dim]")
            for i, t in enumerate(session.turns, 1):
                console.print(f"[dim]{i}.[/dim] [bold]Q:[/bold] {t.question}")
                console.print(f"   [bold]A:[/bold] {t.answer[:200]}…" if len(t.answer) > 200 else f"   [bold]A:[/bold] {t.answer}")
            continue
        if question.startswith("/collection "):
            new_col = question.split(" ", 1)[1].strip()
            current_collection = new_col
            console.print(f"[green]Switched to collection:[/green] [yellow]{current_collection}[/yellow]")
            continue

        # Resolve references
        resolved = session.resolve_references(question, backend.complete_raw)
        if resolved != question:
            console.print(f"[dim]Resolved: {resolved}[/dim]")

        request = QueryRequest(
            question=resolved, collection=current_collection,
            top_k=top_k, mode=QueryMode.HYBRID,
        )

        with Progress(SpinnerColumn(), TextColumn("[dim]Thinking…[/dim]"), console=console, transient=True) as prog:
            prog.add_task("", total=None)
            try:
                response = answer_question(request)
            except (RuntimeError, ValueError) as e:
                console.print(f"[red]Error:[/red] {e}")
                continue

        console.print(f"\n[bold green]Assistant[/bold green] [dim](turn {turn})[/dim]")
        console.print(Panel(response.answer, border_style="green", padding=(0, 2)))

        if response.sources:
            sources_str = "  ".join(f"[dim][{i}] {Path(s.source).name}[/dim]" for i, s in enumerate(response.sources, 1))
            console.print(f"Sources: {sources_str}")

        console.print(f"[dim]{response.latency_ms:.0f}ms · {response.tokens_used} tokens[/dim]")

        session.add_turn(ConversationTurn(
            question=question, answer=response.answer,
            sources=[s.source for s in response.sources],
            collection=current_collection,
            tokens_used=response.tokens_used, latency_ms=response.latency_ms,
        ))

        if len(session.turns) > session.summarize_after:
            session.compress(backend.complete_raw)

        turn += 1


# ── list-collections ──────────────────────────────────────────────────────────


@app.command(name="list-collections")
def list_collections() -> None:
    """[bold]List[/bold] all knowledge base collections."""
    from core.ingestion import list_collections as _list
    from core.graph_rag import get_knowledge_graph

    _print_header()
    cols = _list()
    graph_stats = get_knowledge_graph().stats()

    if not cols:
        console.print("\n[yellow]No collections. Run 'ingest' first.[/yellow]")
        return

    table = Table(title=f"Knowledge Bases ({len(cols)} total)", header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Embedding Model", style="dim")

    for c in cols:
        table.add_row(c["name"], str(c["document_count"]), c["embedding_model"])

    console.print(table)
    console.print(f"\n[dim]Knowledge graph:[/dim] {graph_stats['nodes']} nodes, {graph_stats['edges']} edges\n")


# ── delete-collection ─────────────────────────────────────────────────────────


@app.command(name="delete-collection")
def delete_collection(
    name: Annotated[str, typer.Option("--name", "-n")],
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """[bold]Delete[/bold] a knowledge base permanently."""
    from core.ingestion import delete_collection as _delete
    _print_header()
    if not yes:
        confirm = typer.confirm(f"Delete '{name}'? Irreversible.", default=False)
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)
    deleted = _delete(name)
    if deleted:
        console.print(f"[green]✓[/green] Deleted [yellow]{name!r}[/yellow].")
    else:
        console.print(f"[yellow]Not found: {name!r}[/yellow]")


# ── graph ─────────────────────────────────────────────────────────────────────


graph_app = typer.Typer(name="graph", help="Knowledge graph commands")
app.add_typer(graph_app)


@graph_app.command(name="stats")
def graph_stats() -> None:
    """Show knowledge graph statistics."""
    from core.graph_rag import get_knowledge_graph
    _print_header("knowledge graph")
    stats = get_knowledge_graph().stats()
    console.print(Panel(
        f"Nodes (entities):  [bold cyan]{stats['nodes']}[/bold cyan]\n"
        f"Edges (relations): [bold green]{stats['edges']}[/bold green]\n"
        f"Components:        [bold]{stats['connected_components']}[/bold]",
        title="Knowledge Graph", border_style="yellow",
    ))


@graph_app.command(name="entity")
def graph_entity(
    name: Annotated[str, typer.Argument(help="Entity name to look up")],
    hops: Annotated[int, typer.Option("--hops", help="Relationship hops to traverse")] = 2,
) -> None:
    """Look up an entity's relationships in the knowledge graph."""
    from core.graph_rag import get_knowledge_graph
    _print_header("knowledge graph")
    graph = get_knowledge_graph()
    info = graph.get_entity_summary(name)

    if not info["found"]:
        console.print(f"[yellow]Entity '{name}' not found in graph.[/yellow]")
        return

    table = Table(title=f"Entity: {name}", header_style="bold yellow")
    table.add_column("Direction")
    table.add_column("Entity", style="cyan")
    table.add_column("Relation")
    table.add_column("Source", style="dim")

    for rel in info["outgoing"]:
        table.add_row("→", rel["object"], rel["predicate"], rel.get("source", "?"))
    for rel in info["incoming"]:
        table.add_row("←", rel["subject"], rel["predicate"], rel.get("source", "?"))

    console.print(table)


# ── route ─────────────────────────────────────────────────────────────────────


@app.command()
def route(
    question: Annotated[str, typer.Option("--question", "-q")],
    top_n: Annotated[int, typer.Option("--top-n")] = 2,
) -> None:
    """Show which collection(s) a query would be routed to."""
    from core.router import get_router
    _print_header("query routing")
    router = get_router()
    router.auto_register()
    collections = router.route(question, top_n=top_n)
    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[bold]Recommended collections:[/bold] " + ", ".join(f"[yellow]{c}[/yellow]" for c in collections))


# ── benchmark ─────────────────────────────────────────────────────────────────


@app.command()
def benchmark(
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    n: Annotated[int, typer.Option("--n", help="Number of queries to run")] = 10,
    question: Annotated[str, typer.Option("--question", "-q")] = "What is the main topic of this document?",
) -> None:
    """[bold]Benchmark[/bold] retrieval and generation latency."""
    from core.generation import answer_question

    _print_header("benchmark")
    console.print(f"\nRunning [bold]{n}[/bold] queries against [yellow]{collection!r}[/yellow]…\n")

    latencies = []
    cache_hits = 0

    request = QueryRequest(question=question, collection=collection, top_k=settings.top_k, mode=QueryMode.HYBRID)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Benchmarking…", total=n)
        for i in range(n):
            try:
                start = time.perf_counter()
                response = answer_question(request)
                latencies.append((time.perf_counter() - start) * 1000)
                if response.cache_hit:
                    cache_hits += 1
            except Exception as e:
                console.print(f"[red]Run {i+1} failed: {e}[/red]")
            prog.advance(task)

    if latencies:
        import statistics
        table = Table(title="Benchmark Results", header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Queries", str(n))
        table.add_row("Mean latency", f"{statistics.mean(latencies):.0f}ms")
        table.add_row("Median latency", f"{statistics.median(latencies):.0f}ms")
        table.add_row("Min latency", f"{min(latencies):.0f}ms")
        table.add_row("Max latency", f"{max(latencies):.0f}ms")
        table.add_row("P95 latency", f"{sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms")
        table.add_row("Cache hit rate", f"{cache_hits/n*100:.1f}%")
        console.print(table)


# ── eval ──────────────────────────────────────────────────────────────────────


@app.command()
def eval(
    test_file: Annotated[str, typer.Option("--test-file", "-f")] = "tests/eval_samples.json",
    collection: Annotated[Optional[str], typer.Option("--collection", "-c")] = None,
) -> None:
    """[bold]Evaluate[/bold] RAG quality with RAGAS-style metrics."""
    from core.evaluation import print_eval_summary, run_evaluation
    _print_header("evaluation")
    try:
        with open(test_file, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[red]Cannot load test file: {e}[/red]")
        raise typer.Exit(1) from e
    samples = [EvalSample(**({**item, "collection": collection} if collection else item)) for item in raw]
    summary = run_evaluation(samples)
    print_eval_summary(summary)


# ── serve ─────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = settings.api_host,
    port: Annotated[int, typer.Option("--port")] = settings.api_port,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """[bold]Start[/bold] the FastAPI server (http://localhost:{port}/docs)."""
    import uvicorn
    _print_header("API server")
    console.print(f"\n[bold]Server:[/bold] [cyan]http://{host}:{port}[/cyan]  →  docs at [cyan]/docs[/cyan]\n")
    uvicorn.run("api:app", host=host, port=port, reload=reload, log_level=settings.log_level.lower())


# ── adaptive ──────────────────────────────────────────────────────────────────


@app.command()
def adaptive(
    question: Annotated[str, typer.Option("--question", "-q")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = settings.top_k,
    no_self_rag: Annotated[bool, typer.Option("--no-self-rag", help="Disable Self-RAG reflection")] = False,
    max_hops: Annotated[int, typer.Option("--max-hops")] = 3,
) -> None:
    """
    [bold]Adaptive RAG[/bold] — auto-selects NO_RETRIEVAL / SINGLE_STEP / ITERATIVE strategy.

    Uses Self-RAG reflection tokens: [Retrieve], [IsREL], [IsSUP], [IsUSE].
    Complex multi-hop questions trigger iterative chained retrieval.
    """
    from core.adaptive_rag import adaptive_answer
    from core.generation import get_backend
    from core.retrieval import retrieve
    from models import QueryRequest, QueryMode

    _print_header("adaptive RAG")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    backend = get_backend()

    def _retrieve_fn(q: str, col: str, k: int) -> list:
        req = QueryRequest(question=q, collection=col, top_k=k, mode=QueryMode.HYBRID)
        ctx = retrieve(req, generate_fn=backend.complete_raw)
        return ctx.results

    def _generate_fn(system: str, user: str) -> str:
        return backend.complete_raw(f"{system}\n\n{user}")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Adaptive retrieval…", total=None)
        result = adaptive_answer(
            question=question,
            collection=collection,
            llm_fn=backend.complete_raw,
            retrieve_fn=_retrieve_fn,
            generate_fn=_generate_fn,
            top_k=top_k,
            use_self_rag=not no_self_rag,
            use_iterative=True,
            max_hops=max_hops,
        )
        prog.update(task, completed=True)

    strategy_colors = {"no_retrieval": "yellow", "single_step": "green", "iterative": "cyan"}
    s = result.strategy_used.value
    color = strategy_colors.get(s, "white")

    console.print(Panel(
        result.answer,
        title=f"[bold cyan]Answer[/bold cyan]  [dim]·[/dim]  Strategy: [{color}]{s.upper()}[/{color}]  [dim]·[/dim]  {result.hops} hop(s)  [dim]·[/dim]  {result.latency_ms:.0f}ms",
        border_style="cyan",
        padding=(1, 2),
    ))

    if result.queries_used and len(result.queries_used) > 1:
        console.print("[dim]Sub-queries:[/dim]")
        for i, q in enumerate(result.queries_used):
            console.print(f"  [dim]{i}.[/dim] {q}")

    if result.self_rag_flags:
        flags_str = "  ".join(f"[dim]{k}:[/dim] {v}" for k, v in result.self_rag_flags.items())
        console.print(f"\n[dim]Self-RAG flags:[/dim] {flags_str}\n")


# ── raptor-ingest ─────────────────────────────────────────────────────────────


@app.command(name="raptor-ingest")
def raptor_ingest(
    path: Annotated[str, typer.Option("--path", "-p")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    max_levels: Annotated[int, typer.Option("--levels", help="RAPTOR tree depth")] = 3,
    cluster_size: Annotated[int, typer.Option("--cluster-size")] = 10,
) -> None:
    """
    [bold]RAPTOR ingestion[/bold] — recursive tree summarization for long documents.

    Clusters chunks into semantic groups, summarizes each cluster with LLM,
    then recursively clusters the summaries. Creates a multi-level retrieval tree
    so queries can match at any abstraction level (detail → section → chapter).
    """
    from core.raptor import build_raptor_tree, ingest_raptor_tree
    from core.ingestion import ingest_document
    from core.generation import get_backend

    _print_header("RAPTOR ingestion")
    console.print(f"\n[bold]Source:[/bold] [cyan]{path}[/cyan] → [yellow]{collection!r}[/yellow] (max {max_levels} levels)\n")

    backend = get_backend()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Standard ingestion…", total=None)
        try:
            base_result = ingest_document(path, collection)
        except (ValueError, RuntimeError, OSError) as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1) from e

        prog.update(task, description="Building RAPTOR tree…")
        tree = build_raptor_tree(
            collection_name=collection,
            llm_fn=backend.complete_raw,
            max_levels=max_levels,
            cluster_size=cluster_size,
        )

        prog.update(task, description="Ingesting tree summaries…")
        tree_collection = f"{collection}_raptor"
        stored = ingest_raptor_tree(tree, tree_collection)
        prog.update(task, completed=True)

    console.print(Panel(
        f"[green]✓[/green] Base chunks:       [bold]{base_result.chunks_added}[/bold]\n"
        f"[cyan]✓[/cyan] RAPTOR levels:      [bold]{len(tree.levels)}[/bold]\n"
        f"[cyan]✓[/cyan] Summary chunks:     [bold]{stored}[/bold] → [yellow]{tree_collection}[/yellow]\n"
        f"[dim]∑[/dim] Total nodes:        [bold]{sum(len(nodes) for nodes in tree.levels.values())}[/bold]",
        title="[bold green]RAPTOR Ingestion Complete[/bold green]",
        border_style="green",
    ))
    console.print(
        f"[dim]Query both collections with:[/dim] "
        f"[cyan]--collection {collection}[/cyan] and [cyan]--collection {tree_collection}[/cyan]"
    )


# ── multimodal-ingest ─────────────────────────────────────────────────────────


@app.command(name="multimodal-ingest")
def multimodal_ingest(
    path: Annotated[str, typer.Option("--path", "-p", help="Path to PDF file")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    no_tables: Annotated[bool, typer.Option("--no-tables")] = False,
    no_figures: Annotated[bool, typer.Option("--no-figures")] = False,
    no_describe: Annotated[bool, typer.Option("--no-describe", help="Skip vision LLM description of figures")] = False,
) -> None:
    """
    [bold]Multi-modal ingestion[/bold] — extract tables + figures from PDFs.

    Tables → structured markdown. Figures/charts → searchable text descriptions
    generated by Claude's vision API. Both stored alongside text chunks.

    Requires: pip install pdfplumber pymupdf
    """
    from core.multimodal import ingest_pdf_multimodal

    _print_header("multi-modal ingestion")

    p = Path(path)
    if not p.exists() or not p.suffix.lower() == ".pdf":
        console.print(f"[red]Path must be an existing PDF file: {path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]PDF:[/bold] [cyan]{path}[/cyan] → [yellow]{collection!r}[/yellow]\n")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Extracting tables + figures…", total=None)
        summary = ingest_pdf_multimodal(
            pdf_path=path,
            collection_name=collection,
            extract_tables=not no_tables,
            extract_figures=not no_figures,
            describe_figures=not no_describe,
        )
        prog.update(task, completed=True)

    console.print(Panel(
        f"[green]✓[/green] Tables found:       [bold]{summary['tables_found']}[/bold]\n"
        f"[cyan]✓[/cyan] Figures found:      [bold]{summary['figures_found']}[/bold]\n"
        f"[cyan]✓[/cyan] Elements stored:    [bold]{summary['elements_stored']}[/bold]\n"
        f"Collection:          [yellow]{collection}[/yellow]",
        title="[bold green]Multi-Modal Ingestion Complete[/bold green]",
        border_style="green",
    ))


# ── feedback ──────────────────────────────────────────────────────────────────


feedback_app = typer.Typer(name="feedback", help="Feedback analytics and management")
app.add_typer(feedback_app)


@feedback_app.command(name="stats")
def feedback_stats(
    collection: Annotated[Optional[str], typer.Option("--collection", "-c")] = None,
) -> None:
    """Show feedback analytics: satisfaction rate, top failing queries, best/worst sources."""
    from core.feedback import get_feedback_store
    _print_header("feedback analytics")
    store = get_feedback_store()
    summary = store.get_summary(collection)

    console.print(Panel(
        f"Total feedback:     [bold]{summary.total_feedback}[/bold]\n"
        f"Thumbs up:          [green]{summary.thumbs_up}[/green]\n"
        f"Thumbs down:        [red]{summary.thumbs_down}[/red]\n"
        f"Satisfaction rate:  [bold]{'[green]' if summary.satisfaction_rate >= 0.7 else '[yellow]'}"
        f"{summary.satisfaction_rate:.1%}[/bold]\n"
        f"Corrections:        [bold]{summary.corrections_count}[/bold]",
        title="[bold cyan]Feedback Summary[/bold cyan]",
        border_style="cyan",
    ))

    if summary.top_failing_queries:
        tbl = Table(title="Top Failing Queries", header_style="bold red")
        tbl.add_column("#", width=4, style="dim")
        tbl.add_column("Question", style="red")
        for i, q in enumerate(summary.top_failing_queries, 1):
            tbl.add_row(str(i), q)
        console.print(tbl)

    if summary.top_helpful_sources:
        tbl = Table(title="Best Sources", header_style="bold green")
        tbl.add_column("#", width=4, style="dim")
        tbl.add_column("Source", style="green")
        for i, s in enumerate(summary.top_helpful_sources, 1):
            tbl.add_row(str(i), s)
        console.print(tbl)


@feedback_app.command(name="record")
def feedback_record(
    question: Annotated[str, typer.Option("--question", "-q")],
    answer: Annotated[str, typer.Option("--answer", "-a")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    thumbs: Annotated[str, typer.Option("--thumbs", help="up|down")] = "up",
    correction: Annotated[Optional[str], typer.Option("--correction")] = None,
) -> None:
    """Record user feedback on a RAG response."""
    from core.feedback import get_feedback_store, FeedbackEntry, FeedbackType
    store = get_feedback_store()

    ft = FeedbackType.THUMBS_UP if thumbs.lower() == "up" else FeedbackType.THUMBS_DOWN
    if correction:
        ft = FeedbackType.CORRECTION

    entry = FeedbackEntry(
        question=question, answer=answer, collection=collection,
        feedback_type=ft, correction=correction,
    )
    fid = store.record(entry)
    console.print(f"[green]✓[/green] Recorded feedback [dim]{fid}[/dim]")


@feedback_app.command(name="export")
def feedback_export(
    output: Annotated[str, typer.Option("--output", "-o")] = "feedback_export.jsonl",
    collection: Annotated[Optional[str], typer.Option("--collection", "-c")] = None,
) -> None:
    """Export all feedback to JSONL for offline analysis or fine-tuning."""
    from core.feedback import get_feedback_store
    store = get_feedback_store()
    n = store.export_jsonl(Path(output), collection)
    console.print(f"[green]✓[/green] Exported [bold]{n}[/bold] entries → [cyan]{output}[/cyan]")


# ── finetune ──────────────────────────────────────────────────────────────────


@app.command()
def finetune(
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 3,
    no_synthetic: Annotated[bool, typer.Option("--no-synthetic", help="Skip LLM-generated training pairs")] = False,
    no_feedback: Annotated[bool, typer.Option("--no-feedback", help="Skip feedback-derived pairs")] = False,
) -> None:
    """
    [bold]Fine-tune[/bold] the embedding model on domain-specific data.

    1. Collects training pairs from feedback corrections + synthetic LLM-generated Q&A
    2. Mines hard negatives (semantically similar but incorrect chunks)
    3. Fine-tunes with MultipleNegativesRankingLoss (MNR)
    4. Evaluates improvement via MRR vs baseline
    5. Saves model to ./data/finetuned_embeddings/

    Requires: pip install sentence-transformers[train]
    """
    from core.embedding_finetuner import run_finetuning_pipeline
    from core.generation import get_backend
    from core.ingestion import embed_texts

    _print_header("embedding fine-tuning")
    console.print(f"\nCollection: [yellow]{collection}[/yellow]  |  Epochs: [bold]{epochs}[/bold]\n")

    backend = get_backend()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Collecting training pairs…", total=None)
        try:
            results = run_finetuning_pipeline(
                collection_name=collection,
                llm_fn=backend.complete_raw,
                embed_fn=embed_texts,
                use_feedback=not no_feedback,
                use_synthetic=not no_synthetic,
                epochs=epochs,
            )
        except Exception as e:
            console.print(f"[red]Fine-tuning failed: {e}[/red]")
            raise typer.Exit(1) from e
        prog.update(task, completed=True)

    if "error" in results:
        console.print(f"[yellow]Warning:[/yellow] {results['error']}")
        raise typer.Exit(0)

    improved = results.get("improvement_pct", 0)
    imp_color = "green" if improved > 0 else "red"

    console.print(Panel(
        f"[green]✓[/green] Model saved:       [bold cyan]{results.get('model_path', '?')}[/bold cyan]\n"
        f"Training pairs:     [bold]{results.get('training_pairs', 0)}[/bold]\n"
        f"Test pairs:         [bold]{results.get('test_pairs', 0)}[/bold]\n"
        f"Baseline MRR:       [bold]{results.get('baseline_mrr', 0):.4f}[/bold]\n"
        f"Fine-tuned MRR:     [bold]{results.get('finetuned_mrr', 0):.4f}[/bold]\n"
        f"Improvement:        [{imp_color}][bold]{improved:+.2f}%[/bold][/{imp_color}]",
        title="[bold green]Fine-Tuning Complete[/bold green]",
        border_style="green",
    ))
    console.print(
        f"\n[dim]To use the fine-tuned model, set in .env:[/dim]\n"
        f"[cyan]EMBEDDING_MODEL={results.get('model_path', '')}[/cyan]\n"
    )


# ── agent ─────────────────────────────────────────────────────────────────────


@app.command()
def agent(
    question: Annotated[str, typer.Option("--question", "-q")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    max_iterations: Annotated[int, typer.Option("--max-iter")] = 8,
    show_tools: Annotated[bool, typer.Option("--show-tools")] = True,
) -> None:
    """
    [bold]Agentic RAG[/bold] — LLM decides which tools to call (docs, web, SQL, calculator).

    Uses Claude's native tool_use API. The agent chains tools until it has
    a complete answer: search_docs → search_web → query_sql → calculate.
    Requires ANTHROPIC_API_KEY.
    """
    from core.agent import run_agent
    from core.retrieval import retrieve

    _print_header("agentic RAG")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    def _retrieve_fn(req):
        from core.generation import get_backend
        return retrieve(req, generate_fn=get_backend().complete_raw)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Agent thinking…", total=None)
        result = run_agent(
            question=question,
            collection=collection,
            retrieve_fn=_retrieve_fn,
            max_iterations=max_iterations,
        )
        prog.update(task, completed=True)

    console.print(Panel(
        result.answer,
        title=f"[bold cyan]Agent Answer[/bold cyan]  [dim]·[/dim]  {result.iterations} iteration(s)  [dim]·[/dim]  {result.total_tokens} tokens  [dim]·[/dim]  {result.latency_ms:.0f}ms",
        border_style="cyan",
        padding=(1, 2),
    ))

    if show_tools and result.tool_calls:
        tbl = Table(title="Tool Calls", header_style="bold magenta", show_lines=True)
        tbl.add_column("Step", width=5, style="dim")
        tbl.add_column("Tool", style="cyan")
        tbl.add_column("Input", max_width=40)
        tbl.add_column("Result", max_width=50)
        tbl.add_column("ms", justify="right", width=7)
        for i, tc in enumerate(result.tool_calls, 1):
            inp = str(tc.tool_input)[:40]
            res = tc.result[:50].replace("\n", " ")
            tbl.add_row(str(i), tc.tool_name, inp, res, f"{tc.latency_ms:.0f}")
        console.print(tbl)


# ── sql-query ─────────────────────────────────────────────────────────────────


sql_app = typer.Typer(name="sql", help="Text-to-SQL structured data retrieval")
app.add_typer(sql_app)


@sql_app.command(name="query")
def sql_query(
    question: Annotated[str, typer.Option("--question", "-q")],
    database: Annotated[Optional[str], typer.Option("--database", "-d", help="DB URL or path")] = None,
) -> None:
    """Query a SQL database using natural language → SQL generation."""
    from core.sql_retrieval import query_natural_language

    _print_header("text-to-SQL")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Generating + executing SQL…", total=None)
        result = query_natural_language(question, database)
        prog.update(task, completed=True)

    console.print(Panel(result, title="[bold green]SQL Result[/bold green]", border_style="green"))


@sql_app.command(name="setup-sample")
def sql_setup_sample() -> None:
    """Create a sample SQLite database with products/customers/orders for demos."""
    from core.sql_retrieval import create_sample_db
    path = create_sample_db()
    console.print(f"[green]✓[/green] Sample database created: [cyan]{path}[/cyan]")
    console.print(f"[dim]Add to .env:[/dim] [cyan]SQL_DATABASE_URL=sqlite:///{path}[/cyan]")


@sql_app.command(name="schema")
def sql_schema(
    database: Annotated[Optional[str], typer.Option("--database", "-d")] = None,
) -> None:
    """Show the schema of the configured SQL database."""
    from core.sql_retrieval import get_schema
    schema = get_schema(database)
    console.print(Panel(schema, title="[bold]Database Schema[/bold]", border_style="yellow"))


# ── security ──────────────────────────────────────────────────────────────────


security_app = typer.Typer(name="security", help="PII, injection detection, and audit logs")
app.add_typer(security_app)


@security_app.command(name="audit")
def security_audit(
    days: Annotated[int, typer.Option("--days", "-d")] = 7,
) -> None:
    """Show security audit summary: PII, injection attempts, sensitive queries."""
    from core.security import get_audit_summary
    _print_header("security audit")
    summary = get_audit_summary(days)

    inj = summary.get("injection_attempts", 0)
    pii = summary.get("pii_in_queries", 0)
    inj_color = "red" if inj > 0 else "green"
    pii_color = "red" if pii > 0 else "green"

    console.print(Panel(
        f"Period:              [bold]{days}[/bold] days\n"
        f"Total queries:       [bold]{summary.get('total_queries', 0)}[/bold]\n"
        f"PII in queries:      [{pii_color}][bold]{pii}[/bold][/{pii_color}]\n"
        f"Injection attempts:  [{inj_color}][bold]{inj}[/bold][/{inj_color}]\n"
        f"Sensitive queries:   [bold]{summary.get('sensitive_queries', 0)}[/bold]\n"
        f"Answers with PII:    [bold]{summary.get('answers_with_pii', 0)}[/bold]\n"
        f"PII rate:            [bold]{summary.get('pii_rate', 0):.1%}[/bold]\n"
        f"Injection rate:      [bold]{summary.get('injection_rate', 0):.1%}[/bold]",
        title="[bold red]Security Audit[/bold red]",
        border_style="red",
    ))


@security_app.command(name="scan")
def security_scan(
    text: Annotated[str, typer.Option("--text", "-t", help="Text to scan for PII/injection")],
    redact: Annotated[bool, typer.Option("--redact")] = False,
) -> None:
    """Scan text for PII and prompt injection patterns."""
    from core.security import detect_pii, detect_injection, redact_pii
    _print_header("security scan")

    pii = redact_pii(text) if redact else detect_pii(text)
    inj = detect_injection(text)

    pii_color = "red" if pii.has_pii else "green"
    inj_color = "red" if inj.is_injection else "green"

    console.print(Panel(
        f"PII detected:   [{pii_color}][bold]{'YES' if pii.has_pii else 'NO'}[/bold][/{pii_color}]"
        + (f"  ({', '.join(pii.pii_types)})" if pii.pii_types else "") + "\n"
        f"Injection risk: [{inj_color}][bold]{'YES' if inj.is_injection else 'NO'}[/bold][/{inj_color}]"
        + (f"  (score={inj.risk_score:.1f})" if inj.is_injection else ""),
        title="[bold]Scan Results[/bold]",
        border_style=pii_color if pii.has_pii or inj.is_injection else "green",
    ))

    if redact and pii.has_pii:
        console.print(Panel(pii.redacted_text, title="[yellow]Redacted Text[/yellow]", border_style="yellow"))


# ── graph community commands ──────────────────────────────────────────────────


@graph_app.command(name="communities")
def graph_communities(
    summarize: Annotated[bool, typer.Option("--summarize", help="Generate LLM summaries for each community")] = False,
) -> None:
    """Detect entity communities in the knowledge graph (Microsoft GraphRAG style)."""
    from core.graph_rag import get_knowledge_graph
    _print_header("graph communities")
    graph = get_knowledge_graph()
    communities = graph.detect_communities()

    if not communities:
        console.print("[yellow]No communities detected. Ingest documents with --graph first.[/yellow]")
        return

    tbl = Table(title=f"{len(communities)} Communities", header_style="bold yellow")
    tbl.add_column("ID", width=5, style="dim")
    tbl.add_column("Size", justify="right", width=7)
    tbl.add_column("Top Entities", style="cyan")
    for cid, entities in sorted(communities.items(), key=lambda x: -len(x[1]))[:20]:
        tbl.add_row(str(cid), str(len(entities)), ", ".join(entities[:6]))
    console.print(tbl)

    if summarize:
        console.print("\n[dim]Generating community summaries…[/dim]")
        from core.generation import get_backend
        backend = get_backend()
        summaries = graph.build_community_summaries(backend.complete_raw)
        for cid, summary in list(summaries.items())[:5]:
            console.print(Panel(summary, title=f"[yellow]Community {cid}[/yellow]", border_style="dim yellow"))


@graph_app.command(name="global-query")
def graph_global_query(
    question: Annotated[str, typer.Option("--question", "-q")],
) -> None:
    """Answer a high-level question using GraphRAG community summaries."""
    from core.graph_rag import get_knowledge_graph
    from core.generation import get_backend
    _print_header("global graph query")

    graph = get_knowledge_graph()
    backend = get_backend()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Building community summaries…", total=None)
        summaries = graph.build_community_summaries(backend.complete_raw)
        prog.update(task, description="Querying communities…")
        answer = graph.global_query(question, summaries, backend.complete_raw)
        prog.update(task, completed=True)

    console.print(Panel(answer, title="[bold cyan]Global Graph Answer[/bold cyan]", border_style="cyan", padding=(1, 2)))


# ── query-structured ──────────────────────────────────────────────────────────


@app.command(name="query-structured")
def query_structured(
    question: Annotated[str, typer.Option("--question", "-q")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    schema: Annotated[str, typer.Option("--schema", "-s", help="JSON schema string or path to .json file")] = "{}",
) -> None:
    """
    [bold]Structured query[/bold] — extract typed JSON from retrieved context.

    Example schema: '{"revenue": "number", "period": "string", "currency": "string"}'
    """
    import json
    from core.generation import answer_structured

    _print_header("structured extraction")

    # Parse schema — accept inline JSON or file path
    try:
        schema_path = Path(schema)
        if schema_path.exists():
            with open(schema_path) as f:
                schema_dict = json.load(f)
        else:
            schema_dict = json.loads(schema)
    except (json.JSONDecodeError, OSError):
        schema_dict = {}

    request = QueryRequest(question=question, collection=collection, top_k=settings.top_k, mode=QueryMode.HYBRID)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Extracting structured data…", total=None)
        result = answer_structured(request, schema_dict)
        prog.update(task, completed=True)

    console.print(Panel(
        json.dumps(result, indent=2),
        title="[bold green]Structured Result[/bold green]",
        border_style="green",
    ))


# ── cot (Chain-of-Thought RAG) ────────────────────────────────────────────────


@app.command()
def cot(
    question: Annotated[str, typer.Option("--question", "-q")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    max_steps: Annotated[int, typer.Option("--max-steps")] = settings.cot_max_steps,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = settings.top_k,
    show_trace: Annotated[bool, typer.Option("--show-trace")] = True,
) -> None:
    """
    [bold]CoT-RAG[/bold] — Chain-of-Thought multi-hop reasoning (EMNLP 2025).

    Decomposes the question into reasoning steps, retrieves targeted context
    for each step, and synthesizes with the full reasoning trace visible.
    Best for complex multi-hop questions.
    """
    from core.cot_rag import run_cot_rag
    from core.retrieval import retrieve
    from core.generation import get_backend

    _print_header("CoT-RAG")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    backend = get_backend()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Decomposing into reasoning steps…", total=None)
        try:
            result = run_cot_rag(
                question=question,
                collection=collection,
                retrieve_fn=retrieve,
                llm_fn=backend.complete_raw,
                max_steps=max_steps,
                top_k_per_step=settings.cot_top_k_per_step,
            )
        except Exception as e:
            console.print(f"\n[red]CoT-RAG failed: {e}[/red]")
            raise typer.Exit(1) from e
        prog.update(task, completed=True)

    for w in result.warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")

    if show_trace and result.reasoning_steps:
        step_table = Table(title=f"Reasoning Trace ({result.num_steps} steps)", header_style="bold yellow", show_lines=True)
        step_table.add_column("#", width=4, style="dim")
        step_table.add_column("Thought", style="cyan", max_width=35)
        step_table.add_column("Chunks", justify="right", width=7)
        step_table.add_column("Finding", max_width=50)
        step_table.add_column("ms", justify="right", width=7)

        for step in result.reasoning_steps:
            step_table.add_row(
                str(step.step_number),
                step.thought[:35],
                str(len(step.retrieved)),
                step.intermediate[:50] + "…" if len(step.intermediate) > 50 else step.intermediate,
                f"{step.latency_ms:.0f}",
            )
        console.print(step_table)

    console.print(Panel(
        result.answer,
        title=f"[bold cyan]CoT Answer[/bold cyan]  [dim]·[/dim]  {result.num_steps} steps  [dim]·[/dim]  {result.total_chunks} chunks  [dim]·[/dim]  {result.latency_ms:.0f}ms",
        border_style="cyan",
        padding=(1, 2),
    ))

    if result.all_sources:
        console.print(f"[dim]Sources:[/dim] {', '.join(result.all_sources)}\n")


# ── ttrag ─────────────────────────────────────────────────────────────────────


@app.command()
def ttrag(
    question: Annotated[str, typer.Option("--question", "-q")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    max_iterations: Annotated[int, typer.Option("--max-iterations")] = 4,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = settings.top_k,
    threshold: Annotated[float, typer.Option("--threshold")] = 0.55,
    show_iterations: Annotated[bool, typer.Option("--show-iterations")] = True,
) -> None:
    """
    [bold]TTRAG[/bold] — Test-Time Compute Scaling for RAG (ICLR 2025).

    Iteratively rewrites the query and re-retrieves until sufficient context
    is found, instead of one-shot retrieval. More compute at inference time
    = better answers on hard questions.
    """
    from core.ttrag import run_ttrag
    from core.retrieval import retrieve
    from core.generation import get_backend, build_user_prompt, SYSTEM_PROMPT

    _print_header("TTRAG — Test-Time Compute Scaling")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    backend = get_backend()

    def _generate(q: str, ctx) -> tuple[str, int]:
        prompt = build_user_prompt(ctx)
        answer, tokens, _ = backend.complete(SYSTEM_PROMPT, prompt)
        return answer, tokens

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Iterative retrieval in progress…", total=None)
        try:
            result = run_ttrag(
                question=question,
                collection=collection,
                retrieve_fn=retrieve,
                llm_fn=backend.complete_raw,
                generate_fn=_generate,
                max_iterations=max_iterations,
                top_k=top_k,
                sufficiency_threshold=threshold,
            )
        except Exception as e:
            console.print(f"\n[red]TTRAG failed: {e}[/red]")
            raise typer.Exit(1) from e
        prog.update(task, completed=True)

    if show_iterations and result.iterations:
        iter_table = Table(
            title=f"Retrieval Iterations ({result.num_iterations})",
            header_style="bold magenta",
            show_lines=True,
        )
        iter_table.add_column("#", width=4, style="dim")
        iter_table.add_column("Query used", max_width=40, style="cyan")
        iter_table.add_column("New chunks", justify="right", width=10)
        iter_table.add_column("Sufficiency", justify="right", width=12)
        iter_table.add_column("ms", justify="right", width=7)

        for it in result.iterations:
            suf_pct = f"{it.sufficiency.overall_score:.0%}"
            suf_color = "green" if it.sufficiency.is_sufficient else ("yellow" if it.sufficiency.overall_score >= 0.35 else "red")
            iter_table.add_row(
                str(it.iteration),
                it.query_used[:40],
                str(len(it.retrieved)),
                f"[{suf_color}]{suf_pct}[/{suf_color}]",
                f"{it.latency_ms:.0f}",
            )
        console.print(iter_table)

    converged_str = "[green]converged[/green]" if result.converged else "[yellow]max iterations[/yellow]"
    console.print(Panel(
        result.answer,
        title=(
            f"[bold magenta]TTRAG Answer[/bold magenta]  [dim]·[/dim]  "
            f"{result.num_iterations} iter  [dim]·[/dim]  "
            f"{result.unique_chunks_used} unique chunks  [dim]·[/dim]  "
            f"{result.final_sufficiency:.0%} sufficiency  [dim]·[/dim]  "
            f"{converged_str}  [dim]·[/dim]  {result.latency_ms:.0f}ms"
        ),
        border_style="magenta",
        padding=(1, 2),
    ))


# ── lightrag ──────────────────────────────────────────────────────────────────


lightrag_app = typer.Typer(name="lightrag", help="LightRAG dual-level graph retrieval (EMNLP 2025)")
app.add_typer(lightrag_app)


@lightrag_app.command(name="query")
def lightrag_query(
    question: Annotated[str, typer.Option("--question", "-q")],
    level: Annotated[str, typer.Option("--level", "-l", help="low|high|auto|combined")] = "auto",
) -> None:
    """[bold]LightRAG query[/bold] — dual-level graph retrieval (entity/community)."""
    from core.light_rag import get_light_rag

    _print_header("LightRAG")
    console.print(f"\n[bold]Question:[/bold] {question}  [dim](level={level})[/dim]\n")

    lr = get_light_rag()

    if level == "low":
        chunks = lr.low_level_retrieve(question)
        entities = lr._match_entities(question)
        communities: list[str] = []
        resolved = "low"
    elif level == "high":
        chunks = lr.high_level_retrieve(question)
        entities = []
        communities = [f"community_{i}" for i in range(len(chunks))]
        resolved = "high"
    elif level == "combined":
        result = lr.combined_retrieve(question)
        chunks = result.context_chunks
        entities = result.entities_used
        communities = result.communities_used
        resolved = "combined"
    else:
        result = lr.auto_retrieve(question)
        chunks = result.context_chunks
        entities = result.entities_used
        communities = result.communities_used
        resolved = result.resolved_level

    if not chunks:
        console.print("[yellow]No results — run 'rag ingest --graph' first to build the knowledge graph.[/yellow]")
        return

    level_color = {"low": "green", "high": "cyan", "combined": "magenta"}.get(resolved, "white")
    console.print(f"[dim]Resolved level:[/dim] [{level_color}]{resolved}[/{level_color}]  [dim]|[/dim]  {len(chunks)} chunks  [dim]|[/dim]  {len(entities)} entities\n")

    for i, chunk in enumerate(chunks[:8], 1):
        console.print(Panel(chunk[:300], title=f"[dim]Result {i}[/dim]", border_style="dim", padding=(0, 1)))


@lightrag_app.command(name="stats")
def lightrag_stats() -> None:
    """Show LightRAG graph and index statistics."""
    from core.light_rag import get_light_rag
    _print_header("LightRAG stats")
    lr = get_light_rag()
    stats = lr.stats()

    tbl = Table(title="LightRAG Index", header_style="bold yellow")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right")
    for k, v in stats.items():
        tbl.add_row(k.replace("_", " ").title(), str(v))
    console.print(tbl)


@lightrag_app.command(name="update")
def lightrag_update(
    path: Annotated[str, typer.Option("--path", "-p", help="Document to extract triples from")],
) -> None:
    """Incrementally update the LightRAG graph with triples from a new document."""
    from core.light_rag import get_light_rag
    from core.graph_rag import extract_triples
    from core.generation import get_backend
    from core.ingestion import load_document

    _print_header("LightRAG incremental update")
    backend = get_backend()
    lr = get_light_rag()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Extracting triples…", total=None)
        try:
            pages, _ = load_document(path)
            all_triples: list[tuple[str, str, str]] = []
            for page_text, _ in pages[:10]:
                triples = extract_triples(page_text, path, backend.complete_raw)
                all_triples.extend((t.subject, t.predicate, t.obj) for t in triples)
            added = lr.incremental_update(all_triples, source=path)
        except Exception as e:
            console.print(f"\n[red]Update failed: {e}[/red]")
            raise typer.Exit(1) from e
        prog.update(task, completed=True)

    console.print(f"[green]✓[/green] Added [bold]{added}[/bold] new edges from [cyan]{path}[/cyan]")


# ── sufficiency (check context quality before generating) ─────────────────────


@app.command()
def check_context(
    question: Annotated[str, typer.Option("--question", "-q")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = settings.default_collection,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = settings.top_k,
    self_rating: Annotated[bool, typer.Option("--self-rating")] = False,
) -> None:
    """
    [bold]Sufficient Context check[/bold] — score whether retrieved context is sufficient before generating.

    Based on Google ICLR 2025. Returns a confidence score and recommendation:
    generate / retrieve_more / web_search / abstain.
    """
    from core.retrieval import retrieve
    from core.generation import get_backend, make_crag_evaluator
    from core.sufficient_context import check_sufficiency

    _print_header("sufficient context")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    backend = get_backend()
    req = QueryRequest(question=question, collection=collection, top_k=top_k, mode=QueryMode.HYBRID)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("Retrieving…", total=None)
        evaluate_fn = make_crag_evaluator(backend) if settings.use_hybrid_search else None
        context = retrieve(req, generate_fn=backend.complete_raw, evaluate_fn=evaluate_fn)
        prog.update(task, description="Scoring sufficiency…")
        result = check_sufficiency(
            question=question,
            context=context,
            llm_fn=backend.complete_raw if self_rating else None,
            enable_self_rating=self_rating,
        )
        prog.update(task, completed=True)

    reco_colors = {
        "generate":      "green",
        "retrieve_more": "yellow",
        "web_search":    "cyan",
        "abstain":       "red",
    }
    reco_color = reco_colors.get(result.recommendation, "white")

    console.print(Panel(
        f"Overall score:    [bold]{result.overall_score:.3f}[/bold]\n"
        f"Density:          [bold]{result.density_score:.3f}[/bold]\n"
        f"Coverage:         [bold]{result.coverage_score:.3f}[/bold]\n"
        f"Chunks retrieved: [bold]{result.num_chunks}[/bold]\n"
        f"Self-rating:      [bold]{result.self_rating if result.self_rating is not None else 'n/a'}[/bold]\n"
        f"Recommendation:   [{reco_color}][bold]{result.recommendation.upper()}[/bold][/{reco_color}]\n\n"
        f"[dim]{result.explanation}[/dim]",
        title=f"[bold]Context Sufficiency[/bold]  [dim]·[/dim]  {'Sufficient' if result.is_sufficient else 'Insufficient'}",
        border_style=reco_color,
    ))


if __name__ == "__main__":
    app()
