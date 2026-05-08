"""
Personal Second Brain — a dedicated knowledge collection inside the RAG system.

Features:
- Add notes (text), files (PDF/MD/TXT/DOCX), or URLs with custom tags
- Time-aware queries: filter retrieval by how recently content was added
- Tag-filtered retrieval: only search content matching specific tags
- Source browser: list everything in your brain with metadata
- Daily digest: summarize recent additions with the LLM
- Folder watcher: auto-ingest files dropped into a watched directory
"""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import settings
from core.ingestion import get_chroma_client, get_embedding_model, ingest_document
from models import (
    IngestResult,
    QueryMode,
    QueryResponse,
    RetrievalContext,
    RetrievalResult,
    SourceCitation,
)

logger = logging.getLogger(__name__)

BRAIN_COLLECTION = "second_brain"


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_brain_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=BRAIN_COLLECTION,
        metadata={"embedding_model": settings.embedding_model, "hnsw:space": "cosine"},
    )


def _stamp_metadata(source: str, brain_type: str, title: str, tags: list[str]) -> None:
    """Update all chunks from `source` with brain-specific metadata fields."""
    col = _get_brain_collection()
    try:
        results = col.get(where={"source_file": source}, include=["metadatas"])
    except Exception:
        results = col.get(include=["metadatas"])
        results["ids"] = [
            rid for rid, m in zip(results["ids"], results["metadatas"])
            if m.get("source_file") == source
        ]
        results["metadatas"] = [m for m in results["metadatas"] if m.get("source_file") == source]

    if not results["ids"]:
        return

    tags_str = ",".join(t.lower().strip() for t in tags if t.strip())
    stamp = {
        "brain_type": brain_type,
        "brain_title": title or source,
        "brain_tags": tags_str,
        "brain_ingested_at": int(time.time()),
    }
    updated = [{**m, **stamp} for m in results["metadatas"]]
    col.update(ids=results["ids"], metadatas=updated)


def _build_context(question: str, docs: list[str], metas: list[dict], distances: list[float]) -> RetrievalContext:
    """Build a RetrievalContext from raw ChromaDB query results."""
    results = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        score = max(0.0, min(1.0, 1.0 - dist))
        results.append(RetrievalResult(
            chunk_text=doc,
            source=meta.get("source_file", meta.get("brain_title", "brain")),
            similarity_score=score,
            chunk_index=i,
            page_number=meta.get("page_number") if meta.get("page_number", -1) != -1 else None,
            section_title=meta.get("section_title") or None,
            metadata=meta,
        ))
    return RetrievalContext(query=question, results=results, query_mode=QueryMode.HYBRID)


# ── Public API ────────────────────────────────────────────────────────────────


def add_note(text: str, title: str = "", tags: list[str] | None = None) -> IngestResult:
    """Ingest a plain-text note into the Second Brain."""
    tags = tags or []
    safe_title = (title or "note")[:40].replace("/", "-").replace(" ", "_")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix=f"brain_{safe_title}_", delete=False, encoding="utf-8"
    ) as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write(text)
        tmp_path = f.name

    try:
        result = ingest_document(tmp_path, collection_name=BRAIN_COLLECTION)
        _stamp_metadata(tmp_path, brain_type="note", title=title or "Note", tags=tags)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


def add_source(path_or_url: str, tags: list[str] | None = None, title: str = "") -> IngestResult:
    """Ingest a file path or URL into the Second Brain."""
    tags = tags or []
    is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")
    brain_type = "url" if is_url else "file"
    derived_title = title or (path_or_url if is_url else Path(path_or_url).name)

    result = ingest_document(path_or_url, collection_name=BRAIN_COLLECTION)
    _stamp_metadata(path_or_url, brain_type=brain_type, title=derived_title, tags=tags)
    return result


def query_brain(
    question: str,
    tags: list[str] | None = None,
    days: int | None = None,
    top_k: int = 8,
) -> QueryResponse:
    """
    Query the Second Brain with optional tag and time filters.

    Directly queries ChromaDB with a time where-clause, then applies
    tag filtering in Python, then runs LLM generation on filtered chunks.
    """
    import time as _time
    from core.generation import get_backend, build_user_prompt, SYSTEM_PROMPT

    tags = tags or []
    col = _get_brain_collection()

    if col.count() == 0:
        return QueryResponse(
            question=question,
            answer="Your Second Brain is empty. Add some notes, files, or URLs first.",
            tokens_used=0,
            latency_ms=0.0,
            collection=BRAIN_COLLECTION,
            llm_backend=settings.llm_backend.value,
            model_used="",
        )

    start = _time.perf_counter()
    model = get_embedding_model()
    q_emb = model.encode([question], normalize_embeddings=True)[0].tolist()

    # Build where clause for time filter
    where: dict | None = None
    if days is not None:
        cutoff = int(_time.time()) - days * 86400
        where = {"brain_ingested_at": {"$gte": cutoff}}

    n_results = min(top_k * 3, max(col.count(), 1))
    try:
        raw = col.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        # where clause can fail if no docs have that metadata field
        logger.warning("Brain query with where clause failed (%s), falling back to unfiltered", e)
        raw = col.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    docs = raw["documents"][0] if raw["documents"] else []
    metas = raw["metadatas"][0] if raw["metadatas"] else []
    dists = raw["distances"][0] if raw["distances"] else []

    # Python-side tag filter
    if tags:
        filtered = [
            (d, m, s)
            for d, m, s in zip(docs, metas, dists)
            if any(t.lower() in m.get("brain_tags", "").lower() for t in tags)
        ]
        if filtered:
            docs, metas, dists = zip(*filtered)  # type: ignore[assignment]
        else:
            docs, metas, dists = [], [], []

    docs = list(docs)[:top_k]
    metas = list(metas)[:top_k]
    dists = list(dists)[:top_k]

    if not docs:
        return QueryResponse(
            question=question,
            answer="No matching content found in your Second Brain for those filters. Try broadening your tags or time range.",
            tokens_used=0,
            latency_ms=(_time.perf_counter() - start) * 1000,
            collection=BRAIN_COLLECTION,
            llm_backend=settings.llm_backend.value,
            model_used="",
        )

    context = _build_context(question, docs, metas, dists)
    backend = get_backend()
    user_prompt = build_user_prompt(context)
    answer, tokens, model_used = backend.complete(SYSTEM_PROMPT, user_prompt)

    sources = [
        SourceCitation(
            source=r.source,
            chunk_index=r.chunk_index,
            page_number=r.page_number,
            similarity_score=r.similarity_score,
            excerpt=r.chunk_text[:200],
        )
        for r in context.results
    ]

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        tokens_used=tokens,
        latency_ms=(_time.perf_counter() - start) * 1000,
        collection=BRAIN_COLLECTION,
        llm_backend=settings.llm_backend.value,
        model_used=model_used,
        retrieval_context=context,
    )


def list_sources(
    tags: list[str] | None = None,
    days: int | None = None,
    limit: int = 50,
) -> list[dict]:
    """Return deduplicated source entries from the brain, newest first."""
    col = _get_brain_collection()
    if col.count() == 0:
        return []

    results = col.get(include=["metadatas"])
    seen: set[str] = set()
    sources: list[dict] = []
    cutoff = (int(time.time()) - days * 86400) if days else 0

    for meta in results["metadatas"]:
        src_key = meta.get("source_file", "")
        if src_key in seen:
            continue
        seen.add(src_key)

        ingested_at = meta.get("brain_ingested_at", 0)
        if days and ingested_at < cutoff:
            continue

        if tags:
            chunk_tags = meta.get("brain_tags", "")
            if not any(t.lower() in chunk_tags.lower() for t in tags):
                continue

        sources.append(meta)

    sources.sort(key=lambda m: m.get("brain_ingested_at", 0), reverse=True)
    return sources[:limit]


def get_all_tags() -> dict[str, int]:
    """Return {tag: chunk_count} for all tags in the brain."""
    col = _get_brain_collection()
    if col.count() == 0:
        return {}

    results = col.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in results["metadatas"]:
        for tag in meta.get("brain_tags", "").split(","):
            tag = tag.strip()
            if tag:
                counts[tag] = counts.get(tag, 0) + 1
    return counts


def daily_digest(days: int = 1) -> str:
    """Ask the LLM to summarize what was added to the brain in the last N days."""
    from core.generation import get_backend

    sources = list_sources(days=days, limit=30)
    if not sources:
        label = "today" if days == 1 else f"the last {days} days"
        return f"Nothing was added to your Second Brain in {label}."

    lines = []
    seen: set[str] = set()
    for m in sources:
        title = m.get("brain_title", m.get("source_file", "Unknown"))
        if title in seen:
            continue
        seen.add(title)
        brain_type = m.get("brain_type", "item")
        tags = m.get("brain_tags", "")
        ts = m.get("brain_ingested_at", 0)
        dt = datetime.fromtimestamp(ts).strftime("%b %d %H:%M") if ts else "unknown time"
        lines.append(f"- [{brain_type}] {title} (added {dt})" + (f" — tags: {tags}" if tags else ""))

    items_list = "\n".join(lines)
    label = "today" if days == 1 else f"the last {days} days"
    prompt = (
        f"Here are the items added to my personal knowledge base in {label}:\n\n"
        f"{items_list}\n\n"
        "Write a brief, friendly digest summarizing what was captured and any interesting patterns "
        "or themes across the new content. Keep it under 200 words."
    )

    backend = get_backend()
    answer, _, _ = backend.complete(
        "You are a helpful personal knowledge assistant.",
        prompt,
    )
    return answer


def watch_folder(
    directory: str | Path,
    tags: list[str] | None = None,
    poll_interval: float = 2.0,
) -> None:
    """
    Watch a directory and auto-ingest any new files into the Second Brain.
    Blocks until KeyboardInterrupt. Run in a background thread for non-blocking use.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        raise ImportError(
            "watchdog is required for folder watching. Install it with: pip install watchdog"
        )

    directory = Path(directory)
    tags = tags or []
    supported = {".pdf", ".txt", ".md", ".docx", ".markdown"}

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in supported:
                return
            logger.info("Brain watcher: new file detected: %s", path)
            try:
                result = add_source(str(path), tags=tags)
                logger.info("Brain watcher: ingested %d chunks from %s", result.chunks_added, path.name)
            except Exception as e:
                logger.error("Brain watcher: failed to ingest %s: %s", path, e)

    observer = Observer()
    observer.schedule(_Handler(), str(directory), recursive=False)
    observer.start()
    logger.info("Brain watcher started on '%s'. Press Ctrl+C to stop.", directory)
    try:
        while observer.is_alive():
            observer.join(timeout=poll_interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logger.info("Brain watcher stopped.")
