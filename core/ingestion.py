"""
Document ingestion pipeline.

Handles: PDF, TXT, DOCX, Markdown, URLs
Pipeline: load → chunk → embed → deduplicate → store in ChromaDB

Key design decisions:
- SentenceTransformer embeddings run locally (zero API cost)
- SHA-256 content hashing for exact deduplication
- Collection-per-knowledge-base for clean multi-tenant separation
- Lazy model loading (singleton) to avoid repeated GPU/CPU warm-up
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import chromadb
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from sentence_transformers import SentenceTransformer

from config import settings
from models import DocumentChunk, DocumentType, IngestResult
from utils.chunking import get_chunker

logger = logging.getLogger(__name__)

# ── Singleton embedding model ─────────────────────────────────────────────────
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer instance (lazy load on first call)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s' on device '%s'…", settings.embedding_model, settings.embedding_device)
        _embedding_model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
        logger.info("Embedding model loaded.")
    return _embedding_model


# ── ChromaDB client ───────────────────────────────────────────────────────────
_chroma_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Return cached ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        logger.info("ChromaDB client initialized at '%s'", settings.chroma_persist_dir)
    return _chroma_client


def get_or_create_collection(name: str) -> chromadb.Collection:
    """Get or create a named ChromaDB collection."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "embedding_model": settings.embedding_model,
            "hnsw:space": "cosine",
        },
    )
    return collection


# ── Document loaders ──────────────────────────────────────────────────────────


def _detect_doc_type(path: str) -> DocumentType:
    """Infer document type from extension or URL scheme."""
    lower = path.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        return DocumentType.URL
    suffix = Path(lower).suffix
    return {
        ".pdf": DocumentType.PDF,
        ".txt": DocumentType.TXT,
        ".docx": DocumentType.DOCX,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
    }.get(suffix, DocumentType.UNKNOWN)


def _load_url(url: str) -> list[tuple[str, int | None]]:
    """
    Scrape a URL and return (text, page_number=None) tuples.
    Strips boilerplate using BeautifulSoup; respects a 10-second timeout.
    """
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "RAGBot/1.0"})
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch URL '{url}': {e}") from e

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return [(text, None)]


def _load_pdf(path: str) -> list[tuple[str, int | None]]:
    """Load PDF pages using LangChain's PyPDFLoader. Returns (text, page_num) per page."""
    try:
        loader = PyPDFLoader(path)
        pages = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load PDF '{path}': {e}") from e
    return [(page.page_content, page.metadata.get("page", i) + 1) for i, page in enumerate(pages)]


def _load_docx(path: str) -> list[tuple[str, int | None]]:
    """Load DOCX using LangChain's Docx2txtLoader."""
    try:
        loader = Docx2txtLoader(path)
        docs = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load DOCX '{path}': {e}") from e
    return [(doc.page_content, None) for doc in docs]


def _load_text(path: str) -> list[tuple[str, int | None]]:
    """Load plain text / markdown files."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        raise ValueError(f"Failed to read file '{path}': {e}") from e
    return [(content, None)]


def load_document(source: str) -> tuple[list[tuple[str, int | None]], DocumentType]:
    """
    Load a document from any supported source.

    Returns:
        Tuple of (page_contents, doc_type) where page_contents is a list
        of (text, page_number) tuples.
    """
    doc_type = _detect_doc_type(source)

    dispatch = {
        DocumentType.PDF: _load_pdf,
        DocumentType.URL: _load_url,
        DocumentType.DOCX: _load_docx,
        DocumentType.TXT: _load_text,
        DocumentType.MARKDOWN: _load_text,
        DocumentType.UNKNOWN: _load_text,
    }

    loader_fn = dispatch[doc_type]
    pages = loader_fn(source)
    logger.info("Loaded %d page(s) from '%s' (type=%s)", len(pages), source, doc_type.value)
    return pages, doc_type


# ── Embedding ─────────────────────────────────────────────────────────────────


def embed_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """
    Generate embeddings for a list of chunks in batch.
    Mutates chunks in-place (sets .embedding) and returns them.
    """
    model = get_embedding_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb.tolist()
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of raw strings. Used by retrieval and HyDE."""
    model = get_embedding_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


# ── Deduplication ─────────────────────────────────────────────────────────────


def _get_existing_hashes(collection: chromadb.Collection) -> set[str]:
    """Fetch all content hashes already stored in the collection."""
    try:
        result = collection.get(include=["metadatas"])
        return {
            m.get("content_hash", "") for m in (result.get("metadatas") or []) if m
        }
    except Exception as e:
        logger.warning("Could not fetch existing hashes: %s", e)
        return set()


# ── Core ingestion logic ──────────────────────────────────────────────────────


def ingest_document(
    source: str,
    collection_name: str = "default",
    overwrite: bool = False,
    chunking_strategy: str = "recursive",
    use_contextual_retrieval: bool | None = None,
) -> IngestResult:
    """
    Full ingestion pipeline for a single file or URL.

    Steps:
      1. Load document pages
      2. Chunk each page
      3. Embed chunks
      4. Deduplicate against existing collection
      5. Upsert new chunks into ChromaDB

    Args:
        source: file path or URL
        collection_name: target ChromaDB collection
        overwrite: if True, skip deduplication and force re-ingest
        chunking_strategy: "recursive" | "semantic" | "hierarchical"
        use_contextual_retrieval: prepend LLM context to each chunk (Anthropic Nov 2024).
            None = use settings.use_contextual_retrieval

    Returns:
        IngestResult with counts and timing
    """
    start_time = time.perf_counter()
    collection = get_or_create_collection(collection_name)

    # Resolve contextual retrieval flag
    _use_contextual = use_contextual_retrieval
    if _use_contextual is None:
        _use_contextual = settings.use_contextual_retrieval

    # Load
    pages, doc_type = load_document(source)

    # Build chunker (pass embed_fn for semantic strategy)
    embed_fn = embed_texts if chunking_strategy == "semantic" else None
    chunker = get_chunker(
        strategy=chunking_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embed_fn=embed_fn,
    )

    # Chunk all pages
    all_chunks: list[DocumentChunk] = []
    for page_text, page_number in pages:
        if not page_text.strip():
            continue
        page_chunks = chunker.chunk(
            text=page_text,
            source_file=source,
            doc_type=doc_type,
            page_number=page_number,
        )
        all_chunks.extend(page_chunks)

    if not all_chunks:
        logger.warning("No chunks produced from '%s'", source)
        return IngestResult(
            collection=collection_name,
            source=source,
            chunks_added=0,
            duplicates_skipped=0,
            total_chunks_processed=0,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    total = len(all_chunks)
    logger.info("Produced %d chunks from '%s'", total, source)

    # Deduplicate
    if not overwrite:
        existing_hashes = _get_existing_hashes(collection)
        new_chunks = [c for c in all_chunks if c.metadata.content_hash not in existing_hashes]
        skipped = total - len(new_chunks)
    else:
        new_chunks = all_chunks
        skipped = 0

    if not new_chunks:
        logger.info("All %d chunks already exist in '%s'. Skipping.", total, collection_name)
        return IngestResult(
            collection=collection_name,
            source=source,
            chunks_added=0,
            duplicates_skipped=skipped,
            total_chunks_processed=total,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    # ── Contextual Retrieval (Anthropic Nov 2024) ─────────────────────────────
    # Prepend LLM-generated context to each chunk before embedding.
    # Reduces retrieval failures by ~49% (Anthropic published result).
    if _use_contextual:
        try:
            from core.contextual_retrieval import contextualize_chunks
            from core.generation import get_backend
            backend = get_backend()
            full_doc_text = "\n\n".join(text for text, _ in pages if text.strip())
            chunk_texts = [c.text for c in new_chunks]
            contextualized = contextualize_chunks(
                chunks=chunk_texts,
                document_text=full_doc_text,
                llm_fn=backend.complete_raw,
                use_cheap_model=settings.contextual_retrieval_use_haiku,
            )
            for chunk, ctx_text in zip(new_chunks, contextualized):
                chunk.text = ctx_text
            logger.info("Contextual retrieval: enhanced %d chunks", len(new_chunks))
        except Exception as e:
            logger.warning("Contextual retrieval failed (continuing without): %s", e)

    # ── PII Redaction ─────────────────────────────────────────────────────────
    if settings.enable_pii_redaction:
        try:
            from core.security import redact_pii
            total_redactions = 0
            for chunk in new_chunks:
                result = redact_pii(chunk.text, use_presidio=settings.enable_pii_presidio)
                if result.has_pii:
                    chunk.text = result.redacted_text
                    total_redactions += result.redaction_count
            if total_redactions:
                logger.info("PII redaction: %d redactions across %d chunks", total_redactions, len(new_chunks))
        except Exception as e:
            logger.warning("PII redaction failed: %s", e)

    # Embed
    new_chunks = embed_chunks(new_chunks)

    # Prepare ChromaDB upsert payload
    ids = [c.chunk_id for c in new_chunks]
    embeddings = [c.embedding for c in new_chunks]  # type: ignore[misc]
    documents = [c.text for c in new_chunks]
    metadatas = []
    for c in new_chunks:
        meta = c.metadata.model_dump()
        # ChromaDB requires flat dict with primitive values
        meta["timestamp_ingested"] = meta["timestamp_ingested"].isoformat()
        meta["doc_type"] = meta["doc_type"].value if hasattr(meta["doc_type"], "value") else str(meta["doc_type"])
        meta["page_number"] = meta["page_number"] if meta["page_number"] is not None else -1
        meta["section_title"] = meta["section_title"] or ""
        metadatas.append(meta)

    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as e:
        raise RuntimeError(f"ChromaDB upsert failed for '{source}': {e}") from e

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Ingested %d new chunks into '%s' (skipped %d duplicates) in %.2fs",
        len(new_chunks), collection_name, skipped, elapsed,
    )

    # Invalidate BM25 cache for this collection so next query rebuilds the index
    if len(new_chunks) > 0:
        try:
            from core.retrieval import invalidate_bm25_cache
            invalidate_bm25_cache(collection_name)
        except Exception:
            pass  # non-fatal — old index will still work, just stale

    return IngestResult(
        collection=collection_name,
        source=source,
        chunks_added=len(new_chunks),
        duplicates_skipped=skipped,
        total_chunks_processed=total,
        elapsed_seconds=elapsed,
    )


def ingest_directory(
    dir_path: str,
    collection_name: str = "default",
    overwrite: bool = False,
    chunking_strategy: str = "recursive",
    extensions: list[str] | None = None,
) -> list[IngestResult]:
    """
    Recursively ingest all supported documents in a directory.

    Args:
        dir_path: root directory to scan
        collection_name: target ChromaDB collection
        overwrite: force re-ingest even if duplicates exist
        chunking_strategy: chunking strategy to use
        extensions: file extensions to include (default: all supported types)

    Returns:
        List of IngestResult, one per file
    """
    supported = extensions or [".pdf", ".txt", ".docx", ".md", ".markdown"]
    root = Path(dir_path)

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: '{dir_path}'")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: '{dir_path}'")

    files = [p for p in root.rglob("*") if p.suffix.lower() in supported and p.is_file()]

    if not files:
        logger.warning("No supported files found in '%s'", dir_path)
        return []

    logger.info("Found %d files to ingest in '%s'", len(files), dir_path)
    results = []
    for file_path in files:
        try:
            result = ingest_document(
                source=str(file_path),
                collection_name=collection_name,
                overwrite=overwrite,
                chunking_strategy=chunking_strategy,
            )
            results.append(result)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Failed to ingest '%s': %s", file_path, e)

    return results


def list_collections() -> list[dict]:
    """Return metadata for all collections in the ChromaDB store."""
    client = get_chroma_client()
    collections = client.list_collections()
    result = []
    for col in collections:
        try:
            count = col.count()
            meta = col.metadata or {}
            result.append({
                "name": col.name,
                "document_count": count,
                "embedding_model": meta.get("embedding_model", "unknown"),
            })
        except Exception as e:
            logger.warning("Could not get info for collection '%s': %s", col.name, e)
    return result


def delete_collection(name: str) -> bool:
    """Delete a named collection. Returns True if deleted, False if not found."""
    client = get_chroma_client()
    try:
        client.delete_collection(name)
        logger.info("Deleted collection '%s'", name)
        return True
    except ValueError:
        logger.warning("Collection '%s' not found.", name)
        return False
    except Exception as e:
        raise RuntimeError(f"Failed to delete collection '{name}': {e}") from e
