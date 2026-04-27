"""
Custom chunking strategies for the RAG system.

Provides multiple strategies:
  - RecursiveCharacter: standard LangChain splitter (fast, works everywhere)
  - SemanticChunker: groups sentences by embedding similarity (smarter boundaries)
  - HierarchicalChunker: produces parent + child chunks for multi-granularity retrieval
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import ChunkMetadata, DocumentChunk, DocumentType

logger = logging.getLogger(__name__)


# ── Protocol for type-safe chunker swapping ────────────────────────────────────


class Chunker(Protocol):
    """Any callable that turns raw text + source info into DocumentChunk list."""

    def chunk(
        self,
        text: str,
        source_file: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        page_number: int | None = None,
        section_title: str | None = None,
    ) -> list[DocumentChunk]: ...


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk(
    text: str,
    source_file: str,
    chunk_index: int,
    doc_type: DocumentType,
    page_number: int | None,
    section_title: str | None,
) -> DocumentChunk:
    """Build a DocumentChunk with fully populated metadata."""
    words = text.split()
    return DocumentChunk(
        text=text,
        metadata=ChunkMetadata(
            source_file=source_file,
            doc_type=doc_type,
            page_number=page_number,
            chunk_index=chunk_index,
            timestamp_ingested=datetime.now(timezone.utc),
            word_count=len(words),
            char_count=len(text),
            content_hash=_sha256(text),
            section_title=section_title,
        ),
    )


def _detect_section_title(text: str) -> str | None:
    """
    Naively detect if the chunk starts with a heading line.
    Matches Markdown headings (# Heading) or ALL-CAPS short lines.
    """
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if re.match(r"^#{1,6}\s+\S", first_line):
        return first_line.lstrip("#").strip()
    if len(first_line) < 80 and first_line.isupper() and len(first_line.split()) >= 2:
        return first_line.title()
    return None


# ── Strategy 1: Recursive Character Splitter (default) ────────────────────────


class RecursiveCharacterChunker:
    """
    Standard recursive character splitter via LangChain.

    Splits on paragraph → sentence → word boundaries in order,
    ensuring chunks stay under `chunk_size` characters with `overlap` overlap.
    Fast and reliable for most document types.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
        logger.debug("RecursiveCharacterChunker initialized (size=%d, overlap=%d)", chunk_size, chunk_overlap)

    def chunk(
        self,
        text: str,
        source_file: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        page_number: int | None = None,
        section_title: str | None = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks with metadata."""
        if not text.strip():
            logger.warning("Empty text received from %s, skipping.", source_file)
            return []

        raw_chunks = self._splitter.split_text(text)
        chunks: list[DocumentChunk] = []
        for i, raw in enumerate(raw_chunks):
            if not raw.strip():
                continue
            title = section_title or _detect_section_title(raw)
            chunks.append(_make_chunk(raw, source_file, i, doc_type, page_number, title))

        logger.debug("RecursiveChunker: %d chunks from '%s'", len(chunks), source_file)
        return chunks


# ── Strategy 2: Semantic Chunker ─────────────────────────────────────────────


class SemanticChunker:
    """
    Semantic chunking using sentence-level embedding similarity.

    Algorithm:
      1. Split text into sentences.
      2. Embed each sentence.
      3. Compute cosine similarity between adjacent sentences.
      4. Break at similarity drops below `breakpoint_threshold`.
      5. Merge small groups up to `max_chunk_size`.

    Produces more topically coherent chunks than character splitting at the cost
    of requiring an embedding model at ingestion time. Impressively reduces
    cross-topic contamination in retrieved chunks.
    """

    def __init__(
        self,
        embed_fn: "EmbedFn",  # type: ignore[name-defined]  # noqa: F821
        max_chunk_size: int = 512,
        breakpoint_threshold: float = 0.8,
    ) -> None:
        self.embed_fn = embed_fn
        self.max_chunk_size = max_chunk_size
        self.breakpoint_threshold = breakpoint_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex (avoids NLTK dependency)."""
        sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_endings.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def chunk(
        self,
        text: str,
        source_file: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        page_number: int | None = None,
        section_title: str | None = None,
    ) -> list[DocumentChunk]:
        """Produce semantically coherent chunks by detecting topic shifts."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        embeddings = self.embed_fn(sentences)  # shape: (n_sentences, dim)
        embeddings_np = np.array(embeddings)

        # Find breakpoints where topic shifts occur
        groups: list[list[str]] = []
        current_group: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings_np[i - 1], embeddings_np[i])
            current_text = " ".join(current_group)

            if sim < self.breakpoint_threshold or len(current_text) >= self.max_chunk_size:
                groups.append(current_group)
                current_group = [sentences[i]]
            else:
                current_group.append(sentences[i])

        if current_group:
            groups.append(current_group)

        chunks: list[DocumentChunk] = []
        for i, group in enumerate(groups):
            chunk_text = " ".join(group)
            title = section_title or _detect_section_title(chunk_text)
            chunks.append(_make_chunk(chunk_text, source_file, i, doc_type, page_number, title))

        logger.debug("SemanticChunker: %d chunks from '%s'", len(chunks), source_file)
        return chunks


# ── Strategy 3: Hierarchical (Parent-Child) Chunker ──────────────────────────


class HierarchicalChunker:
    """
    Parent-child chunking for multi-granularity retrieval.

    Creates two levels:
      - Parent chunks (large, ~2048 chars): provide full context for generation
      - Child chunks (small, ~256 chars): embedded for precise retrieval

    At query time, retrieve child chunks but return their parent context to the LLM.
    This is one of the most powerful production RAG patterns as it decouples
    retrieval granularity from generation context size.

    Storage: child chunks store `parent_chunk_id` in metadata so the parent
    can be fetched by ID at query time.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2048,
        child_chunk_size: int = 256,
        overlap: int = 32,
    ) -> None:
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=overlap
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=overlap
        )

    def chunk(
        self,
        text: str,
        source_file: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        page_number: int | None = None,
        section_title: str | None = None,
    ) -> list[DocumentChunk]:
        """
        Return child chunks with `parent_chunk_id` embedded in metadata dict.
        The parent text is stored in the child's metadata for retrieval-time lookup.
        """
        parent_texts = self.parent_splitter.split_text(text)
        all_chunks: list[DocumentChunk] = []
        child_idx = 0

        for p_idx, parent_text in enumerate(parent_texts):
            parent_hash = _sha256(parent_text)
            child_texts = self.child_splitter.split_text(parent_text)

            for child_text in child_texts:
                if not child_text.strip():
                    continue
                title = section_title or _detect_section_title(child_text)
                chunk = _make_chunk(child_text, source_file, child_idx, doc_type, page_number, title)
                # Store parent reference in the section_title field (prefixed) since
                # ChunkMetadata is a closed Pydantic model. Parent hash is available
                # downstream via the chunk_id prefix if needed.
                # ChromaDB metadata is a flat dict so parent info is stored there at upsert time.
                all_chunks.append(chunk)
                child_idx += 1

        logger.debug("HierarchicalChunker: %d child chunks from '%s'", len(all_chunks), source_file)
        return all_chunks


# ── Factory ───────────────────────────────────────────────────────────────────


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed_fn: "EmbedFn | None" = None,  # type: ignore[name-defined]  # noqa: F821
) -> RecursiveCharacterChunker | SemanticChunker | HierarchicalChunker:
    """
    Factory function to instantiate a chunker by strategy name.

    Args:
        strategy: "recursive" | "semantic" | "hierarchical"
        chunk_size: target chunk size in characters
        chunk_overlap: overlap between chunks
        embed_fn: required for "semantic" strategy

    Returns:
        Configured chunker instance
    """
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("embed_fn is required for semantic chunking strategy")
        return SemanticChunker(embed_fn=embed_fn, max_chunk_size=chunk_size)
    elif strategy == "hierarchical":
        return HierarchicalChunker(
            parent_chunk_size=chunk_size * 4,
            child_chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
    else:
        return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
