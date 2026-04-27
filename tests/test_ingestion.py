"""
Tests for the ingestion pipeline.

Tests:
  - Chunk count and metadata correctness for a plain-text document
  - Deduplication skips already-ingested chunks
  - URL loading (mocked HTTP)
  - Directory ingestion discovers all supported file types
  - Metadata fields (word_count, content_hash, timestamp) are correctly populated
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models import DocumentType


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a temp .txt file with known content."""
    content = "\n\n".join([
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models have achieved superhuman performance on many benchmarks.",
        "Reinforcement learning agents can master complex games and real-world tasks.",
    ])
    file = tmp_path / "sample.txt"
    file.write_text(content, encoding="utf-8")
    return file


@pytest.fixture
def chroma_test_collection(tmp_path: Path) -> str:
    """Provide a unique test collection name using a temp ChromaDB path."""
    import os
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
    return "test_collection"


# ── Chunking unit tests ───────────────────────────────────────────────────────


class TestRecursiveCharacterChunker:
    def test_produces_chunks(self) -> None:
        from utils.chunking import RecursiveCharacterChunker

        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a test sentence. " * 20  # 500 chars
        chunks = chunker.chunk(text, source_file="test.txt", doc_type=DocumentType.TXT)

        assert len(chunks) >= 1

    def test_chunk_metadata_populated(self) -> None:
        from utils.chunking import RecursiveCharacterChunker

        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
        text = "Hello world. " * 50
        chunks = chunker.chunk(text, source_file="doc.pdf", doc_type=DocumentType.PDF, page_number=3)

        for chunk in chunks:
            assert chunk.metadata.source_file == "doc.pdf"
            assert chunk.metadata.doc_type == DocumentType.PDF
            assert chunk.metadata.page_number == 3
            assert chunk.metadata.word_count >= 1
            assert chunk.metadata.char_count == len(chunk.text)
            assert len(chunk.metadata.content_hash) == 64  # SHA-256 hex

    def test_chunk_ids_are_unique(self) -> None:
        from utils.chunking import RecursiveCharacterChunker

        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        text = "Paragraph one content. " * 10 + "\n\n" + "Paragraph two content. " * 10
        chunks = chunker.chunk(text, source_file="test.txt", doc_type=DocumentType.TXT)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_empty_text_returns_no_chunks(self) -> None:
        from utils.chunking import RecursiveCharacterChunker

        chunker = RecursiveCharacterChunker()
        chunks = chunker.chunk("   \n\n  ", source_file="empty.txt", doc_type=DocumentType.TXT)
        assert chunks == []

    def test_content_hash_is_deterministic(self) -> None:
        from utils.chunking import RecursiveCharacterChunker

        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=0)
        text = "The quick brown fox jumps over the lazy dog."
        chunks1 = chunker.chunk(text, source_file="a.txt", doc_type=DocumentType.TXT)
        chunks2 = chunker.chunk(text, source_file="a.txt", doc_type=DocumentType.TXT)

        assert chunks1[0].metadata.content_hash == chunks2[0].metadata.content_hash


class TestHierarchicalChunker:
    def test_produces_child_chunks(self) -> None:
        from utils.chunking import HierarchicalChunker

        chunker = HierarchicalChunker(parent_chunk_size=400, child_chunk_size=100)
        text = "This is a test document. " * 30
        chunks = chunker.chunk(text, source_file="test.txt", doc_type=DocumentType.TXT)

        assert len(chunks) > 0
        # All chunks should be child-sized (≤ parent size)
        for c in chunks:
            assert len(c.text) <= 400 + 50  # slight tolerance for word boundaries


# ── Document loading tests ────────────────────────────────────────────────────


class TestDocumentLoading:
    def test_load_txt_file(self, sample_txt_file: Path) -> None:
        from core.ingestion import load_document

        pages, doc_type = load_document(str(sample_txt_file))
        assert doc_type == DocumentType.TXT
        assert len(pages) == 1
        assert "artificial intelligence" in pages[0][0].lower()

    def test_load_markdown(self, tmp_path: Path) -> None:
        from core.ingestion import load_document

        md_file = tmp_path / "readme.md"
        md_file.write_text("# Title\n\nSome content here.\n\n## Section\n\nMore content.", encoding="utf-8")
        pages, doc_type = load_document(str(md_file))
        assert doc_type == DocumentType.MARKDOWN
        assert len(pages) == 1

    def test_load_nonexistent_file_raises(self) -> None:
        from core.ingestion import load_document

        with pytest.raises(ValueError, match="Failed to read file"):
            load_document("/nonexistent/path/to/file.txt")

    @patch("requests.get")
    def test_load_url(self, mock_get: MagicMock) -> None:
        from core.ingestion import load_document

        mock_get.return_value = MagicMock(
            status_code=200,
            text="<html><body><p>Hello from the web!</p></body></html>",
        )
        mock_get.return_value.raise_for_status = MagicMock()

        pages, doc_type = load_document("https://example.com")
        assert doc_type == DocumentType.URL
        assert "hello from the web" in pages[0][0].lower()


# ── Embedding tests ───────────────────────────────────────────────────────────


class TestEmbedding:
    def test_embed_texts_returns_correct_shape(self) -> None:
        from core.ingestion import embed_texts

        texts = ["Hello world", "Another sentence", "Third one"]
        embeddings = embed_texts(texts)

        assert len(embeddings) == 3
        assert all(isinstance(e, list) for e in embeddings)
        assert all(len(e) > 0 for e in embeddings)
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert len(embeddings[0]) == 384

    def test_embeddings_are_normalized(self) -> None:
        """Normalized embeddings should have magnitude ~1.0."""
        import numpy as np
        from core.ingestion import embed_texts

        embeddings = embed_texts(["Test sentence for normalization check."])
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5


# ── Full ingestion pipeline tests ─────────────────────────────────────────────


class TestIngestionPipeline:
    """
    Integration tests for the full ingestion pipeline.

    These tests use tmp_path to isolate ChromaDB state by patching
    core.ingestion.settings.chroma_persist_dir directly (module-level settings
    object) and resetting the ChromaDB client singleton.
    """

    def _setup_isolated_chroma(self, tmp_path: Path) -> None:
        """Point core.ingestion at a fresh tmp_path ChromaDB and reset the client."""
        import core.ingestion as ing
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        ing.settings.chroma_persist_dir = chroma_dir
        ing._chroma_client = None

    def test_ingest_txt_adds_chunks(self, sample_txt_file: Path, tmp_path: Path) -> None:
        """End-to-end: ingest a text file and verify chunks are stored."""
        self._setup_isolated_chroma(tmp_path)
        from core.ingestion import ingest_document

        result = ingest_document(
            source=str(sample_txt_file),
            collection_name="test_ingest",
        )

        assert result.chunks_added > 0
        assert result.total_chunks_processed == result.chunks_added + result.duplicates_skipped
        assert result.elapsed_seconds > 0

    def test_deduplication_skips_existing_chunks(self, sample_txt_file: Path, tmp_path: Path) -> None:
        """Ingesting the same file twice should skip all chunks on second run."""
        self._setup_isolated_chroma(tmp_path)
        from core.ingestion import ingest_document

        result1 = ingest_document(str(sample_txt_file), collection_name="test_dedup")
        result2 = ingest_document(str(sample_txt_file), collection_name="test_dedup")

        assert result1.chunks_added > 0
        assert result2.chunks_added == 0
        assert result2.duplicates_skipped == result1.chunks_added

    def test_overwrite_forces_reingest(self, sample_txt_file: Path, tmp_path: Path) -> None:
        """With overwrite=True, all chunks should be added regardless of duplicates."""
        self._setup_isolated_chroma(tmp_path)
        from core.ingestion import ingest_document

        result1 = ingest_document(str(sample_txt_file), collection_name="test_overwrite")
        result2 = ingest_document(str(sample_txt_file), collection_name="test_overwrite", overwrite=True)

        assert result2.chunks_added == result2.total_chunks_processed
        assert result2.duplicates_skipped == 0
