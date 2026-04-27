"""
API endpoint tests using FastAPI TestClient.

Tests:
  - GET  /health
  - POST /ingest
  - POST /query
  - GET  /collections
  - DELETE /collection/{name}
  - GET  /cache/stats
  - DELETE /cache
  - Error handling (404, 422, 500)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import app
from models import QueryMode


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "llm_backend" in data
        assert "embedding_model" in data
        assert data["status"] == "ok"


# ── Ingest ────────────────────────────────────────────────────────────────────


class TestIngest:
    def test_ingest_success(self, client: TestClient) -> None:
        from models import IngestResult

        mock_result = IngestResult(
            collection="test_kb",
            source="test.txt",
            chunks_added=10,
            duplicates_skipped=2,
            total_chunks_processed=12,
            elapsed_seconds=0.5,
        )

        with patch("core.ingestion.ingest_document", return_value=mock_result):
            response = client.post("/ingest", json={
                "file_path": "test.txt",
                "collection": "test_kb",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["chunks_added"] == 10
        assert data["duplicates_skipped"] == 2
        assert data["collection"] == "test_kb"

    def test_ingest_missing_file_path(self, client: TestClient) -> None:
        response = client.post("/ingest", json={"collection": "test_kb"})
        assert response.status_code == 422  # Pydantic validation error

    def test_ingest_file_not_found(self, client: TestClient) -> None:
        with patch("core.ingestion.ingest_document", side_effect=FileNotFoundError("not found")):
            response = client.post("/ingest", json={
                "file_path": "/bad/path.txt",
                "collection": "kb",
            })
        assert response.status_code == 404

    def test_ingest_processing_error(self, client: TestClient) -> None:
        with patch("core.ingestion.ingest_document", side_effect=RuntimeError("ChromaDB upsert failed")):
            response = client.post("/ingest", json={
                "file_path": "broken.pdf",
                "collection": "kb",
            })
        assert response.status_code == 500


# ── Query ─────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_query_success(self, client: TestClient) -> None:
        from models import QueryResponse, SourceCitation

        mock_result = QueryResponse(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation [Source: docs.txt, chunk 0].",
            sources=[
                SourceCitation(
                    source="docs.txt",
                    chunk_index=0,
                    similarity_score=0.87,
                    excerpt="RAG stands for Retrieval-Augmented Generation...",
                )
            ],
            tokens_used=150,
            latency_ms=320.5,
            collection="kb",
            llm_backend="ollama",
            model_used="llama3.2",
            cache_hit=False,
        )

        with patch("core.generation.answer_question", return_value=mock_result):
            response = client.post("/query", json={
                "question": "What is RAG?",
                "collection": "kb",
                "top_k": 5,
            })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data
        assert "latency_ms" in data
        assert data["cache_hit"] is False

    def test_query_missing_question(self, client: TestClient) -> None:
        response = client.post("/query", json={"collection": "kb"})
        assert response.status_code == 422

    def test_query_empty_question_rejected(self, client: TestClient) -> None:
        response = client.post("/query", json={"question": "", "collection": "kb"})
        assert response.status_code == 422

    def test_query_top_k_clamped(self, client: TestClient) -> None:
        """top_k must be between 1 and 50."""
        response = client.post("/query", json={
            "question": "test question",
            "collection": "kb",
            "top_k": 9999,
        })
        assert response.status_code == 422

    def test_query_runtime_error_returns_500(self, client: TestClient) -> None:
        with patch("core.generation.answer_question", side_effect=RuntimeError("LLM backend unavailable")):
            response = client.post("/query", json={
                "question": "Will this fail?",
                "collection": "kb",
            })
        assert response.status_code == 500


# ── Collections ───────────────────────────────────────────────────────────────


class TestCollections:
    def test_list_collections_empty(self, client: TestClient) -> None:
        with patch("core.ingestion.list_collections", return_value=[]):
            response = client.get("/collections")

        assert response.status_code == 200
        data = response.json()
        assert data["collections"] == []
        assert data["total"] == 0

    def test_list_collections_returns_all(self, client: TestClient) -> None:
        mock_cols = [
            {"name": "kb1", "document_count": 42, "embedding_model": "all-MiniLM-L6-v2"},
            {"name": "kb2", "document_count": 100, "embedding_model": "all-MiniLM-L6-v2"},
        ]
        with patch("core.ingestion.list_collections", return_value=mock_cols):
            response = client.get("/collections")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        names = [c["name"] for c in data["collections"]]
        assert "kb1" in names
        assert "kb2" in names


# ── Delete collection ─────────────────────────────────────────────────────────


class TestDeleteCollection:
    def test_delete_existing_collection(self, client: TestClient) -> None:
        with patch("core.ingestion.delete_collection", return_value=True):
            response = client.delete("/collection/my_kb")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["name"] == "my_kb"

    def test_delete_nonexistent_collection_returns_404(self, client: TestClient) -> None:
        with patch("core.ingestion.delete_collection", return_value=False):
            response = client.delete("/collection/ghost_kb")

        assert response.status_code == 404


# ── Cache endpoints ───────────────────────────────────────────────────────────


class TestCache:
    def test_cache_stats_returns_structure(self, client: TestClient) -> None:
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "size" in data
        assert "max_size" in data
        assert "threshold" in data

    def test_clear_cache_returns_message(self, client: TestClient) -> None:
        response = client.delete("/cache")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


# ── OpenAPI schema ────────────────────────────────────────────────────────────


class TestOpenAPI:
    def test_openapi_schema_accessible(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/ingest" in schema["paths"]
        assert "/query" in schema["paths"]
        assert "/collections" in schema["paths"]

    def test_docs_accessible(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200
