"""
CI evaluation setup — creates a minimal test corpus for the eval gate.

Creates a small ChromaDB collection with synthetic documents and
ingests them so the eval harness has something to query against.
Run before scripts/benchmark_suite.py in CI.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

# ── Synthetic corpus ──────────────────────────────────────────────────────────

EVAL_DOCUMENTS = [
    {
        "filename": "ml_basics.txt",
        "content": """
Machine learning is a subset of artificial intelligence that enables systems to
learn and improve from experience without being explicitly programmed. It focuses
on developing computer programs that can access data and use it to learn for themselves.

Supervised learning involves training a model on labeled data where the desired output
is known. Common algorithms include linear regression, decision trees, random forests,
and support vector machines. The model learns to map inputs to outputs.

Unsupervised learning finds hidden patterns in data without labeled responses.
Clustering algorithms like K-means group similar data points together. Dimensionality
reduction techniques like PCA reduce the number of features while preserving information.

Deep learning uses neural networks with many layers to learn representations from raw data.
Convolutional neural networks (CNNs) excel at image recognition. Recurrent neural networks
(RNNs) and transformers are used for sequential data like text and time series.

Transfer learning allows models pre-trained on large datasets to be fine-tuned on smaller
domain-specific datasets. This dramatically reduces the amount of labeled data required
and training time needed to achieve good performance.
        """.strip(),
    },
    {
        "filename": "rag_systems.txt",
        "content": """
Retrieval-Augmented Generation (RAG) combines information retrieval with large language
model generation. Instead of relying solely on parametric knowledge, RAG retrieves
relevant documents from an external knowledge base and uses them as context for generation.

The basic RAG pipeline has three stages: indexing, retrieval, and generation.
During indexing, documents are split into chunks, embedded using a model like
sentence-transformers, and stored in a vector database such as ChromaDB or Pinecone.

Hybrid retrieval combines dense vector search with sparse BM25 keyword matching.
Dense retrieval captures semantic similarity while BM25 captures exact keyword matches.
Reciprocal Rank Fusion (RRF) combines the rankings from both methods.

HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to the query,
embeds it, and uses that embedding for retrieval instead of the raw query embedding.
This improves retrieval for queries that are phrased differently from documents.

CRAG (Corrective RAG) adds a self-evaluation step where the model scores retrieved
context quality. If below a threshold, it rewrites the query or falls back to web search.
This reduces hallucinations caused by retrieving irrelevant context.

Cross-encoder reranking passes retrieved chunk and query pairs through a more powerful
model to rerank the results by relevance. This improves precision at the cost of latency.
The ms-marco-MiniLM model is a popular choice for this step.
        """.strip(),
    },
    {
        "filename": "vector_databases.txt",
        "content": """
Vector databases store high-dimensional embeddings and support efficient similarity search.
They are the backbone of modern RAG systems and semantic search applications.

ChromaDB is an open-source embedding database that runs locally or as a hosted service.
It supports multiple distance metrics including cosine similarity, L2, and inner product.
ChromaDB can persist embeddings to disk for production use.

Pinecone is a managed vector database service that scales to billions of vectors.
It supports metadata filtering, namespaces, and hybrid search with sparse-dense combinations.
Pinecone charges based on the number of vectors stored and queries made.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.
It supports GPU acceleration and various index types including IVF and HNSW.
FAISS is often used as the backend for other vector stores.

The HNSW (Hierarchical Navigable Small World) algorithm provides approximate nearest
neighbor search with high recall and low latency. It builds a multi-layer graph structure
where each layer is a subset of the previous, enabling fast greedy search.

Indexing strategies matter significantly for production performance. Flat indices provide
exact search but scale poorly. IVF (Inverted File Index) clusters vectors and searches
only the nearest clusters, trading recall for speed. Product quantization compresses
vectors to reduce memory usage.
        """.strip(),
    },
]

QA_PAIRS = [
    {
        "question": "What is supervised learning?",
        "expected_answer": "Supervised learning involves training a model on labeled data where the desired output is known.",
        "relevant_sources": ["ml_basics.txt"],
        "collection": "eval_test",
    },
    {
        "question": "What is HyDE in RAG systems?",
        "expected_answer": "HyDE generates a hypothetical answer to the query, embeds it, and uses that embedding for retrieval instead of the raw query embedding.",
        "relevant_sources": ["rag_systems.txt"],
        "collection": "eval_test",
    },
    {
        "question": "What is ChromaDB used for?",
        "expected_answer": "ChromaDB is an open-source embedding database that stores high-dimensional embeddings and supports efficient similarity search.",
        "relevant_sources": ["vector_databases.txt"],
        "collection": "eval_test",
    },
    {
        "question": "How does CRAG improve RAG quality?",
        "expected_answer": "CRAG adds a self-evaluation step where the model scores retrieved context quality and rewrites the query or falls back to web search if quality is low.",
        "relevant_sources": ["rag_systems.txt"],
        "collection": "eval_test",
    },
    {
        "question": "What is the HNSW algorithm?",
        "expected_answer": "HNSW provides approximate nearest neighbor search by building a multi-layer graph structure enabling fast greedy search with high recall.",
        "relevant_sources": ["vector_databases.txt"],
        "collection": "eval_test",
    },
]


def setup_eval_corpus() -> None:
    """Ingest the synthetic corpus into ChromaDB for CI evaluation."""
    print("Setting up evaluation corpus...")

    try:
        from core.ingestion import ingest_document
    except ImportError as e:
        print(f"Could not import ingestion module: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        for doc in EVAL_DOCUMENTS:
            filepath = Path(tmpdir) / doc["filename"]
            filepath.write_text(doc["content"])

            try:
                result = ingest_document(
                    source=str(filepath),
                    collection_name="eval_test",
                    overwrite=True,
                )
                print(f"  Ingested {doc['filename']}: {result.chunks_added} chunks")
            except Exception as e:
                print(f"  Failed to ingest {doc['filename']}: {e}")

    print("Eval corpus ready.")


def save_qa_pairs(output_path: str = "eval_qa_pairs.json") -> None:
    """Save QA pairs to JSON for the benchmark runner."""
    import json
    with open(output_path, "w") as f:
        json.dump(QA_PAIRS, f, indent=2)
    print(f"Saved {len(QA_PAIRS)} QA pairs to {output_path}")


if __name__ == "__main__":
    setup_eval_corpus()
    save_qa_pairs()
    print("Done.")
