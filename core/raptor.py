"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Paper: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
       (Sarthi et al., Stanford 2024) — https://arxiv.org/abs/2401.18059

The Problem RAPTOR Solves:
  Standard RAG retrieves small, local chunks. This works for detail questions
  ("What year was X founded?") but fails for broad synthesis questions
  ("What are the main themes across these documents?") because the answer
  requires understanding across many chunks simultaneously.

RAPTOR's Solution:
  Build a hierarchical summary tree BOTTOM-UP at ingestion time:
    Level 0: original chunks (leaf nodes)
    Level 1: cluster nearby chunks, summarize each cluster → parent nodes
    Level 2: cluster Level-1 summaries, summarize again → grandparent nodes
    ...up to root (single document summary)

  At query time, retrieve from ALL levels. Simple factual questions match leaf
  chunks; thematic questions match high-level summaries.

Implementation:
  1. Embed all leaf chunks
  2. Cluster using Gaussian Mixture Models (soft clustering — chunks can belong
     to multiple clusters, unlike k-means)
  3. For each cluster, concatenate texts and LLM-summarize
  4. Embed the summaries → new "virtual" nodes
  5. Repeat until one cluster remains
  6. Store all levels in ChromaDB under a special collection suffix

This dramatically improves performance on multi-document synthesis,
thematic analysis, and "big picture" questions — the hardest RAG queries.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RaptorNode:
    """A node in the RAPTOR tree (either a leaf chunk or a summary)."""

    text: str
    level: int                        # 0 = leaf, 1+ = summary
    children: list["RaptorNode"] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    cluster_id: int = -1
    node_id: str = ""

    @property
    def is_leaf(self) -> bool:
        return self.level == 0


@dataclass
class RaptorTree:
    """The complete hierarchical tree for a document collection."""

    leaves: list[RaptorNode]
    all_nodes: list[RaptorNode]       # all levels, for retrieval
    root_summaries: list[RaptorNode]  # highest level nodes
    max_level: int = 0


# ── Gaussian Mixture Model clustering ────────────────────────────────────────


def gaussian_mixture_cluster(
    embeddings: np.ndarray,
    n_components: int | None = None,
    threshold: float = 0.5,
    random_state: int = 42,
) -> list[int]:
    """
    Soft-cluster embeddings using Gaussian Mixture Models.

    GMM is preferred over k-means for RAPTOR because:
      - Soft assignments: a chunk can belong to multiple clusters
      - Better handles overlapping semantic topics
      - BIC/AIC model selection finds optimal n_components automatically

    Args:
        embeddings: (n_samples, embedding_dim) matrix
        n_components: if None, auto-select via BIC
        threshold: probability threshold for cluster assignment
        random_state: for reproducibility

    Returns:
        List of cluster assignments (one per embedding)
    """
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import normalize

        n_samples = len(embeddings)
        if n_samples < 2:
            return [0] * n_samples

        # Reduce dimensionality for GMM stability (GMM struggles in high-dim)
        reduced = _reduce_dimensions(embeddings, target_dim=min(10, n_samples - 1))

        # Auto-select n_components via BIC if not specified
        if n_components is None:
            max_k = min(int(math.sqrt(n_samples)), 10)
            best_bic, best_k = float("inf"), 2
            for k in range(2, max_k + 1):
                try:
                    gm = GaussianMixture(n_components=k, random_state=random_state, n_init=2)
                    gm.fit(reduced)
                    bic = gm.bic(reduced)
                    if bic < best_bic:
                        best_bic, best_k = bic, k
                except Exception:
                    break
            n_components = best_k

        gm = GaussianMixture(n_components=n_components, random_state=random_state)
        gm.fit(reduced)
        labels = gm.predict(reduced)
        return labels.tolist()

    except ImportError:
        logger.warning("scikit-learn not installed. Falling back to simple partitioning for RAPTOR.")
        # Fallback: simple partition into sqrt(n) groups
        n = len(embeddings)
        k = max(2, int(math.sqrt(n)))
        return [i % k for i in range(n)]


def _reduce_dimensions(embeddings: np.ndarray, target_dim: int = 10) -> np.ndarray:
    """Reduce embedding dimensions using UMAP if available, else PCA."""
    try:
        import umap
        reducer = umap.UMAP(n_components=target_dim, metric="cosine", random_state=42)
        return reducer.fit_transform(embeddings)
    except ImportError:
        pass
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(target_dim, embeddings.shape[1], embeddings.shape[0] - 1))
        return pca.fit_transform(embeddings)
    except ImportError:
        return embeddings


# ── Summarization ─────────────────────────────────────────────────────────────


def summarize_cluster(
    texts: list[str],
    cluster_id: int,
    level: int,
    llm_fn: Callable[[str], str],
    max_chars: int = 4000,
) -> str:
    """
    Summarize a cluster of texts into a single coherent summary node.

    Args:
        texts: chunk texts in this cluster
        cluster_id: cluster index (for logging)
        level: current tree level (used to adjust summary style)
        llm_fn: LLM callable
        max_chars: truncate combined texts to this length

    Returns:
        Summary string
    """
    combined = "\n\n---\n\n".join(texts)[:max_chars]

    if level == 1:
        instruction = "Write a concise summary (3-5 sentences) capturing the main points:"
    else:
        instruction = "Write a high-level abstract (2-3 sentences) of the key themes and conclusions:"

    prompt = (
        f"{instruction}\n\n"
        f"{combined}\n\n"
        "Summary:"
    )
    try:
        summary = llm_fn(prompt).strip()
        logger.debug("RAPTOR L%d cluster %d → %d chars", level, cluster_id, len(summary))
        return summary
    except Exception as e:
        logger.warning("RAPTOR summarization failed for cluster %d: %s", cluster_id, e)
        return combined[:500]  # fallback: truncate


# ── Tree construction ──────────────────────────────────────────────────────────


def build_raptor_tree(
    leaf_chunks: list[str],
    leaf_embeddings: list[list[float]],
    llm_fn: Callable[[str], str],
    embed_fn: Callable[[list[str]], list[list[float]]],
    max_levels: int = 3,
    min_cluster_size: int = 3,
) -> RaptorTree:
    """
    Build the RAPTOR tree from leaf chunks bottom-up.

    Args:
        leaf_chunks: original text chunks
        leaf_embeddings: pre-computed embeddings for leaf chunks
        llm_fn: LLM callable for summarization
        embed_fn: embedding callable for new summary nodes
        max_levels: maximum tree depth (usually 2-3 is sufficient)
        min_cluster_size: skip clustering if fewer chunks than this

    Returns:
        RaptorTree with all nodes across all levels
    """
    import uuid

    all_nodes: list[RaptorNode] = []

    # Level 0: leaf nodes
    current_nodes = []
    for i, (text, emb) in enumerate(zip(leaf_chunks, leaf_embeddings)):
        node = RaptorNode(
            text=text, level=0, embedding=emb, node_id=f"leaf_{i}"
        )
        current_nodes.append(node)
        all_nodes.append(node)

    leaves = list(current_nodes)
    max_level_reached = 0

    for level in range(1, max_levels + 1):
        if len(current_nodes) < min_cluster_size:
            logger.info("RAPTOR: stopping at level %d (only %d nodes)", level, len(current_nodes))
            break

        logger.info("RAPTOR: building level %d from %d nodes…", level, len(current_nodes))
        embeddings_matrix = np.array([n.embedding for n in current_nodes])
        cluster_labels = gaussian_mixture_cluster(embeddings_matrix)

        # Group nodes by cluster
        clusters: dict[int, list[RaptorNode]] = {}
        for node, label in zip(current_nodes, cluster_labels):
            node.cluster_id = label
            clusters.setdefault(label, []).append(node)

        # Build summary nodes for each cluster
        new_level_nodes = []
        for cluster_id, cluster_nodes in clusters.items():
            texts = [n.text for n in cluster_nodes]
            summary_text = summarize_cluster(texts, cluster_id, level, llm_fn)
            summary_emb = embed_fn([summary_text])[0]

            summary_node = RaptorNode(
                text=summary_text,
                level=level,
                children=cluster_nodes,
                embedding=summary_emb,
                cluster_id=cluster_id,
                node_id=f"summary_L{level}_{cluster_id}_{uuid.uuid4().hex[:6]}",
            )
            new_level_nodes.append(summary_node)
            all_nodes.append(summary_node)

        current_nodes = new_level_nodes
        max_level_reached = level

    root_summaries = current_nodes

    logger.info(
        "RAPTOR tree built: %d leaves, %d total nodes, %d levels",
        len(leaves), len(all_nodes), max_level_reached,
    )
    return RaptorTree(
        leaves=leaves,
        all_nodes=all_nodes,
        root_summaries=root_summaries,
        max_level=max_level_reached,
    )


# ── ChromaDB storage integration ──────────────────────────────────────────────


def store_raptor_tree(
    tree: RaptorTree,
    collection_name: str,
    source: str,
) -> int:
    """
    Store all RAPTOR tree nodes (all levels) into ChromaDB.

    Summary nodes are stored alongside leaf chunks in a special
    collection named `{collection_name}__raptor`. At query time,
    both collections are searched for collapsed tree retrieval.

    Returns:
        Number of nodes stored
    """
    from core.ingestion import get_or_create_collection
    from datetime import datetime, timezone

    raptor_collection_name = f"{collection_name}__raptor"
    col = get_or_create_collection(raptor_collection_name)

    ids, embeddings, documents, metadatas = [], [], [], []

    for node in tree.all_nodes:
        if not node.embedding:
            continue
        ids.append(node.node_id)
        embeddings.append(node.embedding)
        documents.append(node.text)
        metadatas.append({
            "level": node.level,
            "is_leaf": node.is_leaf,
            "cluster_id": node.cluster_id,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "child_count": len(node.children),
        })

    if ids:
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    logger.info("RAPTOR: stored %d nodes to '%s'", len(ids), raptor_collection_name)
    return len(ids)


def retrieve_from_raptor(
    query_embedding: list[float],
    collection_name: str,
    top_k: int = 6,
    similarity_threshold: float = 0.35,
) -> list[tuple[str, dict, float]]:
    """
    Query the RAPTOR tree collection for multi-level context.

    Returns nodes from all levels (mix of leaves and summaries),
    giving the LLM both granular details and big-picture context.

    Returns:
        List of (text, metadata, similarity_score) tuples
    """
    from core.ingestion import get_or_create_collection

    raptor_collection_name = f"{collection_name}__raptor"
    col = get_or_create_collection(raptor_collection_name)

    if col.count() == 0:
        return []

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    out = []
    for doc, meta, dist in zip(docs, metas, distances):
        sim = max(0.0, 1.0 - float(dist))
        if sim >= similarity_threshold:
            out.append((doc, meta, sim))

    return out
