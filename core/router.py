"""
Intelligent query router — automatically selects the best collection(s) to search.

When a user has multiple knowledge bases (e.g. "finance", "legal", "product_docs"),
the router uses embedding similarity and LLM classification to decide which
collections to query, rather than requiring the user to specify one manually.

Strategies:
  1. Embedding similarity: embed the query and compare against collection summaries
  2. LLM classification: ask the LLM to pick the best collection given descriptions
  3. Broadcast: query all collections and merge results (fallback)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from core.ingestion import get_embedding_model, list_collections

logger = logging.getLogger(__name__)


# ── Collection profile ────────────────────────────────────────────────────────


class CollectionProfile:
    """
    A lightweight semantic profile of a collection for routing decisions.
    Built from a natural-language description or auto-generated from document summaries.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._embedding: list[float] | None = None

    def get_embedding(self) -> list[float]:
        if self._embedding is None:
            model = get_embedding_model()
            self._embedding = model.encode([self.description], normalize_embeddings=True)[0].tolist()
        return self._embedding


# ── Embedding-based router ────────────────────────────────────────────────────


def route_by_embedding(
    query: str,
    profiles: list[CollectionProfile],
    top_n: int = 2,
) -> list[tuple[str, float]]:
    """
    Score collections by cosine similarity between query embedding and collection descriptions.

    Args:
        query: user's question
        profiles: collection profiles with description embeddings
        top_n: number of top collections to return

    Returns:
        Sorted list of (collection_name, similarity_score) tuples
    """
    model = get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    scores: list[tuple[str, float]] = []
    for profile in profiles:
        c_emb = np.array(profile.get_embedding())
        sim = float(np.dot(q_emb, c_emb))
        scores.append((profile.name, round(sim, 4)))

    scores.sort(key=lambda x: x[1], reverse=True)
    logger.debug("Router embedding scores: %s", scores[:top_n])
    return scores[:top_n]


# ── LLM-based router ──────────────────────────────────────────────────────────


def route_by_llm(
    query: str,
    collection_names: list[str],
    llm_fn: Callable[[str], str],
) -> str:
    """
    Ask the LLM to select the single most relevant collection for the query.

    Falls back to the first collection if parsing fails.

    Args:
        query: user's question
        collection_names: available collection names
        llm_fn: LLM callable

    Returns:
        Selected collection name
    """
    if len(collection_names) == 1:
        return collection_names[0]

    options = "\n".join(f"- {name}" for name in collection_names)
    prompt = (
        "Select the single most relevant knowledge base for the following question.\n"
        f"Available knowledge bases:\n{options}\n\n"
        f"Question: {query}\n\n"
        "Reply with ONLY the knowledge base name (exact match from the list above):"
    )
    try:
        selected = llm_fn(prompt).strip().strip('"').strip("'")
        if selected in collection_names:
            logger.info("LLM router selected: '%s'", selected)
            return selected
        # Fuzzy match
        for name in collection_names:
            if name.lower() in selected.lower():
                return name
    except Exception as e:
        logger.warning("LLM routing failed: %s", e)

    return collection_names[0]


# ── Main router ───────────────────────────────────────────────────────────────


class QueryRouter:
    """
    Routes queries to the most relevant collection(s).

    Uses a two-stage approach:
      1. Embedding similarity against collection profiles (fast, no LLM cost)
      2. Optional LLM refinement for ambiguous cases

    Usage:
        router = QueryRouter()
        router.register("finance", "Financial reports, earnings, budgets, and investment docs")
        router.register("legal", "Contracts, compliance, regulations, and legal filings")
        collection = router.route("What was our Q3 revenue?")
        # → "finance"
    """

    def __init__(self) -> None:
        self.profiles: dict[str, CollectionProfile] = {}

    def register(self, collection_name: str, description: str) -> None:
        """Register a collection with a descriptive profile for routing."""
        self.profiles[collection_name] = CollectionProfile(collection_name, description)
        logger.info("Router: registered collection '%s'", collection_name)

    def auto_register(self) -> None:
        """Auto-register all existing ChromaDB collections using their names as descriptions."""
        for col in list_collections():
            name = col["name"]
            if name not in self.profiles:
                # Use the collection name as the description (best we can do without user input)
                self.profiles[name] = CollectionProfile(name, f"Knowledge base: {name.replace('_', ' ')}")
        logger.info("Router: auto-registered %d collections", len(self.profiles))

    def route(
        self,
        query: str,
        top_n: int = 1,
        use_llm: bool = False,
        llm_fn: Callable[[str], str] | None = None,
    ) -> list[str]:
        """
        Route a query to the top-N most relevant collections.

        Args:
            query: user's question
            top_n: number of collections to return
            use_llm: use LLM for final disambiguation
            llm_fn: required if use_llm=True

        Returns:
            List of collection names, ordered by relevance
        """
        if not self.profiles:
            self.auto_register()

        if not self.profiles:
            return ["default"]

        profiles = list(self.profiles.values())
        scored = route_by_embedding(query, profiles, top_n=top_n * 2)

        # If top 2 scores are close, use LLM to disambiguate
        if use_llm and llm_fn and len(scored) >= 2:
            top_score = scored[0][1]
            second_score = scored[1][1]
            if abs(top_score - second_score) < 0.05:  # ambiguous
                candidates = [name for name, _ in scored[:3]]
                winner = route_by_llm(query, candidates, llm_fn)
                return [winner]

        return [name for name, _ in scored[:top_n]]

    def route_single(
        self,
        query: str,
        use_llm: bool = False,
        llm_fn: Callable[[str], str] | None = None,
    ) -> str:
        """Route to a single best-matching collection."""
        results = self.route(query, top_n=1, use_llm=use_llm, llm_fn=llm_fn)
        return results[0] if results else "default"


# Module-level singleton
_router: QueryRouter | None = None


def get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
