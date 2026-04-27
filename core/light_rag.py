"""
LightRAG — Dual-level graph retrieval.

Based on "LightRAG: Simple and Fast Retrieval-Augmented Generation" (EMNLP 2025).
GitHub: github.com/HKUDS/LightRAG (70K+ stars)

Extends the existing GraphRAG with two complementary retrieval modes:

  LOW-LEVEL  : Precise entity/relationship queries — "What is X?", "Who did Y?"
               Traverses the KG starting from query entities, returns specific facts.

  HIGH-LEVEL : Thematic/community queries — "How does X relate to the domain?"
               Uses community summaries to answer broad conceptual questions.

  AUTO       : Classifies the query and routes to the appropriate level.

Also adds INCREMENTAL GRAPH UPDATES — rather than rebuilding the entire graph on
each ingest, new triples are merged into the existing graph while preserving all
prior relationships. This is critical for production systems where documents are
added continuously.

The dual-level approach enables:
  - Specific fact retrieval (low-level): fast, precise, no hallucination
  - Thematic synthesis (high-level): broader understanding, connected reasoning

Architecture:
  LightRAGRetriever
    ├── low_level_retrieve(query) → list[str]   (entity-anchored facts)
    ├── high_level_retrieve(query) → list[str]  (community-level themes)
    ├── auto_retrieve(query) → list[str]        (routed automatically)
    └── incremental_update(triples)             (merges new graph data)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_PATH = Path("./data/knowledge_graph.json")
LIGHT_RAG_PATH = Path("./data/light_rag_index.json")


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class LightRAGResult:
    """Result from a LightRAG dual-level retrieval."""

    query: str
    level: str                  # "low", "high", "auto"
    resolved_level: str         # actual level used after auto-routing
    context_chunks: list[str]
    entities_used: list[str]
    communities_used: list[str]
    confidence: float           # rough estimate: 1.0 = many entity/community hits


# ── Query classifier ──────────────────────────────────────────────────────────

# Patterns that suggest specific fact-seeking (→ low-level)
_LOW_LEVEL_SIGNALS = re.compile(
    r"\b(who|what is|what are|where|when|how many|how much|define|"
    r"name of|list|which|describe specifically)\b",
    re.IGNORECASE,
)

# Patterns that suggest broad/thematic questions (→ high-level)
_HIGH_LEVEL_SIGNALS = re.compile(
    r"\b(how does|why|overall|summarize|explain|compare|relationship|"
    r"broad|theme|impact|role of|significance|in general|overview)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> str:
    """
    Route a query to 'low' or 'high' level retrieval.

    Low-level: entity/fact-specific questions
    High-level: thematic, relational, summarization questions
    Default: low (more precise, safer)
    """
    low_hits  = len(_LOW_LEVEL_SIGNALS.findall(query))
    high_hits = len(_HIGH_LEVEL_SIGNALS.findall(query))

    if high_hits > low_hits:
        return "high"
    return "low"


# ── LightRAG Retriever ────────────────────────────────────────────────────────


class LightRAGRetriever:
    """
    Dual-level graph retriever implementing the LightRAG (EMNLP 2025) pattern.

    Loads the knowledge graph built by core/graph_rag.py and provides two
    retrieval strategies over it, plus an incremental update mechanism.
    """

    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        index_path: Path = LIGHT_RAG_PATH,
        max_low_level_hops: int = 2,
        max_low_level_nodes: int = 10,
        max_high_level_communities: int = 5,
    ) -> None:
        self.graph_path = graph_path
        self.index_path = index_path
        self.max_low_level_hops = max_low_level_hops
        self.max_low_level_nodes = max_low_level_nodes
        self.max_high_level_communities = max_high_level_communities

        self._graph: nx.Graph = nx.Graph()
        self._community_summaries: dict[int, str] = {}
        self._entity_index: dict[str, list[str]] = {}  # entity → [facts]
        self._load()

    # ── Load / persist ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load graph from disk if it exists."""
        if self.graph_path.exists():
            try:
                from core.graph_rag import load_graph
                self._graph = load_graph(self.graph_path)
                logger.info("LightRAG: loaded graph with %d nodes, %d edges",
                            self._graph.number_of_nodes(), self._graph.number_of_edges())
            except Exception as e:
                logger.warning("LightRAG: could not load graph from %s: %s", self.graph_path, e)

        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text())
                self._community_summaries = {int(k): v for k, v in data.get("communities", {}).items()}
                self._entity_index = data.get("entity_index", {})
                logger.info("LightRAG: loaded %d community summaries, %d entity entries",
                            len(self._community_summaries), len(self._entity_index))
            except Exception as e:
                logger.warning("LightRAG: could not load index from %s: %s", self.index_path, e)

    def _save_index(self) -> None:
        """Persist the LightRAG index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "communities": {str(k): v for k, v in self._community_summaries.items()},
            "entity_index": self._entity_index,
        }
        self.index_path.write_text(json.dumps(data, indent=2))

    # ── Incremental update ────────────────────────────────────────────────────

    def incremental_update(
        self,
        new_triples: list[tuple[str, str, str]],
        source: str = "unknown",
    ) -> int:
        """
        Merge new (subject, predicate, object) triples into the existing graph.

        Unlike a full rebuild, this preserves all existing relationships and
        simply adds new nodes/edges. Critical for continuous document ingestion.

        Args:
            new_triples: List of (subject, predicate, object) tuples
            source:      Source document identifier for provenance

        Returns:
            Number of new edges added
        """
        added = 0
        for subj, pred, obj in new_triples:
            subj_norm = subj.lower().strip()
            obj_norm  = obj.lower().strip()

            # Add or update nodes
            if subj_norm not in self._graph:
                self._graph.add_node(subj_norm, label=subj, sources=[source])
            else:
                self._graph.nodes[subj_norm].setdefault("sources", [])
                if source not in self._graph.nodes[subj_norm]["sources"]:
                    self._graph.nodes[subj_norm]["sources"].append(source)

            if obj_norm not in self._graph:
                self._graph.add_node(obj_norm, label=obj, sources=[source])
            else:
                self._graph.nodes[obj_norm].setdefault("sources", [])
                if source not in self._graph.nodes[obj_norm]["sources"]:
                    self._graph.nodes[obj_norm]["sources"].append(source)

            # Add edge (merge predicate if edge exists)
            if self._graph.has_edge(subj_norm, obj_norm):
                existing = self._graph[subj_norm][obj_norm].get("predicates", [])
                if pred not in existing:
                    existing.append(pred)
                self._graph[subj_norm][obj_norm]["predicates"] = existing
            else:
                self._graph.add_edge(subj_norm, obj_norm, predicates=[pred], source=source)
                added += 1

            # Update entity index
            fact = f"{subj} {pred} {obj}"
            for key in [subj_norm, obj_norm]:
                self._entity_index.setdefault(key, [])
                if fact not in self._entity_index[key]:
                    self._entity_index[key].append(fact)

        if added > 0:
            self._save_index()
            logger.info("LightRAG: incremental update added %d edges from '%s'", added, source)

        return added

    def build_community_index(
        self,
        community_summaries: dict[int, str],
    ) -> None:
        """
        Load pre-built community summaries into the LightRAG index.

        These are typically generated by core/graph_rag.build_community_summaries().
        Storing them here enables fast high-level retrieval without rebuilding.
        """
        self._community_summaries = community_summaries
        self._save_index()
        logger.info("LightRAG: indexed %d community summaries", len(community_summaries))

    # ── Entity matching ───────────────────────────────────────────────────────

    def _match_entities(self, query: str) -> list[str]:
        """
        Find graph nodes that appear in the query.

        Uses substring matching (normalized) — fast, no LLM needed.
        Returns entity names sorted by length (longer = more specific).
        """
        q = query.lower()
        matches = [
            node for node in self._graph.nodes
            if len(node) >= 3 and node in q
        ]
        # Sort by length descending (more specific entities first)
        return sorted(matches, key=len, reverse=True)

    # ── Low-level retrieval ───────────────────────────────────────────────────

    def low_level_retrieve(self, query: str) -> list[str]:
        """
        Precise entity/relationship retrieval.

        1. Identify entities in the query
        2. Traverse the KG neighborhood (up to N hops)
        3. Return facts about those entities and their relationships

        Best for: "What is X?", "Who created Y?", "List all Z"
        """
        if self._graph.number_of_nodes() == 0:
            logger.warning(
                "LightRAG low-level: knowledge graph is empty. "
                "Run `rag ingest --graph` to build the graph before querying."
            )
            return []

        matched = self._match_entities(query)

        if not matched:
            # Fallback: return facts from entity_index for any query word
            words = [w.lower() for w in query.split() if len(w) > 4]
            for word in words:
                for entity_key in self._entity_index:
                    if word in entity_key:
                        matched.append(entity_key)
                        break

        if not matched:
            return []

        facts: list[str] = []
        seen_nodes: set[str] = set()

        for entity in matched[:3]:  # cap at 3 seed entities
            if entity not in self._graph:
                continue
            # BFS neighborhood
            try:
                neighbors = list(nx.single_source_shortest_path_length(
                    self._graph, entity, cutoff=self.max_low_level_hops
                ).keys())
            except Exception:
                neighbors = [entity]

            for node in neighbors[:self.max_low_level_nodes]:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)

                # Collect facts from entity index
                node_facts = self._entity_index.get(node, [])
                facts.extend(node_facts[:3])

                # Add edge predicates as facts
                for nbr in self._graph.neighbors(node):
                    preds = self._graph[node][nbr].get("predicates", [])
                    node_label = self._graph.nodes[node].get("label", node)
                    nbr_label  = self._graph.nodes[nbr].get("label", nbr)
                    for pred in preds:
                        facts.append(f"{node_label} {pred} {nbr_label}")

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_facts: list[str] = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                unique_facts.append(f)

        logger.debug("LightRAG low-level: %d entities matched → %d facts", len(matched), len(unique_facts))
        return unique_facts[:20]

    # ── High-level retrieval ──────────────────────────────────────────────────

    def high_level_retrieve(self, query: str) -> list[str]:
        """
        Thematic/community-level retrieval.

        Scores each community summary against the query using keyword overlap
        and returns the most relevant community contexts.

        Best for: "How does X relate to Y?", "Summarize", "Why does...", "Overview of"
        """
        if not self._community_summaries:
            logger.warning(
                "LightRAG high-level: no community summaries found. "
                "Run `rag graph communities --summarize` to build them first."
            )
            return []

        q_words = set(query.lower().split())

        # Score each community by keyword overlap with the query
        scored: list[tuple[float, int, str]] = []
        for comm_id, summary in self._community_summaries.items():
            summary_words = set(summary.lower().split())
            overlap = len(q_words & summary_words)
            # Normalize by summary length to avoid favoring long summaries
            score = overlap / max(len(summary_words), 1) * 10
            scored.append((score, comm_id, summary))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.max_high_level_communities]

        results = []
        for score, comm_id, summary in top:
            if score > 0:
                results.append(f"[Community {comm_id} | relevance={score:.2f}]\n{summary}")

        logger.debug("LightRAG high-level: scored %d communities, returning %d", len(scored), len(results))
        return results

    # ── Auto-routing ──────────────────────────────────────────────────────────

    def auto_retrieve(
        self,
        query: str,
        llm_fn: Callable[[str], str] | None = None,
    ) -> LightRAGResult:
        """
        Automatically route to low-level or high-level retrieval based on query type.

        If an llm_fn is provided, uses LLM classification for higher accuracy.
        Otherwise falls back to keyword-based classification.

        Args:
            query:   User query
            llm_fn:  Optional LLM function for query classification

        Returns:
            LightRAGResult with context chunks and metadata
        """
        # Classify query
        if llm_fn:
            resolved = self._llm_classify(query, llm_fn)
        else:
            resolved = classify_query(query)

        # Retrieve at appropriate level
        if resolved == "high":
            chunks       = self.high_level_retrieve(query)
            entities     = []
            communities  = [f"community_{i}" for i in range(len(chunks))]
        else:
            chunks       = self.low_level_retrieve(query)
            entities     = self._match_entities(query)
            communities  = []

        # Confidence: fraction of results returned (more results = more confident)
        max_expected = self.max_low_level_nodes if resolved == "low" else self.max_high_level_communities
        confidence = min(1.0, len(chunks) / max(max_expected, 1))

        return LightRAGResult(
            query=query,
            level="auto",
            resolved_level=resolved,
            context_chunks=chunks,
            entities_used=entities,
            communities_used=communities,
            confidence=confidence,
        )

    def _llm_classify(self, query: str, llm_fn: Callable[[str], str]) -> str:
        """Use LLM to classify query as low or high level."""
        prompt = (
            "Classify this query as either 'low' (specific fact-seeking) or 'high' (thematic/conceptual).\n\n"
            "LOW examples: 'Who founded X?', 'What year did Y happen?', 'List the features of Z'\n"
            "HIGH examples: 'How does X relate to Y?', 'Why is Z important?', 'Summarize the impact of...'\n\n"
            f"Query: {query}\n\n"
            "Reply with ONLY the word 'low' or 'high':"
        )
        try:
            result = llm_fn(prompt).strip().lower()
            return "high" if "high" in result else "low"
        except Exception:
            return classify_query(query)  # fallback to keyword

    # ── Combined retrieval (low + high merged) ────────────────────────────────

    def combined_retrieve(self, query: str) -> LightRAGResult:
        """
        Retrieve from both levels and merge results.

        Useful when query is ambiguous or when maximum context coverage is desired.
        Low-level results appear first (more specific), high-level context follows.
        """
        low_chunks  = self.low_level_retrieve(query)
        high_chunks = self.high_level_retrieve(query)

        # Interleave: 2 low + 1 high pattern for balance
        merged: list[str] = []
        li, hi = 0, 0
        while li < len(low_chunks) or hi < len(high_chunks):
            if li < len(low_chunks):
                merged.append(low_chunks[li]); li += 1
            if li < len(low_chunks):
                merged.append(low_chunks[li]); li += 1
            if hi < len(high_chunks):
                merged.append(high_chunks[hi]); hi += 1

        entities    = self._match_entities(query)
        communities = [f"community_{i}" for i in range(len(high_chunks))]
        confidence  = min(1.0, (len(low_chunks) + len(high_chunks)) / 15)

        return LightRAGResult(
            query=query,
            level="combined",
            resolved_level="combined",
            context_chunks=merged,
            entities_used=entities,
            communities_used=communities,
            confidence=confidence,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return graph and index statistics."""
        return {
            "graph_nodes":        self._graph.number_of_nodes(),
            "graph_edges":        self._graph.number_of_edges(),
            "community_summaries": len(self._community_summaries),
            "entity_index_size":  len(self._entity_index),
            "is_connected":       nx.is_connected(self._graph) if self._graph.number_of_nodes() > 0 else False,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_retriever: LightRAGRetriever | None = None


def get_light_rag() -> LightRAGRetriever:
    """Return the module-level LightRAGRetriever singleton, configured from settings."""
    global _retriever
    if _retriever is None:
        from config import settings
        _retriever = LightRAGRetriever(
            max_low_level_hops=settings.lightrag_max_low_level_hops,
            max_high_level_communities=settings.lightrag_max_communities,
        )
    return _retriever
