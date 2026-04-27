"""
GraphRAG — Knowledge Graph construction and traversal.

Extracts entities and relationships from ingested documents, builds a
persistent knowledge graph with NetworkX, and combines graph-based context
retrieval with standard vector retrieval for richer answers.

This is one of the most impressive production RAG features of 2024-2025,
adopted by Microsoft Research, enterprise knowledge management, and legal tech.

Pipeline:
  Ingest docs → extract (entity, relation, entity) triples via LLM
              → store in NetworkX graph + persist to JSON
              → at query time: detect entities in query
              → traverse graph neighborhood (1-2 hops)
              → merge graph context with vector context
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

import networkx as nx

logger = logging.getLogger(__name__)

# Default graph storage path
GRAPH_PATH = Path("./data/knowledge_graph.json")


# ── Data structures ───────────────────────────────────────────────────────────


class Triple(NamedTuple):
    """A subject–predicate–object knowledge triple."""
    subject: str
    predicate: str
    obj: str
    source: str  # which document this came from


class GraphContext(NamedTuple):
    """Entities and relationships retrieved for a query."""
    entities_found: list[str]
    triples: list[Triple]
    narrative: str  # human-readable summary of graph context


# ── Entity and relation extraction via LLM ───────────────────────────────────


def extract_triples(
    text: str,
    source: str,
    llm_fn: Callable[[str], str],
    max_triples: int = 20,
) -> list[Triple]:
    """
    Extract (subject, predicate, object) triples from a text chunk using the LLM.

    Uses a structured JSON prompt to ensure parseable output. Falls back
    gracefully to empty list if the LLM returns malformed JSON.

    Args:
        text: document chunk text
        source: source filename for provenance
        llm_fn: callable that takes a prompt and returns text
        max_triples: max triples to extract per chunk (cost control)

    Returns:
        List of Triple namedtuples
    """
    prompt = (
        f"Extract up to {max_triples} factual relationships from the text below.\n"
        "Return ONLY valid JSON as a list of objects with keys: subject, predicate, object.\n"
        "Use short noun phrases for subjects/objects. Use verb phrases for predicates.\n"
        'Example: [{"subject": "Claude", "predicate": "is developed by", "object": "Anthropic"}]\n\n'
        f"TEXT:\n{text[:2000]}\n\n"
        "JSON triples:"
    )
    try:
        raw = llm_fn(prompt).strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
        # Find first JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        triples = []
        for item in data:
            if not isinstance(item, dict):
                continue
            s = str(item.get("subject", "")).strip()
            p = str(item.get("predicate", "")).strip()
            o = str(item.get("object", "")).strip()
            if s and p and o:
                triples.append(Triple(subject=s.lower(), predicate=p.lower(), obj=o.lower(), source=source))
        return triples
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug("Triple extraction failed for '%s': %s", source, e)
        return []


# ── Knowledge graph ───────────────────────────────────────────────────────────


class KnowledgeGraph:
    """
    Directed knowledge graph backed by NetworkX with JSON persistence.

    Nodes = entities (noun phrases)
    Edges = relationships (verb phrases) with source provenance

    Supports:
      - Adding triples from document ingestion
      - Entity-centric neighborhood queries (1-2 hops)
      - Shortest path discovery between entities
      - Subgraph extraction for multi-hop reasoning
      - Persistence to JSON for cross-session retention
    """

    def __init__(self, graph_path: Path = GRAPH_PATH) -> None:
        self.graph_path = graph_path
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._load()

    def _load(self) -> None:
        """Load graph from JSON if it exists."""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                logger.info(
                    "Loaded knowledge graph: %d nodes, %d edges",
                    self.graph.number_of_nodes(), self.graph.number_of_edges(),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Could not load knowledge graph: %s. Starting fresh.", e)
                self.graph = nx.MultiDiGraph()

    def save(self) -> None:
        """Persist graph to JSON."""
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(nx.node_link_data(self.graph), f, indent=2)
        logger.debug("Knowledge graph saved (%d nodes, %d edges)", self.graph.number_of_nodes(), self.graph.number_of_edges())

    def add_triples(self, triples: list[Triple]) -> int:
        """
        Add triples to the graph. Deduplicates by (subject, predicate, object).

        Returns:
            Number of new edges added
        """
        added = 0
        for triple in triples:
            # Add nodes with labels
            if triple.subject not in self.graph:
                self.graph.add_node(triple.subject, label=triple.subject)
            if triple.obj not in self.graph:
                self.graph.add_node(triple.obj, label=triple.obj)

            # Check for duplicate edges
            existing_edges = self.graph.edges(triple.subject, data=True, keys=True)
            duplicate = any(
                d.get("predicate") == triple.predicate and v == triple.obj
                for _, v, _, d in existing_edges
            )
            if not duplicate:
                self.graph.add_edge(
                    triple.subject,
                    triple.obj,
                    predicate=triple.predicate,
                    source=triple.source,
                )
                added += 1

        return added

    def query_entity(self, entity: str, hops: int = 2) -> list[Triple]:
        """
        Retrieve all triples within `hops` of an entity.

        Args:
            entity: entity name (case-insensitive)
            hops: number of relationship hops to traverse

        Returns:
            All triples in the neighborhood subgraph
        """
        entity = entity.lower()
        if entity not in self.graph:
            return []

        # Get all nodes within `hops` hops (both directions)
        neighbors_out = nx.ego_graph(self.graph, entity, radius=hops, undirected=False)
        neighbors_in = nx.ego_graph(self.graph.reverse(), entity, radius=hops, undirected=False)
        subgraph_nodes = set(neighbors_out.nodes) | set(neighbors_in.nodes)
        subgraph = self.graph.subgraph(subgraph_nodes)

        triples = []
        for u, v, data in subgraph.edges(data=True):
            triples.append(Triple(
                subject=u,
                predicate=data.get("predicate", "relates to"),
                obj=v,
                source=data.get("source", "unknown"),
            ))
        return triples

    def find_path(self, entity1: str, entity2: str) -> list[str] | None:
        """
        Find the shortest path between two entities in the graph.

        Returns list of entity names along the path, or None if unreachable.
        """
        e1, e2 = entity1.lower(), entity2.lower()
        if e1 not in self.graph or e2 not in self.graph:
            return None
        try:
            path = nx.shortest_path(self.graph.to_undirected(), e1, e2)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_entity_summary(self, entity: str) -> dict:
        """Return a dict of outgoing and incoming relationships for an entity."""
        entity = entity.lower()
        if entity not in self.graph:
            return {"entity": entity, "found": False, "outgoing": [], "incoming": []}

        outgoing = [
            {"predicate": d.get("predicate"), "object": v, "source": d.get("source")}
            for _, v, d in self.graph.out_edges(entity, data=True)
        ]
        incoming = [
            {"subject": u, "predicate": d.get("predicate"), "source": d.get("source")}
            for u, _, d in self.graph.in_edges(entity, data=True)
        ]
        return {"entity": entity, "found": True, "outgoing": outgoing, "incoming": incoming}

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "connected_components": nx.number_weakly_connected_components(self.graph),
        }

    def detect_communities(self) -> dict[int, list[str]]:
        """
        Detect entity communities using the Louvain method (via NetworkX).

        Communities are clusters of entities that are densely connected to each
        other — they represent coherent topics or concept groups in the KG.

        Returns:
            Dict mapping community_id → list of entity names
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        try:
            # NetworkX community detection (Louvain or greedy modularity)
            undirected = self.graph.to_undirected()
            try:
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(undirected)
            except (ImportError, AttributeError):
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(undirected))

            result: dict[int, list[str]] = {}
            for i, community in enumerate(communities):
                result[i] = sorted(community)

            logger.info("Detected %d communities in knowledge graph", len(result))
            return result

        except Exception as e:
            logger.warning("Community detection failed: %s", e)
            return {}

    def build_community_summaries(
        self,
        llm_fn: "Callable[[str], str]",  # type: ignore[name-defined]
        max_communities: int = 20,
    ) -> dict[int, str]:
        """
        Generate LLM summaries for each community (Microsoft GraphRAG pattern).

        Each community is summarized into a short paragraph describing the
        entities it contains and their relationships. These summaries enable
        "global" queries that reason over the entire knowledge graph's thematic
        structure rather than individual entity lookups.

        Args:
            llm_fn: LLM callable for summary generation
            max_communities: max communities to summarize (cost control)

        Returns:
            Dict mapping community_id → summary text
        """
        communities = self.detect_communities()
        summaries: dict[int, str] = {}

        for cid, entities in list(communities.items())[:max_communities]:
            # Collect all edges within this community
            subgraph = self.graph.subgraph(entities)
            edges_text = []
            for u, v, data in subgraph.edges(data=True):
                edges_text.append(f"{u} {data.get('predicate', 'relates to')} {v}")

            if not edges_text:
                continue

            prompt = (
                f"Summarize the following knowledge graph community in 2-3 sentences. "
                f"Describe what these entities have in common and their key relationships.\n\n"
                f"Entities: {', '.join(entities[:20])}\n"
                f"Relationships:\n" + "\n".join(edges_text[:30]) + "\n\n"
                f"Summary:"
            )
            try:
                summary = llm_fn(prompt).strip()
                summaries[cid] = summary
            except Exception as e:
                logger.warning("Community %d summary failed: %s", cid, e)

        logger.info("Built %d community summaries", len(summaries))
        return summaries

    def global_query(
        self,
        question: str,
        community_summaries: dict[int, str],
        llm_fn: "Callable[[str], str]",  # type: ignore[name-defined]
        top_communities: int = 5,
    ) -> str:
        """
        Answer a "global" query using community summaries (Microsoft GraphRAG).

        Global queries need high-level synthesis — they can't be answered by
        looking up a single entity. Instead, we:
          1. Embed the question and all community summaries
          2. Find the most relevant communities
          3. Pass their summaries as context for the LLM to synthesize an answer

        Args:
            question: high-level question (e.g., "What are the main themes?")
            community_summaries: pre-built community summaries
            llm_fn: LLM callable
            top_communities: number of most relevant communities to include

        Returns:
            Synthesized answer from community context
        """
        if not community_summaries:
            return "No community summaries available. Run build_community_summaries() first."

        # Simple relevance ranking by keyword overlap (replace with embedding sim for production)
        q_words = set(question.lower().split())
        scored = []
        for cid, summary in community_summaries.items():
            s_words = set(summary.lower().split())
            overlap = len(q_words & s_words)
            scored.append((cid, overlap, summary))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_communities]

        if not top:
            return "No relevant communities found for this question."

        context = "\n\n".join(
            f"[Community {cid} (overlap={overlap})]:\n{summary}"
            for cid, overlap, summary in top
        )

        prompt = (
            f"Based on the following knowledge graph community summaries, answer the question.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        try:
            return llm_fn(prompt)
        except Exception as e:
            return f"Global query failed: {e}"


# ── Entity detection in queries ───────────────────────────────────────────────


def detect_entities_in_query(query: str, graph: KnowledgeGraph, min_overlap: int = 3) -> list[str]:
    """
    Find graph entities mentioned in the query using substring matching.

    Args:
        query: user question
        graph: knowledge graph to search
        min_overlap: minimum characters in entity name to match

    Returns:
        List of entity names found in the query
    """
    query_lower = query.lower()
    found = []
    for node in graph.graph.nodes():
        if len(node) >= min_overlap and node in query_lower:
            found.append(node)
    # Sort by length descending to prefer more specific entities
    return sorted(found, key=len, reverse=True)[:10]


# ── Graph context retrieval ───────────────────────────────────────────────────


def retrieve_graph_context(query: str, graph: KnowledgeGraph, hops: int = 2) -> GraphContext:
    """
    Detect entities in query, traverse the knowledge graph, return context.

    Args:
        query: user question
        graph: populated knowledge graph
        hops: relationship hops to traverse per entity

    Returns:
        GraphContext with entities, triples, and narrative summary
    """
    entities = detect_entities_in_query(query, graph)

    if not entities:
        return GraphContext(entities_found=[], triples=[], narrative="")

    all_triples: list[Triple] = []
    seen: set[tuple] = set()

    for entity in entities[:5]:  # cap to 5 entities to control context size
        entity_triples = graph.query_entity(entity, hops=hops)
        for t in entity_triples:
            key = (t.subject, t.predicate, t.obj)
            if key not in seen:
                all_triples.append(t)
                seen.add(key)

    # Build a human-readable narrative
    if all_triples:
        lines = [f"- {t.subject} {t.predicate} {t.obj} [from: {t.source}]" for t in all_triples[:30]]
        narrative = "Knowledge Graph Context:\n" + "\n".join(lines)
    else:
        narrative = ""

    logger.info("GraphRAG: found %d entities, %d triples for query", len(entities), len(all_triples))
    return GraphContext(entities_found=entities, triples=all_triples, narrative=narrative)


# ── Module-level singleton ────────────────────────────────────────────────────

_graph: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Return the singleton knowledge graph (lazy load + auto-save on first use)."""
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph()
    return _graph


def load_graph(path: Path) -> nx.Graph:
    """
    Load a persisted knowledge graph from JSON and return it as an undirected Graph.

    Used by LightRAGRetriever to load the GraphRAG knowledge graph for dual-level
    retrieval. Converts MultiDiGraph → undirected Graph for LightRAG traversal.

    Args:
        path: Path to the JSON file saved by KnowledgeGraph.save()

    Returns:
        nx.Graph (undirected) with nodes and edges from the persisted graph
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    multi_di = nx.node_link_graph(data)
    # Convert to undirected simple Graph for LightRAG BFS traversal
    return nx.Graph(multi_di)
