"""
Retrieval pipeline with state-of-the-art techniques:

  1. Dense vector search (ChromaDB cosine similarity)
  2. Sparse BM25 keyword search (rank-bm25)
  3. Reciprocal Rank Fusion (RRF) for hybrid score combination
  4. Cross-encoder reranking (ms-marco-MiniLM — sentence-transformers)
  5. Maximal Marginal Relevance (MMR) for diversity
  6. HyDE — Hypothetical Document Embeddings
  7. Multi-query expansion for higher recall
  8. CRAG — Corrective RAG self-evaluation loop
  9. Semantic query cache for near-zero latency on repeated questions
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import Settings, settings
from core.ingestion import get_chroma_client, get_embedding_model, get_or_create_collection
from models import (
    CacheEntry,
    QueryMode,
    QueryRequest,
    RetrievalContext,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

# ── Singleton cross-encoder ───────────────────────────────────────────────────
_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    """Lazy-load the cross-encoder model (warm-up only on first rerank call)."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder '%s'…", settings.reranker_model)
        _cross_encoder = CrossEncoder(settings.reranker_model)
        logger.info("Cross-encoder ready.")
    return _cross_encoder


# ── Semantic cache ────────────────────────────────────────────────────────────


class SemanticCache:
    """
    In-memory semantic cache for query-answer pairs.

    Embeds every query we answer. On new queries, computes cosine similarity
    against all cached embeddings. If any exceeds `threshold`, returns the
    cached response immediately — skipping retrieval and generation entirely.

    This mirrors production systems where 30-40% of queries are near-duplicates.
    Evicts LRU entries when `max_size` is reached.
    """

    def __init__(self, max_size: int = 500, threshold: float = 0.95) -> None:
        self.max_size = max_size
        self.threshold = threshold
        self._entries: list[CacheEntry] = []

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def get(self, query: str, query_embedding: list[float]) -> CacheEntry | None:
        """Return cached entry if a semantically similar query exists."""
        best_score, best_entry = 0.0, None
        for entry in self._entries:
            score = self._cosine_sim(query_embedding, entry.embedding)
            if score > best_score:
                best_score, best_entry = score, entry
        if best_score >= self.threshold and best_entry is not None:
            best_entry.hit_count += 1
            logger.debug("Cache HIT (score=%.4f) for query: '%s'", best_score, query[:60])
            return best_entry
        return None

    def put(self, query: str, response: "QueryResponse", embedding: list[float]) -> None:  # type: ignore[name-defined]  # noqa: F821
        """Store a new cache entry, evicting LRU (by hit_count) if at capacity."""
        if len(self._entries) >= self.max_size:
            # O(n) scan to find the min-hit entry — avoids O(n log n) sort on every insert
            min_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].hit_count)
            self._entries.pop(min_idx)
        self._entries.append(CacheEntry(question=query, response=response, embedding=embedding))

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


# Module-level cache singleton
_cache = SemanticCache(
    max_size=settings.cache_max_size,
    threshold=settings.cache_similarity_threshold,
) if settings.enable_cache else None


# ── BM25 index (cached per collection) ────────────────────────────────────────

# Cache BM25 indexes per collection name. Invalidated on ingest via invalidate_bm25_cache().
_bm25_cache: dict[str, "BM25Index"] = {}


def invalidate_bm25_cache(collection_name: str | None = None) -> None:
    """
    Invalidate the BM25 cache for a collection (or all collections).

    Call this after ingesting new documents so the next query rebuilds the index.
    Already called automatically by the ingestion module when chunks are added.
    """
    global _bm25_cache
    if collection_name is None:
        _bm25_cache.clear()
        logger.debug("BM25 cache: cleared all collections")
    elif collection_name in _bm25_cache:
        del _bm25_cache[collection_name]
        logger.debug("BM25 cache: invalidated '%s'", collection_name)


def get_cache() -> SemanticCache | None:
    """Return the module-level semantic cache."""
    return _cache


class BM25Index:
    """
    Lightweight BM25 keyword index backed by rank-bm25.

    Rebuilt each query from the full collection (since ChromaDB doesn't
    natively support sparse search). For large collections (>100K docs)
    this should be pre-computed and cached — see notes below.
    """

    def __init__(self, docs: list[str], ids: list[str]) -> None:
        tokenized = [doc.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.ids = ids
        self.docs = docs

    def query(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (doc_id, normalized_score) pairs for top_k BM25 matches."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
        if max_score == 0:
            return []
        return [(self.ids[i], float(scores[i]) / max_score) for i in top_indices if scores[i] > 0]


def _build_bm25_index(collection_name: str) -> BM25Index | None:
    """
    Fetch all documents from collection and build a BM25 index.

    Results are cached in _bm25_cache keyed by collection name.
    Invalidate with invalidate_bm25_cache(collection_name) after ingesting new docs.
    """
    global _bm25_cache
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]
    try:
        col = get_or_create_collection(collection_name)
        result = col.get(include=["documents"])
        docs = result.get("documents") or []
        ids = result.get("ids") or []
        if not docs:
            return None
        index = BM25Index(docs=docs, ids=ids)
        _bm25_cache[collection_name] = index
        logger.debug("BM25 cache: built index for '%s' (%d docs)", collection_name, len(docs))
        return index
    except Exception as e:
        logger.warning("Could not build BM25 index for '%s': %s", collection_name, e)
        return None


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i) across all retrievers.
    k=60 is the standard default from the original Cormack et al. paper.

    Args:
        rankings: list of ranked [(doc_id, score)] lists from different retrievers
        k: RRF constant (higher k = smaller penalty for low ranks)

    Returns:
        Merged ranked list of (doc_id, rrf_score), sorted descending
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for ranked_list in rankings:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ── MMR (Maximal Marginal Relevance) ─────────────────────────────────────────


def apply_mmr(
    query_embedding: list[float],
    results: list[RetrievalResult],
    embeddings: dict[str, list[float]],
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[RetrievalResult]:
    """
    Apply Maximal Marginal Relevance to balance relevance + diversity.

    MMR score = λ · sim(doc, query) - (1-λ) · max(sim(doc, selected))

    Args:
        query_embedding: embedded query vector
        results: candidate results (pre-filtered by threshold)
        embeddings: mapping of chunk_text -> embedding vector
        top_k: number of results to return
        lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity

    Returns:
        Diverse subset of results, reordered by MMR score
    """
    if not results:
        return []

    def cosine(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    selected: list[RetrievalResult] = []
    remaining = list(results)

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda r: r.similarity_score)
        else:
            # Subsequent picks: MMR score
            selected_embs = [embeddings.get(r.chunk_text, []) for r in selected]

            def mmr_score(r: RetrievalResult) -> float:
                emb = embeddings.get(r.chunk_text, [])
                if not emb or len(emb) == 0:
                    return 0.0
                relevance = cosine(emb, query_embedding)
                redundancy = max((cosine(emb, s) for s in selected_embs if s and len(s) > 0), default=0.0)
                return lambda_mult * relevance - (1 - lambda_mult) * redundancy

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


# ── Core dense retrieval ──────────────────────────────────────────────────────


def _dense_retrieve(
    query_embedding: list[float],
    collection_name: str,
    top_k: int,
) -> list[tuple[str, dict, float]]:
    """
    Query ChromaDB for top-k chunks by cosine similarity.

    Returns list of (doc_text, metadata, similarity_score) tuples.
    """
    col = get_or_create_collection(collection_name)
    if col.count() == 0:
        return []

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []
    raw_embs = results.get("embeddings", None)
    embeddings_raw = raw_embs[0] if (raw_embs is not None and len(raw_embs) > 0) else []

    out = []
    for doc, meta, dist, emb in zip(docs, metas, distances, embeddings_raw):
        # ChromaDB cosine distance → similarity: sim = 1 - dist (for normalized vecs)
        similarity = max(0.0, 1.0 - float(dist))
        # Convert numpy arrays to plain Python lists to avoid ambiguous truth value errors
        emb_list = emb.tolist() if hasattr(emb, "tolist") else (list(emb) if emb is not None else [])
        out.append((doc, meta, similarity, emb_list))
    return out


# ── Query transformation: HyDE ────────────────────────────────────────────────


def generate_hypothetical_document(question: str, generate_fn: Callable[[str], str]) -> str:
    """
    HyDE: Generate a hypothetical answer document and embed it instead of the raw query.

    The hypothesis doc is embedded to find chunks that look like answers,
    rather than chunks that look like questions. This bridges the vocabulary
    mismatch between short queries and long document passages.

    Args:
        question: user's original query
        generate_fn: callable that takes a prompt string and returns generated text

    Returns:
        Hypothetical answer paragraph (to be embedded)
    """
    prompt = (
        "Write a detailed, factual paragraph that would directly answer the following question. "
        "Focus on key facts and technical details. Do not include the question itself.\n\n"
        f"Question: {question}\n\nAnswer paragraph:"
    )
    hypothesis = generate_fn(prompt)
    logger.debug("HyDE hypothesis: %s…", hypothesis[:100])
    return hypothesis


# ── Query transformation: Multi-query expansion ───────────────────────────────


def expand_query(question: str, generate_fn: Callable[[str], str], n: int = 3) -> list[str]:
    """
    Generate N alternative phrasings of the query to increase recall.

    Different phrasings retrieve different chunks that may all be relevant
    but use different vocabulary than the original question.

    Args:
        question: original user query
        generate_fn: LLM callable (prompt -> text)
        n: number of alternative queries to generate

    Returns:
        List of alternative queries (including the original)
    """
    prompt = (
        f"Generate {n} different ways to ask the following question. "
        "Return ONLY the alternative questions, one per line, no numbering or extra text.\n\n"
        f"Original question: {question}\n\nAlternative questions:"
    )
    raw = generate_fn(prompt)
    alternatives = [q.strip() for q in raw.strip().splitlines() if q.strip() and q.strip() != question]
    all_queries = [question] + alternatives[:n]
    logger.debug("Multi-query expansion: %d queries", len(all_queries))
    return all_queries


# ── CRAG: Corrective RAG relevance evaluation ─────────────────────────────────


def evaluate_retrieval_quality(
    question: str,
    results: list[RetrievalResult],
    evaluate_fn: Callable[[str, list[str]], float],
) -> float:
    """
    CRAG: Have the LLM score how relevant the retrieved context is to the question.

    Returns a score in [0, 1]. If below 0.5, the caller should trigger
    query rewriting and re-retrieval.

    Args:
        question: user query
        results: retrieved chunks
        evaluate_fn: callable(question, chunk_texts) -> float [0,1]

    Returns:
        Average relevance score across all retrieved chunks
    """
    if not results:
        return 0.0
    chunk_texts = [r.chunk_text for r in results]
    score = evaluate_fn(question, chunk_texts)
    logger.debug("CRAG relevance score: %.3f", score)
    return score


# ── Main retrieval orchestrator ───────────────────────────────────────────────


def retrieve(
    request: QueryRequest,
    generate_fn: Callable[[str], str] | None = None,
    evaluate_fn: Callable[[str, list[str]], float] | None = None,
) -> RetrievalContext:
    """
    Full retrieval pipeline:
      1. (Optional) HyDE or multi-query expansion
      2. Dense vector search
      3. (Optional) BM25 sparse search + RRF fusion
      4. Filter by similarity threshold
      5. Cross-encoder reranking
      6. MMR diversity filtering
      7. (Optional) CRAG self-correction loop

    Args:
        request: QueryRequest with question, collection, top_k, mode, flags
        generate_fn: LLM callable for HyDE and multi-query (required if use_hyde or use_multi_query)
        evaluate_fn: LLM callable for CRAG relevance scoring (optional)

    Returns:
        RetrievalContext with ranked, diverse, filtered results
    """
    start = time.perf_counter()
    model = get_embedding_model()

    question = request.question
    expanded_queries: list[str] = []
    hyde_hypothesis: str | None = None

    # ── 1. Query transformation ────────────────────────────────────────────────
    queries_to_embed: list[str] = [question]

    use_hyde = getattr(request, "use_hyde", False) or settings.use_hyde
    use_multi_query = getattr(request, "use_multi_query", False) or settings.use_multi_query

    if use_hyde and generate_fn:
        hyde_hypothesis = generate_hypothetical_document(question, generate_fn)
        queries_to_embed = [hyde_hypothesis]  # embed hypothesis instead of raw query

    if use_multi_query and generate_fn:
        expanded_queries = expand_query(question, generate_fn, n=3)
        queries_to_embed = list(set(queries_to_embed + expanded_queries))

    # ── 2. Embed all queries ───────────────────────────────────────────────────
    all_embeddings = model.encode(queries_to_embed, normalize_embeddings=True)
    primary_embedding = all_embeddings[0].tolist()  # main query embedding

    # ── 3. Dense retrieval (all query variants, merge with RRF) ───────────────
    fetch_k = request.top_k * 4  # over-fetch for reranking
    dense_rankings: list[list[tuple[str, float]]] = []
    raw_docs: dict[str, tuple[str, dict, list[float]]] = {}  # id -> (text, meta, emb)

    for q_emb in all_embeddings:
        raw = _dense_retrieve(q_emb.tolist(), request.collection, fetch_k)
        ranked_list = []
        for doc, meta, score, emb in raw:
            doc_id = meta.get("content_hash", doc[:32])
            raw_docs[doc_id] = (doc, meta, emb if (emb is not None and len(emb) > 0) else [])
            ranked_list.append((doc_id, score))
        dense_rankings.append(ranked_list)

    if not raw_docs:
        logger.warning("No documents found in collection '%s'", request.collection)
        return RetrievalContext(
            query=question,
            results=[],
            query_mode=request.mode,
            expanded_queries=expanded_queries,
            hyde_hypothesis=hyde_hypothesis,
        )

    # ── 4. Sparse BM25 retrieval (if hybrid mode) ─────────────────────────────
    use_hybrid = (request.mode == QueryMode.HYBRID) and settings.use_hybrid_search
    bm25_ranking: list[tuple[str, float]] = []

    if use_hybrid:
        bm25_index = _build_bm25_index(request.collection)
        if bm25_index:
            bm25_hits = bm25_index.query(question, top_k=fetch_k)
            bm25_ranking = bm25_hits

    # ── 5. RRF fusion ─────────────────────────────────────────────────────────
    all_rankings = dense_rankings
    if bm25_ranking:
        all_rankings = dense_rankings + [bm25_ranking]

    fused = reciprocal_rank_fusion(all_rankings)

    # ── 6. Build RetrievalResult list with similarity scores ──────────────────
    # Map back from doc_id to score using weighted average of dense sim + bm25
    dense_score_map: dict[str, float] = {}
    for ranked_list in dense_rankings:
        for doc_id, score in ranked_list:
            dense_score_map[doc_id] = max(dense_score_map.get(doc_id, 0.0), score)

    candidates: list[RetrievalResult] = []
    embeddings_lookup: dict[str, list[float]] = {}

    for doc_id, rrf_score in fused:
        if doc_id not in raw_docs:
            continue
        doc_text, meta, emb = raw_docs[doc_id]
        sim_score = dense_score_map.get(doc_id, rrf_score)

        if sim_score < settings.similarity_threshold:
            continue

        embeddings_lookup[doc_text] = emb

        candidates.append(RetrievalResult(
            chunk_text=doc_text,
            source=meta.get("source_file", "unknown"),
            similarity_score=round(sim_score, 4),
            chunk_index=int(meta.get("chunk_index", 0)),
            page_number=int(meta["page_number"]) if meta.get("page_number", -1) != -1 else None,
            section_title=meta.get("section_title") or None,
            metadata={k: v for k, v in meta.items()},
        ))

    if not candidates:
        logger.info("No results above similarity threshold %.2f", settings.similarity_threshold)
        return RetrievalContext(
            query=question,
            results=[],
            query_mode=request.mode,
            expanded_queries=expanded_queries,
            hyde_hypothesis=hyde_hypothesis,
        )

    # ── 7. Cross-encoder reranking ────────────────────────────────────────────
    if settings.use_reranker and len(candidates) > 1:
        try:
            cross_encoder = get_cross_encoder()
            pairs = [(question, r.chunk_text) for r in candidates]
            rerank_scores = cross_encoder.predict(pairs)
            for result, score in zip(candidates, rerank_scores):
                result.rerank_score = float(score)
            candidates.sort(key=lambda r: r.rerank_score or 0, reverse=True)
            logger.debug("Cross-encoder reranked %d candidates", len(candidates))
        except Exception as e:
            logger.warning("Cross-encoder reranking failed: %s. Falling back to dense scores.", e)

    # ── 8. MMR diversity filtering ─────────────────────────────────────────────
    final_results = apply_mmr(
        query_embedding=primary_embedding,
        results=candidates,
        embeddings=embeddings_lookup,
        top_k=request.top_k,
        lambda_mult=settings.mmr_lambda,
    )

    # ── 9. CRAG self-correction + web search fallback ─────────────────────────
    crag_triggered = False
    web_fallback_triggered = False
    if evaluate_fn and final_results:
        quality = evaluate_retrieval_quality(question, final_results, evaluate_fn)
        if quality < 0.5:
            logger.info("CRAG: low quality score (%.2f) — attempting query rewrite", quality)
            crag_triggered = True
            if generate_fn:
                rewrite_prompt = (
                    f"The following question returned poor search results. "
                    f"Rewrite it to be more specific and keyword-rich for document search.\n\n"
                    f"Original: {question}\n\nRewritten:"
                )
                rewritten = generate_fn(rewrite_prompt).strip()
                logger.info("CRAG rewritten query: '%s'", rewritten)

                # Second-pass retrieval with rewritten query
                rewrite_emb = model.encode([rewritten], normalize_embeddings=True)[0].tolist()
                raw2 = _dense_retrieve(rewrite_emb, request.collection, fetch_k)
                new_candidates = []
                for doc, meta, score, emb in raw2:
                    if score < settings.similarity_threshold:
                        continue
                    new_candidates.append(RetrievalResult(
                        chunk_text=doc,
                        source=meta.get("source_file", "unknown"),
                        similarity_score=round(score, 4),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        page_number=int(meta["page_number"]) if meta.get("page_number", -1) != -1 else None,
                        section_title=meta.get("section_title") or None,
                        metadata=meta,
                    ))
                if new_candidates:
                    final_results = new_candidates[:request.top_k]
                    logger.info("CRAG: second-pass returned %d results", len(final_results))

            # ── Web search fallback (if CRAG still low quality) ───────────────
            if settings.web_search_fallback:
                # Re-evaluate after rewrite
                rewrite_quality = evaluate_retrieval_quality(question, final_results, evaluate_fn) if final_results else 0.0
                if rewrite_quality < settings.crag_web_fallback_threshold or not final_results:
                    logger.info("CRAG: triggering web search fallback (score=%.2f)", rewrite_quality)
                    try:
                        from core.web_search import web_search, web_results_to_retrieval
                        web_hits = web_search(question, max_results=4, provider=settings.web_search_provider)
                        if web_hits:
                            web_results = web_results_to_retrieval(web_hits)
                            # Merge: local results first, web results after
                            final_results = (final_results + web_results)[:request.top_k]
                            web_fallback_triggered = True
                            logger.info("Web fallback: added %d web results", len(web_hits))
                    except Exception as e:
                        logger.warning("Web search fallback failed: %s", e)

    # ── Prompt injection detection on retrieved chunks ─────────────────────────
    if settings.enable_injection_detection and final_results:
        try:
            from core.security import detect_injection
            clean_results = []
            blocked = 0
            for r in final_results:
                injection = detect_injection(r.chunk_text)
                if injection.is_injection:
                    blocked += 1
                    logger.warning("Blocked chunk from '%s' (injection score=%.2f)", r.source, injection.risk_score)
                    # Replace chunk text with warning rather than silently dropping
                    r = r.model_copy(update={"chunk_text": "[CHUNK SANITIZED: potential injection detected]"})
                clean_results.append(r)
            if blocked:
                logger.info("Injection detection: blocked/sanitized %d chunk(s)", blocked)
            final_results = clean_results
        except Exception as e:
            logger.warning("Injection detection failed: %s", e)

    elapsed = time.perf_counter() - start
    logger.info(
        "Retrieval: %d results in %.3fs (hybrid=%s, reranked=%s, crag=%s)",
        len(final_results), elapsed, use_hybrid, settings.use_reranker, crag_triggered,
    )

    return RetrievalContext(
        query=question,
        results=final_results,
        query_mode=request.mode,
        expanded_queries=expanded_queries,
        hyde_hypothesis=hyde_hypothesis,
    )


# ── Cache-aware retrieval entry point ─────────────────────────────────────────


def retrieve_with_cache(
    request: QueryRequest,
    generate_fn: Callable[[str], str] | None = None,
    evaluate_fn: Callable[[str, list[str]], float] | None = None,
) -> tuple[RetrievalContext, bool]:
    """
    Retrieve with semantic cache check.

    Returns:
        (RetrievalContext, cache_hit: bool)
    """
    if not settings.enable_cache or _cache is None:
        return retrieve(request, generate_fn, evaluate_fn), False

    model = get_embedding_model()
    q_emb = model.encode([request.question], normalize_embeddings=True)[0].tolist()

    # Note: cache stores full QueryResponse; we return a flag so generation
    # can bypass LLM if cache has a full response. Cache lookup here for
    # retrieval context is handled at the orchestration layer in generation.py.
    return retrieve(request, generate_fn, evaluate_fn), False


# get_cache defined above near the cache singleton definition
