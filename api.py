"""
FastAPI REST API for the RAG system — fully expanded.

Endpoints:
  POST   /ingest                    — ingest a document
  POST   /ingest/multimodal         — ingest PDF with tables + figures (vision LLM)
  POST   /raptor/ingest             — RAPTOR recursive tree ingestion
  POST   /query                     — standard Q&A with citations
  POST   /query/adaptive            — adaptive RAG (auto-selects strategy)
  POST   /query/stream              — streaming token-by-token response (SSE)
  POST   /chat/{session_id}         — multi-turn conversation
  DELETE /chat/{session_id}         — clear a conversation session
  GET    /chat/{session_id}/history — get conversation history
  GET    /collections               — list all knowledge bases
  DELETE /collection/{name}         — delete a knowledge base
  GET    /health                    — health check
  GET    /cache/stats               — semantic cache stats
  DELETE /cache                     — clear cache
  GET    /graph/stats               — knowledge graph stats
  GET    /graph/entity/{name}       — entity relationships from knowledge graph
  GET    /metrics                   — Prometheus metrics (if installed)
  POST   /route                     — auto-route a query to the best collection
  GET    /document/analyze          — analyze a document without ingesting
  POST   /feedback                  — record user feedback on a response
  GET    /feedback/summary          — feedback analytics
  GET    /feedback/export           — export feedback as JSONL
  POST   /finetune                  — trigger embedding fine-tuning pipeline
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, AsyncGenerator

from fastapi import FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from models import (
    CollectionInfo,
    CollectionListResponse,
    DeleteCollectionResponse,
    IngestRequest,
    IngestResult,
    QueryRequest,
    QueryResponse,
)
from monitoring import instrument_app, log_ingest_event, log_query_event, record_query

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("API starting — warming up models…")
    try:
        from core.ingestion import get_embedding_model
        get_embedding_model()
    except Exception as e:
        logger.warning("Embedding warm-up failed: %s", e)
    try:
        from core.generation import get_backend
        get_backend()
    except Exception as e:
        logger.warning("LLM backend warm-up failed: %s", e)
    try:
        from core.graph_rag import get_knowledge_graph
        get_knowledge_graph()
    except Exception as e:
        logger.warning("Graph warm-up failed: %s", e)
    yield
    logger.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="RAG System API",
    description=(
        "Production-grade Retrieval-Augmented Generation. "
        "Multi-backend (Ollama/Claude/OpenAI), hybrid retrieval, cross-encoder reranking, "
        "GraphRAG, streaming, multi-turn conversation, semantic caching, RAGAS evaluation, "
        "Adaptive RAG (Self-RAG), RAPTOR tree ingestion, multi-modal PDF (tables + vision), "
        "user feedback loop, and embedding fine-tuning."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach Prometheus metrics if available
instrument_app(app)


# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    llm_backend: str
    embedding_model: str
    chroma_path: str
    cache_enabled: bool
    graph_nodes: int = 0
    graph_edges: int = 0
    version: str = "4.0.0"


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """System health check with component status."""
    graph_stats = {"nodes": 0, "edges": 0}
    try:
        from core.graph_rag import get_knowledge_graph
        graph_stats = get_knowledge_graph().stats()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        llm_backend=settings.llm_backend.value,
        embedding_model=settings.embedding_model,
        chroma_path=str(settings.chroma_persist_dir),
        cache_enabled=settings.enable_cache,
        graph_nodes=graph_stats.get("nodes", 0),
        graph_edges=graph_stats.get("edges", 0),
    )


# ── Ingest ────────────────────────────────────────────────────────────────────


class IngestRequestExtended(IngestRequest):
    """Extended ingest request with advanced options."""
    chunking_strategy: str = Field(default="recursive", description="recursive | semantic | hierarchical")
    extract_graph: bool = Field(default=False, description="Extract entities/relations into knowledge graph")
    generate_summary: bool = Field(default=False, description="Generate LLM summary at ingest time")
    analyze_document: bool = Field(default=True, description="Run document quality analysis")


class IngestResultExtended(IngestResult):
    """Extended ingest result with analysis and graph stats."""
    quality_score: float = Field(default=1.0)
    language: str = Field(default="unknown")
    pii_warnings: list[str] = Field(default_factory=list)
    sections_detected: int = Field(default=0)
    graph_triples_added: int = Field(default=0)
    summary: str = Field(default="")


@app.post("/ingest", response_model=IngestResultExtended, tags=["Ingestion"])
async def ingest(request: IngestRequestExtended) -> IngestResultExtended:
    """
    Ingest a document with optional document analysis and knowledge graph extraction.
    Supports PDF, TXT, DOCX, Markdown, and URLs.
    """
    from core.ingestion import ingest_document, load_document
    from core.document_processor import analyze_document
    from core.graph_rag import get_knowledge_graph, extract_triples

    try:
        result = ingest_document(
            source=request.file_path,
            collection_name=request.collection,
            overwrite=request.overwrite,
            chunking_strategy=request.chunking_strategy,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Document analysis
    quality_score, language, pii_warnings, sections_count, summary = 1.0, "unknown", [], 0, ""
    if request.analyze_document or request.generate_summary:
        try:
            pages, _ = load_document(request.file_path)
            full_text = "\n\n".join(text for text, _ in pages)

            llm_fn = None
            if request.generate_summary:
                from core.generation import get_backend
                llm_fn = get_backend().complete_raw

            analysis = analyze_document(full_text, request.file_path, llm_fn=llm_fn, generate_summary=request.generate_summary)
            quality_score = analysis.quality_score
            language = analysis.language
            pii_warnings = analysis.pii_warnings
            sections_count = len(analysis.detected_sections)
            summary = analysis.summary
        except Exception as e:
            logger.warning("Document analysis failed: %s", e)

    # Knowledge graph extraction
    graph_triples_added = 0
    if request.extract_graph and result.chunks_added > 0:
        try:
            from core.generation import get_backend
            graph = get_knowledge_graph()
            backend = get_backend()
            pages, _ = load_document(request.file_path)
            # Sample first 5 pages to extract graph triples
            for page_text, _ in pages[:5]:
                triples = extract_triples(page_text, request.file_path, backend.complete_raw)
                graph_triples_added += graph.add_triples(triples)
            graph.save()
            logger.info("Graph: added %d triples from '%s'", graph_triples_added, request.file_path)
        except Exception as e:
            logger.warning("Graph extraction failed: %s", e)

    log_ingest_event(request.file_path, request.collection, result.chunks_added, result.elapsed_seconds)

    return IngestResultExtended(
        **result.model_dump(),
        quality_score=quality_score,
        language=language,
        pii_warnings=pii_warnings,
        sections_detected=sections_count,
        graph_triples_added=graph_triples_added,
        summary=summary,
    )


# ── Query ─────────────────────────────────────────────────────────────────────


class QueryRequestExtended(QueryRequest):
    """Extended query with graph RAG and routing options."""
    use_graph: bool = Field(default=False, description="Augment with knowledge graph context")
    auto_route: bool = Field(default=False, description="Auto-select collection based on query")
    session_id: str | None = Field(default=None, description="Session ID for conversation context")


class QueryResponseExtended(QueryResponse):
    """Extended response with graph and routing info."""
    graph_entities_found: list[str] = Field(default_factory=list)
    graph_triples_used: int = Field(default=0)
    routed_to: str | None = Field(default=None)


@app.post("/query", response_model=QueryResponseExtended, tags=["Query"])
async def query(request: QueryRequestExtended) -> QueryResponseExtended:
    """
    Query with hybrid retrieval, reranking, optional GraphRAG, and conversation context.
    """
    from core.generation import answer_question, get_backend, SYSTEM_PROMPT, build_user_prompt, extract_sources
    from core.retrieval import retrieve, get_cache
    from core.graph_rag import get_knowledge_graph, retrieve_graph_context
    from core.conversation import get_or_create_session
    from core.router import get_router
    from models import QueryMode
    import time

    start = time.perf_counter()
    backend = get_backend()

    # Auto-routing
    routed_to = None
    if request.auto_route:
        router = get_router()
        routed_to = router.route_single(request.question, use_llm=True, llm_fn=backend.complete_raw)
        request = request.model_copy(update={"collection": routed_to})

    # Conversation context
    conv_context = ""
    if request.session_id:
        session = get_or_create_session(request.session_id)
        # Resolve references ("it", "that") using conversation history
        resolved_q = session.resolve_references(request.question, backend.complete_raw)
        request = request.model_copy(update={"question": resolved_q})
        conv_context = session.build_context_prompt()

    try:
        response = answer_question(request)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Graph RAG augmentation
    graph_entities: list[str] = []
    graph_triples_used = 0
    if request.use_graph:
        try:
            graph = get_knowledge_graph()
            graph_ctx = retrieve_graph_context(request.question, graph, hops=2)
            graph_entities = graph_ctx.entities_found
            graph_triples_used = len(graph_ctx.triples)
            if graph_ctx.narrative and not response.answer.startswith("I don't have"):
                # Append graph context note to answer
                response = response.model_copy(update={
                    "answer": response.answer + f"\n\n---\n*Graph context: {graph_ctx.narrative[:500]}*"
                })
        except Exception as e:
            logger.warning("GraphRAG augmentation failed: %s", e)

    # Store in conversation memory
    if request.session_id:
        from core.conversation import get_or_create_session, ConversationTurn
        session = get_or_create_session(request.session_id)
        session.add_turn(ConversationTurn(
            question=request.question,
            answer=response.answer,
            sources=[s.source for s in response.sources],
            collection=request.collection,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
        ))

    # Prometheus metrics
    avg_sim = sum(s.similarity_score for s in response.sources) / max(len(response.sources), 1)
    record_query(len(response.sources), avg_sim, response.tokens_used, settings.llm_backend.value, response.model_used, response.cache_hit)
    log_query_event(request.question, request.collection, len(response.sources), response.tokens_used, response.latency_ms, response.cache_hit, settings.llm_backend.value)

    return QueryResponseExtended(
        **response.model_dump(),
        graph_entities_found=graph_entities,
        graph_triples_used=graph_triples_used,
        routed_to=routed_to,
    )


# ── Streaming query ───────────────────────────────────────────────────────────


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    Streaming RAG query using Server-Sent Events (SSE).

    Retrieval happens synchronously upfront, then tokens stream in real-time
    from the LLM. Compatible with Ollama streaming and Claude streaming APIs.

    Client usage:
        const es = new EventSource('/query/stream', {method: 'POST', body: JSON.stringify(req)})
        es.onmessage = (e) => { if (e.data !== '[DONE]') appendToken(JSON.parse(e.data).token) }
    """
    from core.retrieval import retrieve
    from core.generation import get_backend, SYSTEM_PROMPT, build_user_prompt, extract_sources
    import json as _json

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            backend = get_backend()

            # Retrieval (non-streaming)
            context = retrieve(request, generate_fn=backend.complete_raw)

            # Send retrieval metadata first
            sources = extract_sources(context)
            meta_event = _json.dumps({
                "event": "metadata",
                "sources": [s.model_dump() for s in sources],
                "chunks_retrieved": len(context.results),
            })
            yield f"data: {meta_event}\n\n"

            user_prompt = build_user_prompt(context)

            # Stream from Ollama
            if settings.llm_backend.value == "ollama":
                import requests as _req
                payload = {
                    "model": settings.ollama_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": True,
                    "options": {"temperature": settings.temperature},
                }
                with _req.post(f"{settings.ollama_base_url}/api/chat", json=payload, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line:
                            chunk = _json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield f"data: {_json.dumps({'token': token})}\n\n"
                            if chunk.get("done"):
                                break

            # Stream from Claude
            elif settings.llm_backend.value == "claude":
                import anthropic
                client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                with client.messages.stream(
                    model=settings.claude_model,
                    max_tokens=settings.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        yield f"data: {_json.dumps({'token': text})}\n\n"
                        await asyncio.sleep(0)  # yield to event loop

            else:
                # OpenAI streaming
                from openai import OpenAI
                client = OpenAI(api_key=settings.openai_api_key)
                stream = client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=True,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        yield f"data: {_json.dumps({'token': token})}\n\n"
                        await asyncio.sleep(0)

            yield "data: [DONE]\n\n"

        except Exception as e:
            import json as _j
            yield f"data: {_j.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Conversation (multi-turn chat) ────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    top_k: int = Field(default=6, ge=1, le=50)


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[dict] = Field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0
    turn_number: int = 0


@app.post("/chat/{session_id}", response_model=ChatResponse, tags=["Conversation"])
async def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    """
    Multi-turn conversational RAG with automatic reference resolution.

    Maintains conversation history across calls using the session_id.
    Automatically resolves references like "it", "that document", "the above".
    Compresses history when it grows too long to stay within context limits.
    """
    from core.conversation import get_or_create_session, ConversationTurn
    from core.generation import answer_question, get_backend
    from models import QueryMode
    import time

    session = get_or_create_session(session_id)
    backend = get_backend()

    # Resolve ambiguous references
    resolved_q = session.resolve_references(request.question, backend.complete_raw)

    q_request = QueryRequest(
        question=resolved_q,
        collection=request.collection,
        top_k=request.top_k,
        mode=QueryMode.HYBRID,
    )

    try:
        response = answer_question(q_request)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Store in session
    session.add_turn(ConversationTurn(
        question=request.question,
        answer=response.answer,
        sources=[s.source for s in response.sources],
        collection=request.collection,
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
    ))

    # Auto-compress if needed
    if len(session.turns) > session.summarize_after:
        try:
            session.compress(backend.complete_raw)
        except Exception as e:
            logger.warning("Session compression failed: %s", e)

    return ChatResponse(
        session_id=session_id,
        question=request.question,
        answer=response.answer,
        sources=[s.model_dump() for s in response.sources],
        tokens_used=response.tokens_used,
        latency_ms=response.latency_ms,
        turn_number=len(session.turns),
    )


@app.get("/chat/{session_id}/history", tags=["Conversation"])
async def get_chat_history(session_id: str) -> dict:
    """Get the full conversation history for a session."""
    from core.conversation import get_or_create_session
    session = get_or_create_session(session_id)
    return session.to_dict()


@app.delete("/chat/{session_id}", tags=["Conversation"])
async def clear_chat(session_id: str) -> dict:
    """Clear a conversation session's history."""
    from core.conversation import delete_session
    deleted = delete_session(session_id)
    return {"session_id": session_id, "cleared": deleted}


# ── Collections ───────────────────────────────────────────────────────────────


@app.get("/collections", response_model=CollectionListResponse, tags=["Collections"])
async def list_collections() -> CollectionListResponse:
    from core.ingestion import list_collections as _list
    try:
        raw = _list()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    collections = [CollectionInfo(name=c["name"], document_count=c["document_count"], embedding_model=c["embedding_model"]) for c in raw]
    return CollectionListResponse(collections=collections, total=len(collections))


@app.delete("/collection/{name}", response_model=DeleteCollectionResponse, tags=["Collections"])
async def delete_collection(name: str = Path(..., min_length=1)) -> DeleteCollectionResponse:
    from core.ingestion import delete_collection as _delete
    try:
        deleted = _delete(name)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found.")
    return DeleteCollectionResponse(name=name, deleted=True, message=f"Collection '{name}' deleted.")


# ── Knowledge Graph ───────────────────────────────────────────────────────────


@app.get("/graph/stats", tags=["Knowledge Graph"])
async def graph_stats() -> dict:
    """Return knowledge graph statistics."""
    from core.graph_rag import get_knowledge_graph
    graph = get_knowledge_graph()
    return graph.stats()


@app.get("/graph/entity/{entity_name}", tags=["Knowledge Graph"])
async def graph_entity(entity_name: str) -> dict:
    """Get all relationships for a specific entity from the knowledge graph."""
    from core.graph_rag import get_knowledge_graph
    graph = get_knowledge_graph()
    return graph.get_entity_summary(entity_name)


# ── Routing ───────────────────────────────────────────────────────────────────


class RouteRequest(BaseModel):
    question: str
    top_n: int = Field(default=1, ge=1, le=5)


@app.post("/route", tags=["Collections"])
async def route_query(request: RouteRequest) -> dict:
    """Auto-select the best collection(s) for a query using embedding similarity."""
    from core.router import get_router
    from core.generation import get_backend
    router = get_router()
    router.auto_register()
    collections = router.route(request.question, top_n=request.top_n)
    return {"question": request.question, "recommended_collections": collections}


# ── Cache ─────────────────────────────────────────────────────────────────────


class CacheStats(BaseModel):
    enabled: bool
    size: int
    max_size: int
    threshold: float


@app.get("/cache/stats", response_model=CacheStats, tags=["System"])
async def cache_stats() -> CacheStats:
    from core.retrieval import get_cache
    cache = get_cache()
    return CacheStats(enabled=settings.enable_cache, size=len(cache) if cache else 0, max_size=settings.cache_max_size, threshold=settings.cache_similarity_threshold)


@app.delete("/cache", tags=["System"])
async def clear_cache() -> dict:
    from core.retrieval import get_cache
    cache = get_cache()
    if cache:
        size = len(cache)
        cache.clear()
        return {"message": f"Cleared {size} entries."}
    return {"message": "Cache empty or disabled."}


# ── Multi-modal ingestion ─────────────────────────────────────────────────────


class MultimodalIngestRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute or relative path to a PDF file")
    collection: str = Field(default="default")
    extract_tables: bool = Field(default=True, description="Extract and store tables as markdown")
    extract_figures: bool = Field(default=True, description="Extract embedded images")
    describe_figures: bool = Field(default=True, description="Use Claude vision to describe figures")


class MultimodalIngestResponse(BaseModel):
    pdf: str
    tables_found: int
    figures_found: int
    elements_stored: int
    collection: str


@app.post("/ingest/multimodal", response_model=MultimodalIngestResponse, tags=["Ingestion"])
async def ingest_multimodal(request: MultimodalIngestRequest) -> MultimodalIngestResponse:
    """
    Extract tables (as markdown) and figures (described via vision LLM) from a PDF
    and store them as searchable chunks alongside text content.

    Requires pdfplumber and pymupdf: pip install pdfplumber pymupdf
    """
    from core.multimodal import ingest_pdf_multimodal

    try:
        summary = ingest_pdf_multimodal(
            pdf_path=request.pdf_path,
            collection_name=request.collection,
            extract_tables=request.extract_tables,
            extract_figures=request.extract_figures,
            describe_figures=request.describe_figures,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return MultimodalIngestResponse(**summary, collection=request.collection)


# ── RAPTOR ingestion ──────────────────────────────────────────────────────────


class RaptorIngestRequest(BaseModel):
    collection: str = Field(..., description="Source collection to build RAPTOR tree from")
    output_collection: str | None = Field(default=None, description="Target collection for summaries (defaults to <collection>_raptor)")
    max_levels: int = Field(default=3, ge=1, le=5)
    cluster_size: int = Field(default=10, ge=3, le=50)


class RaptorIngestResponse(BaseModel):
    source_collection: str
    raptor_collection: str
    levels_built: int
    summaries_stored: int
    total_nodes: int


@app.post("/raptor/ingest", response_model=RaptorIngestResponse, tags=["Ingestion"])
async def raptor_ingest(request: RaptorIngestRequest) -> RaptorIngestResponse:
    """
    Build a RAPTOR recursive tree from an existing collection.

    Clusters semantically similar chunks, summarizes each cluster with LLM,
    then recursively builds higher-level summaries. Enables multi-granularity
    retrieval: specific facts, section summaries, and document-level overviews.
    """
    from core.raptor import build_raptor_tree, ingest_raptor_tree
    from core.generation import get_backend

    backend = get_backend()
    out_col = request.output_collection or f"{request.collection}_raptor"

    try:
        tree = build_raptor_tree(
            collection_name=request.collection,
            llm_fn=backend.complete_raw,
            max_levels=request.max_levels,
            cluster_size=request.cluster_size,
        )
        stored = ingest_raptor_tree(tree, out_col)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return RaptorIngestResponse(
        source_collection=request.collection,
        raptor_collection=out_col,
        levels_built=len(tree.levels),
        summaries_stored=stored,
        total_nodes=sum(len(nodes) for nodes in tree.levels.values()),
    )


# ── Adaptive RAG query ────────────────────────────────────────────────────────


class AdaptiveQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    top_k: int = Field(default=6, ge=1, le=50)
    use_self_rag: bool = Field(default=True, description="Apply Self-RAG reflection tokens")
    max_hops: int = Field(default=3, ge=1, le=5, description="Max iterative retrieval hops")


class AdaptiveQueryResponse(BaseModel):
    answer: str
    strategy_used: str
    queries_used: list[str]
    hops: int
    latency_ms: float
    self_rag_flags: dict
    chunks_retrieved: int


@app.post("/query/adaptive", response_model=AdaptiveQueryResponse, tags=["Query"])
async def query_adaptive(request: AdaptiveQueryRequest) -> AdaptiveQueryResponse:
    """
    Adaptive RAG query that automatically selects the optimal retrieval strategy:

    - **NO_RETRIEVAL**: answers from model knowledge (math, general facts)
    - **SINGLE_STEP**: standard single-pass vector retrieval
    - **ITERATIVE**: multi-hop chained retrieval for complex questions

    Includes Self-RAG reflection: [Retrieve], [IsREL], [IsSUP], [IsUSE] tokens.
    """
    from core.adaptive_rag import adaptive_answer
    from core.generation import get_backend
    from core.retrieval import retrieve
    from models import QueryRequest, QueryMode

    backend = get_backend()

    def _retrieve_fn(q: str, col: str, k: int) -> list:
        req = QueryRequest(question=q, collection=col, top_k=k, mode=QueryMode.HYBRID)
        ctx = retrieve(req, generate_fn=backend.complete_raw)
        return ctx.results

    def _generate_fn(system: str, user: str) -> str:
        return backend.complete_raw(f"{system}\n\n{user}")

    try:
        result = adaptive_answer(
            question=request.question,
            collection=request.collection,
            llm_fn=backend.complete_raw,
            retrieve_fn=_retrieve_fn,
            generate_fn=_generate_fn,
            top_k=request.top_k,
            use_self_rag=request.use_self_rag,
            use_iterative=True,
            max_hops=request.max_hops,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return AdaptiveQueryResponse(
        answer=result.answer,
        strategy_used=result.strategy_used.value,
        queries_used=result.queries_used,
        hops=result.hops,
        latency_ms=result.latency_ms,
        self_rag_flags=result.self_rag_flags,
        chunks_retrieved=len(result.chunks_retrieved),
    )


# ── Feedback ──────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    collection: str = Field(default="default")
    feedback_type: str = Field(
        ...,
        description="thumbs_up | thumbs_down | correction | source_irrelevant | source_helpful | incomplete"
    )
    correction: str | None = Field(default=None, description="Correct answer (for correction feedback)")
    source_feedback: str | None = Field(default=None, description="Specific source file (for source feedback)")
    rating: int | None = Field(default=None, ge=1, le=5, description="Optional 1-5 star rating")
    sources_used: list[str] = Field(default_factory=list)
    session_id: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    recorded: bool


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Record user feedback on a RAG response.

    Feedback is persisted to SQLite and used for:
    - Analytics (satisfaction rate, failing queries, source quality)
    - Embedding fine-tuning via contrastive learning
    - Retrieval reranking bias (boost good sources, penalize bad ones)
    """
    from core.feedback import get_feedback_store, FeedbackEntry, FeedbackType

    try:
        ft = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid feedback_type: {request.feedback_type}")

    store = get_feedback_store()
    entry = FeedbackEntry(
        question=request.question,
        answer=request.answer,
        collection=request.collection,
        sources_used=request.sources_used,
        feedback_type=ft,
        correction=request.correction,
        source_feedback=request.source_feedback,
        rating=request.rating,
        session_id=request.session_id,
    )
    fid = store.record(entry)
    return FeedbackResponse(feedback_id=fid, recorded=True)


@app.get("/feedback/summary", tags=["Feedback"])
async def feedback_summary(collection: str | None = None) -> dict:
    """
    Aggregate feedback analytics: satisfaction rate, top failing queries,
    best and worst sources.
    """
    from core.feedback import get_feedback_store
    store = get_feedback_store()
    summary = store.get_summary(collection)
    return summary.model_dump()


@app.get("/feedback/export", tags=["Feedback"])
async def feedback_export(collection: str | None = None) -> dict:
    """Export all feedback entries as a list (for offline analysis or fine-tuning)."""
    from core.feedback import get_feedback_store
    import tempfile, json as _json
    from pathlib import Path

    store = get_feedback_store()
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    n = store.export_jsonl(tmp, collection)
    records = []
    if tmp.exists():
        with open(tmp) as f:
            for line in f:
                records.append(_json.loads(line))
        tmp.unlink()
    return {"count": n, "records": records}


@app.get("/feedback/boost-factors", tags=["Feedback"])
async def feedback_boost_factors(collection: str = "default") -> dict:
    """
    Per-source boost/penalty factors computed from historical feedback.
    Sources > 1.0 are boosted (high thumbs-up). Sources < 1.0 are penalized.
    """
    from core.feedback import get_feedback_store, get_source_boost_factors
    store = get_feedback_store()
    factors = get_source_boost_factors(collection, store)
    return {"collection": collection, "boost_factors": factors}


# ── Embedding fine-tuning ─────────────────────────────────────────────────────


class FinetuneRequest(BaseModel):
    collection: str = Field(..., description="Collection to generate synthetic pairs from")
    epochs: int = Field(default=3, ge=1, le=20)
    use_feedback: bool = Field(default=True)
    use_synthetic: bool = Field(default=True)


class FinetuneResponse(BaseModel):
    model_path: str
    training_pairs: int
    test_pairs: int
    baseline_mrr: float
    finetuned_mrr: float
    improvement_pct: float


@app.post("/finetune", response_model=FinetuneResponse, tags=["Training"])
async def finetune_embeddings(request: FinetuneRequest) -> FinetuneResponse:
    """
    Fine-tune the embedding model on domain-specific data.

    1. Collects training pairs from feedback corrections and synthetic LLM-generated Q&A
    2. Mines hard negatives (semantically similar but wrong chunks)
    3. Fine-tunes with MultipleNegativesRankingLoss
    4. Evaluates improvement via MRR

    This is a long-running operation (minutes). Consider running async in production.
    Requires: pip install sentence-transformers[train]
    """
    from core.embedding_finetuner import run_finetuning_pipeline
    from core.generation import get_backend
    from core.ingestion import embed_texts

    backend = get_backend()

    try:
        results = run_finetuning_pipeline(
            collection_name=request.collection,
            llm_fn=backend.complete_raw,
            embed_fn=embed_texts,
            use_feedback=request.use_feedback,
            use_synthetic=request.use_synthetic,
            epochs=request.epochs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if "error" in results:
        raise HTTPException(status_code=422, detail=results["error"])

    return FinetuneResponse(
        model_path=results.get("model_path", ""),
        training_pairs=results.get("training_pairs", 0),
        test_pairs=results.get("test_pairs", 0),
        baseline_mrr=results.get("baseline_mrr", 0.0),
        finetuned_mrr=results.get("finetuned_mrr", 0.0),
        improvement_pct=results.get("improvement_pct", 0.0),
    )


# ── Agentic RAG ───────────────────────────────────────────────────────────────


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    max_iterations: int = Field(default=8, ge=1, le=20)


class AgentResponse(BaseModel):
    answer: str
    tool_calls: list[dict] = Field(default_factory=list)
    total_tokens: int = 0
    latency_ms: float = 0.0
    iterations: int = 0
    model_used: str = ""


@app.post("/query/agent", response_model=AgentResponse, tags=["Query"])
async def query_agent(request: AgentRequest) -> AgentResponse:
    """
    Agentic RAG — the LLM decides which tools to call in sequence.

    Available tools: search_docs, search_web, query_sql, calculate, get_date, summarize_collection.
    The agent chains tools until it has a complete answer. Requires ANTHROPIC_API_KEY.
    """
    from core.agent import run_agent
    from core.retrieval import retrieve
    from core.generation import get_backend

    backend = get_backend()

    def _retrieve_fn(req):
        return retrieve(req, generate_fn=backend.complete_raw)

    try:
        result = run_agent(
            question=request.question,
            collection=request.collection,
            retrieve_fn=_retrieve_fn,
            max_iterations=request.max_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return AgentResponse(
        answer=result.answer,
        tool_calls=[{"tool": tc.tool_name, "input": tc.tool_input, "result_preview": tc.result[:200], "latency_ms": tc.latency_ms} for tc in result.tool_calls],
        total_tokens=result.total_tokens,
        latency_ms=result.latency_ms,
        iterations=result.iterations,
        model_used=result.model_used,
    )


# ── Structured extraction ─────────────────────────────────────────────────────


class StructuredQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    top_k: int = Field(default=6)
    output_schema: dict = Field(
        default_factory=dict,
        description="JSON Schema describing the desired output structure",
        examples=[{"revenue": {"type": "number"}, "period": {"type": "string"}}],
    )


@app.post("/query/structured", tags=["Query"])
async def query_structured(request: StructuredQueryRequest) -> dict:
    """
    Extract structured JSON from retrieved context instead of prose.

    Useful for dashboards, downstream APIs, or automated pipelines that
    need typed data rather than text answers.

    Example: revenue figures → {"revenue": 2.3, "unit": "billion", "change": "+15%"}
    """
    from core.generation import answer_structured
    from models import QueryMode

    q_request = QueryRequest(
        question=request.question,
        collection=request.collection,
        top_k=request.top_k,
        mode=QueryMode.HYBRID,
    )
    try:
        result = answer_structured(q_request, request.output_schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return result


# ── SQL / Structured data ─────────────────────────────────────────────────────


class SQLQueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    database: str | None = Field(default=None, description="DB URL or path (uses SQL_DATABASE_URL from config if omitted)")


class SQLQueryResponse(BaseModel):
    question: str
    result: str
    database: str


@app.post("/sql/query", response_model=SQLQueryResponse, tags=["SQL"])
async def sql_query(request: SQLQueryRequest) -> SQLQueryResponse:
    """
    Text-to-SQL: generate and execute a SQL query from natural language.
    Combines with vector retrieval for hybrid structured+unstructured answers.
    """
    from core.sql_retrieval import query_natural_language, get_db_url
    try:
        result = query_natural_language(request.question, request.database)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return SQLQueryResponse(question=request.question, result=result, database=get_db_url(request.database))


@app.get("/sql/schema", tags=["SQL"])
async def sql_schema(database: str | None = None) -> dict:
    """Return the schema of the configured SQL database."""
    from core.sql_retrieval import get_schema, get_db_url
    return {"schema": get_schema(database), "database": get_db_url(database)}


@app.post("/sql/setup-sample", tags=["SQL"])
async def sql_setup_sample() -> dict:
    """Create a sample SQLite database with products/customers/orders for demos."""
    from core.sql_retrieval import create_sample_db
    path = create_sample_db()
    return {"message": "Sample database created.", "path": str(path), "hint": f"Set SQL_DATABASE_URL=sqlite:///{path} in .env"}


# ── Security ──────────────────────────────────────────────────────────────────


@app.get("/security/audit", tags=["Security"])
async def security_audit(days: int = 7) -> dict:
    """Return security audit summary: PII rates, injection attempts, sensitive queries."""
    from core.security import get_audit_summary
    return get_audit_summary(days)


class ScanRequest(BaseModel):
    text: str
    redact: bool = Field(default=False, description="If true, return redacted version of the text")


@app.post("/security/scan", tags=["Security"])
async def security_scan(request: ScanRequest) -> dict:
    """Scan text for PII patterns and prompt injection attempts."""
    from core.security import detect_pii, detect_injection, redact_pii
    pii = redact_pii(request.text) if request.redact else detect_pii(request.text)
    inj = detect_injection(request.text)
    return {
        "has_pii": pii.has_pii,
        "pii_types": pii.pii_types,
        "redacted_text": pii.redacted_text if request.redact else None,
        "redaction_count": pii.redaction_count if request.redact else 0,
        "is_injection": inj.is_injection,
        "injection_patterns": inj.matched_patterns,
        "injection_risk_score": inj.risk_score,
    }


# ── GraphRAG Communities ──────────────────────────────────────────────────────


@app.post("/graph/communities", tags=["Knowledge Graph"])
async def graph_communities(summarize: bool = False) -> dict:
    """
    Detect entity communities in the knowledge graph using Louvain method.
    Optionally generate LLM summaries for each community (Microsoft GraphRAG pattern).
    """
    from core.graph_rag import get_knowledge_graph
    graph = get_knowledge_graph()
    communities = graph.detect_communities()
    result = {
        "community_count": len(communities),
        "communities": {str(k): v[:10] for k, v in communities.items()},  # top 10 entities per community
    }
    if summarize:
        from core.generation import get_backend
        backend = get_backend()
        summaries = graph.build_community_summaries(backend.complete_raw)
        result["summaries"] = {str(k): v for k, v in summaries.items()}
    return result


class GlobalQueryRequest(BaseModel):
    question: str = Field(..., description="High-level question to answer via community summaries")
    top_communities: int = Field(default=5, ge=1, le=20)


@app.post("/graph/global-query", tags=["Knowledge Graph"])
async def graph_global_query(request: GlobalQueryRequest) -> dict:
    """
    Answer a global question using GraphRAG community summaries.
    Better than entity lookup for thematic questions like 'What are the main topics?'
    """
    from core.graph_rag import get_knowledge_graph
    from core.generation import get_backend
    graph = get_knowledge_graph()
    backend = get_backend()
    summaries = graph.build_community_summaries(backend.complete_raw)
    answer = graph.global_query(request.question, summaries, backend.complete_raw, top_communities=request.top_communities)
    return {"question": request.question, "answer": answer, "communities_used": min(request.top_communities, len(summaries))}


# ── Observability ─────────────────────────────────────────────────────────────


@app.get("/observability/status", tags=["System"])
async def observability_status() -> dict:
    """Check if Langfuse tracing is active."""
    from core.observability import is_enabled
    return {"langfuse_enabled": is_enabled()}


@app.post("/observability/score", tags=["System"])
async def score_trace_endpoint(trace_id: str, score: float, name: str = "user_feedback") -> dict:
    """Attach a user feedback score to a Langfuse trace (1.0=thumbs up, 0.0=thumbs down)."""
    from core.observability import score_trace
    score_trace(trace_id, score, name)
    return {"trace_id": trace_id, "score": score, "name": name}


# ── CoT-RAG (Chain-of-Thought, EMNLP 2025) ────────────────────────────────────


class CoTRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer with chain-of-thought reasoning")
    collection: str = Field(default="default")
    top_k: int = Field(default=6, ge=1, le=30)
    max_steps: int = Field(default=4, ge=1, le=8, description="Max reasoning steps to decompose into")
    top_k_per_step: int = Field(default=3, ge=1, le=10, description="Chunks to retrieve per step")


class CoTStepResponse(BaseModel):
    step_number: int
    thought: str
    sub_query: str
    retrieved_count: int
    intermediate: str
    latency_ms: float


class CoTResponse(BaseModel):
    question: str
    answer: str
    reasoning_steps: list[CoTStepResponse]
    all_sources: list[str]
    total_chunks: int
    tokens_used: int
    latency_ms: float
    num_steps: int
    warnings: list[str] = Field(default_factory=list)


@app.post("/query/cot", response_model=CoTResponse, tags=["Query"])
async def query_cot(request: CoTRequest) -> CoTResponse:
    """
    Chain-of-Thought RAG (CoT-RAG) — EMNLP 2025.

    Decomposes the question into explicit reasoning steps, retrieves targeted
    context for each step, and synthesizes with the full reasoning trace.

    Best for: multi-hop questions, questions requiring facts from multiple
    document sections, complex analytical queries.

    Returns both the answer and the step-by-step reasoning trace.
    """
    from core.cot_rag import run_cot_rag
    from core.retrieval import retrieve
    from core.generation import get_backend

    backend = get_backend()

    try:
        result = run_cot_rag(
            question=request.question,
            collection=request.collection,
            retrieve_fn=retrieve,
            llm_fn=backend.complete_raw,
            max_steps=request.max_steps,
            top_k_per_step=request.top_k_per_step,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return CoTResponse(
        question=result.question,
        answer=result.answer,
        reasoning_steps=[
            CoTStepResponse(
                step_number=s.step_number,
                thought=s.thought,
                sub_query=s.sub_query,
                retrieved_count=len(s.retrieved),
                intermediate=s.intermediate,
                latency_ms=s.latency_ms,
            )
            for s in result.reasoning_steps
        ],
        all_sources=result.all_sources,
        total_chunks=result.total_chunks,
        tokens_used=result.tokens_used,
        latency_ms=result.latency_ms,
        num_steps=result.num_steps,
        warnings=result.warnings,
    )


# ── TTRAG (ICLR 2025) ─────────────────────────────────────────────────────────


class TTRAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    max_iterations: int = Field(default=4, ge=1, le=8)
    top_k: int = Field(default=6, ge=1, le=20)
    sufficiency_threshold: float = Field(default=0.55, ge=0.1, le=0.95)


class TTRAGIterationResponse(BaseModel):
    iteration: int
    query_used: str
    rewrite_reason: str
    new_chunks_retrieved: int
    sufficiency_score: float
    latency_ms: float


class TTRAGResponse(BaseModel):
    question: str
    answer: str
    iterations: list[TTRAGIterationResponse]
    num_iterations: int
    unique_chunks_used: int
    final_sufficiency: float
    converged: bool
    tokens_used: int
    latency_ms: float


@app.post("/query/ttrag", response_model=TTRAGResponse, tags=["Query"])
async def query_ttrag(request: TTRAGRequest) -> TTRAGResponse:
    """
    TTRAG — Test-Time Compute Scaling for RAG (ICLR 2025).

    Iteratively rewrites the query and re-retrieves until sufficient context
    is found. Each iteration uses LLM-guided query rewriting to target gaps
    in the previously retrieved content.

    Best for: hard questions where one-shot retrieval misses the right chunks,
    queries with ambiguous terminology, and cases where Sufficient Context
    would otherwise abstain.
    """
    from core.ttrag import run_ttrag
    from core.retrieval import retrieve
    from core.generation import get_backend, build_user_prompt, SYSTEM_PROMPT

    backend = get_backend()

    def _generate(q: str, ctx) -> tuple[str, int]:
        prompt = build_user_prompt(ctx)
        answer, tokens, _ = backend.complete(SYSTEM_PROMPT, prompt)
        return answer, tokens

    try:
        result = run_ttrag(
            question=request.question,
            collection=request.collection,
            retrieve_fn=retrieve,
            llm_fn=backend.complete_raw,
            generate_fn=_generate,
            max_iterations=request.max_iterations,
            top_k=request.top_k,
            sufficiency_threshold=request.sufficiency_threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return TTRAGResponse(
        question=result.question,
        answer=result.answer,
        iterations=[
            TTRAGIterationResponse(
                iteration=it.iteration,
                query_used=it.query_used,
                rewrite_reason=it.rewrite_reason,
                new_chunks_retrieved=len(it.retrieved),
                sufficiency_score=it.sufficiency.overall_score,
                latency_ms=it.latency_ms,
            )
            for it in result.iterations
        ],
        num_iterations=result.num_iterations,
        unique_chunks_used=result.unique_chunks_used,
        final_sufficiency=result.final_sufficiency,
        converged=result.converged,
        tokens_used=result.tokens_used,
        latency_ms=result.latency_ms,
    )


# ── Speculative RAG (Google Research 2024) ───────────────────────────────────


class SpeculativeRAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    num_drafts: int = Field(default=3, ge=2, le=6)
    top_k: int = Field(default=9, ge=3, le=30)


class SpeculativeDraftResponse(BaseModel):
    draft_id: int
    confidence_score: float
    answer: str
    num_chunks: int
    latency_ms: float


class SpeculativeRAGResponse(BaseModel):
    question: str
    answer: str
    selected_draft_id: int
    all_drafts: list[SpeculativeDraftResponse]
    num_drafts: int
    total_chunks_retrieved: int
    latency_reduction_pct: float
    tokens_used: int
    latency_ms: float


@app.post("/query/speculative", response_model=SpeculativeRAGResponse, tags=["Query"])
async def query_speculative(request: SpeculativeRAGRequest) -> SpeculativeRAGResponse:
    """
    Speculative RAG — Google Research (2024).

    Generates N independent draft answers from document subsets, scores each
    with LLM self-rating, and returns the highest-confidence draft.
    Achieves ~51% latency reduction vs. full-context generation.
    """
    from core.speculative_rag import run_speculative_rag
    from core.retrieval import retrieve
    from core.generation import get_backend

    backend = get_backend()

    try:
        result = run_speculative_rag(
            question=request.question,
            collection=request.collection,
            retrieve_fn=retrieve,
            llm_complete_fn=backend.complete,
            llm_raw_fn=backend.complete_raw,
            num_drafts=request.num_drafts,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return SpeculativeRAGResponse(
        question=result.question,
        answer=result.answer,
        selected_draft_id=result.selected_draft_id,
        all_drafts=[
            SpeculativeDraftResponse(
                draft_id=d.draft_id,
                confidence_score=d.confidence_score,
                answer=d.answer,
                num_chunks=len(d.chunks),
                latency_ms=d.latency_ms,
            )
            for d in result.all_drafts
        ],
        num_drafts=result.num_drafts,
        total_chunks_retrieved=result.total_chunks_retrieved,
        latency_reduction_pct=result.latency_reduction_pct,
        tokens_used=result.tokens_used,
        latency_ms=result.latency_ms,
    )


# ── A-RAG (Feb 2026) ─────────────────────────────────────────────────────────


class ARAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    max_steps: int = Field(default=5, ge=1, le=10)
    top_k_per_step: int = Field(default=4, ge=1, le=10)


class ARAGStepResponse(BaseModel):
    step: int
    tool_chosen: str
    query: str
    reasoning: str
    new_chunks_retrieved: int
    latency_ms: float


class ARAGResponse(BaseModel):
    question: str
    answer: str
    steps: list[ARAGStepResponse]
    num_steps: int
    unique_chunks: int
    tools_used: list[str]
    tokens_used: int
    latency_ms: float


@app.post("/query/arag", response_model=ARAGResponse, tags=["Query"])
async def query_arag(request: ARAGRequest) -> ARAGResponse:
    """
    A-RAG — Hierarchical Retrieval Interfaces (Feb 2026).

    The agent dynamically selects the retrieval interface at each step:
    keyword_search (BM25), semantic_search (dense), hybrid_search, or
    read_section (fetch from a specific source). Most cutting-edge agentic
    RAG pattern — treats retrieval as a decision, not a fixed pipeline.
    """
    from core.arag import run_arag
    from core.retrieval import retrieve
    from core.generation import get_backend

    backend = get_backend()

    try:
        result = run_arag(
            question=request.question,
            collection=request.collection,
            retrieve_fn=retrieve,
            llm_raw_fn=backend.complete_raw,
            llm_complete_fn=backend.complete,
            max_steps=request.max_steps,
            top_k_per_step=request.top_k_per_step,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return ARAGResponse(
        question=result.question,
        answer=result.answer,
        steps=[
            ARAGStepResponse(
                step=s.step,
                tool_chosen=s.tool_chosen,
                query=s.query,
                reasoning=s.reasoning,
                new_chunks_retrieved=len(s.retrieved),
                latency_ms=s.latency_ms,
            )
            for s in result.steps
        ],
        num_steps=result.num_steps,
        unique_chunks=result.unique_chunks,
        tools_used=result.tools_used,
        tokens_used=result.tokens_used,
        latency_ms=result.latency_ms,
    )


# ── LightRAG (EMNLP 2025) ─────────────────────────────────────────────────────


class LightRAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    level: str = Field(
        default="auto",
        description="Retrieval level: auto | low | high | combined",
        pattern="^(auto|low|high|combined)$",
    )


class LightRAGResponse(BaseModel):
    question: str
    level: str
    resolved_level: str
    context_chunks: list[str]
    entities_used: list[str]
    communities_used: list[str]
    confidence: float


@app.post("/lightrag/query", response_model=LightRAGResponse, tags=["Knowledge Graph"])
async def lightrag_query(request: LightRAGRequest) -> LightRAGResponse:
    """
    LightRAG dual-level graph retrieval (EMNLP 2025).

    Two retrieval modes over the knowledge graph:
    - **low**: precise entity/relationship queries (specific facts)
    - **high**: thematic community-level queries (broad concepts)
    - **auto**: automatically routes based on query type
    - **combined**: merges both modes (maximum coverage)

    Requires a knowledge graph built via POST /ingest with extract_graph=true.
    """
    from core.light_rag import get_light_rag
    from core.generation import get_backend

    lr = get_light_rag()
    backend = get_backend()

    try:
        if request.level == "low":
            chunks   = lr.low_level_retrieve(request.question)
            entities = lr._match_entities(request.question)
            comms: list[str] = []
            resolved = "low"
        elif request.level == "high":
            chunks   = lr.high_level_retrieve(request.question)
            entities = []
            comms    = [f"community_{i}" for i in range(len(chunks))]
            resolved = "high"
        elif request.level == "combined":
            result   = lr.combined_retrieve(request.question)
            chunks   = result.context_chunks
            entities = result.entities_used
            comms    = result.communities_used
            resolved = "combined"
        else:  # auto
            result   = lr.auto_retrieve(request.question, llm_fn=backend.complete_raw)
            chunks   = result.context_chunks
            entities = result.entities_used
            comms    = result.communities_used
            resolved = result.resolved_level
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    confidence = min(1.0, len(chunks) / max(10, 1))
    return LightRAGResponse(
        question=request.question,
        level=request.level,
        resolved_level=resolved,
        context_chunks=chunks[:10],  # cap for API response size
        entities_used=entities,
        communities_used=comms,
        confidence=round(confidence, 3),
    )


@app.get("/lightrag/stats", tags=["Knowledge Graph"])
async def lightrag_stats() -> dict:
    """LightRAG graph and index statistics."""
    from core.light_rag import get_light_rag
    return get_light_rag().stats()


# ── Sufficient Context check (Google ICLR 2025) ───────────────────────────────


class SufficiencyRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collection: str = Field(default="default")
    top_k: int = Field(default=6, ge=1, le=30)
    enable_self_rating: bool = Field(
        default=False,
        description="Ask LLM to self-rate confidence (adds ~200ms latency)",
    )


class SufficiencyResponse(BaseModel):
    is_sufficient: bool
    overall_score: float
    density_score: float
    coverage_score: float
    num_chunks: int
    recommendation: str
    explanation: str
    component_scores: dict


@app.post("/check-context", response_model=SufficiencyResponse, tags=["Query"])
async def check_context_sufficiency(request: SufficiencyRequest) -> SufficiencyResponse:
    """
    Sufficient Context check — Google ICLR 2025.

    Scores whether retrieved context is sufficient to answer the question
    before committing to a full LLM generation call.

    Returns a recommendation: generate | retrieve_more | web_search | abstain.

    Use this to build confidence-gated UIs, route low-confidence queries
    to escalation paths, or implement cost controls.
    """
    from core.retrieval import retrieve
    from core.generation import get_backend, make_crag_evaluator
    from core.sufficient_context import check_sufficiency
    from models import QueryMode

    backend = get_backend()
    req = QueryRequest(
        question=request.question,
        collection=request.collection,
        top_k=request.top_k,
        mode=QueryMode.HYBRID,
    )
    try:
        evaluate_fn = make_crag_evaluator(backend) if settings.use_hybrid_search else None
        context = retrieve(req, generate_fn=backend.complete_raw, evaluate_fn=evaluate_fn)
        result  = check_sufficiency(
            question=request.question,
            context=context,
            llm_fn=backend.complete_raw if request.enable_self_rating else None,
            enable_self_rating=request.enable_self_rating,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return SufficiencyResponse(
        is_sufficient=result.is_sufficient,
        overall_score=result.overall_score,
        density_score=result.density_score,
        coverage_score=result.coverage_score,
        num_chunks=result.num_chunks,
        recommendation=result.recommendation,
        explanation=result.explanation,
        component_scores=result.component_scores,
    )


# ── Token budget diagnostics ──────────────────────────────────────────────────


@app.post("/debug/token-budget", tags=["System"])
async def debug_token_budget(question: str, collection: str = "default", top_k: int = 10) -> dict:
    """
    Debug endpoint: show token budget optimization results for a query.

    Returns how many chunks were kept/dropped, token savings percentage,
    and estimated token counts before/after optimization.
    """
    from core.retrieval import retrieve
    from core.generation import get_backend, SYSTEM_PROMPT
    from core.token_budget import optimize_context, estimate_tokens
    from models import QueryMode

    backend = get_backend()
    req = QueryRequest(
        question=question, collection=collection,
        top_k=top_k, mode=QueryMode.HYBRID,
    )
    context = retrieve(req, generate_fn=backend.complete_raw)

    before_tokens = sum(estimate_tokens(r.chunk_text) for r in context.results)
    optimized, budget_info = optimize_context(
        context=context,
        question=question,
        system_prompt=SYSTEM_PROMPT,
        model_name=settings.claude_model if settings.llm_backend.value == "claude" else settings.ollama_model,
    )
    return {
        "question":         question,
        "chunks_before":    len(context.results),
        "chunks_after":     budget_info.included,
        "chunks_dropped":   budget_info.excluded,
        "tokens_before":    before_tokens,
        "tokens_after":     budget_info.estimated_tokens,
        "savings_pct":      budget_info.savings_pct,
        "budget_tokens":    budget_info.budget_tokens,
        "truncated":        budget_info.truncated,
    }
