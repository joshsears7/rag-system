"""
RAG System — Interactive Demo

Streamlit app with full retrieval visualization:
  - Source highlighting (which chunks supported which claims)
  - Sufficiency confidence score before each answer
  - CoT reasoning trace (collapsible, step-by-step)
  - Agentic tool call trace (for Agent mode)
  - Mode comparison: Naive vs Hybrid vs CoT vs Agent

Run:
    pip install streamlit
    streamlit run demo.py

Environment:
    Requires same .env as the main system. Set LLM_BACKEND=claude or ollama.
"""

from __future__ import annotations

import time
import sys
import os

# ── Streamlit guard ───────────────────────────────────────────────────────────
try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RAG System Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after streamlit) ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from models import QueryMode, QueryRequest


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.source-card {
    background: #1e1e2e;
    border-left: 4px solid #7c6af7;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: monospace;
    font-size: 0.85em;
}
.source-header {
    color: #cba6f7;
    font-weight: bold;
    margin-bottom: 6px;
}
.score-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: bold;
    margin-left: 8px;
}
.score-high   { background: #a6e3a1; color: #1e1e2e; }
.score-mid    { background: #f9e2af; color: #1e1e2e; }
.score-low    { background: #f38ba8; color: #1e1e2e; }
.step-card {
    background: #181825;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 12px;
    margin: 6px 0;
}
.step-number {
    color: #89b4fa;
    font-weight: bold;
    font-size: 0.9em;
}
.tool-call {
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-family: monospace;
    font-size: 0.82em;
    color: #a6e3a1;
}
.confidence-meter {
    background: #313244;
    border-radius: 4px;
    height: 8px;
    margin: 4px 0;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}
.metric-box {
    text-align: center;
    padding: 12px;
    background: #1e1e2e;
    border-radius: 8px;
    border: 1px solid #313244;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(score: float, max_score: float = 1.0) -> str:
    ratio = score / max_score
    if ratio >= 0.7:
        return "score-high"
    elif ratio >= 0.4:
        return "score-mid"
    return "score-low"


def render_source_card(source, idx: int) -> None:
    """Render a retrieved source chunk as a styled card."""
    sim = source.similarity_score if hasattr(source, "similarity_score") else 0.0
    css_cls = score_color(sim)
    excerpt = getattr(source, "excerpt", "")
    if not excerpt and hasattr(source, "chunk_text"):
        excerpt = source.chunk_text[:300]

    st.markdown(f"""
    <div class="source-card">
        <div class="source-header">
            [{idx}] {source.source}
            <span class="score-badge {css_cls}">sim={sim:.3f}</span>
            {f"p.{source.page_number}" if getattr(source, "page_number", None) else ""}
        </div>
        <div style="color:#cdd6f4; margin-top:4px">{excerpt[:400] if excerpt else "(no excerpt)"}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sufficiency_bar(score: float, label: str = "Context Sufficiency") -> None:
    """Render an animated sufficiency confidence bar."""
    color = "#a6e3a1" if score >= 0.6 else ("#f9e2af" if score >= 0.4 else "#f38ba8")
    pct = int(score * 100)
    st.markdown(f"""
    <div style="margin:8px 0">
        <small style="color:#a6adc8">{label}</small>
        <div class="confidence-meter">
            <div class="confidence-fill" style="width:{pct}%; background:{color}"></div>
        </div>
        <small style="color:{color}; font-weight:bold">{pct}%</small>
    </div>
    """, unsafe_allow_html=True)


def get_collections() -> list[str]:
    """List available ChromaDB collections."""
    try:
        from core.ingestion import get_chroma_client
        client = get_chroma_client()
        colls = client.list_collections()
        return [c.name for c in colls] if colls else ["default"]
    except Exception:
        return ["default"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 RAG System")
    st.caption("Production RAG — 2025 techniques")
    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.subheader("Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        help="Upload your own documents to query against. Files are processed in-session.",
    )

    if uploaded_files:
        import tempfile, pathlib
        from core.ingestion import ingest_document

        for uf in uploaded_files:
            file_key = f"ingested_{uf.name}"
            if file_key not in st.session_state:
                with st.spinner(f"Ingesting {uf.name}…"):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=pathlib.Path(uf.name).suffix,
                        ) as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
                        result = ingest_document(tmp_path, collection_name="user_upload")
                        pathlib.Path(tmp_path).unlink(missing_ok=True)
                        st.session_state[file_key] = result.chunks_added
                        st.success(f"{uf.name}: {result.chunks_added} chunks indexed")
                    except Exception as e:
                        st.error(f"{uf.name}: {e}")

        # Auto-switch to user_upload collection
        if any(f"ingested_{uf.name}" in st.session_state for uf in uploaded_files):
            st.info("Querying your uploaded documents.")
            _upload_collection = "user_upload"
        else:
            _upload_collection = None
    else:
        _upload_collection = None

    st.divider()

    # Collection picker
    st.subheader("Knowledge Base")
    collections = get_collections()
    if _upload_collection and _upload_collection not in collections:
        collections = [_upload_collection] + collections
    default_idx = collections.index(_upload_collection) if _upload_collection and _upload_collection in collections else 0
    collection = st.selectbox("Collection", collections, index=default_idx)

    st.divider()

    # Query mode
    st.subheader("Query Mode")
    mode = st.radio(
        "Mode",
        options=["Hybrid RAG", "CoT-RAG", "TTRAG", "Speculative RAG", "A-RAG", "Agentic RAG", "Compare All"],
        index=0,
        help=(
            "**Hybrid RAG:** Dense+BM25+RRF with cross-encoder reranking\n\n"
            "**CoT-RAG:** Chain-of-thought multi-hop reasoning\n\n"
            "**TTRAG:** Test-time compute scaling — iterative query rewriting until sufficient context found (ICLR 2025)\n\n"
            "**Speculative RAG:** Generates N independent draft answers from document subsets, selects best by confidence — ~51% latency reduction (Google 2024)\n\n"
            "**A-RAG:** Agent picks retrieval interface per step (keyword/semantic/hybrid/section) — hierarchical retrieval interfaces (Feb 2026)\n\n"
            "**Agentic RAG:** Claude tool-use with search_docs, search_web, query_sql, calculate\n\n"
            "**Compare All:** Run all modes and show side-by-side"
        ),
    )

    st.divider()

    # Advanced settings
    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 2, 20, settings.top_k)
    show_cot_trace = st.toggle("Show CoT reasoning trace", value=True)
    show_agent_trace = st.toggle("Show agent tool calls", value=True)
    enable_sufficiency = st.toggle("Sufficiency check", value=True)
    enable_self_rating = st.toggle("LLM self-rating (slower)", value=False)

    st.divider()
    st.caption(f"Backend: `{settings.llm_backend.value}`")
    st.caption(f"Embedding: `{settings.embedding_model}`")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("RAG System — Ask Your Documents")
st.caption(
    "Upload your own PDFs, reports, or contracts in the sidebar — then ask questions about them. "
    "Hybrid retrieval · CoT-RAG · TTRAG · Speculative RAG · A-RAG · Sufficient Context abstention"
)

# Example questions
with st.expander("Example questions", expanded=False):
    examples = [
        "What are the main themes in the ingested documents?",
        "How does X relate to Y? (replace with entities from your docs)",
        "Summarize the key findings",
        "What happened in 2023?",
        "Compare the approaches described in the documents",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state["question_input"] = ex

question = st.text_area(
    "Ask a question",
    value=st.session_state.get("question_input", ""),
    placeholder="Ask anything about your ingested documents…",
    height=80,
    key="question_input",
)

run_btn = st.button("Ask", type="primary", use_container_width=True)


# ── Query execution ───────────────────────────────────────────────────────────

if run_btn and question.strip():
    question = question.strip()

    if mode == "Compare All":
        # ── Side-by-side comparison ───────────────────────────────────────────
        st.subheader("Mode Comparison")
        col_naive, col_hybrid, col_cot = st.columns(3)

        from core.generation import answer_question, get_backend
        from core.retrieval import retrieve

        # Naive RAG (dense only, no reranking)
        with col_naive:
            st.markdown("**Naive RAG** (dense only)")
            with st.spinner("Retrieving…"):
                t0 = time.perf_counter()
                naive_req = QueryRequest(
                    question=question, collection=collection,
                    top_k=top_k, mode=QueryMode.DENSE,
                )
                try:
                    resp = answer_question(naive_req)
                    st.success(resp.answer[:600])
                    st.caption(f"Latency: {resp.latency_ms:.0f}ms | Sources: {len(resp.sources)}")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Hybrid RAG
        with col_hybrid:
            st.markdown("**Hybrid RAG** (dense+BM25+rerank)")
            with st.spinner("Retrieving…"):
                hybrid_req = QueryRequest(
                    question=question, collection=collection,
                    top_k=top_k, mode=QueryMode.HYBRID,
                )
                try:
                    resp = answer_question(hybrid_req)
                    st.success(resp.answer[:600])
                    st.caption(f"Latency: {resp.latency_ms:.0f}ms | Sources: {len(resp.sources)}")
                except Exception as e:
                    st.error(f"Error: {e}")

        # CoT-RAG
        with col_cot:
            st.markdown("**CoT-RAG** (chain-of-thought)")
            with st.spinner("Decomposing + retrieving…"):
                try:
                    from core.cot_rag import run_cot_rag
                    backend = get_backend()
                    cot_result = run_cot_rag(
                        question=question,
                        collection=collection,
                        retrieve_fn=retrieve,
                        llm_fn=backend.complete_raw,
                        max_steps=3,
                        top_k_per_step=top_k // 2 or 2,
                    )
                    st.success(cot_result.answer[:600])
                    st.caption(f"Latency: {cot_result.latency_ms:.0f}ms | Steps: {cot_result.num_steps} | Chunks: {cot_result.total_chunks}")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif mode == "CoT-RAG":
        # ── CoT-RAG mode ─────────────────────────────────────────────────────
        st.subheader("Chain-of-Thought RAG")

        from core.generation import get_backend
        from core.retrieval import retrieve
        from core.cot_rag import run_cot_rag

        with st.spinner("Decomposing question into reasoning steps…"):
            try:
                backend = get_backend()
                cot_result = run_cot_rag(
                    question=question,
                    collection=collection,
                    retrieve_fn=retrieve,
                    llm_fn=backend.complete_raw,
                    max_steps=4,
                    top_k_per_step=max(2, top_k // 2),
                )

                # Sufficiency check
                if enable_sufficiency and cot_result.total_chunks > 0:
                    avg_score = cot_result.total_chunks / (len(cot_result.reasoning_steps) * top_k)
                    render_sufficiency_bar(min(avg_score, 1.0))

                # Warnings
                for w in cot_result.warnings:
                    st.warning(w)

                # Reasoning trace
                if show_cot_trace and cot_result.reasoning_steps:
                    with st.expander(f"Reasoning Trace — {cot_result.num_steps} steps", expanded=True):
                        for step in cot_result.reasoning_steps:
                            st.markdown(f"""
                            <div class="step-card">
                                <div class="step-number">Step {step.step_number}: {step.thought}</div>
                                <div style="color:#89dceb; font-size:0.85em; margin:4px 0">
                                    Sub-query: <code>{step.sub_query}</code>
                                </div>
                                <div style="color:#a6adc8; font-size:0.83em">
                                    Retrieved {len(step.retrieved)} chunks
                                    {f"from: {', '.join(set(step.sources))}" if step.sources else "(nothing found)"}
                                </div>
                                <div style="color:#cdd6f4; margin-top:6px">
                                    <strong>Finding:</strong> {step.intermediate}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Final answer
                st.markdown("### Answer")
                st.markdown(cot_result.answer)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Steps", cot_result.num_steps)
                m2.metric("Chunks retrieved", cot_result.total_chunks)
                m3.metric("Tokens used", cot_result.tokens_used)
                m4.metric("Latency", f"{cot_result.latency_ms:.0f}ms")

                # Sources
                if cot_result.all_sources:
                    with st.expander(f"Sources ({len(cot_result.all_sources)} unique)"):
                        for src in cot_result.all_sources:
                            st.markdown(f"- `{src}`")

            except Exception as e:
                st.error(f"CoT-RAG failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    elif mode == "TTRAG":
        # ── TTRAG mode ────────────────────────────────────────────────────────
        st.subheader("TTRAG — Test-Time Compute Scaling (ICLR 2025)")

        max_iterations = st.sidebar.slider("Max iterations", 1, 8, 4)
        threshold = st.sidebar.slider("Sufficiency threshold", 0.1, 0.9, 0.55, step=0.05)

        with st.spinner("Iterative retrieval in progress…"):
            try:
                from core.ttrag import run_ttrag
                from core.retrieval import retrieve
                from core.generation import get_backend, build_user_prompt, SYSTEM_PROMPT

                backend = get_backend()

                def _generate(q: str, ctx) -> tuple[str, int]:
                    prompt = build_user_prompt(ctx)
                    answer, tokens, _ = backend.complete(SYSTEM_PROMPT, prompt)
                    return answer, tokens

                ttrag_result = run_ttrag(
                    question=question,
                    collection=collection,
                    retrieve_fn=retrieve,
                    llm_fn=backend.complete_raw,
                    generate_fn=_generate,
                    max_iterations=max_iterations,
                    top_k=top_k,
                    sufficiency_threshold=threshold,
                )

                # Iteration trace
                if ttrag_result.iterations:
                    with st.expander(f"Iteration Trace — {ttrag_result.num_iterations} iterations", expanded=True):
                        for it in ttrag_result.iterations:
                            suf_pct = f"{it.sufficiency.overall_score:.0%}"
                            color = "green" if it.sufficiency.is_sufficient else ("orange" if it.sufficiency.overall_score >= 0.35 else "red")
                            st.markdown(f"""
                            <div class="step-card">
                                <div class="step-number">Iteration {it.iteration}</div>
                                <div style="color:#89dceb; font-size:0.85em; margin:4px 0">
                                    Query: <code>{it.query_used}</code>
                                </div>
                                <div style="color:#a6adc8; font-size:0.83em">
                                    New chunks: {len(it.retrieved)} &nbsp;|&nbsp;
                                    Sufficiency: <span style="color:{color}">{suf_pct}</span>
                                    {f" &nbsp;|&nbsp; Rewrite reason: {it.rewrite_reason}" if it.iteration > 1 else ""}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("### Answer")
                st.markdown(ttrag_result.answer)

                converged = "Converged" if ttrag_result.converged else "Max iterations reached"
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Iterations", ttrag_result.num_iterations)
                m2.metric("Unique chunks", ttrag_result.unique_chunks_used)
                m3.metric("Sufficiency", f"{ttrag_result.final_sufficiency:.0%}")
                m4.metric("Tokens", ttrag_result.tokens_used)
                m5.metric("Status", converged)

            except Exception as e:
                st.error(f"TTRAG failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    elif mode == "Speculative RAG":
        # ── Speculative RAG mode ──────────────────────────────────────────────
        st.subheader("Speculative RAG — Google Research (2024)")

        num_drafts = st.sidebar.slider("Number of drafts", 2, 6, 3)

        with st.spinner(f"Generating {num_drafts} speculative drafts…"):
            try:
                from core.speculative_rag import run_speculative_rag
                from core.retrieval import retrieve
                from core.generation import get_backend

                backend = get_backend()

                spec_result = run_speculative_rag(
                    question=question,
                    collection=collection,
                    retrieve_fn=retrieve,
                    llm_complete_fn=backend.complete,
                    llm_raw_fn=backend.complete_raw,
                    num_drafts=num_drafts,
                    top_k=top_k,
                )

                # Draft comparison table
                with st.expander(f"Draft Comparison — {spec_result.num_drafts} drafts", expanded=True):
                    for d in spec_result.all_drafts:
                        selected = d.draft_id == spec_result.selected_draft_id
                        border = "#a6e3a1" if selected else "#313244"
                        label = " ✓ Selected" if selected else ""
                        conf_color = "#a6e3a1" if d.confidence_score >= 0.7 else ("#f9e2af" if d.confidence_score >= 0.4 else "#f38ba8")
                        st.markdown(f"""
                        <div class="source-card" style="border-left-color:{border}">
                            <div class="source-header">
                                Draft {d.draft_id}{label} &nbsp;
                                <span class="score-badge" style="background:{conf_color}; color:#1e1e2e">
                                    confidence {d.confidence_score:.0%}
                                </span>
                                &nbsp; <span style="color:#6c7086; font-size:0.8em">{len(d.chunks)} chunks · {d.latency_ms:.0f}ms</span>
                            </div>
                            <div style="color:#cdd6f4; margin-top:6px">{d.answer[:300]}{'…' if len(d.answer) > 300 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("### Selected Answer")
                st.markdown(spec_result.answer)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Drafts", spec_result.num_drafts)
                m2.metric("Winning draft", f"#{spec_result.selected_draft_id}")
                m3.metric("Est. latency reduction", f"~{spec_result.latency_reduction_pct:.0f}%")
                m4.metric("Total latency", f"{spec_result.latency_ms:.0f}ms")

            except Exception as e:
                st.error(f"Speculative RAG failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    elif mode == "A-RAG":
        # ── A-RAG mode ────────────────────────────────────────────────────────
        st.subheader("A-RAG — Hierarchical Retrieval Interfaces (Feb 2026)")

        max_steps = st.sidebar.slider("Max steps", 1, 10, 5)

        with st.spinner("Agent selecting retrieval interfaces…"):
            try:
                from core.arag import run_arag, RETRIEVAL_TOOLS
                from core.retrieval import retrieve
                from core.generation import get_backend

                backend = get_backend()

                arag_result = run_arag(
                    question=question,
                    collection=collection,
                    retrieve_fn=retrieve,
                    llm_raw_fn=backend.complete_raw,
                    llm_complete_fn=backend.complete,
                    max_steps=max_steps,
                    top_k_per_step=top_k,
                )

                # Step trace
                if arag_result.steps:
                    with st.expander(f"Retrieval Interface Decisions — {arag_result.num_steps} steps", expanded=True):
                        for s in arag_result.steps:
                            tool_color = {
                                "keyword_search": "#f9e2af",
                                "semantic_search": "#89b4fa",
                                "hybrid_search": "#cba6f7",
                                "read_section": "#a6e3a1",
                            }.get(s.tool_chosen, "#cdd6f4")
                            st.markdown(f"""
                            <div class="step-card">
                                <div class="step-number">Step {s.step}</div>
                                <div style="margin:4px 0">
                                    <span style="background:{tool_color}; color:#1e1e2e; padding:2px 8px; border-radius:4px; font-size:0.8em; font-weight:bold">{s.tool_chosen}</span>
                                    &nbsp; <code style="font-size:0.85em">{s.query}</code>
                                </div>
                                <div style="color:#a6adc8; font-size:0.83em">
                                    {len(s.retrieved)} new chunks &nbsp;|&nbsp; {s.latency_ms:.0f}ms
                                    {f" &nbsp;|&nbsp; {s.reasoning[:80]}" if s.reasoning else ""}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("### Answer")
                st.markdown(arag_result.answer)

                tools_str = " → ".join(dict.fromkeys(arag_result.tools_used)) or "none"
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Steps", arag_result.num_steps)
                m2.metric("Unique chunks", arag_result.unique_chunks)
                m3.metric("Tool sequence", tools_str)
                m4.metric("Latency", f"{arag_result.latency_ms:.0f}ms")

            except Exception as e:
                st.error(f"A-RAG failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    elif mode == "Agentic RAG":
        # ── Agentic RAG mode ──────────────────────────────────────────────────
        st.subheader("Agentic RAG — Claude tool_use")

        with st.spinner("Agent running… (may take several iterations)"):
            try:
                from core.agent import run_agent
                from core.retrieval import retrieve
                from core.generation import get_backend
                backend = get_backend()

                # sql_fn signature for agent: (question: str) -> str
                sql_fn = None
                if settings.sql_database_url:
                    from core.sql_retrieval import query_natural_language as _sql_fn
                    def sql_fn(q: str) -> str:  # type: ignore[misc]
                        return _sql_fn(q)

                agent_result = run_agent(
                    question=question,
                    collection=collection,
                    retrieve_fn=retrieve,
                    sql_fn=sql_fn,
                )

                # Tool call trace
                if show_agent_trace and agent_result.tool_calls:
                    with st.expander(f"Tool Call Trace — {len(agent_result.tool_calls)} calls", expanded=True):
                        for i, call in enumerate(agent_result.tool_calls, 1):
                            # ToolCall is a dataclass with .tool_name, .tool_input, .result
                            if hasattr(call, "tool_name"):
                                tool_name   = call.tool_name
                                tool_input  = str(call.tool_input)[:200]
                                tool_output = str(call.result)[:300]
                                latency     = f"{call.latency_ms:.0f}ms"
                            else:
                                # Fallback for dict format (from API)
                                tool_name   = call.get("tool", "unknown")
                                tool_input  = str(call.get("input", {}))[:200]
                                tool_output = str(call.get("output", ""))[:300]
                                latency     = ""
                            st.markdown(f"""
                            <div class="tool-call">
                                [{i}] <strong>{tool_name}</strong>({tool_input})<br>
                                → {tool_output} {latency}
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("### Answer")
                st.markdown(agent_result.answer)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Iterations", agent_result.iterations)
                m2.metric("Tool calls", len(agent_result.tool_calls))
                m3.metric("Tokens", agent_result.total_tokens)
                m4.metric("Latency", f"{agent_result.latency_ms:.0f}ms")

            except Exception as e:
                st.error(f"Agent failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    else:
        # ── Hybrid RAG (default) ──────────────────────────────────────────────
        st.subheader("Hybrid RAG Answer")

        from core.generation import answer_question, get_backend
        from core.retrieval import retrieve
        from core.sufficient_context import check_sufficiency

        with st.spinner("Retrieving and generating…"):
            try:
                req = QueryRequest(
                    question=question,
                    collection=collection,
                    top_k=top_k,
                    mode=QueryMode.HYBRID,
                )
                backend = get_backend()

                # Run retrieval first to check sufficiency
                from core.generation import make_crag_evaluator
                generate_fn  = backend.complete_raw
                evaluate_fn  = make_crag_evaluator(backend) if settings.use_hybrid_search else None
                context = retrieve(req, generate_fn=generate_fn, evaluate_fn=evaluate_fn)

                # Sufficiency check
                if enable_sufficiency:
                    suf = check_sufficiency(
                        question=question,
                        context=context,
                        llm_fn=backend.complete_raw if enable_self_rating else None,
                        enable_self_rating=enable_self_rating,
                    )
                    render_sufficiency_bar(suf.overall_score)

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Density",  f"{suf.density_score:.2f}")
                    col_b.metric("Coverage", f"{suf.coverage_score:.2f}")
                    col_c.metric("Chunks",   suf.num_chunks)

                    if suf.recommendation == "abstain":
                        st.error(f"Insufficient context ({suf.overall_score:.0%}): {suf.explanation}")
                        st.stop()
                    elif suf.recommendation == "web_search":
                        st.warning("Triggering web search fallback…")
                    elif suf.recommendation == "retrieve_more":
                        st.info("Context borderline — attempting to retrieve more chunks.")

                # Full generation
                response = answer_question(req)

                # Answer
                st.markdown("### Answer")
                st.markdown(response.answer)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latency",   f"{response.latency_ms:.0f}ms")
                m2.metric("Tokens",    response.tokens_used)
                m3.metric("Sources",   len(response.sources))
                m4.metric("Cache hit", "Yes" if response.cache_hit else "No")

                # Source cards
                if response.sources:
                    st.markdown("### Retrieved Sources")
                    st.caption(
                        "Each card shows the chunk that contributed to the answer. "
                        "Higher similarity = stronger relevance signal."
                    )
                    for i, src in enumerate(response.sources, 1):
                        render_source_card(src, i)

            except Exception as e:
                st.error(f"Query failed: {e}")
                import traceback
                st.code(traceback.format_exc())

elif run_btn:
    st.warning("Please enter a question.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Techniques: Hybrid dense+BM25+RRF · Cross-encoder reranking · HyDE · CRAG · "
    "MMR · RAPTOR · GraphRAG · LightRAG · Contextual Retrieval · CoT-RAG (EMNLP 2025) · "
    "TTRAG test-time scaling (ICLR 2025) · Sufficient Context (Google ICLR 2025) · "
    "Agentic tool_use · Langfuse tracing · PII redaction · Prompt injection detection"
)
