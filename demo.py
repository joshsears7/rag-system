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


# ── Second Brain page ─────────────────────────────────────────────────────────

def _render_brain_page() -> None:
    """Render the Personal Second Brain page."""
    from datetime import datetime as _dt
    from core.brain import add_note, add_source, query_brain, list_sources, get_all_tags, daily_digest

    st.title("Second Brain — Personal Knowledge Base")
    st.caption(
        "Capture notes, files, and URLs. Query your personal knowledge with time and tag filters. "
        "All content lives in its own isolated collection separate from your document RAG."
    )

    cap_tab, query_tab, browse_tab = st.tabs(["Capture", "Query", "Browse"])

    # ── Capture tab ───────────────────────────────────────────────────────────
    with cap_tab:
        st.subheader("Add to Your Brain")
        capture_type = st.radio("What are you adding?", ["Note", "File", "URL"], horizontal=True)
        tags_input = st.text_input("Tags (comma-separated)", placeholder="ai, research, finance", key="brain_tags_input")
        tags = [t.strip().lower() for t in tags_input.split(",") if t.strip()]

        if capture_type == "Note":
            note_title = st.text_input("Title", placeholder="My note title", key="brain_note_title")
            note_text = st.text_area("Note", placeholder="Write your note here…", height=200, key="brain_note_text")
            if st.button("Save Note", type="primary", key="brain_save_note"):
                if note_text.strip():
                    with st.spinner("Saving…"):
                        try:
                            result = add_note(note_text.strip(), title=note_title, tags=tags)
                            if result.chunks_added > 0:
                                st.success(f"Saved! {result.chunks_added} chunk(s) added to your brain.")
                            else:
                                st.info("Already in your brain (duplicate content).")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                else:
                    st.warning("Note is empty.")

        elif capture_type == "File":
            uploaded = st.file_uploader("Upload a file", type=["pdf", "txt", "md", "docx"], key="brain_file_upload")
            file_title = st.text_input("Title (optional, defaults to filename)", key="brain_file_title")
            if st.button("Add File", type="primary", key="brain_add_file") and uploaded:
                import tempfile as _tf, os as _os
                with st.spinner(f"Processing '{uploaded.name}'…"):
                    try:
                        suffix = Path(uploaded.name).suffix
                        with _tf.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                            f.write(uploaded.getvalue())
                            tmp_path = f.name
                        try:
                            result = add_source(tmp_path, tags=tags, title=file_title or uploaded.name)
                        finally:
                            _os.unlink(tmp_path)
                        if result.chunks_added > 0:
                            st.success(f"Added {result.chunks_added} chunk(s) from '{uploaded.name}'.")
                        else:
                            st.info("Already in your brain (duplicate content).")
                    except Exception as e:
                        st.error(f"Failed to add file: {e}")

        elif capture_type == "URL":
            url_input = st.text_input("URL", placeholder="https://…", key="brain_url_input")
            url_title = st.text_input("Title (optional)", key="brain_url_title")
            if st.button("Add URL", type="primary", key="brain_add_url") and url_input.strip():
                with st.spinner(f"Fetching '{url_input}'…"):
                    try:
                        result = add_source(url_input.strip(), tags=tags, title=url_title or url_input)
                        if result.chunks_added > 0:
                            st.success(f"Added {result.chunks_added} chunk(s) from URL.")
                        else:
                            st.info("Already in your brain (duplicate content).")
                    except Exception as e:
                        st.error(f"Failed to add URL: {e}")

    # ── Query tab ─────────────────────────────────────────────────────────────
    with query_tab:
        st.subheader("Ask Your Brain")
        brain_q = st.text_area(
            "Question", placeholder="What do I know about…?", height=80, key="brain_question"
        )

        col_tags, col_time = st.columns(2)
        with col_tags:
            filter_tags_str = st.text_input("Filter by tags", placeholder="ai, finance", key="brain_filter_tags")
        with col_time:
            time_range = st.selectbox(
                "Time range", ["All time", "Today", "Last 7 days", "Last 30 days"], key="brain_time_range"
            )

        filter_tags = [t.strip() for t in filter_tags_str.split(",") if t.strip()]
        days_map = {"All time": None, "Today": 1, "Last 7 days": 7, "Last 30 days": 30}
        days = days_map[time_range]

        if st.button("Ask Brain", type="primary", key="brain_ask_btn") and brain_q.strip():
            with st.spinner("Searching your knowledge base…"):
                try:
                    response = query_brain(brain_q.strip(), tags=filter_tags, days=days)
                    st.markdown("### Answer")
                    st.markdown(response.answer)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Latency", f"{response.latency_ms:.0f}ms")
                    m2.metric("Sources", len(response.sources))
                    m3.metric("Tokens", response.tokens_used)

                    if response.sources:
                        with st.expander(f"Sources ({len(response.sources)})"):
                            for i, src in enumerate(response.sources, 1):
                                render_source_card(src, i)
                except Exception as e:
                    st.error(f"Query failed: {e}")

    # ── Browse tab ────────────────────────────────────────────────────────────
    with browse_tab:
        st.subheader("Knowledge Browser")

        col_tags_browse, col_digest = st.columns(2)

        with col_tags_browse:
            st.markdown("**Tags in your brain**")
            try:
                tags_map = get_all_tags()
                if tags_map:
                    for tag, count in sorted(tags_map.items(), key=lambda x: -x[1]):
                        st.markdown(f"`{tag}` — {count} chunk(s)")
                else:
                    st.caption("No tags yet. Add content with tags to get started.")
            except Exception as e:
                st.caption(f"Could not load tags: {e}")

        with col_digest:
            st.markdown("**Daily Digest**")
            digest_range = st.selectbox(
                "Summarize activity from",
                ["Today", "Last 7 days", "Last 30 days"],
                key="brain_digest_range",
            )
            digest_days_map = {"Today": 1, "Last 7 days": 7, "Last 30 days": 30}
            if st.button("Generate Digest", key="brain_digest_btn"):
                with st.spinner("Summarizing…"):
                    try:
                        digest = daily_digest(days=digest_days_map[digest_range])
                        st.markdown(digest)
                    except Exception as e:
                        st.error(f"Digest failed: {e}")

        st.divider()
        st.markdown("**Sources**")

        browse_range = st.selectbox(
            "Show from", ["All time", "Last 7 days", "Last 30 days"], key="brain_browse_range"
        )
        browse_days = {"All time": None, "Last 7 days": 7, "Last 30 days": 30}[browse_range]

        try:
            sources = list_sources(days=browse_days, limit=50)
            if sources:
                for m in sources:
                    brain_type = m.get("brain_type", "unknown")
                    title = m.get("brain_title", m.get("source_file", "Unknown"))
                    tags_str = m.get("brain_tags", "")
                    ts = m.get("brain_ingested_at", 0)
                    dt_str = _dt.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "unknown"
                    icon = {"note": "📝", "file": "📄", "url": "🔗"}.get(brain_type, "📌")
                    tags_display = f" &nbsp; `{tags_str}`" if tags_str else ""
                    st.markdown(
                        f"{icon} **{title}** &nbsp;"
                        f"<span style='color:#6c7086; font-size:0.85em'>{dt_str}</span>"
                        f"{tags_display}",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No sources yet. Add notes, files, or URLs in the Capture tab.")
        except Exception as e:
            st.caption(f"Could not load sources: {e}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 RAG System")
    st.caption("Production RAG — 2025 techniques")
    st.divider()

    page = st.radio(
        "View",
        ["Document Q&A", "Second Brain"],
        horizontal=True,
        key="page_selector",
        label_visibility="collapsed",
    )
    st.divider()

    # ── File upload ───────────────────────────────────────────────────────────
    st.subheader("Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        help="Max 200 MB per file. PDF, TXT, DOCX, Markdown supported. Scanned/image PDFs won't extract — use text-based files.",
    )

    if uploaded_files:
        import tempfile, pathlib
        from core.ingestion import ingest_document

        for uf in uploaded_files:
            file_key = f"ingested_{uf.name}"
            if file_key not in st.session_state:
                with st.spinner(f"Ingesting {uf.name}…"):
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=pathlib.Path(uf.name).suffix,
                        ) as tmp:
                            tmp.write(uf.read())
                            tmp_path = tmp.name
                        result = ingest_document(tmp_path, collection_name="user_upload")
                        st.session_state[file_key] = result.chunks_added
                    except Exception as e:
                        st.session_state[file_key] = -1
                        st.session_state[f"ingest_error_{uf.name}"] = str(e)
                    finally:
                        if tmp_path:
                            pathlib.Path(tmp_path).unlink(missing_ok=True)

        # Show persistent status for each file
        total_chunks = 0
        for uf in uploaded_files:
            file_key = f"ingested_{uf.name}"
            chunks = st.session_state.get(file_key, None)
            if chunks is None:
                pass
            elif chunks == -1:
                err = st.session_state.get(f"ingest_error_{uf.name}", "unknown error")
                st.error(f"{uf.name}: ingest failed — {err}")
            elif chunks == 0:
                st.warning(f"{uf.name}: 0 chunks extracted. If this is a scanned/image PDF, text extraction won't work — try a text-based PDF or TXT file.")
            else:
                st.success(f"{uf.name}: {chunks} chunks indexed ✓")
                total_chunks += chunks

        # Auto-switch to user_upload collection only if chunks were actually indexed
        if total_chunks > 0:
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
        options=["Hybrid RAG", "Chat", "CoT-RAG", "TTRAG", "Speculative RAG", "A-RAG", "Agentic RAG", "Compare All"],
        index=0,
        help=(
            "**Hybrid RAG:** Dense+BM25+RRF with cross-encoder reranking — streams response token by token\n\n"
            "**Chat:** Multi-turn conversation with memory — ask follow-up questions naturally\n\n"
            "**CoT-RAG:** Chain-of-thought multi-hop reasoning\n\n"
            "**TTRAG:** Test-time compute scaling — iterative query rewriting until sufficient context found (ICLR 2025)\n\n"
            "**Speculative RAG:** Generates N independent draft answers from document subsets, selects best by confidence (Google 2024)\n\n"
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
    if mode == "Chat":
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("conv_memory", None)
            st.session_state.pop("chat_history", None)
            st.rerun()
    st.caption(f"Backend: `{settings.llm_backend.value}`")
    st.caption(f"Embedding: `{settings.embedding_model}`")


# ── Main area ─────────────────────────────────────────────────────────────────

if st.session_state.get("page_selector") == "Second Brain":
    _render_brain_page()
    st.stop()

st.title("RAG System — Ask Your Documents")
st.caption(
    "Upload your own PDFs, reports, or contracts in the sidebar — then ask questions about them. "
    "Streaming responses · Multi-turn chat · CoT-RAG · TTRAG · Speculative RAG · A-RAG · Sufficient Context abstention"
)

if mode != "Chat":
    with st.expander("Example questions", expanded=False):
        examples = [
            "Summarize this document",
            "What are the main themes?",
            "What is the author's argument or conclusion?",
            "What evidence or examples are provided?",
            "What are the key findings or takeaways?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state["question_input"] = ex

if mode == "Chat":
    question = st.chat_input("Ask a follow-up question…")
    run_btn = bool(question)
else:
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
                import traceback, logging as _logging
                _logging.getLogger(__name__).error(traceback.format_exc())
                with st.expander("Technical details"):
                    st.code(traceback.format_exc().split("site-packages")[0])

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
                import traceback, logging as _logging
                _logging.getLogger(__name__).error(traceback.format_exc())
                with st.expander("Technical details"):
                    st.code(traceback.format_exc().split("site-packages")[0])

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
                import traceback, logging as _logging
                _logging.getLogger(__name__).error(traceback.format_exc())
                with st.expander("Technical details"):
                    st.code(traceback.format_exc().split("site-packages")[0])

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
                import traceback, logging as _logging
                _logging.getLogger(__name__).error(traceback.format_exc())
                with st.expander("Technical details"):
                    st.code(traceback.format_exc().split("site-packages")[0])

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
                import traceback, logging as _logging
                _logging.getLogger(__name__).error(traceback.format_exc())
                with st.expander("Technical details"):
                    st.code(traceback.format_exc().split("site-packages")[0])

    elif mode == "Chat":
        # ── Multi-turn Chat mode ──────────────────────────────────────────────
        from core.generation import get_backend, stream_from_context, extract_sources
        from core.retrieval import retrieve
        from core.conversation import ConversationMemory, ConversationTurn

        if "conv_memory" not in st.session_state:
            st.session_state["conv_memory"] = ConversationMemory(max_turns=10, summarize_after=6)
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []  # list of (role, text, sources)

        memory: ConversationMemory = st.session_state["conv_memory"]
        backend = get_backend()

        # Display conversation history
        for role, text, sources in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.markdown(text)
                if sources and role == "assistant":
                    with st.expander(f"Sources ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            render_source_card(src, i)

        # Process new question
        try:
            resolved_q = memory.resolve_references(question, backend.complete_raw)

            req = QueryRequest(
                question=resolved_q,
                collection=collection,
                top_k=top_k,
                mode=QueryMode.HYBRID,
            )
            from core.generation import make_crag_evaluator
            context = retrieve(req, generate_fn=backend.complete_raw,
                               evaluate_fn=make_crag_evaluator(backend) if settings.use_hybrid_search else None)

            # Sufficiency check
            if enable_sufficiency and not context.is_empty:
                from core.sufficient_context import check_sufficiency
                suf = check_sufficiency(question=resolved_q, context=context,
                                        llm_fn=backend.complete_raw if enable_self_rating else None,
                                        enable_self_rating=enable_self_rating)
                render_sufficiency_bar(suf.overall_score)
                if suf.recommendation == "abstain":
                    st.error(f"Insufficient context ({suf.overall_score:.0%}): {suf.explanation}")
                    st.stop()

            # Show user message and stream assistant reply
            st.session_state["chat_history"].append(("user", question, []))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                conv_ctx = memory.build_context_prompt()
                t0 = time.perf_counter()
                answer = st.write_stream(stream_from_context(context, conv_ctx))
                latency = (time.perf_counter() - t0) * 1000
                sources = extract_sources(context)
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            render_source_card(src, i)
                st.caption(f"{latency:.0f}ms · {len(sources)} sources")

            # Persist to memory
            turn = ConversationTurn(
                question=resolved_q, answer=answer,
                sources=[s.source for s in sources],
                collection=collection,
            )
            memory.add_turn(turn)
            if len(memory.turns) >= memory.summarize_after:
                memory.compress(backend.complete_raw)
            st.session_state["chat_history"].append(("assistant", answer, sources))

        except Exception as e:
            st.error(f"Chat failed: {e}")
            import traceback, logging as _logging
            _logging.getLogger(__name__).error(traceback.format_exc())
            with st.expander("Technical details"):
                st.code(traceback.format_exc().split("site-packages")[0])

    else:
        # ── Hybrid RAG (default, streaming) ──────────────────────────────────
        st.subheader("Hybrid RAG Answer")

        from core.generation import get_backend, stream_from_context, extract_sources
        from core.retrieval import retrieve
        from core.sufficient_context import check_sufficiency

        try:
            req = QueryRequest(
                question=question,
                collection=collection,
                top_k=top_k,
                mode=QueryMode.HYBRID,
            )
            backend = get_backend()

            from core.generation import make_crag_evaluator
            generate_fn = backend.complete_raw
            evaluate_fn = make_crag_evaluator(backend) if settings.use_hybrid_search else None
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
                    st.info("Context is limited — consider enabling Web Search Fallback in your .env for broader coverage.")
                elif suf.recommendation == "retrieve_more":
                    st.info("Context borderline — answer may be partial. Try increasing top_k or using CoT-RAG mode.")

            # Stream the answer
            st.markdown("### Answer")
            t0 = time.perf_counter()
            answer = st.write_stream(stream_from_context(context))
            latency_ms = (time.perf_counter() - t0) * 1000

            col_dl, _ = st.columns([1, 4])
            with col_dl:
                st.download_button(
                    "Download answer",
                    data=answer,
                    file_name="rag_answer.txt",
                    mime="text/plain",
                )

            sources = extract_sources(context)
            m1, m2, m3 = st.columns(3)
            m1.metric("Latency",  f"{latency_ms:.0f}ms")
            m2.metric("Sources",  len(sources))
            m3.metric("Chunks retrieved", len(context.results))

            if sources:
                st.markdown("### Retrieved Sources")
                st.caption("Higher similarity = stronger relevance signal.")
                for i, src in enumerate(sources, 1):
                    render_source_card(src, i)

        except Exception as e:
            st.error(f"Query failed: {e}")
            import traceback, logging as _logging
            _logging.getLogger(__name__).error(traceback.format_exc())
            with st.expander("Technical details"):
                st.code(traceback.format_exc().split("site-packages")[0])

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
