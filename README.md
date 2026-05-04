---
title: RAG System Demo
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: hf_space/app.py
pinned: false
license: mit
short_description: Production RAG — 20+ techniques from 2022-2025 research
---

# RAG System

[![CI](https://github.com/joshsears7/rag-system/actions/workflows/eval.yml/badge.svg)](https://github.com/joshsears7/rag-system/actions/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade Retrieval-Augmented Generation implementing 20+ techniques from 2022–2025 research. Built as a reference architecture for serious AI engineering.

```
python3 main.py ingest --path ./docs
python3 main.py query --question "What are the key findings?"
streamlit run demo.py
```

---

## Why this is different from RAG tutorials

Most RAG tutorials stop at "embed → retrieve → generate." Real systems fail in three ways that tutorials don't cover:

1. **Retrieval misses** — a single dense embedding pass doesn't find lexically specific content (product codes, names, acronyms). Fixed with BM25 + RRF fusion.
2. **Confident hallucinations** — the LLM generates fluently even when retrieved context is wrong or irrelevant. Fixed with CRAG evaluation + Sufficient Context abstention.
3. **Multi-hop failures** — questions requiring facts from multiple sections get partial answers. Fixed with CoT-RAG, Adaptive RAG, and RAPTOR.

This system addresses all three.

---

## Technique Inventory

Each technique is listed with the paper that introduced it, what it fixes, and the tradeoff.

### Retrieval

| Technique | Paper | What it fixes | Latency cost |
|-----------|-------|---------------|--------------|
| **Hybrid dense+BM25+RRF** | — | Lexical gaps in pure vector search | +20ms |
| **Cross-encoder reranking** | Nogueira et al. (2019) | Top-k precision after first-pass retrieval | +150ms |
| **HyDE** | Gao et al. (2022) | Query–document mismatch (asymmetric) | +300ms |
| **Multi-query expansion** | — | Narrow query coverage | +200ms/query |
| **MMR diversity** | Carbonell & Goldstein (1998) | Redundant retrieved chunks | +5ms |
| **Contextual Retrieval** | Anthropic (Nov 2024) | Chunk-level context loss (49% fewer failures) | +2s ingest/chunk |
| **Sufficient Context** | Google ICLR 2025 | Hallucination from low-quality retrieval | +50ms |

### Generation & Reasoning

| Technique | Paper | What it fixes | Latency cost |
|-----------|-------|---------------|--------------|
| **CRAG** | Yan et al. (2024) | Generates when retrieved context is bad | +300ms |
| **Adaptive RAG** | Jeong et al. (2024) | Over-retrieval on simple questions | +100ms |
| **Self-RAG** | Asai et al. (2023) | No reflection on retrieval quality | +200ms |
| **CoT-RAG** | EMNLP 2025 | Single-hop failures on multi-hop questions | +1–3s |
| **TTRAG** | ICLR 2025 | One-shot retrieval misses — iterative query rewriting until sufficient context found | +1–4s/iter |
| **Speculative RAG** | Google Research (2024) | Full-context generation bottleneck — N drafts from subsets, best selected by confidence | ~51% faster |
| **RAPTOR** | Sarthi et al. (2024) | Long document comprehension failures | +ingest time |

### Knowledge Graphs

| Technique | Paper | What it fixes | Notes |
|-----------|-------|---------------|-------|
| **GraphRAG** | Microsoft (2024) | Isolated chunk retrieval, no entity relationships | Community detection |
| **LightRAG** | HKUDS EMNLP 2025 | Static graph, no dual retrieval modes | Low-level + high-level routing |

### Infrastructure

| Module | What it does |
|--------|-------------|
| **Agentic RAG** | Claude `tool_use` API: search_docs, search_web, query_sql, calculate |
| **Text-to-SQL** | Natural language → SQL via LLM + schema injection |
| **Multimodal** | PDF table extraction (pdfplumber) + vision LLM figure descriptions |
| **Embedding fine-tuning** | MNR loss training pipeline on domain-specific feedback pairs |
| **Semantic cache** | Cosine similarity cache — 0ms latency on near-duplicate queries |
| **Langfuse tracing** | Full retrieval + generation traces with span-level latency |
| **Prometheus metrics** | Request count, latency histograms, cache hit rate |
| **PII redaction** | Regex + optional Presidio ML — redacts before embedding |
| **Prompt injection detection** | 10 patterns on retrieved chunks — blocks override attempts |
| **Audit logging** | JSONL audit log: hashed queries, PII flags, session IDs |
| **Feedback loop** | SQLite feedback → contrastive pair mining → fine-tuning |
| **RAGAS eval harness** | Faithfulness, Recall@K, Answer Relevancy, Context Precision |
| **CI/CD quality gate** | GitHub Actions: fail build if faithfulness drops below threshold |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG System                              │
├───────────────┬─────────────────────────┬───────────────────────┤
│   INGESTION   │       RETRIEVAL          │      GENERATION       │
│               │                         │                       │
│ Document load │ Dense vector (ChromaDB) │ Sufficient Context    │
│ 3 chunking    │ + BM25 sparse           │ check (ICLR 2025)    │
│ strategies    │ → RRF fusion            │                       │
│               │ → HyDE expansion        │ CoT-RAG reasoning     │
│ PII redaction │ → Multi-query           │ (EMNLP 2025)          │
│               │ → Cross-encoder rerank  │                       │
│ Contextual    │ → MMR diversity         │ Agentic loop          │
│ Retrieval     │ → CRAG evaluation       │ (tool_use API)        │
│ (Anthropic)   │ → Web search fallback   │                       │
│               │                         │ Streaming SSE         │
│ RAPTOR tree   │ GraphRAG + LightRAG     │ (all 3 backends)      │
│ (recursive    │ (entity + community)    │                       │
│  summarize)   │                         │ Semantic cache        │
│               │ Adaptive RAG routing    │                       │
│               │ (NO_RETRIEVAL/SINGLE/   │ Langfuse traces       │
│               │  ITERATIVE)             │                       │
└───────────────┴─────────────────────────┴───────────────────────┘
         │                                         │
    ┌────▼────┐                             ┌──────▼──────┐
    │ ChromaDB│                             │  FastAPI    │
    │ KG JSON │                             │  35+ routes │
    │ SQLite  │                             │  /docs UI   │
    └─────────┘                             └─────────────┘
```

---

## LLM Backends

Switch via `LLM_BACKEND` in `.env`. Zero code changes.

| Backend | Cost | Setup | Best for |
|---------|------|-------|----------|
| `ollama` | Free | `ollama pull llama3.2` | Development, privacy |
| `claude` | ~$0.003/query | `ANTHROPIC_API_KEY=...` | Production quality |
| `openai` | ~$0.002/query | `OPENAI_API_KEY=...` | GPT models |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set LLM_BACKEND=ollama (free) or add ANTHROPIC_API_KEY

# Ingest documents
python3 main.py ingest --path ./your_docs/

# Query
python3 main.py query --question "What are the main topics?"

# Interactive UI
streamlit run demo.py

# REST API
python3 main.py serve
# → http://localhost:8000/docs
```

---

## CLI Commands

```bash
# Core
rag ingest   --path <file|dir|url> --collection <name> [--graph] [--strategy semantic|hierarchical]
rag query    --question "..." [--mode hybrid|dense|sparse] [--hyde] [--multi-query]
rag chat     --collection <name>          # multi-turn conversation
rag adaptive --question "..."             # auto-selects retrieval strategy
rag cot      --question "..." [--max-steps 4]   # Chain-of-Thought RAG (EMNLP 2025)
rag agent    --question "..."             # agentic tool-use loop

# Knowledge Graph
rag graph    stats | entity --name X | communities [--summarize] | global-query --question "..."
rag lightrag query --question "..." [--level auto|low|high|combined]
rag lightrag stats | update --path <file>

# Quality
rag eval     --file questions.json        # RAGAS-style evaluation
rag check-context --question "..."        # Sufficient Context score (ICLR 2025)
rag benchmark                            # latency/throughput test

# Data
rag raptor-ingest   --path <file>         # recursive tree summarization
rag multimodal-ingest --path <pdf>        # tables + figures
rag query-structured --question "..." --schema '{"field": "type"}'
rag route    --question "..."             # show which collection it routes to

# SQL
rag sql query  --question "..."
rag sql setup-sample                      # demo SQLite with products/orders

# Security & Observability
rag security audit [--days 7]
rag security scan  --text "..." [--redact]

# Feedback & Fine-tuning
rag feedback stats | record | export
rag finetune --collection <name>

# Server
rag serve    [--host 0.0.0.0] [--port 8000]
```

---

## REST API

35+ endpoints. Full Swagger at `/docs` when running `rag serve`.

```bash
# Core
POST /ingest              # ingest a document
POST /query               # RAG query
POST /query/agent         # agentic query
POST /query/structured    # extract typed JSON
POST /stream              # streaming SSE response
GET  /collections         # list knowledge bases

# Advanced
POST /adaptive            # adaptive RAG
POST /cot                 # Chain-of-Thought RAG
POST /graph/entities      # graph entity lookup
POST /graph/communities
POST /graph/global-query
POST /sql/query
GET  /sql/schema
POST /lightrag/query

# Quality
GET  /eval/run
GET  /observability/status
POST /observability/score
GET  /security/audit
POST /security/scan

# Health
GET  /health
GET  /metrics             # Prometheus
```

---

## Configuration (`.env`)

```bash
# Backend
LLM_BACKEND=ollama           # ollama | claude | openai
ANTHROPIC_API_KEY=...        # for claude backend
OPENAI_API_KEY=...           # for openai backend

# Retrieval
USE_HYBRID_SEARCH=true       # dense+BM25+RRF
USE_RERANKER=true            # cross-encoder reranking
USE_HYDE=false               # hypothetical document embeddings
TOP_K=6                      # chunks per query

# Quality & Safety
ENABLE_SUFFICIENT_CONTEXT=true        # Google ICLR 2025
SUFFICIENT_CONTEXT_THRESHOLD=0.45     # abstain below this
ENABLE_PII_REDACTION=false            # redact before embedding
ENABLE_INJECTION_DETECTION=true       # scan retrieved chunks
ENABLE_AUDIT_LOG=true                 # JSONL audit trail

# Advanced
USE_CONTEXTUAL_RETRIEVAL=false        # Anthropic Nov 2024 (slower ingest)
WEB_SEARCH_FALLBACK=false             # fall back to Tavily/DDG
TAVILY_API_KEY=...
LANGFUSE_PUBLIC_KEY=...               # semantic tracing
COT_MAX_STEPS=4                       # CoT-RAG reasoning steps
```

---

## Evaluation

Results on the built-in synthetic eval corpus (5 questions, 3 documents, Claude Sonnet backend):

| Configuration | Recall@K | Relevancy | Latency (p50) |
|---------------|----------|-----------|---------------|
| Naive (dense-only) | 0.80 | 0.54 | 2,548ms |
| Hybrid (dense+BM25+RRF) | 0.80 | 0.53 | 2,870ms |
| Hybrid + cross-encoder reranking | 0.80 | 0.53 | 3,520ms |
| **Full stack (hybrid+rerank+HyDE)** | **1.00** | **0.56** | 11,684ms |

HyDE achieves perfect source recall at the cost of ~4.6× latency. For latency-sensitive applications, hybrid+reranking hits 0.80 recall at 3.5s — a reasonable tradeoff. Run against your own corpus for faithfulness scores representative of your domain.

> **CI eval gate** requires an `ANTHROPIC_API_KEY` secret configured in GitHub Actions — see `.github/workflows/eval.yml`. Lint and unit tests run without it.

```bash
# Run full benchmark (naive → hybrid → hybrid+rerank → full stack)
python3 scripts/benchmark_suite.py --compare

# CI quality gate (used in GitHub Actions)
python3 scripts/benchmark_suite.py \
    --output results.json \
    --min-faithfulness 2.5 \
    --min-recall 0.5
```

---

## Demo

**[Live demo →](https://huggingface.co/spaces/joshuasears/rag-system)** *(Hugging Face Spaces — Claude backend)*

Or run locally:

```bash
streamlit run demo.py
```

Features:
- **Source cards** with similarity scores for every retrieved chunk
- **Sufficiency bar** — confidence score before each answer
- **CoT reasoning trace** — collapsible step-by-step thinking
- **Agent tool call log** — every tool call and result
- **Mode comparison** — Naive vs Hybrid vs CoT side-by-side

#### Deploy to Hugging Face Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space (Streamlit SDK)
# 2. Copy hf_space/ contents to the Space repo
# 3. Set secrets: ANTHROPIC_API_KEY, LLM_BACKEND=claude
# 4. Push — the Space auto-ingests sample docs on first boot
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Vector DB | ChromaDB 0.5 |
| Embeddings | sentence-transformers (local, free) |
| Sparse retrieval | rank-bm25 |
| Reranking | ms-marco-MiniLM cross-encoder |
| Knowledge graph | NetworkX |
| LLM backends | Anthropic Claude, OpenAI GPT, Ollama |
| API | FastAPI + uvicorn |
| CLI | Typer + Rich |
| Observability | Langfuse + Prometheus |
| Fine-tuning | sentence-transformers MNR loss |
| Config | Pydantic Settings v2 |
| Tests | pytest + pytest-asyncio |
| CI/CD | GitHub Actions |
| Deploy | Docker + docker-compose |

---

## Docker

```bash
docker compose up
# API at http://localhost:8000
# Prometheus at http://localhost:9090
```

---

## Research References

- **CRAG**: Yan et al., "Corrective Retrieval Augmented Generation" (2024)
- **HyDE**: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (NAACL 2022)
- **RAPTOR**: Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (ICLR 2024)
- **Self-RAG**: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)
- **GraphRAG**: Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft, 2024)
- **Contextual Retrieval**: Anthropic (November 2024)
- **LightRAG**: Guo et al., "LightRAG: Simple and Fast Retrieval-Augmented Generation" (EMNLP 2025)
- **CoT-RAG**: Liu et al., "CoT-RAG: Integrating Chain of Thought and RAG" (EMNLP 2025 Findings)
- **Sufficient Context**: Levy et al., "Sufficient Context: A New Lens on RAG Systems" (Google ICLR 2025)
- **Adaptive RAG**: Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs" (NAACL 2024)

---

## License

MIT
