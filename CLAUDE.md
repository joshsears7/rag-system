# Claude Code Context — ~/rag_system/

## What this is
Production-grade RAG system implementing 20+ techniques from 2022–2025 research. Model-agnostic (Ollama/Claude/OpenAI). Built as a reference architecture / portfolio project — not a tutorial.

## Run commands
```bash
# Ingest documents
/opt/anaconda3/bin/python3 main.py ingest --path ./docs

# Query
/opt/anaconda3/bin/python3 main.py query --question "What are the key findings?"

# Streamlit demo UI
streamlit run demo.py

# API server (35+ endpoints)
/opt/anaconda3/bin/python3 api.py
# → docs at http://localhost:8000/docs

# Docker
docker-compose up
```

## Architecture: 13,194 lines total

### Entry points
| File | Lines | Role |
|------|-------|------|
| `main.py` | 1,440 | Typer CLI — all commands |
| `api.py` | 1,484 | FastAPI — 35+ REST endpoints |
| `demo.py` | 538 | Streamlit UI — source cards, sufficiency bar, CoT trace, mode comparison |
| `config.py` | ~100 | LLM_BACKEND env var, ChromaDB path, model settings |
| `models.py` | ~150 | Pydantic v2 models |
| `monitoring.py` | ~200 | Prometheus metrics |

### Core modules (core/)
| File | Lines | What it fixes |
|------|-------|---------------|
| `retrieval.py` | 677 | Hybrid dense+BM25+RRF, cross-encoder reranking, HyDE, MMR, multi-query |
| `generation.py` | 573 | Main generation pipeline — routes to correct technique |
| `ingestion.py` | 479 | Document loading, 3 chunking strategies, quality scoring |
| `sufficient_context.py` | 343 | Google ICLR 2025 — abstains instead of hallucinating (density+coverage+CRAG ensemble) |
| `cot_rag.py` | 377 | CoT-RAG EMNLP 2025 — decomposes question → retrieves per step → synthesizes |
| `light_rag.py` | 500 | LightRAG EMNLP 2025 — dual-level graph: low (entity) + high (community) + auto-routing |
| `graph_rag.py` | 495 | Microsoft GraphRAG — KG extraction + community detection + global queries |
| `adaptive_rag.py` | 407 | Adaptive RAG + Self-RAG — NO_RETRIEVAL/SINGLE_STEP/ITERATIVE routing |
| `raptor.py` | 377 | RAPTOR — recursive tree summarization for long docs |
| `agent.py` | 519 | Agentic RAG — Claude tool_use: search_docs, search_web, query_sql, calculate |
| `contextual_retrieval.py` | ~300 | Anthropic Nov 2024 — 49% fewer retrieval failures |
| `reranker.py` | ~300 | LLM reranking (RankGPT), ensemble, ColBERT-lite |
| `feedback.py` | 389 | SQLite feedback loop, contrastive pair mining |
| `embedding_finetuner.py` | 379 | MNR loss fine-tuning pipeline |
| `multimodal.py` | 371 | PDF table extraction + vision LLM figure descriptions |
| `sql_retrieval.py` | 339 | Text-to-SQL hybrid (SQLAlchemy + LLM-generated SQL) |
| `security.py` | 427 | PII redaction, prompt injection detection, audit logging |
| `conversation.py` | ~300 | Multi-turn with reference resolution + compression |
| `token_budget.py` | 369 | Token budget management |
| `web_search.py` | ~200 | Tavily/DuckDuckGo fallback |
| `router.py` | ~200 | Auto-routing to best collection |
| `evaluation.py` | ~300 | RAGAS-style eval harness |
| `observability.py` | ~150 | Langfuse semantic tracing |

### Scripts
| File | Role |
|------|------|
| `scripts/benchmark_suite.py` | Named dataset benchmark: naive→hybrid→rerank→full stack |
| `scripts/eval_setup.py` | CI eval corpus setup — synthetic docs + QA pairs |

## CLI commands (main.py)
`ingest`, `query`, `chat`, `adaptive`, `raptor-ingest`, `multimodal-ingest`, `agent`,
`sql query`, `sql setup-sample`, `feedback stats/record/export`, `finetune`,
`graph stats/entity/communities/global-query`, `security audit/scan`,
`query-structured`, `route`, `benchmark`, `eval`, `serve`,
`cot`, `lightrag query/stats/update`, `check-context`

## LLM backends (set LLM_BACKEND in .env)
- `ollama` — default, free, local
- `claude` — Anthropic API
- `openai` — OpenAI API

## CI/CD
- `.github/workflows/eval.yml` — runs eval harness on every push
- PR comment with metrics, fails if faithfulness < threshold

## Key problems solved (vs naive RAG tutorials)
1. Retrieval misses → BM25 + RRF fusion
2. Confident hallucinations → CRAG + Sufficient Context abstention
3. Multi-hop failures → CoT-RAG, Adaptive RAG, RAPTOR

## Backup system
- Script: `bash ~/rag_system/backup.sh`
- Saves to: `backups/full/YYYYMMDD_HHMMSS/` — captures root .py files (api.py, main.py, demo.py, config.py, models.py, monitoring.py), core/*.py, scripts/*.py, tests/*.py, Dockerfile, docker-compose.yml, requirements.txt, CLAUDE.md, README.md
- Keeps 5 most recent full backups, auto-deletes older ones
- "Save" or "backup" = run `bash ~/rag_system/backup.sh`

## Rules
- Python: `/opt/anaconda3/bin/python3`
- Pydantic v2 for all models
- ChromaDB for vector store
- sentence-transformers for embeddings
- Parameterized SQL only
- PII is auto-redacted via security.py before storage
