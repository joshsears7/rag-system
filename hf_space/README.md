---
title: RAG System Demo
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: hf_space/app.py
pinned: false
license: mit
short_description: Production RAG — 20+ techniques from 2022-2025 research
---

# RAG System — Live Demo

Production-grade Retrieval-Augmented Generation implementing 20+ techniques from recent research.

**Techniques demonstrated:**
- Hybrid dense+BM25+RRF retrieval
- Cross-encoder reranking
- Sufficient Context abstention (Google ICLR 2025)
- CoT-RAG chain-of-thought reasoning (EMNLP 2025)
- CRAG evaluation + web search fallback
- Semantic query cache

**Set these secrets in your Space settings:**
- `ANTHROPIC_API_KEY` — required (Claude backend)
- `LLM_BACKEND` — set to `claude`

**Source:** [github.com/joshsears7/rag-system](https://github.com/joshsears7/rag-system)
