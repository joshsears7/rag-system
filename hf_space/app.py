"""
Hugging Face Spaces entry point.

On first boot: ingests sample documents so the demo works out-of-the-box.
Set ANTHROPIC_API_KEY and LLM_BACKEND=claude as Space secrets.

This file is the same as demo.py but with a startup bootstrap step.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Bootstrap: ingest sample corpus on first run ──────────────────────────────

_MARKER = Path("data/.demo_ingested")

def _bootstrap() -> None:
    """Ingest sample docs if not already done. Runs once per Space cold start."""
    if _MARKER.exists():
        return

    print("First boot: ingesting sample documents for the demo…")
    try:
        from scripts.eval_setup import setup_eval_corpus
        setup_eval_corpus(collection="demo")
        _MARKER.parent.mkdir(parents=True, exist_ok=True)
        _MARKER.touch()
        print("Sample corpus ready.")
    except Exception as e:
        print(f"Warning: could not ingest sample corpus: {e}")
        print("You can still ingest your own documents via the sidebar.")

_bootstrap()

# ── Launch the Streamlit demo ─────────────────────────────────────────────────

# Import everything from demo.py — it contains the full Streamlit app.
# We just needed the bootstrap step above to run first.
exec(open(Path(__file__).parent.parent / "demo.py").read())
