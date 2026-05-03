"""
Hugging Face Spaces entry point.

On first boot: ingests sample documents so the demo works out-of-the-box.
Set ANTHROPIC_API_KEY and LLM_BACKEND=claude as Space secrets.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent

# Works whether the full repo is the Space root (demo.py one level up)
# or hf_space/ was synced standalone (demo.py alongside this file).
_root = _HERE.parent if (_HERE.parent / "demo.py").exists() else _HERE
sys.path.insert(0, str(_root))

# ── Bootstrap: ingest sample corpus on first run ──────────────────────────────

_MARKER = Path("data/.demo_ingested")

def _bootstrap() -> None:
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

_bootstrap()

# ── Launch the Streamlit demo ─────────────────────────────────────────────────

_demo = _root / "demo.py"
exec(open(_demo).read(), {"__file__": str(_demo)})
