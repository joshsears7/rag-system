"""
Pytest configuration and shared fixtures.

Pre-imports all core modules before any test runs to prevent
test-ordering bugs where patch("config.settings") runs before a module
is imported, causing the module to bind settings to the MagicMock permanently.
"""

from __future__ import annotations

# Pre-import all modules that import `from config import settings` at module level.
# This ensures that `settings` in each module is bound to the real Settings object
# before any test patches config.settings.
import core.ingestion   # noqa: F401
import core.retrieval   # noqa: F401
import core.generation  # noqa: F401
