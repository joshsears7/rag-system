"""
User Feedback Loop — Collect, Store, and Learn from RAG Feedback.

Production RAG systems need a feedback loop to continuously improve.
Without it, you're flying blind — you don't know which answers users
found helpful, which sources were wrong, or which queries consistently fail.

What This Module Provides:
  1. Feedback collection: thumbs up/down, corrections, source quality ratings
  2. Persistent storage: SQLite (local) with migration path to PostgreSQL
  3. Analytics: identify failing queries, low-quality sources, retrieval gaps
  4. Contrastive pair mining: turn feedback into (good, bad) training pairs
     for embedding fine-tuning (the most powerful downstream use of feedback)
  5. Retrieval reranking bias: boost sources that historically get thumbs-up

This is what separates a demo from a product. Companies like Notion,
Intercom, and Linear all have feedback loops on their AI features.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Generator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DB_PATH = Path("./data/feedback.db")


# ── Data models ───────────────────────────────────────────────────────────────


class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"          # user provides correct answer
    SOURCE_IRRELEVANT = "source_irrelevant"  # a cited source wasn't relevant
    SOURCE_HELPFUL = "source_helpful"  # a specific source was great
    INCOMPLETE = "incomplete"          # answer was missing info


class FeedbackEntry(BaseModel):
    """A single piece of user feedback on a RAG response."""

    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    collection: str
    sources_used: list[str] = Field(default_factory=list)
    feedback_type: FeedbackType
    correction: str | None = None      # user's preferred answer (if correction)
    source_feedback: str | None = None # which specific source (if source feedback)
    rating: int | None = None          # 1-5 star rating (optional)
    session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class FeedbackSummary(BaseModel):
    """Analytics summary for the feedback system."""

    total_feedback: int
    thumbs_up: int
    thumbs_down: int
    satisfaction_rate: float          # thumbs_up / (thumbs_up + thumbs_down)
    corrections_count: int
    top_failing_queries: list[str]    # most downvoted questions
    top_helpful_sources: list[str]    # most upvoted sources
    top_failing_sources: list[str]    # most flagged as irrelevant


# ── SQLite storage ────────────────────────────────────────────────────────────


class FeedbackStore:
    """
    Persistent feedback store backed by SQLite.

    SQLite is perfectly adequate for thousands to tens-of-thousands of
    feedback entries. Migrate to PostgreSQL when you hit 100k+ entries
    or need multi-process writes.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    sources_used TEXT NOT NULL DEFAULT '[]',
                    feedback_type TEXT NOT NULL,
                    correction TEXT,
                    source_feedback TEXT,
                    rating INTEGER,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_collection
                ON feedback(collection)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_type
                ON feedback(feedback_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_created
                ON feedback(created_at)
            """)
        logger.debug("Feedback schema initialized at '%s'", self.db_path)

    def record(self, entry: FeedbackEntry) -> str:
        """Persist a feedback entry. Returns the feedback_id."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feedback
                (feedback_id, question, answer, collection, sources_used,
                 feedback_type, correction, source_feedback, rating,
                 session_id, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.feedback_id,
                entry.question,
                entry.answer[:2000],
                entry.collection,
                json.dumps(entry.sources_used),
                entry.feedback_type.value,
                entry.correction,
                entry.source_feedback,
                entry.rating,
                entry.session_id,
                entry.created_at.isoformat(),
                json.dumps(entry.metadata),
            ))
        logger.info("Feedback recorded: %s on '%s'", entry.feedback_type.value, entry.question[:60])
        return entry.feedback_id

    def get_summary(self, collection: str | None = None) -> FeedbackSummary:
        """Compute aggregate feedback analytics."""
        filter_clause = "WHERE collection = ?" if collection else ""
        params = (collection,) if collection else ()

        with self._connect() as conn:
            # Totals
            row = conn.execute(
                f"SELECT COUNT(*) as total FROM feedback {filter_clause}", params
            ).fetchone()
            total = row["total"]

            # Type breakdown
            type_rows = conn.execute(
                f"SELECT feedback_type, COUNT(*) as cnt FROM feedback {filter_clause} GROUP BY feedback_type",
                params,
            ).fetchall()
            counts = {r["feedback_type"]: r["cnt"] for r in type_rows}

            thumbs_up = counts.get("thumbs_up", 0)
            thumbs_down = counts.get("thumbs_down", 0)
            denom = thumbs_up + thumbs_down
            satisfaction = thumbs_up / denom if denom > 0 else 0.0

            # Top failing queries (most downvoted)
            failing = conn.execute(
                f"SELECT question, COUNT(*) as cnt FROM feedback "
                f"{filter_clause + ' AND' if filter_clause else 'WHERE'} feedback_type = 'thumbs_down' "
                "GROUP BY question ORDER BY cnt DESC LIMIT 5",
                params + ("",) if not filter_clause else params,
            ).fetchall()
            # Simpler query to avoid nested conditions:
            failing_q = f"""
                SELECT question, COUNT(*) as cnt FROM feedback
                WHERE feedback_type = 'thumbs_down'
                {'AND collection = ?' if collection else ''}
                GROUP BY question ORDER BY cnt DESC LIMIT 5
            """
            failing_rows = conn.execute(failing_q, (collection,) if collection else ()).fetchall()
            top_failing = [r["question"][:100] for r in failing_rows]

            # Top helpful sources
            helpful_q = f"""
                SELECT source_feedback, COUNT(*) as cnt FROM feedback
                WHERE feedback_type = 'source_helpful' AND source_feedback IS NOT NULL
                {'AND collection = ?' if collection else ''}
                GROUP BY source_feedback ORDER BY cnt DESC LIMIT 5
            """
            helpful_rows = conn.execute(helpful_q, (collection,) if collection else ()).fetchall()
            top_helpful = [r["source_feedback"] for r in helpful_rows]

            # Top failing sources
            failing_src_q = f"""
                SELECT source_feedback, COUNT(*) as cnt FROM feedback
                WHERE feedback_type = 'source_irrelevant' AND source_feedback IS NOT NULL
                {'AND collection = ?' if collection else ''}
                GROUP BY source_feedback ORDER BY cnt DESC LIMIT 5
            """
            failing_src_rows = conn.execute(failing_src_q, (collection,) if collection else ()).fetchall()
            top_failing_sources = [r["source_feedback"] for r in failing_src_rows]

        return FeedbackSummary(
            total_feedback=total,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            satisfaction_rate=round(satisfaction, 3),
            corrections_count=counts.get("correction", 0),
            top_failing_queries=top_failing,
            top_helpful_sources=top_helpful,
            top_failing_sources=top_failing_sources,
        )

    def get_corrections(self, collection: str | None = None, limit: int = 100) -> list[dict]:
        """
        Retrieve all user corrections — (question, bad_answer, correct_answer) triples.

        These are gold for fine-tuning embedding models via contrastive learning:
          - Positive pair: (question, correct_answer)
          - Negative pair: (question, bad_answer)
        """
        q = f"""
            SELECT question, answer, correction FROM feedback
            WHERE feedback_type = 'correction' AND correction IS NOT NULL
            {'AND collection = ?' if collection else ''}
            ORDER BY created_at DESC LIMIT ?
        """
        params = (collection, limit) if collection else (limit,)
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
        return [{"question": r["question"], "bad_answer": r["answer"], "correct_answer": r["correction"]} for r in rows]

    def mine_contrastive_pairs(self, collection: str | None = None) -> list[dict]:
        """
        Generate contrastive training pairs from feedback for embedding fine-tuning.

        Returns:
            List of {"anchor": question, "positive": good_chunk, "negative": bad_chunk}
            suitable for training with MultipleNegativesRankingLoss or TripletLoss.
        """
        pairs = []

        # From corrections: (question, correct_answer=positive, bad_answer=negative)
        corrections = self.get_corrections(collection)
        for c in corrections:
            pairs.append({
                "anchor": c["question"],
                "positive": c["correct_answer"],
                "negative": c["bad_answer"],
                "source": "correction",
            })

        # From thumbs: group questions with both thumbs_up and thumbs_down answers
        q = f"""
            SELECT question,
                   GROUP_CONCAT(CASE WHEN feedback_type='thumbs_up' THEN answer END) as good,
                   GROUP_CONCAT(CASE WHEN feedback_type='thumbs_down' THEN answer END) as bad
            FROM feedback
            {'WHERE collection = ?' if collection else ''}
            GROUP BY question
            HAVING good IS NOT NULL AND bad IS NOT NULL
            LIMIT 200
        """
        with self._connect() as conn:
            rows = conn.execute(q, (collection,) if collection else ()).fetchall()

        for row in rows:
            if row["good"] and row["bad"]:
                pairs.append({
                    "anchor": row["question"],
                    "positive": row["good"][:500],
                    "negative": row["bad"][:500],
                    "source": "thumbs",
                })

        logger.info("Mined %d contrastive pairs for fine-tuning", len(pairs))
        return pairs

    def export_jsonl(self, output_path: Path, collection: str | None = None) -> int:
        """
        Export all feedback to JSONL format for offline analysis or fine-tuning.

        Returns:
            Number of records exported
        """
        filter_q = "WHERE collection = ?" if collection else ""
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM feedback {filter_q} ORDER BY created_at", (collection,) if collection else ()).fetchall()

        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + "\n")

        logger.info("Exported %d feedback entries to '%s'", len(rows), output_path)
        return len(rows)


# ── Retrieval bias from feedback ──────────────────────────────────────────────


def get_source_boost_factors(
    collection: str,
    store: FeedbackStore | None = None,
) -> dict[str, float]:
    """
    Compute per-source boost/penalty factors from historical feedback.

    Sources with many thumbs-up get a boost factor > 1.0.
    Sources flagged as irrelevant get a penalty factor < 1.0.
    This is applied as a multiplicative factor on similarity scores at retrieval time.

    Returns:
        Dict mapping source filename → boost factor (1.0 = neutral)
    """
    if store is None:
        return {}

    boost: dict[str, float] = {}

    try:
        with store._connect() as conn:
            helpful = conn.execute(
                "SELECT source_feedback, COUNT(*) as cnt FROM feedback "
                "WHERE feedback_type = 'source_helpful' AND collection = ? AND source_feedback IS NOT NULL "
                "GROUP BY source_feedback",
                (collection,),
            ).fetchall()

            irrelevant = conn.execute(
                "SELECT source_feedback, COUNT(*) as cnt FROM feedback "
                "WHERE feedback_type = 'source_irrelevant' AND collection = ? AND source_feedback IS NOT NULL "
                "GROUP BY source_feedback",
                (collection,),
            ).fetchall()

        for row in helpful:
            source = row["source_feedback"]
            boost[source] = boost.get(source, 1.0) + (row["cnt"] * 0.05)  # +5% per helpful vote

        for row in irrelevant:
            source = row["source_feedback"]
            boost[source] = boost.get(source, 1.0) - (row["cnt"] * 0.08)  # -8% per irrelevant flag

        # Clamp to [0.5, 1.5]
        return {k: max(0.5, min(1.5, v)) for k, v in boost.items()}

    except Exception as e:
        logger.warning("Failed to compute source boost factors: %s", e)
        return {}


# ── Module-level singleton ────────────────────────────────────────────────────

_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
