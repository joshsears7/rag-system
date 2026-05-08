"""
SQLite-backed conversation persistence.

Stores sessions and turns so chat history survives browser refreshes.
On ephemeral environments (HF Spaces) this persists within a container
lifecycle; on local installs it persists indefinitely.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.conversation import ConversationMemory, ConversationTurn

_DB_PATH = Path("./data/chat_sessions.db")


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            collection  TEXT DEFAULT 'default',
            created_at  TEXT,
            updated_at  TEXT,
            turn_count  INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS turns (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL REFERENCES sessions(id),
            question    TEXT,
            answer      TEXT,
            sources     TEXT,
            collection  TEXT,
            tokens_used INTEGER DEFAULT 0,
            latency_ms  REAL    DEFAULT 0.0,
            timestamp   TEXT
        );
    """)
    return c


def new_session(collection: str = "default") -> str:
    sid = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (id, collection, created_at, updated_at) VALUES (?,?,?,?)",
            (sid, collection, now, now),
        )
    return sid


def save_turn(session_id: str, turn: ConversationTurn) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO turns (session_id, question, answer, sources, collection, "
            "tokens_used, latency_ms, timestamp) VALUES (?,?,?,?,?,?,?,?)",
            (
                session_id,
                turn.question,
                turn.answer,
                json.dumps(turn.sources),
                turn.collection,
                turn.tokens_used,
                turn.latency_ms,
                turn.timestamp.isoformat(),
            ),
        )
        c.execute(
            "UPDATE sessions SET updated_at=?, turn_count=turn_count+1 WHERE id=?",
            (now, session_id),
        )


def load_memory(session_id: str) -> ConversationMemory:
    memory = ConversationMemory(max_turns=10, summarize_after=6)
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY id ASC", (session_id,)
        ).fetchall()
    for row in rows:
        t = ConversationTurn(
            question=row["question"],
            answer=row["answer"],
            sources=json.loads(row["sources"] or "[]"),
            collection=row["collection"] or "default",
            tokens_used=row["tokens_used"] or 0,
            latency_ms=row["latency_ms"] or 0.0,
        )
        memory.turns.append(t)
    return memory


def list_sessions(limit: int = 8) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, collection, created_at, updated_at, turn_count "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
