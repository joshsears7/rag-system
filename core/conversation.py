"""
Multi-turn conversation memory for the RAG system.

Enables chat-style interactions where the model can:
  - Remember what was discussed earlier in the session
  - Resolve references ("it", "that document", "the previous answer")
  - Maintain context across follow-up questions
  - Summarize conversation history to prevent context window overflow

Design: Conversation history is stored as a list of turns in memory.
At query time, recent turns are injected into the system prompt. When
history grows long, it's compressed via LLM summarization (sliding window).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single question-answer exchange."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    collection: str = "default"
    tokens_used: int = 0
    latency_ms: float = 0.0


class ConversationMemory:
    """
    Sliding-window conversation memory with automatic summarization.

    Stores recent turns verbatim. When the turn count exceeds `max_turns`,
    older turns are compressed into a summary by calling the LLM. This
    keeps the injected context size bounded regardless of conversation length.

    Thread-unsafe (single-user session use only — use per-request instances
    for multi-user API deployments and store in a session store like Redis).
    """

    def __init__(
        self,
        max_turns: int = 10,
        summarize_after: int = 6,
    ) -> None:
        self.max_turns = max_turns
        self.summarize_after = summarize_after
        self.turns: list[ConversationTurn] = []
        self.summary: str = ""  # compressed history of older turns
        self._session_id: str = f"session_{int(time.time())}"

    def add_turn(self, turn: ConversationTurn) -> None:
        """Append a completed Q&A turn to the conversation history."""
        self.turns.append(turn)
        logger.debug("Conversation turn added. Total turns: %d", len(self.turns))

    def compress(self, llm_fn: Callable[[str], str]) -> None:
        """
        Compress older turns into a summary to keep context bounded.

        Called automatically when turn count exceeds `summarize_after`.
        Older turns are replaced by a summary; recent turns are kept verbatim.
        """
        if len(self.turns) <= self.summarize_after:
            return

        # Keep recent turns, compress the rest
        to_compress = self.turns[: len(self.turns) - self.summarize_after]
        self.turns = self.turns[-self.summarize_after :]

        history_text = "\n\n".join(
            f"Q: {t.question}\nA: {t.answer[:300]}" for t in to_compress
        )
        prompt = (
            "Summarize the following conversation history concisely (3-5 sentences). "
            "Preserve key facts, entities, and conclusions that may be relevant to future questions.\n\n"
            f"HISTORY:\n{history_text}\n\nSUMMARY:"
        )
        try:
            new_summary = llm_fn(prompt).strip()
            if self.summary:
                self.summary = f"{self.summary}\n\n{new_summary}"
            else:
                self.summary = new_summary
            logger.info("Conversation compressed: %d turns → summary", len(to_compress))
        except Exception as e:
            logger.warning("Conversation compression failed: %s", e)

    def build_context_prompt(self) -> str:
        """
        Build a conversation context string to inject into the system prompt.

        Returns empty string if no history yet.
        """
        parts = []

        if self.summary:
            parts.append(f"CONVERSATION SUMMARY (earlier):\n{self.summary}")

        if self.turns:
            recent_lines = []
            for turn in self.turns[-4:]:  # inject last 4 turns verbatim
                recent_lines.append(f"User: {turn.question}")
                recent_lines.append(f"Assistant: {turn.answer[:400]}")
                if turn.sources:
                    recent_lines.append(f"Sources cited: {', '.join(turn.sources[:3])}")
            parts.append("RECENT CONVERSATION:\n" + "\n".join(recent_lines))

        return "\n\n".join(parts)

    def resolve_references(self, question: str, llm_fn: Callable[[str], str]) -> str:
        """
        Rewrite the question to resolve ambiguous references ("it", "that", "the above").

        This is critical for follow-up questions like "Can you elaborate on it?" where
        "it" refers to a concept from a previous turn.

        Args:
            question: possibly ambiguous follow-up question
            llm_fn: LLM callable for rewriting

        Returns:
            Rewritten question with references resolved (or original if no history)
        """
        if not self.turns:
            return question

        last_turn = self.turns[-1]
        context = f"Previous Q: {last_turn.question}\nPrevious A: {last_turn.answer[:500]}"

        prompt = (
            "Given the conversation context, rewrite the follow-up question to be self-contained "
            "by replacing pronouns and references with explicit terms. "
            "If the question is already clear, return it unchanged.\n\n"
            f"Context:\n{context}\n\n"
            f"Follow-up question: {question}\n\n"
            "Rewritten question (return ONLY the question, no explanation):"
        )
        try:
            rewritten = llm_fn(prompt).strip().strip('"').strip("'")
            if rewritten and rewritten != question:
                logger.debug("Reference resolved: '%s' → '%s'", question[:50], rewritten[:50])
                return rewritten
        except Exception as e:
            logger.debug("Reference resolution failed: %s", e)
        return question

    def clear(self) -> None:
        """Reset the conversation to a clean state."""
        self.turns.clear()
        self.summary = ""
        logger.info("Conversation memory cleared.")

    def to_dict(self) -> dict:
        """Serialize for API responses or persistence."""
        return {
            "session_id": self._session_id,
            "turn_count": len(self.turns),
            "has_summary": bool(self.summary),
            "turns": [
                {
                    "question": t.question,
                    "answer": t.answer,
                    "sources": t.sources,
                    "timestamp": t.timestamp.isoformat(),
                    "collection": t.collection,
                }
                for t in self.turns
            ],
        }


# ── Session registry for multi-user API support ───────────────────────────────

_sessions: dict[str, ConversationMemory] = {}


def get_or_create_session(session_id: str, max_turns: int = 10) -> ConversationMemory:
    """Get an existing session or create a new one by ID."""
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(max_turns=max_turns)
        logger.info("New conversation session created: '%s'", session_id)
    return _sessions[session_id]


def delete_session(session_id: str) -> bool:
    """Delete a conversation session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def list_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_sessions.keys())
