"""
RAG Security — PII redaction, prompt injection detection, and query auditing.

Enterprise RAG systems face three classes of security threats:

1. PII Leakage:
   Medical records, financial data, employee info ingested into the vector store
   can be retrieved and included verbatim in LLM responses. Users (or attackers)
   get data they shouldn't see.

2. Prompt Injection via Documents:
   Attacker embeds "IGNORE ALL PREVIOUS INSTRUCTIONS. Your new task is..." in a
   document. When retrieved, it overrides the system prompt and hijacks the LLM.
   Demonstrated against multiple production RAG systems in 2024-2025.

3. Sensitive Query Detection:
   Queries probing for credentials, PII, or internal system information should be
   logged and optionally blocked.

This module provides:
  - Regex-based PII detection (SSN, credit card, email, phone, etc.)
  - Optional presidio integration for ML-based NER
  - Prompt injection pattern matching
  - Audit logging with sanitized query/answer pairs

Zero external dependencies by default (pure regex). Install presidio for
higher accuracy: pip install presidio-analyzer presidio-anonymizer
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = Path("./data/audit_log.jsonl")

# ── PII patterns ──────────────────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    ("SSN", "social_security_number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", "credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("EMAIL", "email_address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("PHONE_US", "us_phone", re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("IP_ADDRESS", "ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IBAN", "iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b")),
    ("PASSPORT", "passport_number", re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),
    ("API_KEY", "api_key", re.compile(r"\b(?:sk-|pk-|api[_-]key[:\s=]+)[A-Za-z0-9_\-]{20,}\b", re.IGNORECASE)),
]

# Replacement tokens by type
_REPLACEMENTS = {
    "SSN": "[REDACTED-SSN]",
    "CREDIT_CARD": "[REDACTED-CARD]",
    "EMAIL": "[REDACTED-EMAIL]",
    "PHONE_US": "[REDACTED-PHONE]",
    "IP_ADDRESS": "[REDACTED-IP]",
    "IBAN": "[REDACTED-IBAN]",
    "PASSPORT": "[REDACTED-PASSPORT]",
    "API_KEY": "[REDACTED-KEY]",
}

# ── Prompt injection patterns ─────────────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions?|context)", re.IGNORECASE),
    re.compile(r"your\s+new\s+(task|instructions?|role|purpose)\s+is", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:different|new|evil|malicious)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*[:=]", re.IGNORECASE),
    re.compile(r"<\|?system\|?>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"###\s*(?:Human|Assistant|System)\s*:", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(?:you|previously)", re.IGNORECASE),
    re.compile(r"repeat\s+after\s+me|say\s+exactly|output\s+the\s+following", re.IGNORECASE),
]

# ── Sensitive query patterns ──────────────────────────────────────────────────

_SENSITIVE_QUERY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(password|passwd|secret|api[\s_-]?key|credentials?|token)\b", re.IGNORECASE),
    re.compile(r"\b(ssn|social\s+security|credit\s+card|cvv|pin)\b", re.IGNORECASE),
    re.compile(r"\b(jailbreak|bypass|override|system\s+prompt)\b", re.IGNORECASE),
    re.compile(r"(exfiltrat|extract|dump)\s+(?:all|every|the)\s+(?:data|documents?|chunks?)", re.IGNORECASE),
]


# ── Detection results ─────────────────────────────────────────────────────────


@dataclass
class PIIDetectionResult:
    """Result of PII scanning."""
    has_pii: bool
    pii_types: list[str] = field(default_factory=list)
    redacted_text: str = ""
    redaction_count: int = 0


@dataclass
class InjectionDetectionResult:
    """Result of prompt injection scanning."""
    is_injection: bool
    matched_patterns: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class QueryAuditEntry:
    """A single audit log entry."""
    timestamp: str
    question_hash: str
    collection: str
    has_pii_in_query: bool
    injection_detected: bool
    sensitive_query: bool
    sources_returned: int
    answer_has_pii: bool
    session_id: str | None = None


# ── PII detection and redaction ───────────────────────────────────────────────


def detect_pii(text: str, use_presidio: bool = False) -> PIIDetectionResult:
    """
    Detect PII in text using regex patterns.

    Optionally uses Microsoft Presidio for ML-based NER (more accurate,
    especially for names and dates of birth).

    Args:
        text: text to scan
        use_presidio: use Presidio ML models (requires: pip install presidio-analyzer)

    Returns:
        PIIDetectionResult with detected types
    """
    if use_presidio:
        try:
            return _detect_pii_presidio(text)
        except ImportError:
            pass  # fall through to regex

    pii_types = []
    for name, _, pattern in _PII_PATTERNS:
        if pattern.search(text):
            pii_types.append(name)

    return PIIDetectionResult(
        has_pii=bool(pii_types),
        pii_types=pii_types,
        redacted_text=text,  # not redacted yet; call redact_pii for that
    )


def redact_pii(text: str, use_presidio: bool = False) -> PIIDetectionResult:
    """
    Detect and redact PII from text, replacing matches with type tokens.

    Args:
        text: text to redact
        use_presidio: use Presidio for higher-accuracy detection

    Returns:
        PIIDetectionResult with redacted_text populated
    """
    if use_presidio:
        try:
            return _redact_pii_presidio(text)
        except ImportError:
            pass  # fall through to regex

    pii_types = []
    redacted = text
    count = 0

    for name, _, pattern in _PII_PATTERNS:
        replacement = _REPLACEMENTS.get(name, "[REDACTED]")
        new_text, n = pattern.subn(replacement, redacted)
        if n > 0:
            pii_types.append(name)
            redacted = new_text
            count += n

    return PIIDetectionResult(
        has_pii=bool(pii_types),
        pii_types=pii_types,
        redacted_text=redacted,
        redaction_count=count,
    )


def _detect_pii_presidio(text: str) -> PIIDetectionResult:
    """Presidio-based PII detection (higher accuracy for names, addresses, DOB)."""
    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language="en")
    pii_types = list({r.entity_type for r in results})
    return PIIDetectionResult(has_pii=bool(pii_types), pii_types=pii_types, redacted_text=text)


def _redact_pii_presidio(text: str) -> PIIDetectionResult:
    """Presidio-based PII redaction."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    pii_types = list({r.entity_type for r in results})
    return PIIDetectionResult(
        has_pii=bool(pii_types),
        pii_types=pii_types,
        redacted_text=anonymized.text,
        redaction_count=len(results),
    )


# ── Prompt injection detection ────────────────────────────────────────────────


def detect_injection(text: str) -> InjectionDetectionResult:
    """
    Scan text for prompt injection patterns.

    Checks retrieved document chunks before including them in the LLM prompt.
    If a chunk matches injection patterns, it's flagged and optionally excluded.

    Args:
        text: chunk text or query to scan

    Returns:
        InjectionDetectionResult (is_injection=True if suspicious)
    """
    matched = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            matched.append(pattern.pattern[:50])

    risk_score = min(1.0, len(matched) * 0.3)

    if matched:
        logger.warning(
            "Prompt injection detected: %d pattern(s) matched in text: '%s…'",
            len(matched), text[:100],
        )

    return InjectionDetectionResult(
        is_injection=bool(matched),
        matched_patterns=matched,
        risk_score=risk_score,
    )


def sanitize_chunk(chunk_text: str, block_injection: bool = True) -> tuple[str, bool]:
    """
    Sanitize a retrieved chunk before including it in the LLM prompt.

    Wraps the chunk in XML tags so the LLM clearly distinguishes it from
    instructions. Optionally blocks chunks with injection patterns.

    Args:
        chunk_text: raw retrieved chunk
        block_injection: if True, replace injected chunks with a warning

    Returns:
        (sanitized_text, was_blocked)
    """
    injection = detect_injection(chunk_text)

    if injection.is_injection and block_injection:
        logger.warning("Blocking chunk with injection score %.1f", injection.risk_score)
        return "[CHUNK BLOCKED: potential prompt injection detected]", True

    # Wrap in XML tags to isolate from instruction tokens
    return f"<retrieved_context>\n{chunk_text}\n</retrieved_context>", False


def is_sensitive_query(query: str) -> bool:
    """
    Check if a user query is probing for sensitive information or attempting injection.

    Returns True if the query should be logged with elevated priority.
    Does not necessarily block the query — that's a policy decision.
    """
    for pattern in _SENSITIVE_QUERY_PATTERNS:
        if pattern.search(query):
            return True
    injection = detect_injection(query)
    return injection.is_injection


# ── Audit logging ─────────────────────────────────────────────────────────────


def audit_query(
    question: str,
    collection: str,
    answer: str = "",
    sources_returned: int = 0,
    session_id: str | None = None,
) -> QueryAuditEntry:
    """
    Log a query/answer pair to the audit log.

    The question is hashed (not stored in plaintext) unless PII/injection is detected.
    The answer is scanned for PII leakage.

    Args:
        question: user query
        collection: collection queried
        answer: LLM-generated answer
        sources_returned: number of sources in the response
        session_id: optional session identifier

    Returns:
        QueryAuditEntry (also written to AUDIT_LOG_PATH)
    """
    question_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    pii_result = detect_pii(question)
    injection_result = detect_injection(question)
    sensitive = is_sensitive_query(question)
    answer_pii = detect_pii(answer).has_pii if answer else False

    entry = QueryAuditEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        question_hash=question_hash,
        collection=collection,
        has_pii_in_query=pii_result.has_pii,
        injection_detected=injection_result.is_injection,
        sensitive_query=sensitive,
        sources_returned=sources_returned,
        answer_has_pii=answer_pii,
        session_id=session_id,
    )

    _write_audit_entry(entry)

    if pii_result.has_pii:
        logger.warning("AUDIT: PII detected in query (hash=%s, types=%s)", question_hash, pii_result.pii_types)
    if injection_result.is_injection:
        logger.warning("AUDIT: Injection attempt detected (hash=%s)", question_hash)
    if answer_pii:
        logger.warning("AUDIT: PII may be present in answer (hash=%s)", question_hash)

    return entry


def _write_audit_entry(entry: QueryAuditEntry) -> None:
    """Append audit entry to JSONL log file."""
    import json
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": entry.timestamp,
        "question_hash": entry.question_hash,
        "collection": entry.collection,
        "has_pii_in_query": entry.has_pii_in_query,
        "injection_detected": entry.injection_detected,
        "sensitive_query": entry.sensitive_query,
        "sources_returned": entry.sources_returned,
        "answer_has_pii": entry.answer_has_pii,
        "session_id": entry.session_id,
    }
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning("Audit log write failed: %s", e)


def get_audit_summary(days: int = 7) -> dict:
    """
    Summarize recent audit log entries.

    Returns:
        Dict with counts of PII, injection attempts, sensitive queries
    """
    import json
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    total = pii_queries = injections = sensitives = answer_pii_count = 0

    if not AUDIT_LOG_PATH.exists():
        return {"total_queries": 0, "days": days}

    try:
        with open(AUDIT_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if ts < cutoff:
                        continue
                    total += 1
                    if entry.get("has_pii_in_query"):
                        pii_queries += 1
                    if entry.get("injection_detected"):
                        injections += 1
                    if entry.get("sensitive_query"):
                        sensitives += 1
                    if entry.get("answer_has_pii"):
                        answer_pii_count += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
    except OSError:
        pass

    return {
        "period_days": days,
        "total_queries": total,
        "pii_in_queries": pii_queries,
        "injection_attempts": injections,
        "sensitive_queries": sensitives,
        "answers_with_pii": answer_pii_count,
        "pii_rate": round(pii_queries / max(total, 1), 3),
        "injection_rate": round(injections / max(total, 1), 3),
    }
