"""
Advanced document preprocessing pipeline.

Goes beyond basic text extraction to handle:
  - Table extraction from PDFs and DOCX (preserves structure)
  - Code block detection and special handling
  - Section/heading hierarchy extraction
  - Automatic document summarization at ingest time
  - Language detection and filtering
  - PII detection warnings (emails, phone numbers, SSNs)
  - Document fingerprinting for change detection

These preprocessing steps dramatically improve retrieval quality by
ensuring chunks have clean, well-structured text without garbled
table cells or lost context from code blocks.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ── Document analysis result ──────────────────────────────────────────────────


@dataclass
class DocumentAnalysis:
    """Rich metadata extracted during document preprocessing."""

    source: str
    total_chars: int = 0
    total_words: int = 0
    language: str = "unknown"
    has_tables: bool = False
    has_code_blocks: bool = False
    detected_sections: list[str] = field(default_factory=list)
    pii_warnings: list[str] = field(default_factory=list)
    summary: str = ""
    document_fingerprint: str = ""
    quality_score: float = 1.0  # 0-1, penalized for garbled text


# ── Text quality assessment ───────────────────────────────────────────────────


def assess_text_quality(text: str) -> float:
    """
    Score text quality from 0.0 to 1.0.

    Penalizes:
      - High ratio of non-ASCII characters (garbled PDF extraction)
      - Excessive whitespace / line breaks
      - Very short paragraphs (table artifacts)
      - Repeated characters (OCR noise)

    Returns:
        Quality score (1.0 = clean, 0.0 = likely garbled)
    """
    if not text:
        return 0.0

    total = len(text)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    ascii_ratio = 1.0 - (non_ascii / total)

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    avg_line_len = sum(len(l) for l in non_empty) / max(len(non_empty), 1)
    line_quality = min(1.0, avg_line_len / 40)  # penalize very short lines

    # Detect repeated character sequences (OCR artifacts)
    repeat_pattern = re.compile(r"(.)\1{5,}")  # same char 6+ times
    repeats = len(repeat_pattern.findall(text))
    repeat_penalty = max(0.0, 1.0 - repeats * 0.05)

    score = (ascii_ratio * 0.5 + line_quality * 0.3 + repeat_penalty * 0.2)
    return round(min(1.0, max(0.0, score)), 3)


# ── PII detection ─────────────────────────────────────────────────────────────


_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}


def detect_pii(text: str) -> list[str]:
    """
    Detect potential PII in document text.

    Returns a list of warning strings (types found), not the actual values.
    Use this to warn users before ingesting sensitive documents.
    """
    warnings = []
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            count = len(pattern.findall(text))
            warnings.append(f"{pii_type}: ~{count} occurrences")
    return warnings


# ── Section/heading extraction ────────────────────────────────────────────────


def extract_sections(text: str) -> list[str]:
    """
    Extract section headings from the document.

    Detects:
      - Markdown headings (# Heading)
      - ALL-CAPS lines (common in PDFs and legal docs)
      - Numbered sections (1. Introduction, 2.1 Background)

    Returns:
        List of detected section titles
    """
    sections = []
    patterns = [
        re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE),             # Markdown
        re.compile(r"^([A-Z][A-Z\s]{5,60})$", re.MULTILINE),       # ALL-CAPS
        re.compile(r"^\d+(?:\.\d+)*\.?\s+([A-Z][^\n]{5,60})$", re.MULTILINE),  # Numbered
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            title = match.group(1).strip()
            if title and title not in sections:
                sections.append(title)
    return sections[:50]  # cap at 50


# ── Table extraction ──────────────────────────────────────────────────────────


def extract_markdown_tables(text: str) -> list[str]:
    """
    Extract markdown-formatted tables from text.

    Returns each table as a clean string block.
    """
    table_pattern = re.compile(
        r"(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)",
        re.MULTILINE,
    )
    return [m.group(1).strip() for m in table_pattern.finditer(text)]


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract code blocks from markdown/text.

    Returns list of (language, code) tuples.
    """
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    return [(m.group(1) or "text", m.group(2).strip()) for m in pattern.finditer(text)]


def clean_pdf_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts.

    Fixes:
      - Hyphenated line breaks (re-join words split across lines)
      - Excessive whitespace
      - Form feed characters
      - Ligature replacements (ﬁ → fi, ﬂ → fl)
    """
    # Rejoin hyphenated words at line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace form feeds and null bytes
    text = text.replace("\x0c", "\n\n").replace("\x00", "")
    # Fix ligatures
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for lig, rep in ligatures.items():
        text = text.replace(lig, rep)
    # Normalize multiple spaces
    text = re.sub(r" {3,}", "  ", text)
    # Normalize multiple blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ── Document fingerprinting ───────────────────────────────────────────────────


def fingerprint_document(text: str) -> str:
    """
    Generate a document-level SHA-256 fingerprint for change detection.

    Useful for detecting when a document has been updated and needs re-ingestion.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ── Language detection (simple, no external deps) ────────────────────────────


def detect_language(text: str) -> str:
    """
    Lightweight language detection using character frequency analysis.

    Returns ISO 639-1 language code ("en", "es", "fr", etc.) or "unknown".
    Only detects major languages — use langdetect for production accuracy.
    """
    sample = text[:500].lower()

    # Common function words by language
    language_hints = {
        "en": ["the", "and", "is", "in", "of", "to", "a", "that"],
        "es": ["el", "la", "de", "que", "y", "en", "los", "se"],
        "fr": ["le", "la", "de", "et", "en", "les", "des", "que"],
        "de": ["der", "die", "das", "und", "ist", "in", "den", "von"],
        "pt": ["o", "a", "de", "que", "e", "do", "da", "em"],
    }

    words = re.findall(r"\b\w+\b", sample)
    word_set = set(words)
    scores = {lang: sum(1 for w in hints if w in word_set) for lang, hints in language_hints.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "unknown"


# ── Document summarization ────────────────────────────────────────────────────


def summarize_document(
    text: str,
    source: str,
    llm_fn: Callable[[str], str],
    max_summary_length: int = 300,
) -> str:
    """
    Generate a concise document summary at ingestion time.

    The summary is stored alongside chunks in ChromaDB metadata, enabling
    "collection overview" queries and improving routing accuracy.

    Args:
        text: full document text (truncated to first 3000 chars for efficiency)
        source: source filename for context
        llm_fn: LLM callable
        max_summary_length: target summary length in words

    Returns:
        Summary string, or empty string if generation fails
    """
    # Use first 3000 chars to keep cost low
    sample = text[:3000]
    prompt = (
        f"Write a concise summary (under {max_summary_length} words) of the following document. "
        f"Include: main topic, key points, and any important entities or figures.\n\n"
        f"Document ({source}):\n{sample}\n\nSummary:"
    )
    try:
        summary = llm_fn(prompt).strip()
        logger.debug("Generated summary for '%s': %d chars", source, len(summary))
        return summary
    except Exception as e:
        logger.warning("Document summarization failed for '%s': %s", source, e)
        return ""


# ── Full analysis pipeline ────────────────────────────────────────────────────


def analyze_document(
    text: str,
    source: str,
    llm_fn: Callable[[str], str] | None = None,
    generate_summary: bool = True,
) -> DocumentAnalysis:
    """
    Run the full document analysis pipeline.

    Performs quality assessment, PII detection, section extraction,
    language detection, and optional LLM-generated summary.

    Args:
        text: raw document text
        source: source filename or URL
        llm_fn: optional LLM callable for summarization
        generate_summary: whether to generate an LLM summary

    Returns:
        DocumentAnalysis with all extracted metadata
    """
    cleaned = clean_pdf_text(text)

    analysis = DocumentAnalysis(
        source=source,
        total_chars=len(cleaned),
        total_words=len(cleaned.split()),
        language=detect_language(cleaned),
        has_tables=bool(extract_markdown_tables(cleaned)),
        has_code_blocks=bool(extract_code_blocks(cleaned)),
        detected_sections=extract_sections(cleaned),
        pii_warnings=detect_pii(cleaned),
        document_fingerprint=fingerprint_document(cleaned),
        quality_score=assess_text_quality(cleaned),
    )

    if analysis.pii_warnings:
        logger.warning(
            "PII detected in '%s': %s", source, ", ".join(analysis.pii_warnings)
        )

    if generate_summary and llm_fn and len(cleaned) > 200:
        analysis.summary = summarize_document(cleaned, source, llm_fn)

    logger.info(
        "Document analysis: '%s' | %d words | lang=%s | quality=%.2f | sections=%d",
        source, analysis.total_words, analysis.language,
        analysis.quality_score, len(analysis.detected_sections),
    )
    return analysis
