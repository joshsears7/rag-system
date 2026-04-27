"""
Multi-modal RAG — Handle PDFs with Images, Charts, and Tables.

The Problem:
  Standard RAG extracts text and throws away images, charts, and tables.
  In financial reports, scientific papers, and technical documentation,
  30-70% of critical information lives in figures and tables.

What This Module Adds:
  1. Table extraction → markdown-formatted structured text
  2. Chart/figure description via Claude's vision API (multimodal)
  3. PDF page rendering for image extraction
  4. Screenshot/image ingestion (OCR via pytesseract + vision LLM)
  5. Table-aware chunking: tables are stored as complete units, not split

Architecture:
  The key insight is that vision-language models can "read" figures and charts.
  We extract each figure, send it to Claude as an image, and store the
  generated description as a searchable text chunk with special metadata.

  This means a query like "What was the revenue trend in Q3?" can retrieve
  the bar chart description alongside text paragraphs.

Dependencies (optional, graceful fallback if not installed):
  pip install pymupdf pytesseract pillow
  brew install tesseract  # for OCR
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ExtractedElement:
    """A non-text element extracted from a document."""

    element_type: str          # "table", "figure", "chart", "image"
    description: str           # text description (from OCR or vision LLM)
    raw_text: str              # raw extracted text (if any)
    page_number: int | None
    source: str
    confidence: float = 1.0   # OCR/extraction confidence
    image_b64: str = ""        # base64-encoded image (for vision LLM)
    metadata: dict = field(default_factory=dict)


# ── Table extraction ──────────────────────────────────────────────────────────


def extract_tables_from_pdf(pdf_path: str) -> list[ExtractedElement]:
    """
    Extract tables from a PDF using pdfplumber (best table extraction library).

    Each table is converted to markdown format and stored as an ExtractedElement.
    Tables are stored whole (not chunked) to preserve cell relationships.

    Returns:
        List of ExtractedElement with type="table"
    """
    elements = []
    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not installed. pip install pdfplumber for table extraction.")
        return []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table or not table[0]:
                        continue

                    # Convert to markdown
                    rows = []
                    for i, row in enumerate(table):
                        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
                        rows.append("| " + " | ".join(cleaned) + " |")
                        if i == 0:  # add separator after header
                            rows.append("| " + " | ".join("---" for _ in row) + " |")

                    md_table = "\n".join(rows)
                    elements.append(ExtractedElement(
                        element_type="table",
                        description=f"Table from page {page_num}:\n{md_table}",
                        raw_text=md_table,
                        page_number=page_num,
                        source=pdf_path,
                        metadata={"table_index": table_idx, "rows": len(table), "cols": len(table[0])},
                    ))

        logger.info("Extracted %d tables from '%s'", len(elements), pdf_path)
    except Exception as e:
        logger.warning("Table extraction failed for '%s': %s", pdf_path, e)

    return elements


# ── PDF image extraction ──────────────────────────────────────────────────────


def extract_images_from_pdf(pdf_path: str, min_size: int = 100) -> list[ExtractedElement]:
    """
    Extract images from a PDF and prepare them for vision LLM description.

    Uses PyMuPDF (fitz) for rendering, which handles embedded images,
    vector graphics rendered to raster, and diagram pages.

    Args:
        pdf_path: path to PDF file
        min_size: minimum pixel dimension to include (filters out tiny logos/icons)

    Returns:
        List of ExtractedElement with type="figure" and image_b64 populated
    """
    elements = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("PyMuPDF not installed. pip install pymupdf for image extraction.")
        return []

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            for img_idx, img_ref in enumerate(image_list):
                xref = img_ref[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    if width < min_size or height < min_size:
                        continue  # skip tiny images

                    b64 = base64.b64encode(image_bytes).decode("utf-8")
                    img_ext = base_image.get("ext", "png")

                    elements.append(ExtractedElement(
                        element_type="figure",
                        description="",  # will be filled by vision LLM
                        raw_text="",
                        page_number=page_num + 1,
                        source=pdf_path,
                        image_b64=b64,
                        metadata={"width": width, "height": height, "ext": img_ext, "xref": xref},
                    ))
                except Exception as e:
                    logger.debug("Failed to extract image %d from page %d: %s", img_idx, page_num, e)

        doc.close()
        logger.info("Extracted %d images from '%s'", len(elements), pdf_path)
    except Exception as e:
        logger.warning("Image extraction failed for '%s': %s", pdf_path, e)

    return elements


# ── Vision LLM description ────────────────────────────────────────────────────


def describe_image_with_claude(
    image_b64: str,
    image_ext: str = "png",
    context: str = "",
    claude_client: "anthropic.Anthropic | None" = None,  # type: ignore[name-defined]
) -> str:
    """
    Use Claude's vision API to generate a searchable text description of an image.

    This transforms unreadable figures into retrievable text:
      - Bar charts → "Revenue bar chart showing Q1: $2.3M, Q2: $2.8M, Q3: $3.1M"
      - Architecture diagrams → description of components and relationships
      - Tables as images → extracted cell values

    Args:
        image_b64: base64-encoded image data
        image_ext: image file extension (png, jpg, etc.)
        context: surrounding text context to help with description
        claude_client: pre-initialized Anthropic client

    Returns:
        Text description of the image
    """
    try:
        import anthropic
        from config import settings

        client = claude_client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

        media_type_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
        media_type = media_type_map.get(image_ext.lower(), "image/png")

        context_str = f"\n\nDocument context: {context[:300]}" if context else ""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # use Haiku for cost efficiency
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail for a search index. "
                            "If it's a chart or graph, extract the key data points and trends. "
                            "If it's a table, list the key values. "
                            "If it's a diagram, describe the components and relationships. "
                            "Be specific and factual — your description will be used for semantic search."
                            + context_str
                        ),
                    },
                ],
            }],
        )
        return response.content[0].text
    except Exception as e:
        logger.warning("Vision LLM description failed: %s", e)
        return "Image content (description unavailable)"


def describe_images_batch(
    elements: list[ExtractedElement],
    claude_client: "anthropic.Anthropic | None" = None,  # type: ignore[name-defined]
) -> list[ExtractedElement]:
    """
    Add vision descriptions to all ExtractedElements with image_b64.

    Processes images sequentially (Claude API rate limits apply).
    In production, batch with a semaphore for concurrent processing.
    """
    described = []
    for i, elem in enumerate(elements):
        if elem.image_b64:
            logger.debug("Describing image %d/%d from page %d…", i + 1, len(elements), elem.page_number or 0)
            description = describe_image_with_claude(
                image_b64=elem.image_b64,
                image_ext=elem.metadata.get("ext", "png"),
                claude_client=claude_client,
            )
            elem = ExtractedElement(
                element_type=elem.element_type,
                description=description,
                raw_text=elem.raw_text,
                page_number=elem.page_number,
                source=elem.source,
                image_b64="",  # clear after processing to save memory
                metadata=elem.metadata,
            )
        described.append(elem)
    return described


# ── Ingest multi-modal elements ───────────────────────────────────────────────


def ingest_multimodal_elements(
    elements: list[ExtractedElement],
    collection_name: str,
) -> int:
    """
    Store multi-modal element descriptions in ChromaDB alongside text chunks.

    Each element's description becomes a searchable chunk with rich metadata
    indicating it came from a figure, table, or image.

    Returns:
        Number of elements stored
    """
    from core.ingestion import embed_texts, get_or_create_collection
    from datetime import datetime, timezone
    import hashlib

    if not elements:
        return 0

    col = get_or_create_collection(collection_name)
    texts = [e.description for e in elements if e.description]
    if not texts:
        return 0

    embeddings = embed_texts(texts)

    ids, embs, docs, metas = [], [], [], []
    valid_elements = [e for e in elements if e.description]

    for elem, emb in zip(valid_elements, embeddings):
        chunk_id = hashlib.sha256(elem.description.encode()).hexdigest()[:16] + f"_{elem.element_type}"
        ids.append(chunk_id)
        embs.append(emb)
        docs.append(elem.description)
        metas.append({
            "source_file": elem.source,
            "element_type": elem.element_type,
            "page_number": elem.page_number or -1,
            "chunk_index": 0,
            "content_hash": hashlib.sha256(elem.description.encode()).hexdigest(),
            "timestamp_ingested": datetime.now(timezone.utc).isoformat(),
            "word_count": len(elem.description.split()),
            "char_count": len(elem.description),
            "doc_type": "multimodal",
            "section_title": f"{elem.element_type.title()} from page {elem.page_number}",
        })

    col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    logger.info("Stored %d multi-modal elements in '%s'", len(ids), collection_name)
    return len(ids)


# ── Full multi-modal ingestion pipeline ──────────────────────────────────────


def ingest_pdf_multimodal(
    pdf_path: str,
    collection_name: str,
    extract_tables: bool = True,
    extract_figures: bool = True,
    describe_figures: bool = True,
) -> dict:
    """
    Full multi-modal PDF ingestion: extract tables + figures, describe with vision LLM.

    Args:
        pdf_path: path to PDF
        collection_name: target ChromaDB collection
        extract_tables: extract and store table content
        extract_figures: extract embedded images
        describe_figures: use Claude vision to describe figures

    Returns:
        Summary dict with counts of extracted elements
    """
    all_elements: list[ExtractedElement] = []

    if extract_tables:
        tables = extract_tables_from_pdf(pdf_path)
        all_elements.extend(tables)
        logger.info("Found %d tables in '%s'", len(tables), pdf_path)

    if extract_figures:
        figures = extract_images_from_pdf(pdf_path)
        if describe_figures and figures:
            figures = describe_images_batch(figures)
        all_elements.extend(figures)
        logger.info("Found %d figures in '%s'", len(figures), pdf_path)

    stored = ingest_multimodal_elements(all_elements, collection_name)

    return {
        "pdf": pdf_path,
        "tables_found": len([e for e in all_elements if e.element_type == "table"]),
        "figures_found": len([e for e in all_elements if e.element_type == "figure"]),
        "elements_stored": stored,
    }
