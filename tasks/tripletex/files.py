"""
tasks/tripletex/files.py
------------------------
File content extraction for Tripletex task attachments.

Strategy:
  PDF  → pymupdf4llm for layout-preserving Markdown (tables, columns, reading order)
         If no text found (scanned PDF), pass raw PDF to Gemini vision as fallback
  Image → multimodal LLM vision call (Gemini Flash with vision)
  Other → best-effort UTF-8 decode

Entry point: process_attachment(attachment) -> str
"""

from __future__ import annotations

import base64
import logging
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.tripletex.solve import FileAttachment

logger = logging.getLogger(__name__)

# Prompt used when passing an image or PDF to the vision LLM
_VISION_EXTRACTION_PROMPT = """You are a financial document data extractor.
Extract ALL relevant information from this document.

For receipts (kvittering) include:
- Store/vendor name
- Receipt date
- EVERY line item with: description, quantity, unit price, VAT rate, total price (including VAT)
- Subtotal, total VAT, grand total
- Payment method

For invoices include:
- vendor/supplier name and address
- invoice number and date
- due date and payment terms
- line items: description, quantity, unit price, VAT rate, total
- subtotal, VAT amount, total amount (with currency)

For contracts and offer letters: extract all key fields, dates, parties, amounts, addresses, phone numbers.

Format the output as a clear structured text with labeled fields. Use tables for line items.
"""


# ===========================================================================
# PDF EXTRACTION
# ===========================================================================

def extract_pdf_text(content_b64: str) -> str:
    """
    Extract text from a base64-encoded PDF using pymupdf4llm.

    pymupdf4llm preserves layout, tables, and reading order as Markdown.
    Falls back to native PDF vision if no embedded text is found (scanned PDF).
    """
    raw = base64.b64decode(content_b64)

    # --- Fast path: pymupdf4llm layout-preserving Markdown ---
    try:
        import pymupdf4llm

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            md_text = pymupdf4llm.to_markdown(tmp.name)

        if md_text and len(md_text.strip()) > 50:
            logger.info("pymupdf4llm extracted %d chars of Markdown", len(md_text))
            return md_text
        else:
            logger.info("pymupdf4llm returned minimal text (%d chars) — trying vision fallback",
                        len(md_text.strip()) if md_text else 0)
    except Exception as exc:
        logger.warning("pymupdf4llm failed: %s — trying vision fallback", exc)

    # --- Fallback: pass raw PDF directly to Gemini vision ---
    logger.info("Using native PDF vision extraction")
    return _extract_pdf_via_vision(content_b64)


def _extract_pdf_via_vision(content_b64: str) -> str:
    """
    Pass the raw PDF directly to Gemini as a multimodal document.
    Much faster and more accurate than page-by-page PNG rendering.
    """
    try:
        from langchain_core.messages import HumanMessage
        from tasks.language.factory import get_llm

        llm = get_llm()
        message = HumanMessage(content=[
            {"type": "text", "text": _VISION_EXTRACTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:application/pdf;base64,{content_b64}"},
            },
        ])
        response = llm.invoke([message])
        content = response.content if isinstance(response.content, str) else str(response.content)
        logger.info("PDF vision extraction returned %d chars", len(content))
        return content

    except Exception as exc:
        logger.warning("PDF vision extraction failed: %s", exc)
        # Last resort: try old-style pymupdf raw text extraction
        return _extract_pdf_raw_text(content_b64)


def _extract_pdf_raw_text(content_b64: str) -> str:
    """Last-resort raw text extraction using pymupdf (no layout preservation)."""
    try:
        import fitz
        raw = base64.b64decode(content_b64)
        doc = fitz.open(stream=raw, filetype="pdf")
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[Page {i}]\n{text}")
        doc.close()
        if pages:
            return "\n\n".join(pages)
    except Exception as exc:
        logger.warning("Raw pymupdf extraction also failed: %s", exc)

    return "(PDF contained no extractable content)"


# ===========================================================================
# IMAGE EXTRACTION — VISION LLM
# ===========================================================================

def extract_image_content(content_b64: str, mime_type: str) -> str:
    """
    Pass an image to the configured LLM (vision) and extract structured data.
    """
    try:
        from langchain_core.messages import HumanMessage
        from tasks.language.factory import get_llm

        llm = get_llm()
        message = HumanMessage(content=[
            {"type": "text", "text": _VISION_EXTRACTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{content_b64}"},
            },
        ])
        response = llm.invoke([message])
        return response.content if isinstance(response.content, str) else str(response.content)

    except Exception as exc:
        logger.warning("Vision extraction failed for %s: %s", mime_type, exc)
        return f"(Image attachment — vision extraction unavailable: {exc})"


# ===========================================================================
# DISPATCHER
# ===========================================================================

def process_attachment(attachment: "FileAttachment") -> str:
    """
    Extract human-readable content from a file attachment.

    Dispatches based on MIME type:
      application/pdf       → pymupdf4llm Markdown, then Gemini vision fallback
      image/*               → vision LLM extraction
      text/plain, text/csv  → direct UTF-8 decode
      other                 → note the file and skip

    Returns a string ready to be injected into the agent prompt.
    """
    mime = attachment.mime_type.lower()
    filename = attachment.filename

    try:
        if mime == "application/pdf":
            logger.info("Extracting text from PDF: %s", filename)
            text = extract_pdf_text(attachment.content_base64)
            return f"### {filename} (PDF)\n{text}"

        elif mime.startswith("image/"):
            logger.info("Running vision extraction on image: %s", filename)
            extracted = extract_image_content(attachment.content_base64, mime)
            return f"### {filename} (Image — extracted data)\n{extracted}"

        elif mime in ("text/plain", "text/csv", "application/csv"):
            raw = base64.b64decode(attachment.content_base64).decode("utf-8", errors="replace")
            return f"### {filename} (Text)\n{raw}"

        else:
            return f"### {filename} ({attachment.mime_type}) — unsupported type, skipped"

    except Exception as exc:
        logger.error("Failed to process attachment %s: %s", filename, exc)
        return f"### {filename} — extraction failed: {exc}"
