"""
tasks/tripletex/files.py
------------------------
File content extraction for Tripletex task attachments.

Strategy:
  PDF  → pymupdf text extraction (fast, accurate for digital invoices)
  Image → multimodal LLM vision call (GPT-4o / o4-mini with vision)
  Other → best-effort UTF-8 decode

Entry point: process_attachment(attachment) -> str
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.tripletex.solve import FileAttachment

logger = logging.getLogger(__name__)

# Prompt used when passing an image to the vision LLM
_VISION_EXTRACTION_PROMPT = """You are a financial document data extractor.
Extract ALL relevant information from this document and return it as structured JSON.

For invoices and receipts include:
- vendor / supplier name and address
- invoice number and date
- due date and payment terms
- line items: description, quantity, unit price, VAT rate, total
- subtotal, VAT amount, total amount (with currency)
- customer/recipient name
- any bank account or payment reference numbers

For contracts and other documents: extract all key fields, dates, parties, and amounts.

Return ONLY valid JSON. No markdown, no explanation.
"""


# ===========================================================================
# PDF EXTRACTION
# ===========================================================================

def extract_pdf_text(content_b64: str) -> str:
    """
    Extract text from a base64-encoded PDF using pymupdf.

    Returns the full text across all pages. Includes basic structure
    markers (page breaks) so the LLM can reason about multi-page documents.
    """
    import fitz  # pymupdf

    raw = base64.b64decode(content_b64)
    doc = fitz.open(stream=raw, filetype="pdf")

    pages: list[str] = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")

    doc.close()
    return "\n\n".join(pages) if pages else "(PDF contained no extractable text)"


# ===========================================================================
# IMAGE EXTRACTION — VISION LLM
# ===========================================================================

def extract_image_content(content_b64: str, mime_type: str) -> str:
    """
    Pass an image to the configured LLM (vision) and extract structured data.

    Uses the LangChain multimodal message format (data URI).
    Returns a JSON string with the extracted fields, or a fallback string on error.
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
        return response.content

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
      application/pdf       → pymupdf text extraction
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
            return f"### {filename} (Text)\n{raw[:5000]}"

        else:
            return f"### {filename} ({attachment.mime_type}) — unsupported type, skipped"

    except Exception as exc:
        logger.error("Failed to process attachment %s: %s", filename, exc)
        return f"### {filename} — extraction failed: {exc}"
