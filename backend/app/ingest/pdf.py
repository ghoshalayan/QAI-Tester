"""PDF parser — uses ``pymupdf4llm`` to convert PDF → Markdown.

``pymupdf4llm`` is a wrapper around PyMuPDF specifically tuned for LLM
ingestion. It detects headings via font-size analysis, preserves tables as
GFM, and handles multi-column layouts better than naive text extraction.

License note: PyMuPDF (and pymupdf4llm) are AGPL-or-commercial. Fine for the
local non-distributed MVP. If we ever distribute, swap to ``pdfminer.six``
(MIT, slightly weaker layout extraction).
"""

from __future__ import annotations

from pathlib import Path

from app.ingest.markdown import normalize_markdown


def parse_pdf(file_path: Path) -> str:
    """Convert a PDF file to canonical Markdown."""
    try:
        import pymupdf4llm
    except ImportError as e:
        raise RuntimeError(
            "pymupdf4llm not installed. Run `uv sync` in v2/backend.",
        ) from e

    try:
        md = pymupdf4llm.to_markdown(str(file_path))
    except Exception as e:
        raise RuntimeError(f"PDF parse failed: {type(e).__name__}: {e}") from e

    return normalize_markdown(md)
