"""Document ingest pipeline.

Each parser converts a source format into **canonical Markdown** — a single
string that:

- Uses ``#``/``##``/``###`` headings for hierarchy
- Has paragraphs separated by blank lines
- Has Unix line endings, no BOM, no trailing whitespace per line
- Contains GFM tables for tabular data when the source had any

The chunker (step 4) walks the headings and assigns each chunk an
``anchor`` (slugified heading_path) — anchors are NOT injected into the
Markdown itself, so the canonical MD stays clean and renders normally.
"""

from app.ingest.chunker import Chunk, chunk_markdown, slugify
from app.ingest.docx import parse_docx
from app.ingest.markdown import normalize_markdown, parse_markdown
from app.ingest.pdf import parse_pdf
from app.ingest.text import parse_paste

__all__ = [
    "Chunk",
    "chunk_markdown",
    "normalize_markdown",
    "parse_docx",
    "parse_markdown",
    "parse_paste",
    "parse_pdf",
    "slugify",
]
