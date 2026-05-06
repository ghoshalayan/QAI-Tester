"""DOCX parser — uses ``python-docx`` to convert DOCX → Markdown.

Walks the body's child blocks in document order so paragraphs and tables
stay interleaved correctly (the convenience iterators ``doc.paragraphs``
and ``doc.tables`` lose this ordering).

Heading detection uses Word's built-in style names (``Heading 1`` ..
``Heading 9``). List items use ``List Bullet`` / ``List Number`` /
``List Paragraph`` styles — they all become ``- `` prefixed for MVP
(numbered lists won't auto-increment, but the content is captured).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Union

from app.ingest.markdown import normalize_markdown

if TYPE_CHECKING:
    from docx.document import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph


def _iter_block_items(
    doc: "DocxDocument",
) -> "Iterator[Union[Paragraph, Table]]":
    """Yield paragraphs and tables from the document body in source order."""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _heading_level(style_name: str) -> int | None:
    """Return 1..6 if the style is a Word heading, else None."""
    if not style_name or not style_name.startswith("Heading "):
        return None
    try:
        n = int(style_name.split(" ", 1)[1])
    except (ValueError, IndexError):
        return None
    return min(max(n, 1), 6)


def _is_list_style(style_name: str) -> bool:
    if not style_name:
        return False
    return style_name in {"List Bullet", "List Number", "List Paragraph"}


def _render_paragraph(p: "Paragraph") -> str | None:
    """Convert a paragraph to a Markdown line. Returns None for empty paragraphs."""
    text = p.text.strip()
    if not text:
        return None

    style_name = p.style.name if p.style else ""

    level = _heading_level(style_name)
    if level is not None:
        return f"{'#' * level} {text}"

    if _is_list_style(style_name):
        return f"- {text}"

    return text


def _render_table(t: "Table") -> str:
    """Convert a docx table to a GFM Markdown table."""
    rows = [
        [cell.text.strip().replace("\n", " ").replace("|", "\\|") for cell in row.cells]
        for row in t.rows
    ]
    if not rows or not rows[0]:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]  # pad to uniform width

    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, sep, *body])


def parse_docx(file_path: Path) -> str:
    """Convert a .docx file to canonical Markdown."""
    try:
        from docx import Document as DocxOpen
    except ImportError as e:
        raise RuntimeError(
            "python-docx not installed. Run `uv sync` in v2/backend.",
        ) from e

    try:
        doc = DocxOpen(str(file_path))
    except Exception as e:
        raise RuntimeError(f"DOCX parse failed: {type(e).__name__}: {e}") from e

    parts: list[str] = []
    for block in _iter_block_items(doc):
        # Late imports keep the type-checker happy without forcing the dep
        # into module-import time.
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        if isinstance(block, Paragraph):
            line = _render_paragraph(block)
            if line is not None:
                parts.append(line)
                parts.append("")
        elif isinstance(block, Table):
            rendered = _render_table(block)
            if rendered:
                parts.append(rendered)
                parts.append("")

    return normalize_markdown("\n".join(parts))
