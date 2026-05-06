"""Paste-text parser.

The paste box accepts any text. If the user is technical they may paste
real Markdown (``# Heading``, lists, code blocks); if not, they may paste
prose. We don't try to auto-detect headings from prose — that's
unreliable. We just:

1. Wrap the text as Markdown (it already is, structurally — paragraphs are
   blank-line separated which is GFM convention)
2. Optionally prepend a user-provided ``title`` as a top-level heading
3. Run through the same normalizer the markdown parser uses
"""

from __future__ import annotations

from app.ingest.markdown import normalize_markdown


def parse_paste(text: str, title: str | None = None) -> str:
    """Wrap pasted text as canonical Markdown.

    Args:
        text: Raw pasted content. Treated as already-Markdown — paragraphs
            inferred from blank lines, existing ``#`` headings preserved.
        title: Optional top-level heading to prepend. If the user pasted a
            doc that already has its own ``# Title``, leave this as None.
    """
    body = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if title and title.strip():
        return normalize_markdown(f"# {title.strip()}\n\n{body}")
    return normalize_markdown(body)
