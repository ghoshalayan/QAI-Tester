"""Heading-aware Markdown chunker.

Walks canonical Markdown and splits it into chunks suitable for embedding.

Algorithm
---------
1. Parse the MD line by line, tracking the current heading stack.
2. Accumulate content under each heading. When a new heading appears, flush
   the previous (path, content) as a *section*.
3. For each section, split its content into pieces of ~``target_size``
   characters with ``overlap`` overlap. Splits prefer paragraph boundaries,
   then line breaks, then sentence ends.
4. Each emitted chunk has the heading path prepended as a plain
   ``"A > B > C\\n\\n<content>"`` line — the embedder picks up the topical
   context that way without polluting the raw content with markdown syntax.

Code-block awareness
--------------------
``#`` lines inside fenced code blocks (\\`\\`\\`...) are preserved as
content, not parsed as headings.

Anchors
-------
The anchor is the slugified heading path (or ``section-N`` for content
above the first heading). Multiple chunks in the same leaf section share
the same anchor — uniqueness comes from ``(document_id, ordinal)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^```")
_SLUG_BAD_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Chunk:
    ordinal: int
    heading_path: str   # e.g. "Authentication > Login Flow"
    anchor: str          # slugified heading_path
    text: str            # heading_path prepended + section content
    char_count: int


@dataclass
class _Section:
    path: tuple[str, ...]
    content: str


def slugify(text: str) -> str:
    """Convert ``"Authentication > Forgot Password"`` to ``"authentication-forgot-password"``."""
    s = _SLUG_BAD_RE.sub("-", text.lower()).strip("-")
    return s[:240]  # leave headroom under the 256-char DB column


def _walk_sections(md: str) -> list[_Section]:
    """Yield one ``_Section`` per leaf section (path + content between headings)."""
    sections: list[_Section] = []
    stack: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        if not buf:
            return
        content = "\n".join(buf).strip()
        if content:
            sections.append(_Section(tuple(stack), content))
        buf.clear()

    for line in md.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue

        if in_fence:
            buf.append(line)
            continue

        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            flush()
            # Trim stack to parent depth, then push
            stack[:] = stack[: level - 1]
            stack.append(heading_text)
        else:
            buf.append(line)

    flush()
    return sections


def _find_break(text: str, start: int, end: int) -> int | None:
    """Find a sensible split point in ``text[start:end]`` — paragraph, line, then sentence."""
    if start >= end or end > len(text):
        return None

    idx = text.rfind("\n\n", start, end)
    if idx >= 0:
        return idx + 2

    idx = text.rfind("\n", start, end)
    if idx >= 0:
        return idx + 1

    idx = text.rfind(". ", start, end)
    if idx >= 0:
        return idx + 2

    return None


def _split_text(text: str, target: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of approximately ``target`` chars with ``overlap`` overlap."""
    if len(text) <= target:
        return [text]

    if overlap >= target:
        raise ValueError("overlap must be smaller than target")

    chunks: list[str] = []
    pos = 0
    n = len(text)

    while pos < n:
        end = min(pos + target, n)
        if end < n:
            search_start = max(pos + int(target * 0.7), pos + 1)
            search_end = min(pos + int(target * 1.3), n)
            split = _find_break(text, search_start, search_end)
            if split is not None and split > pos:
                end = split

        piece = text[pos:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break
        pos = max(pos + 1, end - overlap)

    return chunks


def chunk_markdown(
    md: str,
    target_size: int = 800,
    overlap: int = 100,
) -> list[Chunk]:
    """Convert canonical Markdown into a list of chunks ready for embedding."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if overlap < 0 or overlap >= target_size:
        raise ValueError("overlap must satisfy 0 <= overlap < target_size")

    sections = _walk_sections(md)
    out: list[Chunk] = []
    ordinal = 0

    for section_idx, section in enumerate(sections):
        heading_path = " > ".join(section.path) if section.path else ""
        anchor = slugify(heading_path) if heading_path else f"section-{section_idx}"
        if not anchor:
            anchor = f"section-{section_idx}"

        for piece in _split_text(section.content, target_size, overlap):
            text = (
                f"{heading_path}\n\n{piece}".strip()
                if heading_path
                else piece.strip()
            )
            if not text:
                continue
            out.append(
                Chunk(
                    ordinal=ordinal,
                    heading_path=heading_path,
                    anchor=anchor,
                    text=text,
                    char_count=len(text),
                ),
            )
            ordinal += 1

    return out
