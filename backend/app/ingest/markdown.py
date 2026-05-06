"""Markdown parser — passthrough with normalization.

Idempotent: ``normalize_markdown(normalize_markdown(x)) == normalize_markdown(x)``.

Normalization steps:
1. Decode bytes as UTF-8 (replace errors so a corrupt input doesn't crash ingest)
2. Strip a leading UTF-8 BOM if present
3. Convert CRLF / CR to LF
4. Strip trailing whitespace from every line
5. Collapse 3+ consecutive blank lines down to 2 (one paragraph break)
6. Ensure exactly one trailing newline
"""

from __future__ import annotations


def normalize_markdown(text: str) -> str:
    """Normalize whitespace + line endings of an already-decoded markdown string."""
    if text.startswith("﻿"):
        text = text[1:]

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    out: list[str] = []
    blank_run = 0
    for line in text.split("\n"):
        line = line.rstrip()
        if line:
            blank_run = 0
            out.append(line)
        else:
            blank_run += 1
            if blank_run <= 2:
                out.append("")

    return "\n".join(out).strip() + "\n"


def parse_markdown(raw_bytes: bytes) -> str:
    """Decode raw bytes as UTF-8 and normalize. Robust to corrupt / mixed encodings."""
    text = raw_bytes.decode("utf-8", errors="replace")
    return normalize_markdown(text)
