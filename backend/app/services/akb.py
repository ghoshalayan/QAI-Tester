"""Production-α/β/γ — App Knowledge Base service.

Single retrieval surface for everything the runtime agent needs to
know about a target application:

- BRD / FRD chunks (α-3 ingestion hook)
- Reconnaissance notes from scout walks (β-1)
- Pattern packs (β-2)
- Dispute outcomes confirmed by the user (γ-1)
- Frozen-path summaries (γ-2)

Storage layout
--------------
- ``app_knowledge`` SQLite table — metadata + raw content. Always
  authoritative (FAISS rebuilds from it on key drift).
- One FAISS ``IndexIDMap`` over ``IndexFlatIP`` per
  ``target_url_pattern``. Saved at:
      data/akb_faiss/<sha256(pattern)>.faiss
  Vector ID == ``app_knowledge.id``. Reuses the BGE embedder
  (1024-dim, L2-normalized → cosine == IP).

Why a separate index from ``app/faiss_store/`` (which is keyed by
project_id + namespace) — AKB is per-target_url so knowledge about
``amazon.com`` flows across projects. Different key space = different
index files. Same on-disk shape, no infra duplication beyond the
tiny dispatcher in this module.

Public API
----------
``query_akb(target_url, query, k=5, kinds=None)`` — top-k chunks
ranked by relevance × confidence. Filters by kind (e.g. only
pattern rules + recon notes for screen-classifier; everything for
the agent's submodule context).

``write_chunk(target_url_pattern, kind, content, ...)`` — appends a
chunk + embeds it + adds to FAISS. Idempotent on (kind, content)
within the same pattern (deduplicates verbatim re-writes from
recon refresh / repeated dispute resolutions).

``invalidate_pattern(target_url_pattern)`` — drops the index file +
deletes rows. Used by the AKB-browser "clear app knowledge" button.

Defaults
--------
The runtime agent typically calls:
    chunks = query_akb(plan.target_url, submodule_goal_text, k=6)
which returns up to 6 chunks blended from all kinds. The orchestrator
renders them as a `KNOWN ABOUT THIS APP:` prompt block (see
``qa_agent`` AKB injection site, α-6).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from sqlalchemy import select

from app.config import settings
from app.embeddings.bge import get_embedder

if TYPE_CHECKING:
    import faiss
    import numpy as np
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Default confidence per kind ────────────────────────────────────
# Set when the caller doesn't supply an explicit confidence; the
# ranker uses these to break ties in retrieval ordering.
_DEFAULT_CONFIDENCE: dict[str, float] = {
    "pattern_rule": 0.95,
    "dispute_outcome": 0.90,
    "frozen_path_summary": 0.85,
    "brd_chunk": 0.80,
    "recon_note": 0.70,
    "manual_note": 0.80,
}


@dataclass
class AkbChunk:
    """Returned shape from ``query_akb``."""

    id: int
    target_url_pattern: str
    kind: str
    content: str
    tags: list[str]
    confidence: float
    relevance: float  # raw FAISS score (0..1 because L2-normalized)
    source_run_id: int | None = None
    source_doc_id: int | None = None


def _akb_dir() -> Path:
    """Resolve ``data/akb_faiss/`` and ensure it exists."""
    raw = getattr(settings, "data_dir", None) or "data"
    p = Path(raw)
    if not p.is_absolute():
        # services/akb.py → backend root is parents[2]
        p = Path(__file__).resolve().parents[2] / p
    out = p / "akb_faiss"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _index_path(target_url_pattern: str) -> Path:
    """Per-pattern index file. Hashed so weird URL chars don't break
    the filesystem (and so pattern length stays bounded)."""
    h = hashlib.sha256(
        target_url_pattern.encode("utf-8", "replace"),
    ).hexdigest()[:32]
    return _akb_dir() / f"{h}.faiss"


# ── In-process FAISS index registry ────────────────────────────────
# Faiss doesn't load itself when this module imports; first call
# resolves the package + warms the cache. ``threading.Lock`` per
# pattern guards mutating ops on the in-memory index.
_indices: dict[str, "faiss.IndexIDMap"] = {}
_locks: dict[str, threading.Lock] = {}
_dict_lock = threading.Lock()
_DIM = 1024  # BGE-large output dim


def _lock_for(pattern: str) -> threading.Lock:
    with _dict_lock:
        lk = _locks.get(pattern)
        if lk is None:
            lk = threading.Lock()
            _locks[pattern] = lk
        return lk


def _ensure_index(pattern: str) -> "faiss.IndexIDMap":
    """Load (or create) the FAISS index for ``pattern``. Cached
    in-process — reload only after process restart or invalidate."""
    if pattern in _indices:
        return _indices[pattern]
    import faiss  # noqa: PLC0415

    path = _index_path(pattern)
    if path.exists():
        try:
            idx = faiss.read_index(str(path))
        except Exception as e:
            logger.warning(
                "AKB FAISS read failed for %s (%s) — rebuilding empty",
                path, e,
            )
            idx = faiss.IndexIDMap(faiss.IndexFlatIP(_DIM))
    else:
        idx = faiss.IndexIDMap(faiss.IndexFlatIP(_DIM))
    _indices[pattern] = idx
    return idx


def _persist_index(pattern: str) -> None:
    import faiss  # noqa: PLC0415

    idx = _indices.get(pattern)
    if idx is None:
        return
    try:
        faiss.write_index(idx, str(_index_path(pattern)))
    except Exception as e:
        logger.warning("AKB FAISS write failed for %s: %s", pattern, e)


def _normalise_pattern(target_url: str) -> str:
    """Reduce a full URL to a stable AKB key.

    Strategy:
    - Drop scheme + path + query — keep just the host.
    - Lowercase.
    - Drop trailing port.
    - Strip ``www.`` prefix so ``www.amazon.com`` and ``amazon.com``
      share knowledge.
    Preserves subdomain (``account.salesforce.com`` and
    ``trailblazer.salesforce.com`` stay distinct — different apps).
    Multi-tenant SaaS sometimes wants to merge by parent domain;
    that's a manual_note + tag concern, not handled here.
    """
    import re  # noqa: PLC0415

    if not target_url:
        return ""
    s = target_url.strip().lower()
    # Strip scheme.
    s = re.sub(r"^[a-z]+://", "", s)
    # Drop path / query / fragment.
    s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    # Drop port.
    s = s.split(":", 1)[0]
    # Strip leading www.
    if s.startswith("www."):
        s = s[4:]
    return s


def _matches_pattern(stored_pattern: str, target_url: str) -> bool:
    """Substring match for storage-pattern → live-URL.

    A pattern of ``amazon.com`` matches ``amazon.com`` AND
    ``amazon.in`` (`amazon.com.au` etc.). Tighter matching is
    available by storing the full host (e.g. ``account.salesforce.com``).
    """
    if not stored_pattern:
        return True  # global pattern
    return stored_pattern in (target_url or "").lower()


# ── Public API ─────────────────────────────────────────────────────


def write_chunk(
    db: "Session",
    *,
    target_url_pattern: str,
    kind: str,
    content: str,
    tags: list[str] | None = None,
    confidence: float | None = None,
    source_run_id: int | None = None,
    source_doc_id: int | None = None,
) -> int | None:
    """Append a knowledge chunk + embed it. Returns the row id, or
    ``None`` when the input is rejected (empty content, etc.).

    Deduplication: if the same (pattern, kind, content) already exists,
    we update the row's ``updated_at`` and confidence and skip
    re-embedding.
    """
    from app.models.app_knowledge import AppKnowledge  # noqa: PLC0415

    pattern = _normalise_pattern(target_url_pattern)
    body = (content or "").strip()
    if not pattern or not body:
        return None

    # Dedup probe.
    existing = db.execute(
        select(AppKnowledge).where(
            AppKnowledge.target_url_pattern == pattern,
            AppKnowledge.kind == kind,
            AppKnowledge.content == body,
        ),
    ).scalar_one_or_none()
    if existing is not None:
        if confidence is not None and confidence > existing.confidence:
            existing.confidence = confidence
        if tags:
            existing.tags = list(tags)
        db.commit()
        return existing.id

    eff_confidence = (
        confidence
        if confidence is not None
        else _DEFAULT_CONFIDENCE.get(kind, 0.75)
    )
    row = AppKnowledge(
        target_url_pattern=pattern,
        kind=kind,
        content=body,
        tags=list(tags) if tags else None,
        confidence=eff_confidence,
        source_run_id=source_run_id,
        source_doc_id=source_doc_id,
    )
    db.add(row)
    db.flush()  # need row.id for FAISS
    row_id = row.id

    # Embed + add to FAISS.
    try:
        emb = get_embedder().embed_documents([body])
        import numpy as np  # noqa: PLC0415

        ids = np.asarray([row_id], dtype="int64")
        with _lock_for(pattern):
            idx = _ensure_index(pattern)
            idx.add_with_ids(emb.astype("float32"), ids)
            _persist_index(pattern)
    except Exception as e:
        logger.warning(
            "AKB embed failed for chunk %s — row stays but won't "
            "be retrievable until reindex: %s", row_id, e,
        )
    db.commit()
    return row_id


def query_akb(
    db: "Session",
    *,
    target_url: str,
    query: str,
    k: int = 6,
    kinds: Iterable[str] | None = None,
    min_confidence: float = 0.0,
) -> list[AkbChunk]:
    """Top-k chunks ranked by ``relevance × confidence``.

    Behavior:
    - Empty target_url OR no chunks for the pattern → returns [].
    - Empty query → returns the k highest-confidence chunks (lets
      callers fetch "general knowledge about this app").
    - Cross-pattern fallback: if the live URL's normalised host has
      no chunks, we also try its parent domain (``foo.acme.io`` →
      ``acme.io``) so multi-tenant SaaS works without per-tenant
      AKB seeding.
    """
    from app.models.app_knowledge import AppKnowledge  # noqa: PLC0415

    pattern = _normalise_pattern(target_url)
    if not pattern:
        return []

    candidates = _find_pattern_chain(pattern)
    if not candidates:
        return []

    # When the query is empty, just return top by confidence (no FAISS
    # call needed).
    if not (query or "").strip():
        rows_q = (
            select(AppKnowledge)
            .where(
                AppKnowledge.target_url_pattern.in_(candidates),
                AppKnowledge.confidence >= min_confidence,
            )
            .order_by(AppKnowledge.confidence.desc())
            .limit(k)
        )
        if kinds:
            rows_q = rows_q.where(AppKnowledge.kind.in_(list(kinds)))
        rows = db.execute(rows_q).scalars().all()
        return [
            AkbChunk(
                id=r.id,
                target_url_pattern=r.target_url_pattern,
                kind=r.kind,
                content=r.content,
                tags=list(r.tags or []),
                confidence=float(r.confidence),
                relevance=1.0,
                source_run_id=r.source_run_id,
                source_doc_id=r.source_doc_id,
            )
            for r in rows
        ]

    # Embed the query once; search each pattern in the chain.
    try:
        q_emb = get_embedder().embed_query(query)
    except Exception as e:
        logger.warning("AKB query embed failed: %s — falling back", e)
        return []

    import numpy as np  # noqa: PLC0415

    q = q_emb.astype("float32").reshape(1, -1)
    hits: dict[int, float] = {}  # row_id → relevance
    for cand in candidates:
        with _lock_for(cand):
            idx = _ensure_index(cand)
            if idx.ntotal == 0:
                continue
            top = min(k * 3, int(idx.ntotal))
            scores, ids = idx.search(q, top)
        for s, i in zip(scores[0].tolist(), ids[0].tolist()):
            if i < 0:
                continue
            prev = hits.get(int(i), -1e9)
            if s > prev:
                hits[int(i)] = float(s)
    if not hits:
        return []

    rows_q = select(AppKnowledge).where(
        AppKnowledge.id.in_(list(hits.keys())),
        AppKnowledge.confidence >= min_confidence,
    )
    if kinds:
        rows_q = rows_q.where(AppKnowledge.kind.in_(list(kinds)))
    rows = db.execute(rows_q).scalars().all()

    out = [
        AkbChunk(
            id=r.id,
            target_url_pattern=r.target_url_pattern,
            kind=r.kind,
            content=r.content,
            tags=list(r.tags or []),
            confidence=float(r.confidence),
            relevance=hits.get(r.id, 0.0),
            source_run_id=r.source_run_id,
            source_doc_id=r.source_doc_id,
        )
        for r in rows
    ]
    # Final ranker: relevance × confidence; cap at k.
    out.sort(key=lambda c: c.relevance * c.confidence, reverse=True)
    return out[:k]


def _find_pattern_chain(pattern: str) -> list[str]:
    """Return [exact, parent-domain, "global"] candidates that exist
    in the table. Empty when nothing has been seeded.

    A parent fallback lets ``account.salesforce.com`` reuse generic
    salesforce knowledge stored under ``salesforce.com``.
    """
    out: list[str] = [pattern]
    parts = pattern.split(".")
    if len(parts) >= 3:
        # Strip leftmost subdomain → parent.
        out.append(".".join(parts[1:]))
    if len(parts) >= 4:
        out.append(".".join(parts[2:]))
    # ``""`` reserves a spot for global pattern packs (apply to all).
    out.append("")
    return out


def list_chunks_for_pattern(
    db: "Session",
    target_url_pattern: str,
) -> list["AppKnowledge"]:  # noqa: F821
    """For the AKB browser UI."""
    from app.models.app_knowledge import AppKnowledge  # noqa: PLC0415

    pattern = _normalise_pattern(target_url_pattern)
    return list(
        db.execute(
            select(AppKnowledge)
            .where(AppKnowledge.target_url_pattern == pattern)
            .order_by(AppKnowledge.kind, AppKnowledge.created_at.desc()),
        ).scalars(),
    )


def invalidate_pattern(
    db: "Session", target_url_pattern: str,
) -> int:
    """Drop all chunks + the FAISS index for one pattern. Returns the
    number of rows removed. Used by the AKB browser's "reset
    knowledge for this app" action and by recon refresh."""
    from app.models.app_knowledge import AppKnowledge  # noqa: PLC0415

    pattern = _normalise_pattern(target_url_pattern)
    if not pattern:
        return 0
    n = (
        db.query(AppKnowledge)
        .filter(AppKnowledge.target_url_pattern == pattern)
        .delete()
    )
    db.commit()
    with _lock_for(pattern):
        _indices.pop(pattern, None)
        try:
            _index_path(pattern).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(
                "AKB index unlink failed for %s: %s", pattern, e,
            )
    return n
