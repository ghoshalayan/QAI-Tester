"""Wire ``Requirement.status`` transitions to the project's FAISS namespace.

Status → FAISS mapping
----------------------
    proposed → not in FAISS
    edited   → not in FAISS  (auto-demotion when content changes)
    approved → in FAISS
    rejected → not in FAISS

Public API
----------
- :func:`sync_after_change`  — call after PATCH; takes the prior status
- :func:`sync_after_bulk`    — bulk version used by ``POST /bulk-update``
- :func:`remove_before_delete` — drop vector before deleting a row

Transactional model
-------------------
All helpers perform ``db.flush()`` (not commit) so the caller controls the
transaction boundary. If FAISS upsert raises mid-flow, the in-progress
DB transaction never commits — preserving ``embedding_id`` integrity. The
inverse is also true: a FAISS write that succeeds before a DB commit
that later fails will leave a slightly-ahead FAISS — harmless because the
next ``approve`` call upserts and overwrites.

Embedding text contract
-----------------------
``"<code>: <title>\\n\\n<body_md>"`` — including the code (FRD-12) gives
the embedder a strong topical anchor; including the body gives semantic
coverage. The week-4 TC agent retrieves with the same model.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from sqlalchemy.orm import Session

from app.embeddings.bge import get_embedder
from app.faiss_store.store import get_store
from app.models.requirement import Requirement

logger = logging.getLogger(__name__)

NAMESPACE = "frd_requirements"


def _build_embed_text(req: Requirement) -> str:
    return f"{req.code}: {req.title}\n\n{req.body_md}"


# ── Single-row helpers ────────────────────────────────────────────


def _embed_one(db: Session, req: Requirement) -> None:
    """Embed (or re-embed) a single approved requirement. Stamps ``embedding_id``."""
    embedder = get_embedder()
    text = _build_embed_text(req)
    vec = embedder.embed_documents([text])  # shape (1, 1024)

    store = get_store()
    store.upsert(req.project_id, NAMESPACE, [req.id], vec)

    if req.embedding_id != req.id:
        req.embedding_id = req.id
        db.flush()


def _remove_one(db: Session, req: Requirement) -> None:
    """Remove a requirement's vector from FAISS and clear ``embedding_id``."""
    if req.embedding_id is None:
        return
    store = get_store()
    try:
        store.remove(req.project_id, NAMESPACE, [req.embedding_id])
    except Exception as e:
        # Logged but not fatal — DB consistency is what matters
        logger.warning("FAISS remove failed for req %s: %s", req.id, e)
    req.embedding_id = None
    db.flush()


# ── Public API ────────────────────────────────────────────────────


def sync_after_change(
    db: Session,
    req: Requirement,
    *,
    prior_status: str | None,
) -> None:
    """Reconcile FAISS with the requirement's new status.

    Caller is responsible for ``db.commit()`` afterward. This function only
    flushes — embedding_id changes are part of the same uncommitted transaction.

    Behaviors:
    - ``approved`` (was anything) → upsert vector
    - was ``approved`` but no longer → remove vector
    - never approved (proposed/edited/rejected stay) → no-op
    """
    is_approved = req.status == "approved"
    was_approved = prior_status == "approved"

    if is_approved:
        # Covers both first-approval and re-approval after edit (upsert)
        _embed_one(db, req)
    elif was_approved:
        _remove_one(db, req)
    # else: nothing to do (stayed un-approved)


def sync_after_bulk(
    db: Session,
    pairs: Iterable[tuple[Requirement, str | None]],
) -> None:
    """Bulk version: reconcile each ``(requirement, prior_status)`` pair.

    Batches embedding into one ``embed_documents`` call per project for
    efficiency (relevant for "approve all 50 proposed" in one click).
    """
    project_to_add: dict[int, list[Requirement]] = {}
    project_to_remove_ids: dict[int, list[int]] = {}
    rows_to_clear_embedding_id: list[Requirement] = []

    for req, prior_status in pairs:
        is_approved = req.status == "approved"
        was_approved = prior_status == "approved"

        if is_approved:
            project_to_add.setdefault(req.project_id, []).append(req)
        elif was_approved and req.embedding_id is not None:
            project_to_remove_ids.setdefault(req.project_id, []).append(
                req.embedding_id,
            )
            rows_to_clear_embedding_id.append(req)

    if project_to_add:
        embedder = get_embedder()
        store = get_store()
        for pid, reqs_list in project_to_add.items():
            texts = [_build_embed_text(r) for r in reqs_list]
            vectors = embedder.embed_documents(texts)
            ids = [r.id for r in reqs_list]
            store.upsert(pid, NAMESPACE, ids, vectors)
            for r in reqs_list:
                r.embedding_id = r.id

    if project_to_remove_ids:
        store = get_store()
        for pid, ids in project_to_remove_ids.items():
            try:
                store.remove(pid, NAMESPACE, ids)
            except Exception as e:
                logger.warning(
                    "FAISS bulk remove failed for project %s: %s", pid, e,
                )
        for r in rows_to_clear_embedding_id:
            r.embedding_id = None

    db.flush()


def remove_before_delete(req: Requirement) -> None:
    """Drop a requirement's vector from FAISS — call before ``db.delete(req)``.

    Doesn't touch the DB row (caller is deleting it). Tolerant of FAISS
    errors: logs and continues so the row delete still proceeds.
    """
    if req.embedding_id is None:
        return
    try:
        get_store().remove(req.project_id, NAMESPACE, [req.embedding_id])
    except Exception as e:
        logger.warning("FAISS remove-before-delete failed for req %s: %s", req.id, e)
