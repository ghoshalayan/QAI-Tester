"""Production-α — App Knowledge Base.

Per-target_url knowledge chunks the runtime agent queries at
submodule start. Sources are heterogeneous (BRD chunks, recon
notes, pattern packs, dispute outcomes, frozen-path summaries) but
they all flow through the same retrieval surface.

Embeddings live OUTSIDE this table — the service layer maintains a
FAISS index on disk per ``target_url_pattern`` (under
``data/akb_faiss/``). This row carries metadata + raw content; the
FAISS index maps embedding-vector → row id. Same shape as the BRD
indexer for TC-gen, just keyed by app instead of by document.

Confidence semantics
--------------------
Used by the ranker to prefer high-confidence chunks when relevance
+ recency tie. Defaults by source kind:
- pattern_rule    : 0.95 — hand-curated, reviewed
- dispute_outcome : 0.90 — user-confirmed
- brd_chunk       : 0.80 — explicit business spec
- frozen_path_summary : 0.85 — proven-replayable
- recon_note      : 0.70 — agent-observed, may go stale
- manual_note     : variable — set by author

Lifecycle
---------
- Created by: the AKB service (``app/services/akb.py``).
- Read by: the agent's submodule-start prompt, the screen
  classifier when looking up app-specific UX patterns.
- Updated by: dispute resolution (γ), recon refresh (β), manual
  edits via the AKB browser UI.
- Soft-deleted by: source row deletion (FK ON DELETE SET NULL keeps
  the chunk; ``source_run_id`` becoming NULL is the audit signal).
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AppKnowledge(Base):
    __tablename__ = "app_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    target_url_pattern: Mapped[str] = mapped_column(
        String(512), nullable=False, index=True,
    )
    kind: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.8, server_default="0.8",
    )
    source_run_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_doc_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )
