"""Requirement model — FRD items synthesized by the BRD→FRD agent.

A ``Requirement`` is project-scoped (not document-scoped) — the same FRD
applies regardless of which test plan exercises it. ``source_document_id`` and
``source_chunk_ids`` capture which BRD informed the requirement, so reviewers
can drill into the exact paragraph that motivated each FRD.

FAISS coupling
--------------
Only **approved** requirements get embedded into the project's
``frd_requirements`` namespace. ``embedding_id`` mirrors ``id`` when the row
is in FAISS, and is ``NULL`` otherwise. Un-approving (back to ``edited`` /
``rejected``) removes the vector and clears ``embedding_id``.

Statuses
--------
    proposed → fresh from the agent, not yet reviewed
    edited   → user modified body/title (pre-approval)
    approved → user accepted; in FAISS
    rejected → user dismissed; not in FAISS
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Requirement(Base):
    __tablename__ = "requirements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Source linkage — which BRD this came from + which chunks specifically
    source_document_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_chunk_ids: Mapped[list[int]] = mapped_column(
        JSON, default=list, nullable=False,
    )

    # 'FRD' for now; week 4+ may extend to 'TC' or 'BRD-extracted'
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # Auto-assigned by the agent: "FRD-1", "FRD-2", ...
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    body_md: Mapped[str] = mapped_column(Text, nullable=False)

    # Review state
    status: Mapped[str] = mapped_column(
        String(16), default="proposed", nullable=False,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    # NULL = not in FAISS. Equal to ``id`` when embedded into the project's
    # ``frd_requirements`` namespace.
    embedding_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "kind IN ('FRD')",
            name="requirement_kind_valid",
        ),
        CheckConstraint(
            "status IN ('proposed', 'edited', 'approved', 'rejected')",
            name="requirement_status_valid",
        ),
        Index(
            "ix_requirements_project_code",
            "project_id",
            "code",
            unique=True,
        ),
    )
