"""TestPlan + Credentials + Document-link models.

A ``TestPlan`` bundles execution config — the target URL, credentials, scope
of modules to test, and free-text instructions — for a single project. Each
plan may optionally link to one or more BRD/FRD/INSTRUCTIONS documents whose
content the agent will consume during test-case generation.

The credentials table stores **plaintext** username/password per the local
MVP "no master key" policy. OTP shared secrets are never stored — OTP entry
is handled live via HITL intervention every time the agent encounters one
(Phase 2 · Week 6).

Cascade chain
-------------
``projects → test_plans → test_plan_credentials``
``projects → test_plans → test_plan_documents``
``documents → test_plan_documents``  (deleting a doc removes its links)

All FKs are ``ON DELETE CASCADE`` so a single ``DELETE FROM projects WHERE id=?``
flushes the entire content+config tree atomically.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.models.document import Document


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TestPlan(Base):
    __tablename__ = "test_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_url: Mapped[str] = mapped_column(String(2048), nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # JSON list of free-text module names, e.g. ["Authentication", "Dashboard"].
    # The Plan editor's dropdown pre-populates from headings of linked docs,
    # but users can also type custom scope items.
    scope: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list,
    )

    # 'draft' | 'ready' | 'archived'
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="draft",
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

    credentials: Mapped[list["TestPlanCredential"]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TestPlanCredential.id",
    )
    linked_docs: Mapped[list["TestPlanDocument"]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'ready', 'archived')",
            name="test_plan_status_valid",
        ),
    )


class TestPlanCredential(Base):
    __tablename__ = "test_plan_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("test_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    label: Mapped[str] = mapped_column(String(64), nullable=False)
    # ``username`` / ``password`` / ``totp_secret`` are Fernet-encrypted
    # at rest when ``encrypted=True``. Read via vault.decrypt_credential
    # to get plaintext. Direct DB inspection shows ciphertext.
    username: Mapped[str] = mapped_column(String(512), nullable=False)
    password: Mapped[str] = mapped_column(String(512), nullable=False)
    # Phase 3 — TOTP secret (RFC 6238). When set, the agent generates
    # 2FA codes itself instead of prompting for HITL. NULL / empty
    # means "no TOTP; OTP screens fall back to HITL prompt".
    totp_secret: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    # Phase 3 — encryption-at-rest marker. False on legacy MVP rows
    # (plaintext); True on rows written through the hardened vault.
    # Vault read path branches on this flag.
    encrypted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0",
    )

    # Optional URL pattern this credential applies to. NULL → applies to the
    # plan's ``target_url``. Used by the runtime intervention vault when
    # multiple credentials exist (admin vs user vs read-only).
    url_pattern: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    # Optional CSS-selector hints for the agent — useful when the login
    # form has unusual structure or the agent's auto-detection fails.
    username_selector_hint: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    password_selector_hint: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    plan: Mapped[TestPlan] = relationship(back_populates="credentials")


class TestPlanDocument(Base):
    """Many-to-many join: TestPlans ↔ Documents.

    Plans can run with no docs at all (pure ``description`` instructions);
    or one plan can reference multiple docs (BRD + FRD + INSTRUCTIONS).
    """

    __tablename__ = "test_plan_documents"

    plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("test_plans.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    plan: Mapped[TestPlan] = relationship(back_populates="linked_docs")
    document: Mapped[Document] = relationship()
