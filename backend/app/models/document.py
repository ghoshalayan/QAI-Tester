"""Document + DocumentChunk models.

A ``Document`` belongs to a ``Project`` and represents one uploaded BRD or
FRD (PDF / DOCX / Markdown / pasted text). After ingest, its content lives
on disk as canonical Markdown at ``data/docs/<project_id>/<doc_id>/parsed.md``
and as ``DocumentChunk`` rows whose ``id`` doubles as the FAISS vector id
in the project's ``chunks`` namespace.

Lifecycle (driven by ``services.ingest_service``):

    pending → parsing → embedding → parsed
                              \\ → failed (terminal)
"""

from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 'BRD' | 'FRD'
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # 'pdf' | 'docx' | 'md' | 'paste'
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)

    filename: Mapped[str] = mapped_column(String(512), nullable=False)

    # Where the original bytes live (NULL for pasted text)
    original_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Where the parsed canonical Markdown is written after parsing
    parsed_md_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Hex of the original file's sha256 (uploaded files only) — useful for dedup later
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 'pending' | 'parsing' | 'embedding' | 'parsed' | 'failed'
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        # 'INSTRUCTIONS' bypasses the BRD→FRD analysis and feeds directly into
        # test-case generation in week 4 — for users who already know what they
        # want tested and don't want to author a requirements doc first.
        CheckConstraint(
            "kind IN ('BRD', 'FRD', 'INSTRUCTIONS')",
            name="document_kind_valid",
        ),
        CheckConstraint(
            "source_type IN ('pdf', 'docx', 'md', 'paste')",
            name="document_source_type_valid",
        ),
        CheckConstraint(
            "status IN ('pending', 'parsing', 'embedding', 'parsed', 'failed')",
            name="document_status_valid",
        ),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    # e.g. "Authentication > Forgot password" — for citing the chunk in UI
    heading_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # slugified heading_path — stable anchor that survives re-parses
    anchor: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # The chunk text the embedder sees. Includes heading_path prepended.
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_document_chunks_document_ordinal",
            "document_id",
            "ordinal",
            unique=True,
        ),
    )
