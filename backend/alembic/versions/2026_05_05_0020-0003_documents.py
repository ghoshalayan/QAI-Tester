"""documents and document_chunks tables

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-05 00:20:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("original_path", sa.String(length=1024), nullable=True),
        sa.Column("parsed_md_path", sa.String(length=1024), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], ondelete="CASCADE",
        ),
        sa.CheckConstraint("kind IN ('BRD', 'FRD')", name="document_kind_valid"),
        sa.CheckConstraint(
            "source_type IN ('pdf', 'docx', 'md', 'paste')",
            name="document_source_type_valid",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'parsing', 'embedding', 'parsed', 'failed')",
            name="document_status_valid",
        ),
    )
    op.create_index(
        "ix_documents_project_id", "documents", ["project_id"],
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading_path", sa.String(length=1024), nullable=True),
        sa.Column("anchor", sa.String(length=256), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )
    op.create_index(
        "ix_document_chunks_document_ordinal",
        "document_chunks",
        ["document_id", "ordinal"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunks_document_ordinal", table_name="document_chunks",
    )
    op.drop_index(
        "ix_document_chunks_document_id", table_name="document_chunks",
    )
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_project_id", table_name="documents")
    op.drop_table("documents")
