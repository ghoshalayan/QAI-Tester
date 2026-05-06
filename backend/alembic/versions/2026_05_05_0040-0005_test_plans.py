"""test_plans, test_plan_credentials, test_plan_documents

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-05 00:40:00

A TestPlan bundles execution config (target URL + credentials + scope +
instructions) for a project. May optionally link to BRD/FRD/INSTRUCTIONS
documents for the agent to reference during test-case generation.

Credentials store plaintext username/password per the local-MVP "no master
key" policy. OTP shared secrets are NOT persisted — handled live via HITL
intervention every time.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "test_plans",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("target_url", sa.String(length=2048), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("scope", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="draft",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'ready', 'archived')",
            name="test_plan_status_valid",
        ),
    )
    op.create_index("ix_test_plans_project_id", "test_plans", ["project_id"])

    op.create_table(
        "test_plan_credentials",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("plan_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column("username", sa.String(length=512), nullable=False),
        sa.Column("password", sa.String(length=512), nullable=False),
        sa.Column("url_pattern", sa.String(length=2048), nullable=True),
        sa.Column(
            "username_selector_hint", sa.String(length=512), nullable=True,
        ),
        sa.Column(
            "password_selector_hint", sa.String(length=512), nullable=True,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["plan_id"], ["test_plans.id"], ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_test_plan_credentials_plan_id",
        "test_plan_credentials",
        ["plan_id"],
    )

    op.create_table(
        "test_plan_documents",
        sa.Column("plan_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("plan_id", "document_id"),
        sa.ForeignKeyConstraint(
            ["plan_id"], ["test_plans.id"], ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_test_plan_documents_document_id",
        "test_plan_documents",
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_test_plan_documents_document_id",
        table_name="test_plan_documents",
    )
    op.drop_table("test_plan_documents")

    op.drop_index(
        "ix_test_plan_credentials_plan_id",
        table_name="test_plan_credentials",
    )
    op.drop_table("test_plan_credentials")

    op.drop_index("ix_test_plans_project_id", table_name="test_plans")
    op.drop_table("test_plans")
