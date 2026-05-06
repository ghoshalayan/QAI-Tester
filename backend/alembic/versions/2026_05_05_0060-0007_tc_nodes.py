"""tc_nodes table — N-level test-case tree per plan

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-05 01:00:00

Week 4 — FRD→TC synthesis. Each plan owns a tree of test-case nodes:
roots are modules (parent_id IS NULL), depth 1 = submodules, depth 2 = steps.
Schema permits arbitrary depth; the agent generates 3 levels for MVP.

Cascade:
- project / plan delete → wipes the entire tree
- parent_id self-FK CASCADE → deleting a node removes its subtree
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tc_nodes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("plan_id", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False),
        sa.Column("path_cached", sa.String(length=2048), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("description_md", sa.Text(), nullable=True),
        # Step-only fields
        sa.Column("action_type", sa.String(length=64), nullable=True),
        sa.Column("target_hint", sa.Text(), nullable=True),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("expected", sa.Text(), nullable=True),
        sa.Column("data_needs_json", sa.JSON(), nullable=True),
        # Selection + status
        sa.Column(
            "selectable_default",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="draft",
        ),
        # Traceability
        sa.Column("source_requirement_ids", sa.JSON(), nullable=False),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        # FKs
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"], ["test_plans.id"], ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"], ["tc_nodes.id"], ondelete="CASCADE",
        ),
        # Constraints
        sa.CheckConstraint(
            "kind IN ('module', 'submodule', 'step')",
            name="tc_node_kind_valid",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'approved', 'archived')",
            name="tc_node_status_valid",
        ),
    )
    op.create_index("ix_tc_nodes_project_id", "tc_nodes", ["project_id"])
    op.create_index("ix_tc_nodes_plan_id", "tc_nodes", ["plan_id"])
    op.create_index("ix_tc_nodes_parent_id", "tc_nodes", ["parent_id"])
    op.create_index(
        "ix_tc_nodes_plan_parent_ordinal",
        "tc_nodes",
        ["plan_id", "parent_id", "ordinal"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tc_nodes_plan_parent_ordinal", table_name="tc_nodes",
    )
    op.drop_index("ix_tc_nodes_parent_id", table_name="tc_nodes")
    op.drop_index("ix_tc_nodes_plan_id", table_name="tc_nodes")
    op.drop_index("ix_tc_nodes_project_id", table_name="tc_nodes")
    op.drop_table("tc_nodes")
