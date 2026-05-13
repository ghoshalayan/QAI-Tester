"""Phase E — Reusable sub-flow modules.

Revision ID: 0024
Revises: 0023
Create Date: 2026-05-12 00:50:00

Adds the ``sub_flow_modules`` table that stores named, reusable
bundles of proven test steps extracted from passed submodules. The
table is project-scoped; rows carry the v2 frozen segments plus
step summaries so an import can:

1. Replay deterministically (segments → new TcNode.frozen_path)
2. Render the imported submodule in the test-cases viewer
   (step snapshots → new step TcNodes under the imported submodule)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0024"
down_revision: Union[str, None] = "0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sub_flow_modules",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "target_url_pattern", sa.String(length=512), nullable=True,
        ),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("frozen_segments", sa.JSON(), nullable=False),
        sa.Column("step_snapshots", sa.JSON(), nullable=False),
        sa.Column("source_plan_id", sa.Integer(), nullable=True),
        sa.Column(
            "source_submodule_tc_node_id", sa.Integer(), nullable=True,
        ),
        sa.Column("source_run_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_sub_flow_modules_project_id",
        "sub_flow_modules",
        ["project_id"],
    )
    op.create_index(
        "ix_sub_flow_modules_target_url_pattern",
        "sub_flow_modules",
        ["target_url_pattern"],
    )


def downgrade() -> None:
    op.drop_index("ix_sub_flow_modules_target_url_pattern")
    op.drop_index("ix_sub_flow_modules_project_id")
    op.drop_table("sub_flow_modules")
