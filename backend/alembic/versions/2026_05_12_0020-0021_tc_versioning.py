"""Phase C.3 — TC versioning.

Revision ID: 0021
Revises: 0020
Create Date: 2026-05-12 00:20:00

Adds the test-plan TC versioning tables so refinement runs (BRD →
TC initial → app-map-refined → manual edits) each land as a
distinct, immutable snapshot the user can pick between at run-start.

New tables
----------
- ``test_plan_tc_versions``: one row per refinement event.
- ``tc_node_snapshots``: per-node snapshot of the TC tree at that
  version. Self-referential parent_snapshot_id preserves tree
  shape independent of the live tc_nodes table.

New column
----------
- ``test_plans.current_tc_version_id`` (nullable FK to
  ``test_plan_tc_versions.id``). NULL means "run against the live
  TcNode tree"; non-null means "run against the snapshot tree of
  that version".

Compatibility
-------------
Existing plans get NULL ``current_tc_version_id`` and continue
running against the live tc_nodes tree exactly as before. New
versions show up only after the user clicks "Refine from app map".
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0021"
down_revision: Union[str, None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "test_plan_tc_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "plan_id", sa.Integer(), nullable=False,
        ),
        sa.Column(
            "version_number", sa.Integer(), nullable=False,
        ),
        sa.Column(
            "source", sa.String(length=32), nullable=False,
            server_default="manual",
        ),
        sa.Column(
            "label", sa.String(length=120), nullable=False,
            server_default="",
        ),
        sa.Column("notes_json", sa.JSON(), nullable=True),
        sa.Column("source_doc_snapshot", sa.JSON(), nullable=True),
        sa.Column(
            "source_app_map_run_id", sa.Integer(), nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"], ["test_plans.id"], ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_test_plan_tc_versions_plan_id",
        "test_plan_tc_versions",
        ["plan_id"],
    )

    op.create_table(
        "tc_node_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tc_version_id", sa.Integer(), nullable=False,
        ),
        sa.Column(
            "original_tc_node_id", sa.Integer(), nullable=True,
        ),
        sa.Column(
            "parent_snapshot_id", sa.Integer(), nullable=True,
        ),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False),
        sa.Column("path_cached", sa.String(length=2048), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("description_md", sa.Text(), nullable=True),
        sa.Column("action_type", sa.String(length=64), nullable=True),
        sa.Column("target_hint", sa.Text(), nullable=True),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("expected", sa.Text(), nullable=True),
        sa.Column("data_needs_json", sa.JSON(), nullable=True),
        sa.Column(
            "selectable_default", sa.Boolean(), nullable=False,
            server_default="1",
        ),
        sa.Column(
            "change_kind", sa.String(length=24), nullable=False,
            server_default="kept",
        ),
        sa.Column("change_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["tc_version_id"], ["test_plan_tc_versions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["original_tc_node_id"], ["tc_nodes.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["parent_snapshot_id"], ["tc_node_snapshots.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_tc_node_snapshots_tc_version_id",
        "tc_node_snapshots",
        ["tc_version_id"],
    )
    op.create_index(
        "ix_tc_node_snapshots_original_tc_node_id",
        "tc_node_snapshots",
        ["original_tc_node_id"],
    )
    op.create_index(
        "ix_tc_node_snapshots_parent_snapshot_id",
        "tc_node_snapshots",
        ["parent_snapshot_id"],
    )

    with op.batch_alter_table("test_plans") as batch_op:
        batch_op.add_column(
            sa.Column(
                "current_tc_version_id",
                sa.Integer(),
                nullable=True,
            ),
        )
        batch_op.create_foreign_key(
            "fk_test_plans_current_tc_version_id",
            "test_plan_tc_versions",
            ["current_tc_version_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("test_plans") as batch_op:
        batch_op.drop_constraint(
            "fk_test_plans_current_tc_version_id",
            type_="foreignkey",
        )
        batch_op.drop_column("current_tc_version_id")
    op.drop_index("ix_tc_node_snapshots_parent_snapshot_id")
    op.drop_index("ix_tc_node_snapshots_original_tc_node_id")
    op.drop_index("ix_tc_node_snapshots_tc_version_id")
    op.drop_table("tc_node_snapshots")
    op.drop_index("ix_test_plan_tc_versions_plan_id")
    op.drop_table("test_plan_tc_versions")
