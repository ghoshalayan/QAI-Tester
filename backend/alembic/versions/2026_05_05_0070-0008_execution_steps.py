"""execution_steps table — per-step results from the executor agent

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-05 02:00:00

Week 5 — FRD→Test Cases→Execution. Each row tracks one step the executor
ran (or skipped/blocked) under an ``agent_runs`` row whose ``kind='execute'``.
Modules and submodules are not persisted; their pass/fail aggregate is
derived from the leaf rows beneath them at render time.

Snapshot fields (``*_snapshot``) freeze the source TcNode at run-time so
later edits or deletions don't mutate history.

Cascade:
- run_id     CASCADE → run delete wipes its rows
- project_id CASCADE → project delete wipes everything
- plan_id    SET NULL → plan delete keeps run history
- tc_node_id SET NULL → node delete/edit keeps run history
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "execution_steps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("plan_id", sa.Integer(), nullable=True),
        sa.Column("tc_node_id", sa.Integer(), nullable=True),
        # Snapshots (frozen at run-time)
        sa.Column("title_snapshot", sa.String(length=512), nullable=False),
        sa.Column("path_snapshot", sa.String(length=2048), nullable=False),
        sa.Column("action_type_snapshot", sa.String(length=64), nullable=True),
        sa.Column("target_hint_snapshot", sa.Text(), nullable=True),
        sa.Column("expected_snapshot", sa.Text(), nullable=True),
        sa.Column("narrative_snapshot", sa.Text(), nullable=True),
        # Run-time state
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        # Outputs
        sa.Column("screenshot_path", sa.Text(), nullable=True),
        sa.Column("narration", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=False),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        # FKs
        sa.ForeignKeyConstraint(
            ["run_id"], ["agent_runs.id"], ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"], ["test_plans.id"], ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tc_node_id"], ["tc_nodes.id"], ondelete="SET NULL",
        ),
        # Constraints
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'passed', 'failed', "
            "'skipped', 'blocked')",
            name="execution_step_status_valid",
        ),
    )
    op.create_index(
        "ix_execution_steps_run_id", "execution_steps", ["run_id"],
    )
    op.create_index(
        "ix_execution_steps_project_id", "execution_steps", ["project_id"],
    )
    op.create_index(
        "ix_execution_steps_tc_node_id", "execution_steps", ["tc_node_id"],
    )
    op.create_index(
        "ix_execution_steps_run_ordinal",
        "execution_steps",
        ["run_id", "ordinal"],
    )


def downgrade() -> None:
    op.drop_index("ix_execution_steps_run_ordinal", table_name="execution_steps")
    op.drop_index("ix_execution_steps_tc_node_id", table_name="execution_steps")
    op.drop_index("ix_execution_steps_project_id", table_name="execution_steps")
    op.drop_index("ix_execution_steps_run_id", table_name="execution_steps")
    op.drop_table("execution_steps")
