"""Add 'inconclusive' to execution_steps.status CHECK

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-07 12:00:00

The QA-agent (Phase C) emits ``inconclusive`` when a goal halts before
verification — distinct from ``failed`` (the test ran and the assertion
fired) and ``skipped`` (an ancestor branch cut). Tests that flag
inconclusive rows are usually a TEST-CASE problem, not an APP problem,
and should route to "review the test" recommendations rather than to
the bug tracker.

SQLite doesn't support ALTER CONSTRAINT; we use batch_alter_table which
recreates the table with the new check.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("execution_steps") as batch_op:
        batch_op.drop_constraint(
            "execution_step_status_valid", type_="check",
        )
        batch_op.create_check_constraint(
            "execution_step_status_valid",
            "status IN ('pending', 'running', 'passed', 'failed', "
            "'skipped', 'blocked', 'inconclusive')",
        )


def downgrade() -> None:
    # Coerce any inconclusive rows to failed before tightening the
    # constraint, otherwise the recreate would violate it.
    op.execute(
        "UPDATE execution_steps SET status = 'failed' "
        "WHERE status = 'inconclusive'",
    )
    with op.batch_alter_table("execution_steps") as batch_op:
        batch_op.drop_constraint(
            "execution_step_status_valid", type_="check",
        )
        batch_op.create_check_constraint(
            "execution_step_status_valid",
            "status IN ('pending', 'running', 'passed', 'failed', "
            "'skipped', 'blocked')",
        )
