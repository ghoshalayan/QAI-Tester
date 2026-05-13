"""Add 'recon' to agent_runs.kind CHECK constraint.

Revision ID: 0023
Revises: 0022
Create Date: 2026-05-12 00:40:00

The plan-editor "Scout this app" endpoint creates an AgentRun with
``kind='recon'`` so scout activity surfaces in the runs list with
its own cost breakdown. The original CHECK constraint (migration
0006) didn't include 'recon' → INSERT failed with
``CHECK constraint failed: agent_run_kind_valid``.

SQLite doesn't support ALTER CHECK directly; ``batch_alter_table``
rebuilds the table with the new constraint.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0023"
down_revision: Union[str, None] = "0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_OLD_KINDS = (
    "kind IN ('brd_to_frd', 'frd_to_tc', 'execute', 'reporter')"
)
_NEW_KINDS = (
    "kind IN ('brd_to_frd', 'frd_to_tc', 'execute', 'reporter', 'recon')"
)


def upgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_constraint(
            "agent_run_kind_valid", type_="check",
        )
        batch_op.create_check_constraint(
            "agent_run_kind_valid", _NEW_KINDS,
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_constraint(
            "agent_run_kind_valid", type_="check",
        )
        batch_op.create_check_constraint(
            "agent_run_kind_valid", _OLD_KINDS,
        )
