"""Add 'record' to agent_runs.kind CHECK constraint.

Revision ID: 0026
Revises: 0025
Create Date: 2026-05-13 13:10:00

Phase W' — the recording session creates an ``AgentRun`` with
``kind='record'`` so it shows up in the runs list alongside execute
/ recon / brd_to_frd / etc. The CHECK constraint from migration
0023 (which added 'recon') didn't include 'record' → INSERT failed
with ``CHECK constraint failed: agent_run_kind_valid``.

SQLite doesn't support ALTER CHECK directly; ``batch_alter_table``
rebuilds the table with the new constraint.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "0026"
down_revision: Union[str, None] = "0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_OLD_KINDS = (
    "kind IN ('brd_to_frd', 'frd_to_tc', 'execute', 'reporter', 'recon')"
)
_NEW_KINDS = (
    "kind IN ('brd_to_frd', 'frd_to_tc', 'execute', 'reporter', "
    "'recon', 'record')"
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
