"""Add tc_nodes.frozen_path JSON for Phase E (replay mode).

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-08 12:00:00

Phase E.1 — Path freezing. When an agentic run on a submodule passes
vision verification cleanly, we serialize the agent's working tool
sequence (with the actual successful selectors after fuzzy / vision
substitution) onto the submodule. Future runs in mode='replay' walk
this path deterministically — no LLM calls — at ~5% of the agentic
token cost and >95% reliability.

Field is JSON-shaped, NULL when no run has frozen a path yet.
Replay mode falls back to agentic for any submodule with a NULL
frozen_path so partial freeze-ups don't block coverage.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("tc_nodes") as batch_op:
        batch_op.add_column(
            sa.Column(
                "frozen_path",
                sa.JSON(),
                nullable=True,
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("tc_nodes") as batch_op:
        batch_op.drop_column("frozen_path")
