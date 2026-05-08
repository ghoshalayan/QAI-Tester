"""Add agent_runs.agent_strategy for Phase 6 (hybrid vs vision-only).

Revision ID: 0015
Revises: 0014
Create Date: 2026-05-09 12:00:00

Phase 6 — mode flag. Within ``mode='agentic'``, the agent has two
sub-strategies:

- ``hybrid`` (default): existing DOM-first ladder (literal → fuzzy →
  vision-search → coord-click). Cheaper, faster, works on most
  modern apps.

- ``vision_only``: every click / type goes through ``propose_click_
  coordinates`` and ``page.mouse.click`` / ``page.keyboard.type``.
  DOM resolution is bypassed entirely. ~3-5× more vision tokens but
  works on apps where DOM resolution is hopeless (heavy canvas,
  custom widgets, cross-origin iframes the resolver can't pierce,
  hostile sites with rotating class names).

The column lives on ``agent_runs`` so the choice is per-run, not
per-plan — same plan can run hybrid for fast smoke and vision-only
for deep validation.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "agent_strategy",
                sa.String(length=16),
                nullable=False,
                server_default="hybrid",
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_column("agent_strategy")
