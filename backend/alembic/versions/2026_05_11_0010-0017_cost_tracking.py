"""Cost tracking — per-tier token columns + tier pricing.

Revision ID: 0017
Revises: 0016
Create Date: 2026-05-11 09:00:00

Two coordinated additions for the cost-tracking feature:

1. ``agent_runs`` gets:
   - ``strong_input_tokens`` / ``strong_output_tokens``
   - ``cheap_input_tokens``  / ``cheap_output_tokens``
     (four integer counters, default 0). The agent loop's existing
     aggregate ``input_tokens`` / ``output_tokens`` stay for back-
     compat; the new columns split the totals by tier so cost
     reports can apply different ``$/M`` rates to each.
   - ``strong_model_snapshot`` / ``cheap_model_snapshot`` (String
     nullable). Captures WHICH model produced the tokens at run
     time so cost calculations stay correct after the user changes
     their LLM settings. Without this snapshot, switching from
     GPT-4o to GPT-5 would silently re-cost every historical run
     at the new rate.

2. ``app_settings`` gets:
   - ``strong_input_price_per_m`` / ``strong_output_price_per_m``
   - ``cheap_input_price_per_m``  / ``cheap_output_price_per_m``
     (four Float nullable columns). User-supplied USD pricing in
     ``$ per million tokens`` — the industry-standard format.
     NULL means "user hasn't configured pricing for this tier"
     → cost view renders ``$—`` instead of $0.

Backfill (NOT in this migration — handled by the cost service at
read time per the user's locked policy):
- Old runs only have aggregate tokens. The cost helper treats the
  whole aggregate as strong-tier when the per-tier columns are 0
  (the only way they can be 0 is on pre-feature runs OR a run that
  truly used no LLM, both of which collapse to "treat as strong"
  cleanly). So old runs render a cost number once pricing is set,
  no DB backfill needed.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. agent_runs columns ──────────────────────────────────────
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "strong_input_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        batch_op.add_column(
            sa.Column(
                "strong_output_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cheap_input_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cheap_output_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        batch_op.add_column(
            sa.Column(
                "strong_model_snapshot",
                sa.String(length=128),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cheap_model_snapshot",
                sa.String(length=128),
                nullable=True,
            ),
        )

    # ── 2. app_settings columns ────────────────────────────────────
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "strong_input_price_per_m",
                sa.Float(),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "strong_output_price_per_m",
                sa.Float(),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cheap_input_price_per_m",
                sa.Float(),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cheap_output_price_per_m",
                sa.Float(),
                nullable=True,
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.drop_column("cheap_output_price_per_m")
        batch_op.drop_column("cheap_input_price_per_m")
        batch_op.drop_column("strong_output_price_per_m")
        batch_op.drop_column("strong_input_price_per_m")

    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_column("cheap_model_snapshot")
        batch_op.drop_column("strong_model_snapshot")
        batch_op.drop_column("cheap_output_tokens")
        batch_op.drop_column("cheap_input_tokens")
        batch_op.drop_column("strong_output_tokens")
        batch_op.drop_column("strong_input_tokens")
