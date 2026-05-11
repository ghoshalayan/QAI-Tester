"""Cached input token accounting.

Revision ID: 0019
Revises: 0018
Create Date: 2026-05-11 13:00:00

Splits the existing ``input_tokens`` total into ``regular_input`` +
``cached_input`` so the cost surface bills cached portions at the
provider's discounted rate (OpenAI automatically caches prompts
≥ 1024 tokens at ~50%; Gemini's ``cached_content`` API at ~25%).

Without this, the system prompt (~3 KB) re-sent on every planner
turn was billed at 100% even though OpenAI charged ~50% for it.
Resulting over-report on a typical 30-turn run: 30-40%.

Schema notes
------------
- ``cached_input_tokens`` is a SUBSET of the total ``input_tokens``
  that's already stored. Cost calc derives ``regular_input =
  input_tokens - cached_input_tokens`` and applies the matching
  rate to each portion.
- Default = 0 so existing runs read as "no cache hit" and bill
  unchanged (no silent re-cost of historical data).
- ``cached_input_price_per_m`` NULL = "user hasn't configured a
  cached rate" → cost service applies the REGULAR rate to cached
  tokens as a safe default (over-bills by the cache discount,
  but never under-bills). Hint text in the Settings UI nudges
  users to set ~50% of the regular rate for OpenAI.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0019"
down_revision: Union[str, None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── agent_runs: per-tier cached input counters ─────────────────
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "strong_cached_input_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cheap_cached_input_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )

    # ── llm_call_logs: per-call cached counter ─────────────────────
    with op.batch_alter_table("llm_call_logs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "cached_input_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )

    # ── app_settings: per-tier cached rate ─────────────────────────
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "strong_cached_input_price_per_m",
                sa.Float(),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cheap_cached_input_price_per_m",
                sa.Float(),
                nullable=True,
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.drop_column("cheap_cached_input_price_per_m")
        batch_op.drop_column("strong_cached_input_price_per_m")
    with op.batch_alter_table("llm_call_logs") as batch_op:
        batch_op.drop_column("cached_input_tokens")
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_column("cheap_cached_input_tokens")
        batch_op.drop_column("strong_cached_input_tokens")
