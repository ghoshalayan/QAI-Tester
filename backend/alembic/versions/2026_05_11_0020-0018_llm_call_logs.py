"""Per-call LLM telemetry — ``llm_call_logs`` table.

Revision ID: 0018
Revises: 0017
Create Date: 2026-05-11 11:00:00

Production-grade cost telemetry: every LLM round-trip the agent
makes during a run gets a row here. Lets the user drill into
"why did this run cost $1.50?" → see each individual call, its
role, model, tokens, and timing.

Row schema
----------
- ``run_id``      : FK → agent_runs.id (ON DELETE CASCADE — call
                    logs vanish when the run is deleted)
- ``step_id``     : FK → execution_steps.id (nullable; calls made
                    outside a step, e.g. scout's homepage summary,
                    have NULL here)
- ``ordinal``     : monotonic per-run integer for display order
- ``role``        : LLMRole.value at the call site (planner,
                    smart_picker, semantic_verifier, ...)
- ``tier``        : "strong" | "cheap" — which provider answered
- ``model``       : model name AT CALL TIME (snapshot — changing
                    the LLM config later doesn't rewrite history)
- ``input_tokens``, ``output_tokens``
- ``escalated``   : True when the cheap tier was tried first and
                    fell through to strong (the call's tokens are
                    from the WINNING tier — the cheap-tier tokens
                    that preceded it have their own log row)
- ``duration_ms`` : wall-clock for the round-trip
- ``created_at``

Costs are computed at READ time (against current ``app_settings``
pricing) so changing $/M rates re-costs history correctly.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_call_logs",
        sa.Column(
            "id", sa.Integer, primary_key=True, autoincrement=True,
        ),
        sa.Column(
            "run_id", sa.Integer,
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column(
            "step_id", sa.Integer,
            sa.ForeignKey("execution_steps.id", ondelete="SET NULL"),
            nullable=True, index=True,
        ),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("tier", sa.String(length=16), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column(
            "input_tokens", sa.Integer,
            nullable=False, server_default="0",
        ),
        sa.Column(
            "output_tokens", sa.Integer,
            nullable=False, server_default="0",
        ),
        sa.Column(
            "escalated", sa.Boolean,
            nullable=False, server_default=sa.false(),
        ),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_llm_call_logs_run_ordinal",
        "llm_call_logs",
        ["run_id", "ordinal"],
    )


def downgrade() -> None:
    op.drop_index("ix_llm_call_logs_run_ordinal", "llm_call_logs")
    op.drop_table("llm_call_logs")
