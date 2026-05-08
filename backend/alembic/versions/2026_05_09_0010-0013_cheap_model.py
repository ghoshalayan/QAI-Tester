"""Add app_settings.cheap_model for Phase 1 provider tiering.

Revision ID: 0013
Revises: 0012
Create Date: 2026-05-09 10:00:00

Phase 1 — Provider tiering. The agent makes many vision calls per
run (screen classify, popup classify, smart-pick, on-track check,
goal verify, semantic verify) of varying difficulty. Running every
call against the strongest model wastes tokens — most are simple
yes/no or scoring tasks the cheap tier handles fine.

This migration adds a single nullable ``cheap_model`` column to
``app_settings``. When set, the LLM router tries it first for tier-
appropriate roles (screen_classifier, fast_classify, vision_search,
goal_verifier, on_track_check, smart_picker, semantic_verifier).
Roles that need precision (planner, action, coord_proposer) always
go to the existing ``model`` (treated as the STRONG tier).

When ``cheap_model`` is NULL or empty, the router falls back to the
single-model legacy behavior — every call uses ``model``. So the
migration is fully backwards-compatible: nothing changes for users
who don't configure a cheap tier.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "cheap_model",
                sa.String(length=128),
                nullable=True,
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.drop_column("cheap_model")
