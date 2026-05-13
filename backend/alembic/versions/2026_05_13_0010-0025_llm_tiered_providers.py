"""Per-tier LLM provider + OpenRouter support.

Revision ID: 0025
Revises: 0024
Create Date: 2026-05-13 10:00:00

Adds:
- ``openrouter`` as a valid provider value (CHECK constraint widened).
- Per-tier provider/model/api_key/base_url for the cheap, fallback-strong,
  and fallback-cheap tiers so the user can mix providers — e.g. OpenAI
  GPT-5 as strong, OpenRouter DeepSeek as cheap, Gemini 2.5 as fallback.

Schema decisions
----------------
- Each tier's columns are NULLABLE. NULL fields fall back to the primary
  tier's credentials when the provider matches (keeps existing single-
  provider setups working without re-entering credentials per tier).
- The primary tier's columns (``provider``, ``model``, ``api_key``,
  ``base_url``) are NOT renamed — they remain authoritative for the
  strong tier.
- ``cheap_model`` predates this migration; ``cheap_provider`` etc. are
  added alongside it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0025"
down_revision: Union[str, None] = "0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_NEW_TIERS: tuple[str, ...] = (
    "cheap",
    "fallback_strong",
    "fallback_cheap",
)


def upgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        # Per-tier (provider, model, api_key, base_url) for each new
        # tier. `cheap_model` already exists; we add `cheap_provider`
        # / `cheap_api_key` / `cheap_base_url` next to it.
        for tier in _NEW_TIERS:
            batch_op.add_column(
                sa.Column(
                    f"{tier}_provider",
                    sa.String(32),
                    nullable=True,
                ),
            )
            # `cheap_model` already exists from a prior migration; only
            # add for the two new fallback tiers.
            if tier != "cheap":
                batch_op.add_column(
                    sa.Column(
                        f"{tier}_model",
                        sa.String(128),
                        nullable=True,
                    ),
                )
            batch_op.add_column(
                sa.Column(
                    f"{tier}_api_key",
                    sa.String(512),
                    nullable=True,
                ),
            )
            batch_op.add_column(
                sa.Column(
                    f"{tier}_base_url",
                    sa.String(512),
                    nullable=True,
                ),
            )
        # Pricing for the 2 new tiers (cheap already has pricing).
        for tier in ("fallback_strong", "fallback_cheap"):
            for io in ("input", "output"):
                batch_op.add_column(
                    sa.Column(
                        f"{tier}_{io}_price_per_m",
                        sa.Float(),
                        nullable=True,
                    ),
                )
            batch_op.add_column(
                sa.Column(
                    f"{tier}_cached_input_price_per_m",
                    sa.Float(),
                    nullable=True,
                ),
            )

    # SQLite CHECK constraint can't be added in-place; we have to
    # recreate the table. Use batch_alter_table's auto-rebuild path
    # by dropping + re-creating the constraint via DDL.
    with op.batch_alter_table(
        "app_settings",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "app_settings_provider_valid",
            type_="check",
        )
        batch_op.create_check_constraint(
            "app_settings_provider_valid",
            "provider IN ('gemini', 'openai', 'openai_compat', 'openrouter')",
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "app_settings",
        recreate="always",
    ) as batch_op:
        batch_op.drop_constraint(
            "app_settings_provider_valid",
            type_="check",
        )
        batch_op.create_check_constraint(
            "app_settings_provider_valid",
            "provider IN ('gemini', 'openai', 'openai_compat')",
        )
    with op.batch_alter_table("app_settings") as batch_op:
        for tier in ("fallback_cheap", "fallback_strong"):
            batch_op.drop_column(f"{tier}_cached_input_price_per_m")
            batch_op.drop_column(f"{tier}_output_price_per_m")
            batch_op.drop_column(f"{tier}_input_price_per_m")
        for tier in reversed(_NEW_TIERS):
            batch_op.drop_column(f"{tier}_base_url")
            batch_op.drop_column(f"{tier}_api_key")
            if tier != "cheap":
                batch_op.drop_column(f"{tier}_model")
            batch_op.drop_column(f"{tier}_provider")
