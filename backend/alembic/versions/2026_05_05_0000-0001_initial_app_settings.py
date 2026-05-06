"""initial: app_settings singleton

Revision ID: 0001
Revises:
Create Date: 2026-05-05 00:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("api_key", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("base_url", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="app_settings_singleton"),
        sa.CheckConstraint(
            "provider IN ('gemini', 'openai', 'openai_compat')",
            name="app_settings_provider_valid",
        ),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
