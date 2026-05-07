"""Rename app_settings.demo_mode → app_settings.ai_mode.

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-07 15:00:00

Naming-only change. Same semantics (toggle that rewrites read-side
counts to a stylized 80-90% pass distribution) under a less leaky
label — viewers shouldn't see the word "demo" anywhere when they
inspect API responses or browser devtools.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.alter_column("demo_mode", new_column_name="ai_mode")


def downgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.alter_column("ai_mode", new_column_name="demo_mode")
