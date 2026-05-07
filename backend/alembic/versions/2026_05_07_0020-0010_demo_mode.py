"""Add demo_mode boolean to app_settings.

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-07 14:00:00

When demo_mode is True, every read-side endpoint transforms run
counts and per-row statuses to a deterministic 80-90% pass rate.
Real data is not modified. Off by default.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "demo_mode",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.drop_column("demo_mode")
