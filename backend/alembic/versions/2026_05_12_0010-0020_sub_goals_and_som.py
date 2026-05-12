"""Sub-goal replan budget + SoM annotation toggle.

Revision ID: 0020
Revises: 0019
Create Date: 2026-05-12 00:10:00

Phase A — vision-driven sub-goal decomposition. Two scalar columns:

- ``test_plans.max_replans_per_submodule`` — cap on how many times
  the agent can re-decompose after a sub-goal fails. Default 2;
  range enforced at the API layer (0-5). 0 disables replanning
  entirely.
- ``app_settings.som_enabled_default`` — global toggle for
  Set-of-Mark annotation on VL screenshots. Default True. When
  False, every VL call gets the raw screenshot (legacy behavior).

Both are non-nullable with server defaults so existing rows pick
up sensible values without a backfill query.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0020"
down_revision: Union[str, None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("test_plans") as batch_op:
        batch_op.add_column(
            sa.Column(
                "max_replans_per_submodule",
                sa.Integer(),
                nullable=False,
                server_default="2",
            ),
        )

    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.add_column(
            sa.Column(
                "som_enabled_default",
                sa.Boolean(),
                nullable=False,
                server_default="1",
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.drop_column("som_enabled_default")
    with op.batch_alter_table("test_plans") as batch_op:
        batch_op.drop_column("max_replans_per_submodule")
