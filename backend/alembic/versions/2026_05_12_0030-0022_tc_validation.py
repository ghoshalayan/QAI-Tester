"""Phase D — live-UI validation fields on TcNodeSnapshot.

Revision ID: 0022
Revises: 0021
Create Date: 2026-05-12 00:30:00

Adds the per-step validation result columns that the dry-run
validator writes after a refinement. The dialog + test-cases viewer
read these to render confidence badges and warn the operator before
running a submodule with unconfirmed steps.

Columns
-------
- ``validation_status`` (default ``pending``):
  ``pending | confirmed | partial | unresolved | unreachable | skipped``
- ``validation_confidence`` (nullable float, 0.0–1.0)
- ``validation_reason`` (nullable text — tooltip body)
- ``validation_at`` (nullable timestamp — when the probe ran)

Existing snapshot rows backfill with ``pending`` / NULLs; the
validator is opt-in (button in the refinement dialog) so historical
versions stay readable.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("tc_node_snapshots") as batch_op:
        batch_op.add_column(
            sa.Column(
                "validation_status",
                sa.String(length=16),
                nullable=False,
                server_default="pending",
            ),
        )
        batch_op.add_column(
            sa.Column(
                "validation_confidence",
                sa.Float(),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "validation_reason",
                sa.Text(),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "validation_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("tc_node_snapshots") as batch_op:
        batch_op.drop_column("validation_at")
        batch_op.drop_column("validation_reason")
        batch_op.drop_column("validation_confidence")
        batch_op.drop_column("validation_status")
