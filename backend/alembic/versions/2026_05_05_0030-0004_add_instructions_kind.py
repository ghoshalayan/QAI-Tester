"""extend documents.kind CHECK to allow 'INSTRUCTIONS'

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-05 00:30:00

The 'INSTRUCTIONS' kind is for users who want to skip the BRD→FRD analysis
and feed test-case generation directly with plain instructions like
"Test the login form, signup, and password reset". The chunker, embedder,
and FAISS path are unchanged — only the agent in week 4 will branch on
``kind`` to decide whether to consume requirements or treat content as
direct test instructions.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite can't ALTER a CHECK constraint in place; batch mode recreates
    # the table with the new constraint.
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_constraint("document_kind_valid", type_="check")
        batch_op.create_check_constraint(
            "document_kind_valid",
            "kind IN ('BRD', 'FRD', 'INSTRUCTIONS')",
        )


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_constraint("document_kind_valid", type_="check")
        batch_op.create_check_constraint(
            "document_kind_valid",
            "kind IN ('BRD', 'FRD')",
        )
