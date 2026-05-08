"""Add TOTP secret + encrypted-at-rest markers for credential vault.

Revision ID: 0014
Revises: 0013
Create Date: 2026-05-09 11:00:00

Phase 3 — credential vault hardening. Two changes:

1. ``totp_secret`` (NULL when not set). Stores a TOTP seed (RFC 6238)
   so the agent can generate 2FA codes itself when the auth flow
   reaches an OTP screen, eliminating HITL for ~90% of 2FA cases.
   Same security level as the password — both are user secrets.
   Stored encrypted (see #2). Empty / NULL means "no TOTP for this
   credential; OTP screens fall back to HITL".

2. ``encrypted`` boolean flag. Marks rows whose ``username`` /
   ``password`` / ``totp_secret`` are Fernet-encrypted at rest.
   When False, the row is plaintext (legacy MVP rows from before
   this migration). The vault read path detects the flag and
   decrypts on-the-fly; new writes always encrypt.

Encryption key resolution:
    1. ``QAI_VAULT_KEY`` env var → use that.
    2. ``data/.vault_key`` file (mode 0600) → use that.
    3. Generate a fresh key on first vault read, write to file.

This migration does NOT migrate existing plaintext rows — they keep
working via the ``encrypted=False`` path. To re-encrypt, the user
re-saves the credential through the API.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("test_plan_credentials") as batch_op:
        batch_op.add_column(
            sa.Column(
                "totp_secret",
                sa.String(length=512),
                nullable=True,
            ),
        )
        batch_op.add_column(
            sa.Column(
                "encrypted",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("test_plan_credentials") as batch_op:
        batch_op.drop_column("encrypted")
        batch_op.drop_column("totp_secret")
