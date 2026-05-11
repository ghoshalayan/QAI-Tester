"""Production-α/β/γ — App Knowledge Base, WorldState, Goal pre/post.

Revision ID: 0016
Revises: 0015
Create Date: 2026-05-09 13:00:00

Three coordinated additions for the production-grade agent:

1. ``app_knowledge`` table — per-``target_url_pattern`` knowledge
   chunks the agent queries at submodule start (RAG-style). Sources:
   BRD/FRD chunks (α), reconnaissance walker output (β), pattern
   packs (β), resolved-dispute rules (γ), frozen-path summaries (γ).

2. ``agent_runs.world_state_json`` — plan-scoped state carried across
   submodules within one run (cart_count, logged_in_as, last_url,
   etc.). Read at submodule start to assert preconditions; updated
   on verify success.

3. ``tc_nodes.preconditions`` / ``postconditions`` / ``evidence_signals``
   / ``alternative_paths`` — richer goal shape so submodules know
   what state they need, what state proves them done, and how to
   tell when they're done with N-of-M signal voting (vs single-
   token verify).

These columns are nullable + default-empty so existing runs that
predate this migration keep working — the agent reads "empty AKB"
and "no preconditions" as "fall through to legacy behavior."
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. App Knowledge Base table ────────────────────────────────
    op.create_table(
        "app_knowledge",
        sa.Column(
            "id", sa.Integer, primary_key=True, autoincrement=True,
        ),
        # Substring-matched against the run's plan.target_url so a
        # single chunk can serve all *.amazon.com / *.amazon.in /
        # account.salesforce.com test plans. Examples:
        #   "amazon.com"  → matches amazon.com AND amazon.in subdomains
        #   "myapp.acme.io"  → matches that one tenant
        # Indexed for fast lookup at submodule start (every run does
        # at least one query against it).
        sa.Column(
            "target_url_pattern", sa.String(length=512),
            nullable=False, index=True,
        ),
        # Source kind — drives ranking (recon notes < pattern rules
        # for the same query, etc.) and lets the UI filter. One of:
        #   brd_chunk            — BRD/FRD content from project docs
        #   recon_note           — pages observed during scout walk
        #   pattern_rule         — curated pack (sap, salesforce, ...)
        #   dispute_outcome      — rules learned from resolved disputes
        #   frozen_path_summary  — replayable flow distilled to text
        #   manual_note          — human-authored hint
        sa.Column(
            "kind", sa.String(length=32), nullable=False, index=True,
        ),
        # The knowledge content. Free text; embedded for retrieval.
        # Keep under ~2KB per chunk so prompt injection stays bounded.
        sa.Column("content", sa.Text(), nullable=False),
        # Optional structured tags the agent can filter by. Examples:
        #   ["auth", "login"]      — relevant when classifying auth screens
        #   ["cart", "checkout"]   — for cart-related submodules
        #   ["sap-fiori"]          — pack-specific
        sa.Column("tags", sa.JSON(), nullable=True),
        # Confidence 0.0-1.0 — recon notes start at 0.7, pattern
        # packs at 0.9, dispute outcomes after user accept at 0.95.
        # The runtime ranks by confidence × recency × relevance.
        sa.Column(
            "confidence", sa.Float(),
            nullable=False, server_default="0.8",
        ),
        # Provenance. ``source_run_id`` is null for pattern packs and
        # human notes; set for chunks written by recon / disputes /
        # frozen-path summaries so we can audit "where did this rule
        # come from" + invalidate per-run if the source got nuked.
        sa.Column(
            "source_run_id", sa.Integer(),
            sa.ForeignKey("agent_runs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "source_doc_id", sa.Integer(),
            sa.ForeignKey("documents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # Embedding stays out of this table — we keep the FAISS index
        # on disk (``data/akb_faiss/<sha256(target_url_pattern)>.idx``)
        # so the table holds metadata + content only. The service
        # layer maps row id → FAISS vector index.
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            nullable=False, server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_app_knowledge_pattern_kind",
        "app_knowledge",
        ["target_url_pattern", "kind"],
    )

    # ── 2. WorldState column on agent_runs ─────────────────────────
    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "world_state_json", sa.JSON(),
                nullable=True,
            ),
        )

    # ── 3. Goal pre/post + signals on tc_nodes ─────────────────────
    with op.batch_alter_table("tc_nodes") as batch_op:
        # Each is a list[str] of short conditions/criteria. NULL =
        # "not specified at TC-gen time; agent infers from goal
        # description" (legacy behavior preserved).
        batch_op.add_column(
            sa.Column("preconditions", sa.JSON(), nullable=True),
        )
        batch_op.add_column(
            sa.Column("postconditions", sa.JSON(), nullable=True),
        )
        batch_op.add_column(
            sa.Column("evidence_signals", sa.JSON(), nullable=True),
        )
        batch_op.add_column(
            sa.Column("alternative_paths", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    with op.batch_alter_table("tc_nodes") as batch_op:
        batch_op.drop_column("alternative_paths")
        batch_op.drop_column("evidence_signals")
        batch_op.drop_column("postconditions")
        batch_op.drop_column("preconditions")

    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_column("world_state_json")

    op.drop_index("ix_app_knowledge_pattern_kind", "app_knowledge")
    op.drop_table("app_knowledge")
