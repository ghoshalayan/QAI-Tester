"""Cost computation: per-run breakdown + aggregated views.

Token columns ($per-tier-per-direction) live on ``agent_runs``;
``$/M tokens`` rates live on ``app_settings``. This service joins
the two at read time + handles two backward-compat cases:

1. **Pre-feature runs.** All four per-tier counters = 0 because the
   migration default is 0 and the agent loop didn't write them.
   Locked policy is to treat aggregate ``input_tokens`` /
   ``output_tokens`` (from ``output_summary_json``) as strong-tier
   so the user sees a cost number on old runs once pricing is set.

2. **Missing pricing.** When a rate column is NULL, the matching
   line item shows ``$—`` (rendered as ``cost: None`` here; the
   frontend converts). Total skips the unpriced lines.

Embeddings (AKB / BRD ingest / requirement embed) use a local CPU
model and cost $0 — they don't show up here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.models.agent_run import AgentRun
    from app.models.app_settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass
class CostLine:
    """One line item in the breakdown — tier × direction.

    ``direction`` is one of:
    - ``input``        — regular (uncached) prompt tokens
    - ``input_cached`` — cached prompt tokens (subset of total
                         input; billed at the cached rate)
    - ``output``       — completion tokens

    With the cached split the breakdown is 6 lines (3 directions ×
    2 tiers) instead of 4. UIs that don't care about the cached
    nuance can sum direction in {input, input_cached} for "input
    total".
    """

    tier: str  # "strong" | "cheap"
    direction: str  # "input" | "input_cached" | "output"
    tokens: int
    price_per_m: float | None
    # ``None`` when price_per_m is unset; UI renders ``$—``.
    cost_usd: float | None


@dataclass
class RunCost:
    """Complete per-run breakdown."""

    run_id: int
    kind: str
    strong_model: str | None
    cheap_model: str | None
    lines: list[CostLine]
    # Sum of priced lines only. ``None`` when EVERY line is unpriced
    # (settings has no rates) so the UI can show "configure pricing"
    # instead of a misleading $0.00.
    total_cost_usd: float | None
    # True when the breakdown was derived from aggregate tokens
    # (pre-feature run) — UI shows a small "estimate" badge.
    estimated_from_aggregate: bool = False


def _load_settings(db: "Session") -> "AppSettings | None":
    from app.models.app_settings import AppSettings  # noqa: PLC0415

    return db.query(AppSettings).filter(AppSettings.id == 1).first()


def _maybe_cost(tokens: int, rate: float | None) -> float | None:
    if rate is None:
        return None
    return round(tokens * rate / 1_000_000, 6)


def compute_run_cost(
    db: "Session", run: "AgentRun",
) -> RunCost:
    """Build the per-tier × per-direction breakdown for one run.

    Six lines: (input, input_cached, output) × (strong, cheap).
    Regular input = total input − cached input; each portion is
    billed at its own ``$/M`` rate. When ``cached_input_price_per_m``
    is unset, the regular input rate is used for the cached portion
    too (safe over-bill default).
    """
    settings = _load_settings(db)

    # Backfill: if all four per-tier columns are 0 AND the aggregate
    # tokens are non-zero, treat the aggregate as strong-tier. This
    # handles every run that predated the cost-tracking feature
    # without a DB migration.
    s_in = int(getattr(run, "strong_input_tokens", 0) or 0)
    s_out = int(getattr(run, "strong_output_tokens", 0) or 0)
    c_in = int(getattr(run, "cheap_input_tokens", 0) or 0)
    c_out = int(getattr(run, "cheap_output_tokens", 0) or 0)
    s_cached = int(getattr(run, "strong_cached_input_tokens", 0) or 0)
    c_cached = int(getattr(run, "cheap_cached_input_tokens", 0) or 0)
    aggregate_only = (s_in + s_out + c_in + c_out) == 0

    estimated = False
    if aggregate_only:
        summary = run.output_summary_json or {}
        agg_in = int(summary.get("input_tokens") or 0)
        agg_out = int(summary.get("output_tokens") or 0)
        if agg_in or agg_out:
            s_in = agg_in
            s_out = agg_out
            estimated = True
            # Pre-feature runs have no cached info — assume 0.

    s_in_rate = getattr(settings, "strong_input_price_per_m", None) if settings else None
    s_out_rate = getattr(settings, "strong_output_price_per_m", None) if settings else None
    c_in_rate = getattr(settings, "cheap_input_price_per_m", None) if settings else None
    c_out_rate = getattr(settings, "cheap_output_price_per_m", None) if settings else None
    s_cached_rate = (
        getattr(settings, "strong_cached_input_price_per_m", None)
        if settings else None
    )
    c_cached_rate = (
        getattr(settings, "cheap_cached_input_price_per_m", None)
        if settings else None
    )
    # Fallback: when cached rate isn't set, use the regular input
    # rate. This over-bills the cached portion by the cache discount
    # but never under-bills — safe default. Users who care set the
    # cached rate explicitly in Cost Settings (typically ~50% of
    # the regular rate for OpenAI).
    if s_cached_rate is None:
        s_cached_rate = s_in_rate
    if c_cached_rate is None:
        c_cached_rate = c_in_rate

    # Derive regular = total - cached. Clamped at 0 in case of
    # provider misreporting (already clamped in CostCounters.add,
    # but defence in depth).
    s_regular = max(0, s_in - s_cached)
    c_regular = max(0, c_in - c_cached)

    lines = [
        CostLine(
            tier="strong", direction="input",
            tokens=s_regular, price_per_m=s_in_rate,
            cost_usd=_maybe_cost(s_regular, s_in_rate),
        ),
        CostLine(
            tier="strong", direction="input_cached",
            tokens=s_cached, price_per_m=s_cached_rate,
            cost_usd=_maybe_cost(s_cached, s_cached_rate),
        ),
        CostLine(
            tier="strong", direction="output",
            tokens=s_out, price_per_m=s_out_rate,
            cost_usd=_maybe_cost(s_out, s_out_rate),
        ),
        CostLine(
            tier="cheap", direction="input",
            tokens=c_regular, price_per_m=c_in_rate,
            cost_usd=_maybe_cost(c_regular, c_in_rate),
        ),
        CostLine(
            tier="cheap", direction="input_cached",
            tokens=c_cached, price_per_m=c_cached_rate,
            cost_usd=_maybe_cost(c_cached, c_cached_rate),
        ),
        CostLine(
            tier="cheap", direction="output",
            tokens=c_out, price_per_m=c_out_rate,
            cost_usd=_maybe_cost(c_out, c_out_rate),
        ),
    ]
    priced = [ln.cost_usd for ln in lines if ln.cost_usd is not None]
    total = round(sum(priced), 6) if priced else None

    return RunCost(
        run_id=run.id,
        kind=run.kind or "execute",
        strong_model=getattr(run, "strong_model_snapshot", None),
        cheap_model=getattr(run, "cheap_model_snapshot", None),
        lines=lines,
        total_cost_usd=total,
        estimated_from_aggregate=estimated,
    )


@dataclass
class AggregateCost:
    """Roll-up across many runs for the Cost Logs dashboard.

    Token totals are split: the ``..._input_tokens`` fields are
    REGULAR (uncached) totals; ``..._cached_input_tokens`` are the
    cached portions. So ``input_grand_total = input + cached_input``
    if the UI wants to display a non-split number.
    """

    run_count: int
    total_strong_input_tokens: int
    total_strong_output_tokens: int
    total_cheap_input_tokens: int
    total_cheap_output_tokens: int
    # Cached portions (subset of the input totals above were they
    # also reported as cached; here we surface them as separate
    # sums so the UI can show "$X saved by cache" later).
    total_strong_cached_input_tokens: int
    total_cheap_cached_input_tokens: int
    total_cost_usd: float | None
    # Per-kind breakdown — typically {"execute": $X, "recon": $Y,
    # "frd_to_tc": $Z}.
    by_kind: dict[str, float]


def compute_aggregate_cost(
    db: "Session",
    *,
    project_id: int | None = None,
    plan_id: int | None = None,
    limit: int | None = None,
) -> AggregateCost:
    """Sum + per-kind breakdown of run costs for the Cost Logs view.

    ``project_id`` / ``plan_id`` filters are inclusive — pass None to
    skip a filter. ``limit`` caps the rows we walk (most-recent
    first); useful when the user only cares about the last N runs.
    """
    from app.models.agent_run import AgentRun  # noqa: PLC0415

    settings = _load_settings(db)
    stmt = select(AgentRun).order_by(AgentRun.created_at.desc())
    if project_id is not None:
        stmt = stmt.where(AgentRun.project_id == project_id)
    if plan_id is not None:
        stmt = stmt.where(AgentRun.plan_id == plan_id)
    if limit is not None:
        stmt = stmt.limit(limit)
    runs = list(db.execute(stmt).scalars())

    s_in_rate = getattr(settings, "strong_input_price_per_m", None) if settings else None
    s_out_rate = getattr(settings, "strong_output_price_per_m", None) if settings else None
    c_in_rate = getattr(settings, "cheap_input_price_per_m", None) if settings else None
    c_out_rate = getattr(settings, "cheap_output_price_per_m", None) if settings else None

    total_s_in = total_s_out = 0
    total_c_in = total_c_out = 0
    total_s_cached = total_c_cached = 0
    total_cost = 0.0
    any_priced = False
    by_kind: dict[str, float] = {}

    for r in runs:
        rc = compute_run_cost(db, r)
        for ln in rc.lines:
            if ln.tier == "strong" and ln.direction == "input":
                total_s_in += ln.tokens
            elif ln.tier == "strong" and ln.direction == "input_cached":
                total_s_cached += ln.tokens
            elif ln.tier == "strong" and ln.direction == "output":
                total_s_out += ln.tokens
            elif ln.tier == "cheap" and ln.direction == "input":
                total_c_in += ln.tokens
            elif ln.tier == "cheap" and ln.direction == "input_cached":
                total_c_cached += ln.tokens
            elif ln.tier == "cheap" and ln.direction == "output":
                total_c_out += ln.tokens
        if rc.total_cost_usd is not None:
            any_priced = True
            total_cost += rc.total_cost_usd
            by_kind[rc.kind] = by_kind.get(rc.kind, 0.0) + rc.total_cost_usd

    # Silence the unused-import warning on settings rate references —
    # they're read in compute_run_cost which we're aggregating across.
    del s_in_rate, s_out_rate, c_in_rate, c_out_rate

    return AggregateCost(
        run_count=len(runs),
        total_strong_input_tokens=total_s_in,
        total_strong_output_tokens=total_s_out,
        total_cheap_input_tokens=total_c_in,
        total_cheap_output_tokens=total_c_out,
        total_strong_cached_input_tokens=total_s_cached,
        total_cheap_cached_input_tokens=total_c_cached,
        total_cost_usd=round(total_cost, 6) if any_priced else None,
        by_kind={k: round(v, 6) for k, v in by_kind.items()},
    )
