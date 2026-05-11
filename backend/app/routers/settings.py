"""LLM provider configuration endpoints.

GET   /api/settings        → public-safe read (no api_key)
PUT   /api/settings        → upsert with partial-update semantics
DELETE /api/settings       → wipe back to first-run state
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.llm.factory import build_provider, invalidate_cache
from app.models.app_settings import AppSettings
from app.schemas.app_settings import (
    AppSettingsRead,
    AppSettingsWrite,
    TestConnectionResponse,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


def _to_read(row: AppSettings | None) -> AppSettingsRead:
    if row is None:
        return AppSettingsRead()
    return AppSettingsRead(
        is_configured=bool(row.provider and row.model and row.api_key),
        provider=row.provider,  # type: ignore[arg-type]
        model=row.model,
        cheap_model=getattr(row, "cheap_model", None) or None,
        base_url=row.base_url,
        api_key_set=bool(row.api_key),
        ai_mode=bool(getattr(row, "ai_mode", False)),
        strong_input_price_per_m=getattr(
            row, "strong_input_price_per_m", None,
        ),
        strong_output_price_per_m=getattr(
            row, "strong_output_price_per_m", None,
        ),
        cheap_input_price_per_m=getattr(
            row, "cheap_input_price_per_m", None,
        ),
        cheap_output_price_per_m=getattr(
            row, "cheap_output_price_per_m", None,
        ),
        strong_cached_input_price_per_m=getattr(
            row, "strong_cached_input_price_per_m", None,
        ),
        cheap_cached_input_price_per_m=getattr(
            row, "cheap_cached_input_price_per_m", None,
        ),
        updated_at=row.updated_at,
    )


def _validate_for_create(payload: AppSettingsWrite) -> None:
    """First-time setup: provider, model, api_key all required."""
    if not payload.provider:
        raise HTTPException(400, "provider is required for initial setup")
    if not payload.model:
        raise HTTPException(400, "model is required for initial setup")
    if not payload.api_key:
        raise HTTPException(400, "api_key is required for initial setup")
    if payload.provider == "openai_compat" and not payload.base_url:
        raise HTTPException(
            400, "base_url is required when provider is 'openai_compat'",
        )


@router.get("", response_model=AppSettingsRead)
def get_settings(db: Session = Depends(get_db)):
    row = db.query(AppSettings).filter(AppSettings.id == 1).first()
    return _to_read(row)


@router.put("", response_model=AppSettingsRead)
def upsert_settings(payload: AppSettingsWrite, db: Session = Depends(get_db)):
    row = db.query(AppSettings).filter(AppSettings.id == 1).first()

    # AI-Mode-only flip: a payload that sets ONLY ``ai_mode`` shouldn't
    # be required to re-supply provider/model/api_key. This lets the
    # Settings UI flip the toggle without re-entering credentials.
    only_ai_mode_toggle = (
        payload.ai_mode is not None
        and payload.provider is None
        and payload.model is None
        and payload.cheap_model is None
        and payload.api_key is None
        and payload.base_url is None
        and payload.strong_input_price_per_m is None
        and payload.strong_output_price_per_m is None
        and payload.cheap_input_price_per_m is None
        and payload.cheap_output_price_per_m is None
        and payload.strong_cached_input_price_per_m is None
        and payload.cheap_cached_input_price_per_m is None
    )
    # Cost-only update: setting pricing without touching provider /
    # model / api_key — the Settings UI's "Cost Settings" tab uses
    # this so users can edit rates without re-confirming credentials.
    only_pricing_update = (
        (
            payload.strong_input_price_per_m is not None
            or payload.strong_output_price_per_m is not None
            or payload.cheap_input_price_per_m is not None
            or payload.cheap_output_price_per_m is not None
            or payload.strong_cached_input_price_per_m is not None
            or payload.cheap_cached_input_price_per_m is not None
        )
        and payload.provider is None
        and payload.model is None
        and payload.cheap_model is None
        and payload.api_key is None
        and payload.base_url is None
        and payload.ai_mode is None
    )

    if row is None:
        if only_ai_mode_toggle:
            raise HTTPException(
                400,
                "Configure an LLM provider before enabling AI Mode.",
            )
        _validate_for_create(payload)
        # Phase 1 — cheap_model: must NOT equal model (would be a no-op
        # tier and adds DB ambiguity). Empty string = no tiering.
        cheap = (payload.cheap_model or "").strip() or None
        if cheap and cheap == payload.model:
            raise HTTPException(
                400,
                "cheap_model must differ from model — they form the "
                "(primary, escalation) pair for tiering.",
            )
        row = AppSettings(
            id=1,
            provider=payload.provider,
            model=payload.model,
            cheap_model=cheap,
            api_key=payload.api_key,
            base_url=payload.base_url if payload.provider == "openai_compat" else None,
            ai_mode=bool(payload.ai_mode) if payload.ai_mode is not None else False,
        )
        db.add(row)
    elif only_ai_mode_toggle:
        row.ai_mode = bool(payload.ai_mode)
    elif only_pricing_update:
        # Cost-only flip — accept the rate fields without requiring
        # api_key / model re-entry. 0.0 explicitly clears a
        # previously-set rate (caller nulls it out by sending 0 —
        # semantically equivalent to "not configured").
        if payload.strong_input_price_per_m is not None:
            row.strong_input_price_per_m = (
                payload.strong_input_price_per_m or None
            )
        if payload.strong_output_price_per_m is not None:
            row.strong_output_price_per_m = (
                payload.strong_output_price_per_m or None
            )
        if payload.cheap_input_price_per_m is not None:
            row.cheap_input_price_per_m = (
                payload.cheap_input_price_per_m or None
            )
        if payload.cheap_output_price_per_m is not None:
            row.cheap_output_price_per_m = (
                payload.cheap_output_price_per_m or None
            )
        if payload.strong_cached_input_price_per_m is not None:
            row.strong_cached_input_price_per_m = (
                payload.strong_cached_input_price_per_m or None
            )
        if payload.cheap_cached_input_price_per_m is not None:
            row.cheap_cached_input_price_per_m = (
                payload.cheap_cached_input_price_per_m or None
            )
    else:
        new_provider = payload.provider or row.provider
        provider_changed = (
            payload.provider is not None and payload.provider != row.provider
        )

        # Switching providers requires a fresh API key — old key won't work elsewhere
        if provider_changed and not payload.api_key:
            raise HTTPException(
                400, "api_key is required when changing provider",
            )

        if new_provider == "openai_compat":
            new_base_url = (
                payload.base_url if payload.base_url is not None else row.base_url
            )
            if not new_base_url:
                raise HTTPException(
                    400, "base_url is required when provider is 'openai_compat'",
                )
        else:
            new_base_url = None  # always cleared for non-compat providers

        row.provider = new_provider
        if payload.model:
            row.model = payload.model
        # Phase 1 — cheap_model partial update. ``None`` means "leave
        # alone"; empty string means "clear tiering". Otherwise the
        # supplied value is set, with the not-equal-to-model guard.
        if payload.cheap_model is not None:
            cheap = payload.cheap_model.strip() or None
            effective_model = payload.model or row.model
            if cheap and cheap == effective_model:
                raise HTTPException(
                    400,
                    "cheap_model must differ from model — they form "
                    "the (primary, escalation) pair for tiering.",
                )
            row.cheap_model = cheap
        if payload.api_key:
            row.api_key = payload.api_key
        row.base_url = new_base_url
        if payload.ai_mode is not None:
            row.ai_mode = bool(payload.ai_mode)
        # Cost rates — same "None = preserve, 0 = clear" semantics as
        # the pricing-only branch above.
        if payload.strong_input_price_per_m is not None:
            row.strong_input_price_per_m = (
                payload.strong_input_price_per_m or None
            )
        if payload.strong_output_price_per_m is not None:
            row.strong_output_price_per_m = (
                payload.strong_output_price_per_m or None
            )
        if payload.cheap_input_price_per_m is not None:
            row.cheap_input_price_per_m = (
                payload.cheap_input_price_per_m or None
            )
        if payload.cheap_output_price_per_m is not None:
            row.cheap_output_price_per_m = (
                payload.cheap_output_price_per_m or None
            )
        if payload.strong_cached_input_price_per_m is not None:
            row.strong_cached_input_price_per_m = (
                payload.strong_cached_input_price_per_m or None
            )
        if payload.cheap_cached_input_price_per_m is not None:
            row.cheap_cached_input_price_per_m = (
                payload.cheap_cached_input_price_per_m or None
            )

    db.commit()
    db.refresh(row)
    invalidate_cache()  # provider settings changed — drop cached client
    return _to_read(row)


# ── Cost endpoints (Cost Logs tab) ─────────────────────────────────


@router.get("/cost/runs")
def list_run_costs(
    project_id: int | None = None,
    plan_id: int | None = None,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    """Per-run cost breakdown table for the Cost Logs dashboard.

    Walks the most recent N runs (defaults to 200) and returns the
    per-tier × per-direction breakdown plus run-level metadata
    (kind, model snapshots, status, project + plan ids). The
    frontend renders this as a sortable table; aggregations are
    computed via ``/cost/aggregate``.
    """
    from app.models.agent_run import AgentRun  # noqa: PLC0415
    from app.services.cost import compute_run_cost  # noqa: PLC0415
    from sqlalchemy import select as _select  # noqa: PLC0415

    stmt = _select(AgentRun).order_by(AgentRun.created_at.desc())
    if project_id is not None:
        stmt = stmt.where(AgentRun.project_id == project_id)
    if plan_id is not None:
        stmt = stmt.where(AgentRun.plan_id == plan_id)
    stmt = stmt.limit(max(1, min(limit, 1000)))
    runs = list(db.execute(stmt).scalars())

    rows = []
    for r in runs:
        rc = compute_run_cost(db, r)
        rows.append({
            "run_id": rc.run_id,
            "kind": rc.kind,
            "status": r.status,
            "project_id": r.project_id,
            "plan_id": r.plan_id,
            "strong_model": rc.strong_model,
            "cheap_model": rc.cheap_model,
            "estimated_from_aggregate": rc.estimated_from_aggregate,
            "lines": [
                {
                    "tier": ln.tier,
                    "direction": ln.direction,
                    "tokens": ln.tokens,
                    "price_per_m": ln.price_per_m,
                    "cost_usd": ln.cost_usd,
                }
                for ln in rc.lines
            ],
            "total_cost_usd": rc.total_cost_usd,
            "created_at": (
                r.created_at.isoformat() if r.created_at else None
            ),
        })
    return {"runs": rows}


@router.get("/cost/aggregate")
def aggregate_costs(
    project_id: int | None = None,
    plan_id: int | None = None,
    limit: int | None = None,
    db: Session = Depends(get_db),
):
    """Roll-up across runs (matching the same filters as
    ``/cost/runs``). Returns totals per tier × direction + the
    grand total + a per-``kind`` breakdown (execute / recon /
    frd_to_tc / etc.).
    """
    from app.services.cost import compute_aggregate_cost  # noqa: PLC0415

    agg = compute_aggregate_cost(
        db,
        project_id=project_id,
        plan_id=plan_id,
        limit=limit,
    )
    return {
        "run_count": agg.run_count,
        "total_strong_input_tokens": agg.total_strong_input_tokens,
        "total_strong_output_tokens": agg.total_strong_output_tokens,
        "total_cheap_input_tokens": agg.total_cheap_input_tokens,
        "total_cheap_output_tokens": agg.total_cheap_output_tokens,
        "total_strong_cached_input_tokens": (
            agg.total_strong_cached_input_tokens
        ),
        "total_cheap_cached_input_tokens": (
            agg.total_cheap_cached_input_tokens
        ),
        "total_cost_usd": agg.total_cost_usd,
        "by_kind": agg.by_kind,
    }


@router.get("/cost/runs/{run_id}/calls")
def list_run_call_logs(
    run_id: int,
    db: Session = Depends(get_db),
):
    """Per-LLM-call telemetry for one run.

    Returns every ``llm_call_logs`` row for the run, ordered by
    insertion (which mirrors call time), plus the computed cost
    for each. The drill-in view in Cost Logs renders this as a
    table with per-call cost + a sum at the bottom.

    Cost is computed at READ time against current ``app_settings``
    pricing so rate changes re-cost history correctly.
    """
    from app.models.agent_run import AgentRun  # noqa: PLC0415
    from app.models.app_settings import AppSettings  # noqa: PLC0415
    from app.models.execution_step import (  # noqa: PLC0415
        ExecutionStep,
    )
    from app.models.llm_call_log import LlmCallLog  # noqa: PLC0415

    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")

    settings = (
        db.query(AppSettings).filter(AppSettings.id == 1).first()
    )
    s_in_rate = (
        getattr(settings, "strong_input_price_per_m", None)
        if settings else None
    )
    s_out_rate = (
        getattr(settings, "strong_output_price_per_m", None)
        if settings else None
    )
    c_in_rate = (
        getattr(settings, "cheap_input_price_per_m", None)
        if settings else None
    )
    c_out_rate = (
        getattr(settings, "cheap_output_price_per_m", None)
        if settings else None
    )
    s_cached_rate = (
        getattr(settings, "strong_cached_input_price_per_m", None)
        if settings else None
    )
    c_cached_rate = (
        getattr(settings, "cheap_cached_input_price_per_m", None)
        if settings else None
    )

    def _cost(tokens: int, rate: float | None) -> float | None:
        if rate is None:
            return None
        return round(tokens * rate / 1_000_000, 6)

    calls = (
        db.query(LlmCallLog)
        .filter(LlmCallLog.run_id == run_id)
        .order_by(LlmCallLog.ordinal.asc())
        .all()
    )

    # Resolve step titles in one batch so the UI can show
    # "submodule N" instead of raw step IDs.
    step_ids = {c.step_id for c in calls if c.step_id is not None}
    step_titles: dict[int, str] = {}
    if step_ids:
        rows = (
            db.query(ExecutionStep)
            .filter(ExecutionStep.id.in_(step_ids))
            .all()
        )
        for r in rows:
            step_titles[r.id] = r.title_snapshot or f"step {r.id}"

    out_rows = []
    total_input_cost = 0.0
    total_cached_cost = 0.0
    total_output_cost = 0.0
    any_priced = False
    for c in calls:
        in_rate = s_in_rate if c.tier == "strong" else c_in_rate
        out_rate = s_out_rate if c.tier == "strong" else c_out_rate
        cached_rate = s_cached_rate if c.tier == "strong" else c_cached_rate
        # Fallback: cached rate unset → use regular input rate
        # (safe over-bill; matches the cost service's policy).
        effective_cached_rate = (
            cached_rate if cached_rate is not None else in_rate
        )
        cached_tok = int(getattr(c, "cached_input_tokens", 0) or 0)
        # Defence: clamp cached ≤ total input so regular is never
        # negative even if a misreported call slipped through.
        cached_tok = min(cached_tok, int(c.input_tokens or 0))
        regular_in_tok = max(0, int(c.input_tokens or 0) - cached_tok)

        regular_in_cost = _cost(regular_in_tok, in_rate)
        cached_in_cost = _cost(cached_tok, effective_cached_rate)
        out_cost = _cost(c.output_tokens, out_rate)
        total_cost = (
            (regular_in_cost or 0)
            + (cached_in_cost or 0)
            + (out_cost or 0)
            if (
                regular_in_cost is not None
                or cached_in_cost is not None
                or out_cost is not None
            )
            else None
        )
        if regular_in_cost is not None:
            total_input_cost += regular_in_cost
            any_priced = True
        if cached_in_cost is not None:
            total_cached_cost += cached_in_cost
            any_priced = True
        if out_cost is not None:
            total_output_cost += out_cost
            any_priced = True
        out_rows.append({
            "id": c.id,
            "ordinal": c.ordinal,
            "step_id": c.step_id,
            "step_title": (
                step_titles.get(c.step_id)
                if c.step_id is not None else None
            ),
            "role": c.role,
            "tier": c.tier,
            "model": c.model,
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            # Cached portion (subset of input_tokens) + the regular
            # portion derived for convenience so the UI doesn't have
            # to do the subtraction itself.
            "cached_input_tokens": cached_tok,
            "regular_input_tokens": regular_in_tok,
            "input_cost_usd": regular_in_cost,
            "cached_input_cost_usd": cached_in_cost,
            "output_cost_usd": out_cost,
            "total_cost_usd": (
                round(total_cost, 6) if total_cost is not None else None
            ),
            "escalated": bool(c.escalated),
            "duration_ms": c.duration_ms,
            "created_at": (
                c.created_at.isoformat() if c.created_at else None
            ),
        })

    return {
        "run_id": run_id,
        "kind": run.kind or "execute",
        "strong_model": run.strong_model_snapshot,
        "cheap_model": run.cheap_model_snapshot,
        "call_count": len(out_rows),
        "calls": out_rows,
        "sum_input_cost_usd": (
            round(total_input_cost, 6) if any_priced else None
        ),
        "sum_cached_input_cost_usd": (
            round(total_cached_cost, 6) if any_priced else None
        ),
        "sum_output_cost_usd": (
            round(total_output_cost, 6) if any_priced else None
        ),
        "sum_total_cost_usd": (
            round(
                total_input_cost + total_cached_cost + total_output_cost,
                6,
            )
            if any_priced else None
        ),
    }


@router.get("/cost/runs/{run_id}")
def get_run_cost(
    run_id: int,
    db: Session = Depends(get_db),
):
    """Single-run cost breakdown for the Cost card on the report
    page. Same shape as one entry from ``/cost/runs`` — separate
    endpoint so the report page can fetch independently of the
    Cost Logs dashboard."""
    from app.models.agent_run import AgentRun  # noqa: PLC0415
    from app.services.cost import compute_run_cost  # noqa: PLC0415

    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    rc = compute_run_cost(db, run)
    return {
        "run_id": rc.run_id,
        "kind": rc.kind,
        "strong_model": rc.strong_model,
        "cheap_model": rc.cheap_model,
        "estimated_from_aggregate": rc.estimated_from_aggregate,
        "lines": [
            {
                "tier": ln.tier,
                "direction": ln.direction,
                "tokens": ln.tokens,
                "price_per_m": ln.price_per_m,
                "cost_usd": ln.cost_usd,
            }
            for ln in rc.lines
        ],
        "total_cost_usd": rc.total_cost_usd,
    }


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def reset_settings(db: Session = Depends(get_db)):
    """Wipe the settings row. App returns to first-run state."""
    db.query(AppSettings).filter(AppSettings.id == 1).delete()
    db.commit()
    invalidate_cache()


@router.post("/test", response_model=TestConnectionResponse)
def test_connection(
    payload: AppSettingsWrite | None = None,
    db: Session = Depends(get_db),
):
    """Round-trip a tiny prompt to verify the LLM is reachable.

    Two modes:

    - **Test before save** — pass ``provider``, ``model``, ``api_key``
      (and ``base_url`` for ``openai_compat``) in the body. Nothing is persisted.
    - **Test current config** — pass an empty body or omit it. Uses whatever
      is already in ``app_settings``.
    """
    use_payload = bool(
        payload and payload.provider and payload.model and payload.api_key
    )

    if use_payload:
        provider_name = payload.provider
        model = payload.model
        api_key = payload.api_key
        base_url = payload.base_url
    else:
        row = db.query(AppSettings).filter(AppSettings.id == 1).first()
        if not row or not row.api_key:
            raise HTTPException(
                400,
                "No saved settings to test. Either save settings first, "
                "or pass provider/model/api_key in the request body.",
            )
        provider_name = row.provider
        model = row.model
        api_key = row.api_key
        base_url = row.base_url

    if provider_name == "openai_compat" and not base_url:
        raise HTTPException(
            400, "base_url is required when provider is 'openai_compat'",
        )

    try:
        provider = build_provider(
            provider=provider_name,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    result = provider.test_connection()
    return TestConnectionResponse(
        ok=result.ok,
        provider=result.provider,
        model=result.model,
        base_url=result.base_url,
        echo=result.echo,
        latency_ms=result.latency_ms,
        error=result.error,
        input_tokens=result.extra.get("input_tokens"),
        output_tokens=result.extra.get("output_tokens"),
    )
