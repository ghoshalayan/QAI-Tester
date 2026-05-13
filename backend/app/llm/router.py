"""LLM call router with provider tiering + escalation.

Phase 1 — every VL call site has a cost profile:
- planner / action / coord-proposer  : NEED the strong model (precision-critical)
- screen-classify / popup-classify   : cheap tier handles fine
- on-track / vision-search / goal-verify / smart-pick / semantic-verify
                                       : cheap tier with escalation on borderline

Centralizing this here means call sites stop sprinkling escalation
logic — they pass a role + a (strong, cheap) pair, and the router
picks the provider plus re-runs on the strong model when the cheap
result is unreliable (low confidence, structural validator fails,
or the cheap call raised entirely).

Backwards compatibility: when ``cheap`` is ``None`` the router routes
every role to the strong provider — old behavior preserved exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from app.llm.base import ChatMessage, ChatResult, LLMProvider

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class LLMRole(str, Enum):
    """Why a call is being made — used to pick the cost tier.

    Adding a new role: pick whether it's STRONG_ONLY (precision
    matters more than cost) or CHEAP_FIRST (cheap with escalation).
    Wire it in ``_STRONG_ONLY_ROLES`` below.
    """

    # Strong-only.
    PLANNER = "planner"                # the per-turn tool call (qa_agent)
    ACTION = "action"
    COORD_PROPOSER = "coord_proposer"  # propose_click_coordinates (pixels)

    # Cheap-first with escalation.
    SCREEN_CLASSIFIER = "screen_classifier"  # Phase 2
    FAST_CLASSIFY = "fast_classify"          # Phase 10 popup classifier
    VISION_SEARCH = "vision_search"
    GOAL_VERIFIER = "goal_verifier"
    ON_TRACK_CHECK = "on_track_check"
    SMART_PICKER = "smart_picker"            # Phase 14
    SEMANTIC_VERIFIER = "semantic_verifier"  # Phase 9


_STRONG_ONLY_ROLES: frozenset[LLMRole] = frozenset({
    LLMRole.PLANNER,
    LLMRole.ACTION,
    LLMRole.COORD_PROPOSER,
})


# Locked per the user's preference (Phase 1, Q4): escalate when the
# cheap tier returns confidence < 0.7. Validators returning False
# also escalate regardless of confidence.
ESCALATION_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class TieredResult:
    """Wraps a ChatResult with metadata about which tier produced it."""

    chat: ChatResult
    tier: str  # "cheap" | "strong"
    escalated: bool
    # When escalation fired, this is the cheap-tier's rejected output
    # (for telemetry / live-feed surfacing). None on a clean pass.
    cheap_output: Any | None = None
    cheap_reason: str | None = None


def _resolve_tier(
    primary_row: Any,
    *,
    tier_provider: str | None,
    tier_model: str | None,
    tier_api_key: str | None,
    tier_base_url: str | None,
) -> "LLMProvider | None":
    """Migration 0025 — resolve a tier's provider with credential
    fallback to the primary tier.

    Each non-primary tier carries optional (provider, model, api_key,
    base_url). When a field is NULL we fall back to the primary tier's
    field — useful when the user wants e.g. cheap_model=gpt-5-mini on
    the SAME OpenAI key as the strong tier without re-entering it.

    Returns ``None`` when the tier has NO model configured at all
    (caller treats as "tier not enabled").
    """
    from app.llm.factory import get_provider  # noqa: PLC0415

    model = (tier_model or "").strip()
    if not model:
        return None
    provider = (
        (tier_provider or "").strip() or primary_row.provider
    )
    # Credentials fall back to primary only when SAME provider.
    if (tier_api_key or "").strip():
        api_key = tier_api_key
    elif provider == primary_row.provider:
        api_key = primary_row.api_key
    else:
        # Different provider, no key configured → tier disabled.
        return None
    base_url = (tier_base_url or "").strip() or (
        primary_row.base_url if provider == primary_row.provider else None
    )
    try:
        return get_provider(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        # Bad config (missing base_url on openai_compat, unknown
        # provider) → tier disabled but the run can still proceed.
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).warning(
            "Tier resolve failed: provider=%s model=%s — %s: %s",
            provider, model, type(e).__name__, e,
        )
        return None


@dataclass
class TierQuad:
    """Migration 0025 — the four-tier provider bundle.

    - ``strong`` / ``cheap`` mirror the historical pair.
    - ``fallback_strong`` / ``fallback_cheap`` are tried when the
      primary tier raises an exception (rate limit, 5xx, network).
    All four are independent (provider, model, api_key, base_url)
    tuples — the user can mix OpenAI / OpenRouter / Gemini freely.
    """
    strong: "LLMProvider"
    cheap: "LLMProvider | None" = None
    fallback_strong: "LLMProvider | None" = None
    fallback_cheap: "LLMProvider | None" = None


def build_tier_quad(db: "Session") -> TierQuad:
    """Build the four-tier provider bundle from app_settings.

    Replaces ``build_tier_pair`` for new callers; the pair shim below
    keeps existing call sites working without code churn.
    """
    from app.llm.factory import get_provider  # noqa: PLC0415
    from app.models.app_settings import AppSettings  # noqa: PLC0415

    row = db.query(AppSettings).filter(AppSettings.id == 1).first()
    if not row or not row.api_key:
        raise RuntimeError(
            "LLM not configured. Configure it in Settings before "
            "running agents.",
        )
    strong = get_provider(
        provider=row.provider,
        model=row.model,
        api_key=row.api_key,
        base_url=row.base_url,
    )
    cheap = _resolve_tier(
        row,
        tier_provider=getattr(row, "cheap_provider", None) or row.provider,
        tier_model=row.cheap_model,
        tier_api_key=getattr(row, "cheap_api_key", None),
        tier_base_url=getattr(row, "cheap_base_url", None),
    )
    # Avoid running the cheap tier when it resolves to the SAME
    # provider+model as strong — wastes a cache entry and the router
    # already handles "no tiering" cleanly.
    if cheap is not None and (
        getattr(cheap, "model", None) == row.model
        and getattr(cheap, "provider_id", None)
        == getattr(strong, "provider_id", None)
    ):
        cheap = None
    fb_strong = _resolve_tier(
        row,
        tier_provider=getattr(row, "fallback_strong_provider", None),
        tier_model=getattr(row, "fallback_strong_model", None),
        tier_api_key=getattr(row, "fallback_strong_api_key", None),
        tier_base_url=getattr(row, "fallback_strong_base_url", None),
    )
    fb_cheap = _resolve_tier(
        row,
        tier_provider=getattr(row, "fallback_cheap_provider", None),
        tier_model=getattr(row, "fallback_cheap_model", None),
        tier_api_key=getattr(row, "fallback_cheap_api_key", None),
        tier_base_url=getattr(row, "fallback_cheap_base_url", None),
    )
    return TierQuad(
        strong=strong,
        cheap=cheap,
        fallback_strong=fb_strong,
        fallback_cheap=fb_cheap,
    )


def build_tier_pair(
    db: "Session",
) -> tuple[LLMProvider, LLMProvider | None]:
    """Shim — return (strong, cheap) from the quad for legacy callers.

    New code should call :func:`build_tier_quad` so fallback tiers are
    visible too. The fallbacks are still applied automatically when
    ``call_for_role`` is invoked WITHIN an :func:`active_quad` context
    (see ``set_active_quad`` below).
    """
    q = build_tier_quad(db)
    set_active_quad(q)
    return q.strong, q.cheap


# ── Migration 0025 — active-quad context ──────────────────────────
#
# Threading ``fallback_strong`` / ``fallback_cheap`` as kwargs through
# every ``call_for_role`` site would touch ~15 files. Instead, the
# orchestrator sets the active TierQuad at run start, and
# ``call_for_role`` reads the fallback tiers from this context when
# the kwargs aren't supplied. Per-process, thread-local.

import contextvars  # noqa: E402, PLC0415

_active_quad_var: "contextvars.ContextVar[TierQuad | None]" = (
    contextvars.ContextVar("active_tier_quad", default=None)
)


def set_active_quad(quad: TierQuad) -> None:
    """Install the active 4-tier provider bundle for this context.

    Subsequent ``call_for_role`` invocations read ``fallback_strong``
    and ``fallback_cheap`` from here when the caller didn't pass them
    explicitly. Call once at run start (orchestrator-level); the
    contextvar inherits down through threads spawned via the
    standard ``concurrent.futures`` pool.
    """
    _active_quad_var.set(quad)


def get_active_quad() -> "TierQuad | None":
    """Read the current active quad (or None when no orchestrator
    installed one — legacy callers, tests, etc.)."""
    return _active_quad_var.get()


def clear_active_quad() -> None:
    _active_quad_var.set(None)


def _chat_with_fallback(
    primary: LLMProvider,
    fallback: LLMProvider | None,
    *,
    messages: list[ChatMessage],
    schema: dict[str, Any],
    schema_name: str,
    temperature: float | None,
    max_output_tokens: int | None,
    role_value: str,
    tier_label: str,
    on_escalate: Callable[[str, str, str], None] | None = None,
) -> tuple[Any, "LLMProvider", bool]:
    """Migration 0025 — call ``primary``; on exception, retry on
    ``fallback`` if configured. Returns ``(result, provider_used,
    used_fallback)``.

    The fallback fires ONLY when the primary raises (rate limit,
    network, 5xx). Confidence-driven escalation between cheap and
    strong tiers continues to fire separately in ``call_for_role``.
    """
    try:
        result = primary.chat_structured(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return result, primary, False
    except Exception as primary_err:
        if fallback is None:
            raise
        reason = (
            f"{tier_label}-tier raised "
            f"{type(primary_err).__name__}: {str(primary_err)[:160]}"
            " — failing over to fallback"
        )
        logger.warning(
            "%s tier exception, fallback engaged: %s",
            tier_label, reason,
        )
        if on_escalate:
            try:
                on_escalate(
                    role_value,
                    getattr(primary, "model", "unknown"),
                    reason,
                )
            except Exception:
                pass
        result = fallback.chat_structured(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return result, fallback, True


def call_for_role(
    strong: LLMProvider,
    cheap: LLMProvider | None,
    role: LLMRole,
    *,
    messages: list[ChatMessage],
    schema: dict[str, Any],
    schema_name: str = "output",
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    confidence_field: str = "confidence",
    validate: Callable[[Any], bool] | None = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
    fallback_strong: LLMProvider | None = None,
    fallback_cheap: LLMProvider | None = None,
) -> TieredResult:
    """Make a structured chat call routed by role.

    1. Strong-only role OR cheap=None → strong.chat_structured directly.
       Falls over to ``fallback_strong`` if the strong call raises.
    2. Cheap-first role with cheap configured →
       a) call cheap; if it raises, try ``fallback_cheap``; if that
          also raises, escalate to strong (which itself has a
          fallback path).
       b) if parsed isn't a dict → escalate
       c) if confidence < threshold → escalate
       d) if validate(parsed) returns False → escalate
       e) otherwise return cheap result

    on_escalate(role_name, from_model, reason) fires when escalation
    happens — wire it to the live-feed event emitter so the user sees
    the tier transition.
    """
    # Local imports to avoid circular deps with cost_tracker init.
    import time as _time  # noqa: PLC0415

    from app.llm.cost_tracker import record_call  # noqa: PLC0415

    # Migration 0025 — pick up fallback tiers from the active quad
    # set by the orchestrator if the caller didn't pass them.
    if fallback_strong is None or fallback_cheap is None:
        _active = get_active_quad()
        if _active is not None:
            if fallback_strong is None:
                fallback_strong = _active.fallback_strong
            if fallback_cheap is None:
                fallback_cheap = _active.fallback_cheap

    if role in _STRONG_ONLY_ROLES or cheap is None:
        _t0 = _time.monotonic()
        result, provider_used, used_fb = _chat_with_fallback(
            strong, fallback_strong,
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            role_value=role.value,
            tier_label="strong",
            on_escalate=on_escalate,
        )
        # Per-call telemetry — role + model snapshotted so the drill-
        # in UI can show which call did what. ``cached_input_tokens``
        # comes from the provider response (OpenAI's automatic cache,
        # Gemini's cached_content) so the cost surface bills the
        # cached portion at the discounted rate.
        record_call(
            "strong", result.input_tokens, result.output_tokens,
            role=role.value,
            model=getattr(strong, "model", None),
            cached_input_tokens=getattr(result, "cached_input_tokens", None),
            escalated=False,
            duration_ms=int((_time.monotonic() - _t0) * 1000),
        )
        return TieredResult(chat=result, tier="strong", escalated=False)

    # Cheap-first path.
    cheap_reason: str | None = None
    cheap_output: Any | None = None
    _t_cheap = _time.monotonic()
    try:
        # Migration 0025 — cheap call with fallback_cheap on
        # exception (rate limit / network / 5xx). Fallback fires
        # BEFORE the strong escalation; it's same-tier substitution,
        # not tier promotion.
        cheap_result, cheap_provider_used, used_fb_cheap = (
            _chat_with_fallback(
                cheap, fallback_cheap,
                messages=messages,
                schema=schema,
                schema_name=schema_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                role_value=role.value,
                tier_label="cheap",
                on_escalate=on_escalate,
            )
        )
        # Per-call telemetry: cheap-tier attempt is its own row.
        # Whether or not escalation fires, the cheap-tier tokens
        # were already spent and account separately.
        record_call(
            "cheap", cheap_result.input_tokens, cheap_result.output_tokens,
            role=role.value,
            model=getattr(cheap_provider_used, "model", None),
            cached_input_tokens=getattr(
                cheap_result, "cached_input_tokens", None,
            ),
            escalated=False,
            duration_ms=int((_time.monotonic() - _t_cheap) * 1000),
        )
        cheap_output = cheap_result.parsed
    except Exception as e:
        cheap_reason = f"cheap-tier raised {type(e).__name__}: {str(e)[:160]}"
        logger.info(
            "Tier escalation %s: %s — escalating to %s",
            role.value, cheap_reason, strong.model,
        )
        if on_escalate:
            on_escalate(role.value, cheap.model, cheap_reason)
        _t_strong = _time.monotonic()
        # Strong with fallback_strong on exception (same as the
        # strong-only path above).
        strong_result, strong_provider_used, _used_fb_s = (
            _chat_with_fallback(
                strong, fallback_strong,
                messages=messages,
                schema=schema,
                schema_name=schema_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                role_value=role.value,
                tier_label="strong",
                on_escalate=on_escalate,
            )
        )
        record_call(
            "strong",
            strong_result.input_tokens,
            strong_result.output_tokens,
            role=role.value,
            model=getattr(strong_provider_used, "model", None),
            cached_input_tokens=getattr(
                strong_result, "cached_input_tokens", None,
            ),
            escalated=True,
            duration_ms=int((_time.monotonic() - _t_strong) * 1000),
        )
        return TieredResult(
            chat=strong_result,
            tier="strong",
            escalated=True,
            cheap_output=None,
            cheap_reason=cheap_reason,
        )

    # Confidence + structural validation gates.
    parsed = cheap_result.parsed
    needs_escalation = False
    if not isinstance(parsed, dict):
        cheap_reason = "cheap output not a dict"
        needs_escalation = True
    elif confidence_field:
        conf_val = parsed.get(confidence_field)
        try:
            conf = float(conf_val) if conf_val is not None else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        if conf < ESCALATION_CONFIDENCE_THRESHOLD:
            cheap_reason = (
                f"cheap confidence {conf:.2f} < {ESCALATION_CONFIDENCE_THRESHOLD}"
            )
            needs_escalation = True

    if not needs_escalation and validate is not None:
        try:
            ok = validate(parsed)
        except Exception as e:
            ok = False
            cheap_reason = (
                f"cheap validator raised {type(e).__name__}: {str(e)[:120]}"
            )
        if not ok:
            cheap_reason = (
                cheap_reason or "cheap output failed structural validator"
            )
            needs_escalation = True

    if not needs_escalation:
        return TieredResult(chat=cheap_result, tier="cheap", escalated=False)

    logger.info(
        "Tier escalation %s: %s — re-running on %s",
        role.value, cheap_reason, strong.model,
    )
    if on_escalate:
        on_escalate(role.value, cheap.model, cheap_reason or "")
    _t_strong = _time.monotonic()
    # Migration 0025 — strong with fallback_strong on exception.
    strong_result, _strong_used, _used_fb = _chat_with_fallback(
        strong, fallback_strong,
        messages=messages,
        schema=schema,
        schema_name=schema_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        role_value=role.value,
        tier_label="strong",
        on_escalate=on_escalate,
    )
    record_call(
        "strong", strong_result.input_tokens, strong_result.output_tokens,
        role=role.value,
        model=getattr(strong, "model", None),
        cached_input_tokens=getattr(
            strong_result, "cached_input_tokens", None,
        ),
        escalated=True,
        duration_ms=int((_time.monotonic() - _t_strong) * 1000),
    )
    return TieredResult(
        chat=strong_result,
        tier="strong",
        escalated=True,
        cheap_output=cheap_output,
        cheap_reason=cheap_reason,
    )
