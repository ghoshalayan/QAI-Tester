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


def build_tier_pair(
    db: "Session",
) -> tuple[LLMProvider, LLMProvider | None]:
    """Build (strong_provider, cheap_provider_or_None) from app_settings.

    Single helper used by orchestrators (qa_agent, agent_run_service)
    to bootstrap their tiering. Returns ``cheap=None`` when the user
    hasn't configured a cheap model — caller then routes everything
    to strong (legacy behavior).
    """
    # Local imports to avoid cycles.
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
    cheap: LLMProvider | None = None
    cheap_model = (row.cheap_model or "").strip()
    if cheap_model and cheap_model != row.model:
        cheap = get_provider(
            provider=row.provider,
            model=cheap_model,
            api_key=row.api_key,
            base_url=row.base_url,
        )
    return strong, cheap


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
) -> TieredResult:
    """Make a structured chat call routed by role.

    1. Strong-only role OR cheap=None → strong.chat_structured directly.
    2. Cheap-first role with cheap configured →
       a) call cheap; if it raises, escalate to strong
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

    if role in _STRONG_ONLY_ROLES or cheap is None:
        _t0 = _time.monotonic()
        result = strong.chat_structured(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
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
        cheap_result = cheap.chat_structured(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        # Per-call telemetry: cheap-tier attempt is its own row.
        # Whether or not escalation fires, the cheap-tier tokens
        # were already spent and account separately.
        record_call(
            "cheap", cheap_result.input_tokens, cheap_result.output_tokens,
            role=role.value,
            model=getattr(cheap, "model", None),
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
        strong_result = strong.chat_structured(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        record_call(
            "strong",
            strong_result.input_tokens,
            strong_result.output_tokens,
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
    strong_result = strong.chat_structured(
        messages=messages,
        schema=schema,
        schema_name=schema_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
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
