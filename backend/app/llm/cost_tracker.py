"""Per-run, per-tier token accumulator + per-call telemetry buffer.

A run's LLM activity spans dozens of call sites (planner, coord
proposer, screen classifier, popup classifier, smart pick, semantic
verify, on-track check, goal verify, vision search, ...). Each
call passes through one of two paths:

1. ``app.llm.router.call_for_role`` — knows its tier (cheap | strong)
   from the role + provider configuration.
2. Direct ``provider.chat_structured`` — always strong by definition
   (the planner loop uses this, as does ``extract_goal``).

The tracker has TWO outputs:

- **Aggregate counters** (``CostCounters``) — four ints split by
  tier × direction. Persisted to ``agent_runs`` columns; the Cost
  card on the report renders from these.
- **Per-call records** (``CallRecord``) — one entry per round-trip
  with role, model, tokens, escalation flag, and timing. Flushed
  to ``llm_call_logs`` rows at run end so the drill-in UI can
  show "what calls happened, in what order, to which model."

Threading model: ``ContextVar`` keyed → the run's worker thread
sees the context everywhere without explicit plumbing. When no
context is open (legacy callers, scripted-mode runs not tracking
cost), ``record_call`` is a silent no-op.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """One LLM round-trip — the per-row shape for the call log."""

    role: str
    tier: str  # "strong" | "cheap"
    model: str | None
    input_tokens: int
    output_tokens: int
    # Cached portion of input_tokens (SUBSET, not additive). 0 when
    # the prompt didn't hit cache OR the provider didn't report it.
    # OpenAI prompt caching is automatic on ≥ 1024-token prompts;
    # Gemini's ``cached_content`` API is explicit (not yet wired).
    cached_input_tokens: int = 0
    escalated: bool = False
    duration_ms: int | None = None
    # Step the call was made under, snapshotted at record-time so
    # the drill-in view can group calls by submodule. None for
    # pre-step calls (goal extraction before the per-submodule
    # loop) and post-step calls.
    step_id: int | None = None


@dataclass
class CostCounters:
    """Aggregate (four ints) + per-call buffer.

    ``calls`` is the buffer the orchestrator flushes to
    ``llm_call_logs`` at run end. Kept in memory during the run so
    per-call DB writes don't add latency to the LLM hot path.
    """

    strong_input: int = 0
    strong_output: int = 0
    cheap_input: int = 0
    cheap_output: int = 0
    # Cached portions of the per-tier input counters above (SUBSET).
    # Provider responses populate these via record_call's
    # ``cached_input_tokens`` kwarg.
    strong_cached_input: int = 0
    cheap_cached_input: int = 0
    calls: list[CallRecord] = field(default_factory=list)
    run_id: int | None = None
    step_id: int | None = None

    def add(
        self,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        *,
        role: str = "",
        model: str | None = None,
        cached_input_tokens: int | None = None,
        escalated: bool = False,
        duration_ms: int | None = None,
    ) -> None:
        in_n = max(0, int(input_tokens or 0))
        out_n = max(0, int(output_tokens or 0))
        # Clamp cached to ≤ total input (defensive — providers
        # SHOULD respect this, but a misreported value could
        # otherwise drive regular_input negative downstream).
        cached_n = min(in_n, max(0, int(cached_input_tokens or 0)))
        if tier == "cheap":
            self.cheap_input += in_n
            self.cheap_output += out_n
            self.cheap_cached_input += cached_n
        else:
            self.strong_input += in_n
            self.strong_output += out_n
            self.strong_cached_input += cached_n
        # Buffer the per-call record. Skipped when role is empty —
        # legacy direct-provider call sites that haven't been
        # updated to pass role yet still account toward the
        # aggregate counters but don't write to llm_call_logs.
        if role:
            self.calls.append(
                CallRecord(
                    role=role,
                    tier=tier,
                    model=model,
                    input_tokens=in_n,
                    output_tokens=out_n,
                    cached_input_tokens=cached_n,
                    escalated=escalated,
                    duration_ms=duration_ms,
                    # Snapshot the CURRENT step_id at record-time so
                    # calls don't all flush against the LAST step
                    # at end_run.
                    step_id=self.step_id,
                ),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "strong_input": self.strong_input,
            "strong_output": self.strong_output,
            "cheap_input": self.cheap_input,
            "cheap_output": self.cheap_output,
            "strong_cached_input": self.strong_cached_input,
            "cheap_cached_input": self.cheap_cached_input,
        }


_run_cost: contextvars.ContextVar[CostCounters | None] = (
    contextvars.ContextVar("qai_run_cost", default=None)
)


def begin_run(
    run_id: int | None = None,
    step_id: int | None = None,
) -> CostCounters:
    """Open a new run-scoped cost context.

    ``run_id`` lets ``end_run`` write the buffered call records to
    ``llm_call_logs`` with the correct FK. When None (legacy
    callers, scout-only paths that don't have a run row), the
    call log step is skipped — aggregate counters still work.
    """
    counters = CostCounters(run_id=run_id, step_id=step_id)
    _run_cost.set(counters)
    return counters


def set_current_step(step_id: int | None) -> None:
    """Update the current step_id on the open context so subsequent
    call records get the right FK. Called at submodule boundaries.
    Idempotent / no-op when no context is open."""
    c = _run_cost.get()
    if c is not None:
        c.step_id = step_id


def get_counters() -> CostCounters | None:
    return _run_cost.get()


def record_call(
    tier: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    role: str = "",
    model: str | None = None,
    cached_input_tokens: int | None = None,
    escalated: bool = False,
    duration_ms: int | None = None,
) -> None:
    """Add tokens from one LLM round-trip to the run counters.

    Called by:
    - ``app.llm.router.call_for_role`` for every tiered call
      (passes role, model, cached, escalated, duration)
    - Direct provider calls (planner, ``extract_goal``) — also
      pass role + model + cached so the call appears in the
      drill-in view with the right cache breakdown

    ``cached_input_tokens`` is the SUBSET of ``input_tokens`` that
    the provider reported as cache hits. ``None``/0 means "no
    cache hit reported" — the regular input rate applies to the
    full input volume.

    No-op when no context is open.
    """
    c = _run_cost.get()
    if c is None:
        return
    c.add(
        tier,
        input_tokens or 0,
        output_tokens or 0,
        role=role,
        model=model,
        cached_input_tokens=cached_input_tokens,
        escalated=escalated,
        duration_ms=duration_ms,
    )


def end_run(db: "Session | None" = None) -> CostCounters | None:
    """Close the current context, optionally flush call records to
    ``llm_call_logs``, then return the counters.

    Caller persists the aggregate counters to the ``agent_runs``
    row. When ``db`` is supplied AND the context has a ``run_id``,
    we ALSO insert one ``LlmCallLog`` row per buffered ``CallRecord``
    in a single commit so the drill-in view sees the per-call trail.

    Failures here are non-fatal: the per-call log is auxiliary
    telemetry; if the write fails the aggregate counters (the
    primary cost surface) still land on the run row.
    """
    c = _run_cost.get()
    _run_cost.set(None)
    if c is None:
        return None

    if db is not None and c.run_id is not None and c.calls:
        try:
            from app.models.llm_call_log import LlmCallLog  # noqa: PLC0415

            rows = [
                LlmCallLog(
                    run_id=c.run_id,
                    step_id=call.step_id,
                    ordinal=i,
                    role=call.role,
                    tier=call.tier,
                    model=call.model,
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    cached_input_tokens=call.cached_input_tokens,
                    escalated=call.escalated,
                    duration_ms=call.duration_ms,
                )
                for i, call in enumerate(c.calls)
            ]
            db.add_all(rows)
            db.commit()
        except Exception as e:
            logger.warning(
                "llm_call_logs flush failed for run %s (%d calls "
                "dropped — aggregate counters preserved): %s",
                c.run_id, len(c.calls), e,
            )
            try:
                db.rollback()
            except Exception:
                pass
    return c
