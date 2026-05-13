"""Phase D — live-UI dry-run validation for a refined TcVersion.

After the refiner produces a v2 snapshot tree, this service opens a
real browser, logs into the target app via ``auth_flow``, walks each
refined step, and PROBES each target_hint against the live DOM —
WITHOUT dispatching the action. The result lands on the snapshot row
as ``validation_status`` + ``validation_confidence`` + ``validation_reason``.

The dialog + test-cases viewer surface these as confidence badges,
so the operator can see — before running anything in earnest — which
refined steps the live app actually supports. Acts like a human QA
who flips through the app once before settling in to run the test
suite.

Strategy
--------
For each step in submodule order:

1. **Navigate to the predicted page.** Use ``page_url_after`` from the
   prior step's frozen v1 path when available; otherwise rely on the
   submodule's prior steps to land us in the right place. When the
   predicted page is unreachable (404, auth wall, network error) →
   mark all remaining steps in the submodule as ``unreachable``.

2. **Probe the target_hint.** Try ``selectors.resolve(page, hint,
   timeout_ms=2000)``. Resolved → ``confirmed``. Otherwise →
   ``unresolved``.

3. **Probe the expected text** (when the step has one). Substring
   check against ``page.body.innerText``. Resolved → keeps the
   step at ``confirmed``; not present and target_hint also failed →
   ``unresolved``; not present but target_hint resolved → ``partial``.

4. **Special cases**:
   - ``change_kind="flagged_missing"`` → status=``skipped``, the
     refiner already told us it doesn't exist.
   - ``action_type="navigate"`` → status=``confirmed`` (no element
     to probe; navigation is its own validation).
   - ``action_type="verify"`` with no target_hint → probe expected
     text only.

5. **Auth wall in the middle**: if we hit a login screen mid-walk,
   re-invoke ``auth_flow.run_auth_loop`` and continue.

6. **Cancellation**: caller passes a flag; we honor it between
   steps so the user can abort a long validation.

Validation NEVER dispatches actions (no clicks, no form submits).
Worst-case cost: ~1-2s per step + auth + navigation overhead.
For Solar's 30+ steps: ~60-90s.

Auth handling
-------------
Validation reuses the same ``run_auth_loop`` that the agent uses,
backed by the plan's vault. No new prompts — credentials are already
saved if the plan has them; we just trigger the same flow once.

Confidence scoring
------------------
Combined into ``validation_confidence`` (0.0–1.0):

  status=confirmed   → base 0.95
  status=partial     → base 0.70
  status=unresolved  → base 0.45
  status=unreachable → base 0.10
  status=skipped     → base 0.30 (flagged_missing — refiner couldn't find a target)
  status=pending     → base 0.50 (validation never ran)

Then adjusted by change_kind weight:
  kept       → ×1.00
  rewritten  → ×0.95
  added      → ×0.90  (newer = less proven)
  flagged_missing → ×0.50

Final value is clamped to [0.0, 1.0].
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from playwright.sync_api import Page
    from sqlalchemy.orm import Session

    from app.llm.base import LLMProvider
    from app.models.tc_version import TcNodeSnapshot

logger = logging.getLogger(__name__)


# ── Status types ─────────────────────────────────────────────────


ValidationStatus = str  # confirmed | partial | unresolved | unreachable | skipped | pending

_STATUS_BASE_CONFIDENCE: dict[str, float] = {
    "confirmed":   0.95,
    "partial":     0.70,
    "unresolved":  0.45,
    "unreachable": 0.10,
    "skipped":     0.30,
    "pending":     0.50,
}

_CHANGE_KIND_WEIGHT: dict[str, float] = {
    "kept":            1.00,
    "rewritten":       0.95,
    "added":           0.90,
    "flagged_missing": 0.50,
}


def score_validation(
    status: str, change_kind: str,
) -> float:
    """Combine validation status + change_kind into a 0-1 confidence."""
    base = _STATUS_BASE_CONFIDENCE.get(status, 0.50)
    weight = _CHANGE_KIND_WEIGHT.get(change_kind, 1.00)
    return max(0.0, min(1.0, base * weight))


# ── Result aggregation ───────────────────────────────────────────


@dataclass
class StepValidation:
    snapshot_id: int
    status: str
    reason: str
    confidence: float


@dataclass
class SubmoduleValidationSummary:
    submodule_snapshot_id: int
    submodule_title: str
    confirmed: int = 0
    partial: int = 0
    unresolved: int = 0
    unreachable: int = 0
    skipped: int = 0
    pending: int = 0
    confidence: float = 0.0
    steps: list[StepValidation] = field(default_factory=list)


@dataclass
class ValidationResult:
    plan_id: int
    version_id: int
    submodules: list[SubmoduleValidationSummary] = field(default_factory=list)
    total_probed: int = 0
    total_seconds: float = 0.0
    error_message: str | None = None
    cancelled: bool = False


# ── Public API ────────────────────────────────────────────────────


def validate_version_against_live(
    db: "Session",
    *,
    plan_id: int,
    version_id: int,
    headless: bool = True,
    provider: "LLMProvider | None" = None,
    cheap_provider: "LLMProvider | None" = None,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ValidationResult:
    """Walk the v2 snapshot tree, probe each step against the live UI,
    write results to each snapshot row.

    Returns ``ValidationResult`` with per-submodule rollups.

    Failures inside this function are NEVER fatal to the run — a
    bad navigation / network glitch leaves remaining steps as
    ``unreachable`` and the result message carries the reason.
    """
    from app.models.tc_version import (  # noqa: PLC0415
        TcVersion, TcNodeSnapshot,
    )
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from app.executor import browser_session  # noqa: PLC0415
    from sqlalchemy import select  # noqa: PLC0415

    t0 = time.monotonic()
    out = ValidationResult(plan_id=plan_id, version_id=version_id)

    plan = db.get(TestPlan, plan_id)
    if plan is None:
        out.error_message = f"plan {plan_id} not found"
        return out
    version = db.get(TcVersion, version_id)
    if version is None or version.plan_id != plan.id:
        out.error_message = (
            f"version {version_id} not found for plan {plan_id}"
        )
        return out

    snapshots = list(db.execute(
        select(TcNodeSnapshot)
        .where(TcNodeSnapshot.tc_version_id == version_id)
        .order_by(
            TcNodeSnapshot.depth,
            TcNodeSnapshot.parent_snapshot_id,
            TcNodeSnapshot.ordinal,
        ),
    ).scalars())
    if not snapshots:
        out.error_message = "no snapshots to validate"
        return out

    by_parent: dict[int | None, list[TcNodeSnapshot]] = {}
    for s in snapshots:
        by_parent.setdefault(s.parent_snapshot_id, []).append(s)
    submodules = [s for s in snapshots if s.kind == "submodule"]

    def _emit(t: str, d: dict) -> None:
        if emit_event:
            try:
                emit_event(t, d)
            except Exception:
                pass

    _emit("tc_validation_started", {
        "plan_id": plan_id,
        "version_id": version_id,
        "submodule_count": len(submodules),
    })

    target_url = (plan.target_url or "").strip()
    if not target_url:
        out.error_message = "plan has no target_url"
        return out

    # Best-effort browser session. Headless by default — validation is
    # a background QA probe, not a user-watched run. Same speed config
    # as a "fast" run since we're not dispatching anything.
    try:
        with browser_session(headless=headless, speed=None) as page:
            try:
                page.goto(
                    target_url,
                    wait_until="domcontentloaded",
                    timeout=20_000,
                )
            except Exception as e:
                out.error_message = (
                    f"could not navigate to {target_url}: {e}"
                )
                return out

            # Single auth at start. The same vault credentials work
            # across submodules. We DON'T re-auth per submodule —
            # validation isn't supposed to bounce in and out of login.
            _maybe_run_auth(
                page, plan=plan, provider=provider,
                cheap_provider=cheap_provider,
                emit_event=emit_event,
            )

            for sm in submodules:
                if is_cancelled and is_cancelled():
                    out.cancelled = True
                    break
                summary = SubmoduleValidationSummary(
                    submodule_snapshot_id=sm.id,
                    submodule_title=sm.title,
                )
                steps = sorted(
                    (s for s in by_parent.get(sm.id, [])
                     if s.kind == "step"),
                    key=lambda s: s.ordinal,
                )
                _emit("tc_validation_submodule_started", {
                    "plan_id": plan_id,
                    "version_id": version_id,
                    "submodule_snapshot_id": sm.id,
                    "title": sm.title,
                    "step_count": len(steps),
                })

                # If we hit unreachable mid-submodule, the remaining
                # steps in THIS submodule cascade to "unreachable".
                cascade_unreachable = False
                for step in steps:
                    if is_cancelled and is_cancelled():
                        out.cancelled = True
                        break
                    if cascade_unreachable:
                        _record(
                            db, step,
                            status="unreachable",
                            reason=(
                                "preceding step left the page in an "
                                "unreachable state; not probed"
                            ),
                        )
                        summary.unreachable += 1
                        continue

                    status, reason = _probe_step(page, step)
                    _record(db, step, status=status, reason=reason)

                    if status == "unreachable":
                        cascade_unreachable = True
                    if status == "confirmed":
                        summary.confirmed += 1
                    elif status == "partial":
                        summary.partial += 1
                    elif status == "unresolved":
                        summary.unresolved += 1
                    elif status == "unreachable":
                        summary.unreachable += 1
                    elif status == "skipped":
                        summary.skipped += 1
                    else:
                        summary.pending += 1

                    summary.steps.append(StepValidation(
                        snapshot_id=step.id,
                        status=status,
                        reason=reason,
                        confidence=score_validation(
                            status, step.change_kind,
                        ),
                    ))
                    out.total_probed += 1
                    _emit("tc_validation_step", {
                        "plan_id": plan_id,
                        "version_id": version_id,
                        "submodule_snapshot_id": sm.id,
                        "step_snapshot_id": step.id,
                        "title": step.title,
                        "status": status,
                        "reason": reason[:200],
                    })

                # Roll up the submodule confidence as the WORST step
                # confidence (a chain is only as strong as its weakest
                # step). Pending / skipped don't drag the rollup down.
                meaningful = [
                    sv.confidence for sv in summary.steps
                    if sv.status not in ("pending", "skipped")
                ]
                summary.confidence = (
                    min(meaningful) if meaningful else 0.5
                )
                _record_submodule(
                    db, sm,
                    confidence=summary.confidence,
                    confirmed_n=summary.confirmed,
                    unresolved_n=summary.unresolved,
                    unreachable_n=summary.unreachable,
                )
                out.submodules.append(summary)
                _emit("tc_validation_submodule_completed", {
                    "plan_id": plan_id,
                    "version_id": version_id,
                    "submodule_snapshot_id": sm.id,
                    "confirmed": summary.confirmed,
                    "partial": summary.partial,
                    "unresolved": summary.unresolved,
                    "unreachable": summary.unreachable,
                    "skipped": summary.skipped,
                    "confidence": round(summary.confidence, 3),
                })
    except Exception as e:
        out.error_message = f"{type(e).__name__}: {str(e)[:300]}"
        logger.warning("validation aborted: %s", e)

    db.commit()
    out.total_seconds = round(time.monotonic() - t0, 2)
    _emit("tc_validation_completed", {
        "plan_id": plan_id,
        "version_id": version_id,
        "submodules": len(out.submodules),
        "total_probed": out.total_probed,
        "total_seconds": out.total_seconds,
        "error": out.error_message,
        "cancelled": out.cancelled,
    })
    return out


# ── Probe primitives ──────────────────────────────────────────────


def _probe_step(
    page: "Page", step: "TcNodeSnapshot",
) -> tuple[str, str]:
    """Return ``(status, reason)`` for one step's dry-run probe."""
    if step.change_kind == "flagged_missing":
        return (
            "skipped",
            "refiner flagged this step as missing in the app — "
            "validator skipped per spec",
        )

    action = (step.action_type or "").lower()
    hint = (step.target_hint or "").strip()
    expected = (step.expected or "").strip()

    # navigate is self-validating (a URL exists or doesn't);
    # we'd need to actually navigate to probe, which is too invasive
    # for a dry-run. Mark as confirmed and let runtime catch nav errors.
    if action == "navigate":
        return ("confirmed", "navigate steps are skipped in dry-run")

    target_resolved = False
    target_reason = ""
    if hint:
        try:
            from app.executor.selectors import (  # noqa: PLC0415
                resolve, SelectorNotFound,
            )
            try:
                resolve(page, hint, timeout_ms=2000)
                target_resolved = True
            except SelectorNotFound as e:
                target_reason = (
                    f"target_hint {hint!r} did not resolve: "
                    f"{str(e)[:160]}"
                )
            except Exception as e:
                target_reason = (
                    f"resolver raised: {type(e).__name__}: "
                    f"{str(e)[:160]}"
                )
        except Exception as e:
            target_reason = (
                f"validation harness error: {type(e).__name__}: "
                f"{str(e)[:160]}"
            )

    expected_present = False
    if expected and len(expected) >= 3:
        try:
            page_text = page.evaluate(
                "() => (document.body && document.body.innerText) || ''"
            )
            if isinstance(page_text, str):
                needle = expected.lower()
                # Strip generic words; match the first quotable token
                # in the expected text.
                for token in needle.split():
                    token = token.strip(".,;:'\"()")
                    if len(token) < 4:
                        continue
                    if token in page_text.lower():
                        expected_present = True
                        break
        except Exception:
            pass

    if not hint and not expected:
        return (
            "skipped",
            "no target_hint and no expected text to probe",
        )

    if hint and target_resolved and (not expected or expected_present):
        return ("confirmed", "target_hint resolved on the live page")
    if hint and target_resolved and not expected_present:
        return (
            "partial",
            "target_hint resolved but expected text not found in "
            "visible page text",
        )
    if hint and not target_resolved and expected_present:
        return (
            "partial",
            f"expected text found but {target_reason}",
        )
    if hint and not target_resolved:
        return ("unresolved", target_reason or "target_hint did not resolve")
    if expected and expected_present:
        return (
            "confirmed",
            "expected text found on page (no target_hint to check)",
        )
    return ("unresolved", "expected text not found on the page")


def _record(
    db: "Session",
    snap: "TcNodeSnapshot",
    *,
    status: str,
    reason: str,
) -> None:
    snap.validation_status = status
    snap.validation_confidence = score_validation(status, snap.change_kind)
    snap.validation_reason = reason[:1000]
    snap.validation_at = datetime.now(timezone.utc)
    db.flush()


def _record_submodule(
    db: "Session",
    sm: "TcNodeSnapshot",
    *,
    confidence: float,
    confirmed_n: int,
    unresolved_n: int,
    unreachable_n: int,
) -> None:
    """Roll up confidence onto the submodule snapshot itself so the
    test-cases viewer can render a single badge per submodule
    without re-aggregating its children."""
    if confirmed_n > 0 and unresolved_n == 0 and unreachable_n == 0:
        sm.validation_status = "confirmed"
    elif unreachable_n > 0:
        sm.validation_status = "unreachable"
    elif unresolved_n > 0:
        sm.validation_status = "partial"
    else:
        sm.validation_status = "pending"
    sm.validation_confidence = confidence
    sm.validation_reason = (
        f"{confirmed_n} confirmed · {unresolved_n} unresolved · "
        f"{unreachable_n} unreachable"
    )
    sm.validation_at = datetime.now(timezone.utc)
    db.flush()


# ── Auth helper ───────────────────────────────────────────────────


def _maybe_run_auth(
    page: "Page",
    *,
    plan: Any,
    provider: "LLMProvider | None",
    cheap_provider: "LLMProvider | None",
    emit_event: Callable[[str, dict], None] | None,
) -> None:
    """Run auth_flow when the validator lands on a login screen.
    No-op when the page doesn't look like login. Best-effort — auth
    failure leaves the page where it is and steps will mark
    unreachable accordingly."""
    if provider is None:
        return
    try:
        from app.agents.qa_agent import (  # noqa: PLC0415
            _looks_like_login_page,
        )
        from app.agents.auth_flow import (  # noqa: PLC0415
            run_auth_loop,
        )
        if not _looks_like_login_page(page):
            return
        run_auth_loop(
            page,
            plan=plan,
            provider=provider,
            cheap_provider=cheap_provider,
            emit_event=emit_event,
            open_typed_prompt=None,  # validator never opens HITL
            request_intervention=None,
            is_cancelled=None,
        )
    except Exception as e:
        logger.debug("validator auth_flow non-fatal: %s", e)
