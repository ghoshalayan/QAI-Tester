"""Phase H — preflight orchestrator: Scout → Refine → Activate.

The user's observation that triggered this module: today's lifecycle
generates test cases from the BRD without UI grounding, then refines
LAZILY per-submodule at runtime — by which point the agent has already
burned turns chasing buttons that don't exist with labels that don't
match. The Scout + AppMap consolidation + tc_refinement pieces all
exist; what was missing is the orchestrator that runs them *once,
upfront, before any submodule executes*.

What this module does
---------------------
``run_preflight(db, plan_id)`` is the single public entry point:

1. **Bootstrap check** — if the plan already has an
   ``app_map_refined`` TcVersion active AND a non-stale AppMap, returns
   immediately with ``status="skipped"`` so re-runs don't re-pay the
   cost.
2. **Scout (if needed)** — when no AppMap exists for the plan's
   ``target_url``, launches a headless Chromium, runs ``auth_flow``,
   then ``run_authenticated_scout(depth="deep")``, consolidates with
   ``consolidate_app_map``, and saves via ``save_app_map``. Cached at
   the AKB layer so subsequent plans on the same target_url reuse it.
3. **Refine** — calls ``refine_plan`` to materialise per-submodule
   rewrites against the AppMap, persisting them as a new TcVersion
   with ``source="app_map_refined"``.
4. **Activate** — calls ``apply_tc_version_to_live`` so the live
   ``TcNode`` tree IS the refined plan; ``current_tc_version_id`` is
   updated on the plan. Execute() picks this up automatically because
   it reads the live tree.

Live events
-----------
All progress is surfaced through ``emit_event`` with ``preflight_*``
event types so the existing /live page renders the preflight pass
exactly like a normal run. The frontend doesn't need to know it's a
preflight; the events are scoped via the ``phase`` field.

When NOT to use
---------------
- Plans without a ``target_url`` (no scout target).
- Plans intentionally pinned to a specific TcVersion (the user picked
  v3 manually) — we honor ``preflight="skip"`` from the caller.
- Test-fast paths where the tester wants to see raw BRD-generated
  steps fail (debugging the refiner itself).

Cost
----
- First plan against a target_url: 1 scout call (~$0.08-0.15) +
  1 refine call per submodule (~$0.05-0.10 total for Solar). ~$0.20.
- Subsequent plans: refine only when the AppMap changes; otherwise
  the existing refined version is reused. Effectively free.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.agents.app_map import AppMap
    from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


PreflightStatus = Literal[
    "completed", "skipped", "failed", "partial",
]


@dataclass
class PreflightResult:
    plan_id: int
    status: PreflightStatus = "failed"
    # Scout sub-step.
    scout_ran: bool = False
    scout_pages: int = 0
    scout_create_surfaces: int = 0
    # Refine sub-step.
    refine_ran: bool = False
    new_version_id: int | None = None
    refined_submodules: int = 0
    refined_rewritten: int = 0
    refined_added: int = 0
    refined_flagged_missing: int = 0
    # Activation.
    activated_version_id: int | None = None
    # Bookkeeping.
    input_tokens: int = 0
    output_tokens: int = 0
    total_seconds: float = 0.0
    error_message: str | None = None
    notes: list[str] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────


def run_preflight(
    db: "Session",
    *,
    plan_id: int,
    provider: "LLMProvider",
    cheap_provider: "LLMProvider | None" = None,
    force: bool = False,
    skip_scout: bool = False,
    skip_refine: bool = False,
    skip_activation: bool = False,
    scout_depth: Literal["shallow", "deep"] = "deep",
    headless: bool = True,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
) -> PreflightResult:
    """Run preflight for a test plan. See module docstring.

    Args:
        force: When True, scout + refine + activate regardless of
            cache state. The escape hatch for "I edited the BRD,
            rerun everything." When False (default), each sub-step
            short-circuits if a fresh cached artifact exists.
        skip_scout / skip_refine / skip_activation: granular bypass
            flags for testing or partial reruns.
        scout_depth: forwarded to ``run_authenticated_scout``.
        headless: passed through to the browser session for scouting.
            Default ``True`` — preflight is invisible cooperation; the
            user shouldn't have to watch a window pop up just for the
            test cases to refresh.
    """
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from app.models.tc_version import TcVersion  # noqa: PLC0415
    from app.agents.app_map import load_app_map  # noqa: PLC0415
    from app.services.tc_refinement import (  # noqa: PLC0415
        refine_plan, apply_tc_version_to_live,
    )
    from sqlalchemy import select  # noqa: PLC0415

    out = PreflightResult(plan_id=plan_id)
    t0 = time.monotonic()

    def _emit(t: str, d: dict) -> None:
        if emit_event is None:
            return
        try:
            emit_event(t, d)
        except Exception:
            pass

    plan = db.get(TestPlan, plan_id)
    if plan is None:
        out.error_message = f"plan {plan_id} not found"
        return out
    target = (plan.target_url or "").strip()
    if not target:
        out.error_message = "plan has no target_url; cannot scout"
        return out

    _emit("preflight_started", {
        "plan_id": plan_id,
        "target_url": target,
        "force": force,
    })

    # ── Step 1: shortcut when refinement already exists and isn't stale.
    current_version_id = plan.current_tc_version_id
    if not force and current_version_id is not None:
        cur_ver = db.get(TcVersion, current_version_id)
        if cur_ver is not None and cur_ver.source == "app_map_refined":
            # Make sure the AppMap still exists too; if someone purged
            # the AKB row, force a re-scout.
            existing_map = load_app_map(db, target_url=target)
            if existing_map is not None:
                out.status = "skipped"
                out.notes.append(
                    "plan already pinned to app_map_refined version "
                    f"v{cur_ver.version_number}; AppMap present — "
                    "nothing to do",
                )
                out.activated_version_id = cur_ver.id
                out.total_seconds = round(time.monotonic() - t0, 2)
                _emit("preflight_completed", {
                    "plan_id": plan_id,
                    "status": out.status,
                    "version_id": out.activated_version_id,
                    "seconds": out.total_seconds,
                    "notes": out.notes,
                })
                return out

    # ── Step 2: Scout. Only if no AppMap is present or force=True.
    existing_map: "AppMap | None" = load_app_map(db, target_url=target)
    needs_scout = (
        not skip_scout
        and (force or existing_map is None)
    )
    if needs_scout:
        try:
            _scouted = _scout_target(
                db,
                plan=plan,
                provider=provider,
                cheap_provider=cheap_provider,
                scout_depth=scout_depth,
                headless=headless,
                emit_event=_emit,
                is_cancelled=is_cancelled,
                on_escalate=on_escalate,
            )
            if _scouted is None:
                out.error_message = (
                    "scout failed; cannot refine without an AppMap"
                )
                _emit("preflight_failed", {
                    "plan_id": plan_id,
                    "stage": "scout",
                    "error": out.error_message,
                })
                out.total_seconds = round(time.monotonic() - t0, 2)
                return out
            out.scout_ran = True
            out.scout_pages = _scouted.get("pages", 0)
            out.scout_create_surfaces = _scouted.get(
                "create_surfaces", 0,
            )
            out.input_tokens += _scouted.get("input_tokens", 0)
            out.output_tokens += _scouted.get("output_tokens", 0)
            existing_map = load_app_map(db, target_url=target)
        except Exception as e:
            logger.exception(
                "preflight scout failed for plan %s", plan_id,
            )
            out.error_message = f"scout exception: {e}"
            _emit("preflight_failed", {
                "plan_id": plan_id,
                "stage": "scout",
                "error": out.error_message,
            })
            out.total_seconds = round(time.monotonic() - t0, 2)
            return out

    if existing_map is None:
        out.error_message = (
            "no AppMap available; refinement skipped"
        )
        out.status = "partial"
        out.total_seconds = round(time.monotonic() - t0, 2)
        _emit("preflight_completed", {
            "plan_id": plan_id,
            "status": out.status,
            "error": out.error_message,
            "seconds": out.total_seconds,
        })
        return out

    if is_cancelled and is_cancelled():
        out.error_message = "cancelled mid-preflight"
        return out

    # ── Step 3: Refine.
    if skip_refine:
        out.notes.append("refine skipped per caller flag")
    else:
        _emit("preflight_refine_started", {
            "plan_id": plan_id,
            "appmap_modules": len(existing_map.modules),
            "appmap_create_flows": len(existing_map.create_flows),
        })
        try:
            refine_res = refine_plan(
                db,
                plan_id=plan_id,
                provider=provider,
                cheap_provider=cheap_provider,
                on_escalate=on_escalate,
                emit_event=emit_event,
            )
        except Exception as e:
            logger.exception(
                "preflight refine failed for plan %s", plan_id,
            )
            out.error_message = f"refine exception: {e}"
            _emit("preflight_failed", {
                "plan_id": plan_id,
                "stage": "refine",
                "error": out.error_message,
            })
            out.total_seconds = round(time.monotonic() - t0, 2)
            return out

        if refine_res.error_message:
            out.error_message = refine_res.error_message
            _emit("preflight_failed", {
                "plan_id": plan_id,
                "stage": "refine",
                "error": out.error_message,
            })
            out.total_seconds = round(time.monotonic() - t0, 2)
            return out

        out.refine_ran = True
        out.new_version_id = refine_res.new_version_id
        out.refined_submodules = len(refine_res.submodules)
        out.input_tokens += refine_res.total_input_tokens
        out.output_tokens += refine_res.total_output_tokens
        for rs in refine_res.submodules:
            for s in rs.steps:
                if s.change_kind == "rewritten":
                    out.refined_rewritten += 1
                elif s.change_kind == "added":
                    out.refined_added += 1
                elif s.change_kind == "flagged_missing":
                    out.refined_flagged_missing += 1
        _emit("preflight_refine_completed", {
            "plan_id": plan_id,
            "new_version_id": out.new_version_id,
            "submodules": out.refined_submodules,
            "rewritten": out.refined_rewritten,
            "added": out.refined_added,
            "flagged_missing": out.refined_flagged_missing,
        })

    # ── Step 4: Activate.
    target_version_id = out.new_version_id or current_version_id
    if (
        not skip_activation
        and target_version_id is not None
        and target_version_id != current_version_id
    ):
        try:
            _emit("preflight_activation_started", {
                "plan_id": plan_id,
                "version_id": target_version_id,
            })
            counts = apply_tc_version_to_live(
                db,
                plan_id=plan_id,
                version_id=target_version_id,
            )
            plan.current_tc_version_id = target_version_id
            db.flush()
            out.activated_version_id = target_version_id
            _emit("preflight_activation_completed", {
                "plan_id": plan_id,
                "version_id": target_version_id,
                "nodes_created": counts.get("created", 0),
                "nodes_removed": counts.get("removed", 0),
            })
        except Exception as e:
            logger.exception(
                "preflight activation failed for plan %s", plan_id,
            )
            out.error_message = f"activation exception: {e}"
            _emit("preflight_failed", {
                "plan_id": plan_id,
                "stage": "activation",
                "error": out.error_message,
            })
            out.total_seconds = round(time.monotonic() - t0, 2)
            return out
    elif target_version_id is not None:
        out.activated_version_id = target_version_id

    try:
        db.commit()
    except Exception as e:
        logger.warning("preflight commit failed: %s", e)
        try:
            db.rollback()
        except Exception:
            pass

    out.status = "completed"
    out.total_seconds = round(time.monotonic() - t0, 2)
    _emit("preflight_completed", {
        "plan_id": plan_id,
        "status": out.status,
        "version_id": out.activated_version_id,
        "seconds": out.total_seconds,
        "scout_ran": out.scout_ran,
        "refine_ran": out.refine_ran,
        "refined_submodules": out.refined_submodules,
        "rewritten": out.refined_rewritten,
        "added": out.refined_added,
        "flagged_missing": out.refined_flagged_missing,
        "input_tokens": out.input_tokens,
        "output_tokens": out.output_tokens,
        "notes": out.notes,
    })
    return out


# ── Scout helper ──────────────────────────────────────────────────


def _scout_target(
    db: "Session",
    *,
    plan: Any,
    provider: "LLMProvider",
    cheap_provider: "LLMProvider | None",
    scout_depth: Literal["shallow", "deep"],
    headless: bool,
    emit_event: Callable[[str, dict], None] | None,
    is_cancelled: Callable[[], bool] | None,
    on_escalate: Callable[[str, str, str], None] | None,
) -> dict[str, Any] | None:
    """Open a Playwright session → auth_flow → authenticated_scout →
    consolidate + save AppMap. Returns a small dict of stats on success,
    None on failure.

    Headless by default; tester sees progress via live-feed events.
    """
    from app.agents.app_map import (  # noqa: PLC0415
        consolidate_app_map, save_app_map,
    )
    from app.agents.auth_flow import run_auth_loop  # noqa: PLC0415
    from app.agents.authenticated_scout import (  # noqa: PLC0415
        run_authenticated_scout,
    )
    from app.executor.browser import browser_session  # noqa: PLC0415

    target_url = (plan.target_url or "").strip()
    if not target_url:
        return None

    def _emit(t: str, d: dict) -> None:
        if emit_event is None:
            return
        try:
            emit_event(t, d)
        except Exception:
            pass

    _emit("preflight_scout_started", {
        "plan_id": plan.id,
        "target_url": target_url,
        "depth": scout_depth,
        "headless": headless,
    })

    input_tokens = 0
    output_tokens = 0

    try:
        with browser_session(headless=headless) as page:
            # Navigate to the plan's target_url. Generous timeout —
            # admin SPAs over Cloudflare tunnels can take ~10s.
            try:
                page.goto(target_url, timeout=30_000)
            except Exception as e:
                logger.warning(
                    "preflight: goto failed for %s: %s", target_url, e,
                )

            # If we landed on a login screen, drive auth.
            try:
                from app.agents.qa_agent import _looks_like_login_page  # noqa: PLC0415
                needs_auth = _looks_like_login_page(page)
            except Exception:
                # Fallback heuristic: URL contains 'login' or 'signin'.
                cur = (page.url or "").lower()
                needs_auth = (
                    "login" in cur or "signin" in cur or "auth" in cur
                )
            if needs_auth:
                _emit("preflight_scout_auth_started", {
                    "plan_id": plan.id,
                    "url": page.url,
                })
                try:
                    auth = run_auth_loop(
                        page,
                        plan=plan,
                        provider=provider,
                        cheap_provider=cheap_provider,
                        emit_event=emit_event,
                        on_escalate=on_escalate,
                        is_cancelled=is_cancelled,
                    )
                except Exception as e:
                    logger.exception("preflight auth_flow exception")
                    _emit("preflight_scout_auth_failed", {
                        "plan_id": plan.id,
                        "error": str(e)[:200],
                    })
                    return None
                input_tokens += getattr(auth, "input_tokens", 0) or 0
                output_tokens += getattr(auth, "output_tokens", 0) or 0
                if auth.status != "ok":
                    _emit("preflight_scout_auth_failed", {
                        "plan_id": plan.id,
                        "status": auth.status,
                        "error": auth.error_message or "",
                    })
                    return None
                _emit("preflight_scout_auth_completed", {
                    "plan_id": plan.id,
                    "url": page.url,
                    "iterations": auth.iterations,
                })

            # Now authenticated (or no auth was needed). Walk the surface.
            scout = run_authenticated_scout(
                page,
                target_url=target_url,
                depth=scout_depth,
                emit_event=emit_event,
                is_cancelled=is_cancelled,
            )
            if not scout.pages:
                _emit("preflight_scout_empty", {
                    "plan_id": plan.id,
                    "error": scout.error_message or "no pages captured",
                })
                return None

            input_tokens += scout.input_tokens
            output_tokens += scout.output_tokens

            # Consolidate via the strong-tier LLM call.
            app_map, in_tok, out_tok = consolidate_app_map(
                provider,
                scout_result=scout,
                cheap_provider=cheap_provider,
                on_escalate=on_escalate,
            )
            input_tokens += in_tok or 0
            output_tokens += out_tok or 0

            # Persist (overwrites any prior map for this target_url).
            try:
                save_app_map(
                    db,
                    target_url=target_url,
                    app_map=app_map,
                    source_run_id=None,
                )
                db.commit()
            except Exception as e:
                logger.warning(
                    "preflight: save_app_map failed: %s", e,
                )
                try:
                    db.rollback()
                except Exception:
                    pass

            create_surfaces = sum(
                1 for p in scout.pages if p.create_surface is not None
            )

            _emit("preflight_scout_completed", {
                "plan_id": plan.id,
                "pages": len(scout.pages),
                "create_surfaces": create_surfaces,
                "modules": len(app_map.modules),
                "create_flows": len(app_map.create_flows),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
            return {
                "pages": len(scout.pages),
                "create_surfaces": create_surfaces,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
    except Exception as e:
        logger.exception("preflight scout outer failure")
        _emit("preflight_scout_failed", {
            "plan_id": plan.id,
            "error": str(e)[:200],
        })
        return None
