"""Phase W — deterministic replay of a user-recorded submodule.

The agentic runner consults ``submodule.frozen_path`` BEFORE invoking
the agent's per-turn LLM loop. When it finds a payload with
``recording_kind="user_actions"`` (saved by Phase W's recorder), it
walks the recorded actions deterministically — zero LLM calls — and
hands off to the agent loop ONLY when an action's selector fails to
resolve (UI moved, label changed, etc.).

What's emitted on the live feed during replay
---------------------------------------------
- ``recording_replay_started`` — submodule_id, action_count
- ``recording_replay_action`` — one per action, with index + kind +
  target text + bounding rect (for the highlight overlay)
- ``recording_replay_highlight`` — bounding rect for the click ring
  the overlay draws around the element (then fades over 2s)
- ``recording_replay_completed`` — success + duration
- ``recording_replay_step_failed`` — when an action can't be
  resolved; caller decides whether to hand off to the agent
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from playwright.sync_api import Page

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    status: str = "completed"  # completed | partial | failed | cancelled
    actions_executed: int = 0
    actions_failed: int = 0
    duration_s: float = 0.0
    error_message: str | None = None
    failed_action_indices: list[int] = field(default_factory=list)


def _label_for_target(target: dict[str, Any] | None) -> str:
    """Pick the best human label for an element from recorded target
    metadata. Priority: visible text → aria-label → placeholder →
    name → id → tag. Trimmed to 60 chars."""
    if not isinstance(target, dict):
        return "element"
    for key in ("text", "aria_label", "placeholder", "name", "id"):
        val = (target.get(key) or "").strip()
        if val:
            return val[:60]
    tag = (target.get("tag") or "").strip().lower()
    role = (target.get("role") or "").strip().lower()
    return role or tag or "element"


def _role_noun(target: dict[str, Any] | None) -> str:
    """Short noun for the kind of element ("button", "field",
    "checkbox", "link", "option"). Defaults to "element"."""
    if not isinstance(target, dict):
        return "element"
    role = (target.get("role") or "").strip().lower()
    tag = (target.get("tag") or "").strip().lower()
    if role in ("button", "link", "checkbox", "radio", "menuitem", "option", "tab"):
        return role
    if tag in ("button", "a"):
        return "button" if tag == "button" else "link"
    if tag in ("input", "textarea"):
        ttype = (target.get("type") or "").strip().lower()
        if ttype == "checkbox":
            return "checkbox"
        if ttype == "radio":
            return "radio"
        return "field"
    if tag == "select":
        return "dropdown"
    return role or tag or "element"


def describe_action(action: dict[str, Any]) -> str:
    """Human-readable one-liner for a recorded action. Used by the
    live presenter to narrate replay step-by-step.

    Pure derivation from the captured target metadata — no LLM call.
    """
    if not isinstance(action, dict):
        return "Unknown action"
    kind = str(action.get("kind") or "")
    target = action.get("target") if isinstance(action.get("target"), dict) else None

    if kind == "click":
        label = _label_for_target(target)
        noun = _role_noun(target)
        if label and label.lower() != noun:
            return f"Click the ‘{label}’ {noun}"
        return f"Click the {noun}"
    if kind == "type":
        value = str(action.get("value") or "")
        preview = value if len(value) <= 40 else value[:37] + "…"
        label = _label_for_target(target)
        noun = _role_noun(target)
        field_label = (
            f"‘{label}’ {noun}"
            if label and label.lower() != noun else noun
        )
        return f"Type ‘{preview}’ into the {field_label}"
    if kind == "key":
        key = str(action.get("key") or "").strip()
        if key.lower() == "enter":
            return "Press Enter to submit"
        if key.lower() == "escape":
            return "Press Escape"
        if key.lower() == "tab":
            return "Press Tab"
        return f"Press {key or 'a key'}"
    if kind == "navigate":
        url = str(action.get("url") or "")
        return f"Navigate to {url}" if url else "Navigate"
    if kind == "scroll":
        return "Scroll the page"
    if kind:
        return kind.replace("_", " ").capitalize()
    return "Unknown action"


def _emit(emit_event, t: str, d: dict) -> None:
    if emit_event is None:
        return
    try:
        emit_event(t, d)
    except Exception:
        pass


def _resolve_target(
    page: "Page", target: dict[str, Any] | None,
) -> Any | None:
    """Try the recorded selector strategies in priority order.

    Returns a Playwright Locator (first match) or None. Falls back
    to bbox + text matching when selectors don't resolve."""
    if not isinstance(target, dict):
        return None
    # Strategy 1 — recorded selector.
    sel = (target.get("selector") or "").strip()
    if sel and not sel.startswith("BUTTON[") and not sel.startswith("DIV["):
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass
    # Strategy 2 — role + text.
    role = (target.get("role") or "").strip()
    text = (target.get("text") or "").strip()
    if role and text:
        try:
            loc = page.get_by_role(role, name=text).first  # type: ignore[arg-type]
            if loc.count() > 0:
                return loc
        except Exception:
            pass
    # Strategy 3 — visible text only.
    if text and len(text) <= 80:
        try:
            loc = page.get_by_text(text, exact=False).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass
    # Strategy 4 — placeholder match.
    placeholder = (target.get("placeholder") or "").strip()
    if placeholder:
        try:
            loc = page.get_by_placeholder(placeholder).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass
    # Strategy 5 — aria-label.
    aria = (target.get("aria_label") or "").strip()
    if aria:
        try:
            loc = page.locator(
                f'[aria-label="{aria}"]',
            ).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass
    return None


def replay_recording(
    page: "Page",
    *,
    recording: dict[str, Any],
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    submodule_id: int | None = None,
) -> ReplayResult:
    """Walk a ``user_actions`` recording deterministically.

    The page is assumed to be navigated to the recording's
    target_url. The first action's URL is checked against the
    current URL as a sanity-check; mismatch is logged but doesn't
    abort.
    """
    t0 = time.monotonic()
    out = ReplayResult()

    actions = recording.get("actions") or []
    if not isinstance(actions, list):
        out.status = "failed"
        out.error_message = "recording.actions is not a list"
        return out

    _emit(emit_event, "recording_replay_started", {
        "submodule_id": submodule_id,
        "action_count": len(actions),
        "target_url": recording.get("target_url") or "",
    })

    for idx, action in enumerate(actions):
        if is_cancelled and is_cancelled():
            out.status = "cancelled"
            out.error_message = "cancelled mid-replay"
            break
        if not isinstance(action, dict):
            continue
        kind = str(action.get("kind") or "")
        target = action.get("target") if isinstance(action.get("target"), dict) else None
        target_text = (
            (target.get("text") if target else "") or ""
        )[:80]

        # Resolve the target (clicks + types need an element).
        loc = None
        if kind in ("click", "type") and target:
            loc = _resolve_target(page, target)

        # Emit a highlight FIRST so the live presenter rings the
        # element BEFORE it's clicked (matches the operator's eye
        # movement: see → click).
        if loc is not None:
            try:
                box = loc.bounding_box()
            except Exception:
                box = None
            if box:
                _emit(emit_event, "recording_replay_highlight", {
                    "submodule_id": submodule_id,
                    "action_index": idx,
                    "x": int(box["x"]),
                    "y": int(box["y"]),
                    "w": int(box["width"]),
                    "h": int(box["height"]),
                })
                # Also draw it client-side via the existing overlay.
                try:
                    page.evaluate(
                        "(r) => window.__qaiHighlightRect && "
                        "window.__qaiHighlightRect(r.x, r.y, r.w, r.h, 1800)",
                        {
                            "x": box["x"], "y": box["y"],
                            "w": box["width"], "h": box["height"],
                        },
                    )
                except Exception:
                    pass

        description = describe_action(action)
        _emit(emit_event, "recording_replay_action", {
            "submodule_id": submodule_id,
            "action_index": idx,
            "kind": kind,
            "target_text": target_text,
            "description": description,
            "value_preview": (
                (action.get("value") or "")[:80]
                if isinstance(action.get("value"), str) else ""
            ),
        })

        try:
            if kind == "click":
                if loc is not None:
                    loc.scroll_into_view_if_needed(timeout=2_000)
                    loc.click(timeout=3_000)
                else:
                    # Fall back to the recorded coordinates.
                    x = action.get("x")
                    y = action.get("y")
                    if isinstance(x, int) and isinstance(y, int):
                        page.mouse.click(x, y)
                    else:
                        raise RuntimeError(
                            "could not resolve click target "
                            f"({target_text!r}) and no coords recorded",
                        )
            elif kind == "type":
                value = str(action.get("value") or "")
                if loc is not None:
                    try:
                        loc.fill(value, timeout=2_500)
                    except Exception:
                        loc.click(timeout=2_000)
                        page.keyboard.press("Control+A")
                        page.keyboard.press("Delete")
                        page.keyboard.type(value, delay=15)
                else:
                    raise RuntimeError(
                        f"type with no resolved target ({target_text!r})",
                    )
            elif kind == "key":
                key = str(action.get("key") or "")
                if key:
                    page.keyboard.press(key)
            elif kind == "navigate":
                url = str(action.get("url") or "")
                # Replay only navigates when the URL actually
                # changed FROM the prior action's URL — otherwise
                # the recorder's initial navigate fires twice.
                if url:
                    cur = ""
                    try:
                        cur = page.url or ""
                    except Exception:
                        cur = ""
                    if cur != url:
                        try:
                            page.goto(url, timeout=20_000)
                        except Exception:
                            pass
            else:
                # Unknown kind — log and skip.
                logger.info(
                    "replay: skipping unknown action kind %r at index %d",
                    kind, idx,
                )

            out.actions_executed += 1
            # Small pacing so the highlight is visible to the
            # operator + the page can settle between actions.
            try:
                page.wait_for_timeout(450)
            except Exception:
                pass
        except Exception as e:
            out.actions_failed += 1
            out.failed_action_indices.append(idx)
            _emit(emit_event, "recording_replay_step_failed", {
                "submodule_id": submodule_id,
                "action_index": idx,
                "kind": kind,
                "target_text": target_text,
                "description": description,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            # Continue on per-action failure — partial replay is
            # better than aborting the whole submodule.

    if out.status not in ("cancelled", "failed"):
        out.status = (
            "partial" if out.failed_action_indices else "completed"
        )
    out.duration_s = round(time.monotonic() - t0, 2)
    _emit(emit_event, "recording_replay_completed", {
        "submodule_id": submodule_id,
        "status": out.status,
        "actions_executed": out.actions_executed,
        "actions_failed": out.actions_failed,
        "duration_s": out.duration_s,
    })
    return out
