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
    # Phase Z.5 — first frame captured during trace playback, as a
    # path RELATIVE to settings.screenshots_dir (e.g. "130/trace_
    # step_469_start.png"). Caller sets this on
    # ``execution_steps.screenshot_path`` so the report viewer shows
    # the trace's opening frame. None when screenshotting was
    # disabled or every capture attempt failed.
    first_screenshot_relpath: str | None = None
    screenshot_relpaths: list[str] = field(default_factory=list)


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


# Strings whose ``target.text`` carries no semantic value — usually
# the literal "on" off-state of a checkbox, or empty / whitespace.
# When the recorded text is in this set, text-based disambiguation
# can't help and we should fall back to coordinates / role.
_NON_SEMANTIC_TEXTS: set[str] = {
    "", "on", "off", " ", " ",
}


def _looks_generic_selector(sel: str) -> bool:
    """Return True when the recorded CSS selector is the kind that
    matches many elements on the page (no #id anchor, no attribute
    selector, no nth-of-type). E.g. ``button.p-ripple.p-button`` or
    ``span.layout-menuitem-text.ng-star-inserted``. Triggers the
    text-disambiguation path."""
    if not sel:
        return True
    s = sel.strip()
    # ID-anchored selectors are usually unique.
    if "#" in s:
        return False
    # Attribute selectors or nth-of-type usually narrow to one.
    if "[" in s and "=" in s:
        return False
    if ":nth" in s or ":first-of-type" in s or ":last-of-type" in s:
        return False
    # Pure-class / pure-tag selectors are the dangerous ones.
    return True


def _scope_of(page: "Page", target: dict[str, Any]) -> Any:
    """Return a Playwright Locator scoped to the recorded container
    or component anchor — for sub-page lookups. Falls back to the
    page itself when no anchor is present (legacy recordings)."""
    anchor = target.get("component_anchor")
    if isinstance(anchor, dict):
        sel = (anchor.get("selector") or "").strip()
        if sel:
            try:
                scope = page.locator(sel).first
                if scope.count() > 0:
                    return scope
            except Exception:
                pass
    container = target.get("container")
    if isinstance(container, dict):
        sel = (container.get("selector") or "").strip()
        if sel:
            try:
                scope = page.locator(sel).first
                if scope.count() > 0:
                    return scope
            except Exception:
                pass
        cname = (container.get("accessible_name") or "").strip()
        crole = (container.get("role") or "").strip()
        ctag = (container.get("tag") or "").strip()
        if cname and crole:
            try:
                scope = page.get_by_role(
                    crole, name=cname,  # type: ignore[arg-type]
                ).first
                if scope.count() > 0:
                    return scope
            except Exception:
                pass
        if cname and ctag:
            try:
                scope = page.locator(ctag).filter(has_text=cname).first
                if scope.count() > 0:
                    return scope
            except Exception:
                pass
    return page


def _resolve_target(
    page: "Page", target: dict[str, Any] | None,
) -> Any | None:
    """Resolve a recorded target metadata object to a Playwright
    Locator pointing at the SPECIFIC element the operator clicked.

    Strategy order (Tricentis-grade — most specific → least):

    Z.2 — framework-aware strategies (require enriched fingerprint
    from Phase Z.1 recorder; legacy recordings skip these):

    0a. **formControlName** — `[formcontrolname="email"]` — unique
        within a single form, robust to Angular class-hash churn.
    0b. **Component anchor + accessible name** — find the recorded
        anchor element, narrow search within its subtree to the
        named control. Mirrors Tosca's Module-bound replay.
    0c. **Container scope + accessible name** — when no component
        anchor was recorded, scope to the nearest semantic container
        (form / dialog / section / nav) and find by name there.
    0d. **Label association** — `page.get_by_label(labelText)` —
        Playwright's built-in label-for / aria-labelledby walker.

    Legacy strategies (fallback for old recordings or when Z.2 misses):

    1.  **#id selector** — recorded ``#name`` style IDs are usually
        unique.
    2.  **aria-label** — strong semantic anchor.
    3.  **Selector + text filter** — generic CSS classes
        disambiguated by visible text.
    4.  **Role + text** — when target.role is recorded.
    5.  **Tag-inferred role + text** — button/a/input tag → ARIA role.
    6.  **Visible text alone** — for unique-enough strings.
    7.  **Placeholder** — for input fields.
    8.  **Generic selector ``.first``** — last resort.
    """
    if not isinstance(target, dict):
        return None

    sel = (target.get("selector") or "").strip()
    text = (target.get("text") or "").strip()
    text_is_semantic = (
        text and text.lower() not in _NON_SEMANTIC_TEXTS and len(text) <= 80
    )
    role = (target.get("role") or "").strip().lower()
    tag = (target.get("tag") or "").strip().lower()
    aria = (target.get("aria_label") or "").strip()
    placeholder = (target.get("placeholder") or "").strip()

    # Z.1 fingerprint fields (empty on legacy recordings).
    accessible_name = (target.get("accessible_name") or "").strip()
    form_control = (target.get("form_control") or "").strip()
    label_text = (target.get("label_text") or "").strip()

    # ── Z.2 — Framework-aware strategies ─────────────────────────

    # 0a. formControlName — uniquely identifies an Angular form
    # control within its form. Survives class rename.
    if form_control:
        try:
            loc = page.locator(
                f'[formcontrolname="{form_control}"]',
            ).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass
        try:
            loc = page.locator(
                f'[ng-reflect-name="{form_control}"]',
            ).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # Scope to the recorded component anchor / container for the
    # next two strategies. ``_scope_of`` falls back to ``page`` when
    # no anchor was recorded so legacy recordings still work.
    scope = _scope_of(page, target)
    has_scope = scope is not page

    # 0b. Component anchor + accessible name. The accessible name is
    # the same string a screen reader announces — far more stable
    # than visible text (it includes aria-labels, label-for, etc.).
    if has_scope and accessible_name:
        if role:
            try:
                loc = scope.get_by_role(
                    role, name=accessible_name,  # type: ignore[arg-type]
                ).first
                if loc.count() > 0:
                    return loc
            except Exception:
                pass
        if tag in ("button", "a", "input", "textarea", "select"):
            inferred = {
                "button": "button", "a": "link",
                "input": "textbox", "textarea": "textbox",
                "select": "combobox",
            }.get(tag, "")
            if inferred:
                try:
                    loc = scope.get_by_role(
                        inferred, name=accessible_name,  # type: ignore[arg-type]
                    ).first
                    if loc.count() > 0:
                        return loc
                except Exception:
                    pass
        try:
            loc = scope.get_by_text(accessible_name, exact=False).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # 0c. Label association — for form inputs, Playwright's built-in
    # label resolver handles <label for=>, wrapping <label>, and
    # aria-labelledby in one go.
    if label_text:
        try:
            loc = page.get_by_label(label_text, exact=True).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass
        try:
            loc = page.get_by_label(label_text).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 1. #id selector (most reliable when present) ─────────────
    if sel.startswith("#") and " " not in sel and "," not in sel:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 2. aria-label exact match ─────────────────────────────────
    if aria:
        try:
            loc = page.get_by_label(aria, exact=True).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass
        try:
            loc = page.locator(f'[aria-label="{aria}"]').first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 3. Recorded selector + text disambiguation ────────────────
    # The hot path for generic Angular/PrimeNG class selectors:
    # narrow to elements whose visible text matches the recording.
    # When a container scope is available (Z.2 recordings), search
    # within it first — "find the Save button in the Add Role
    # Dialog" is far less ambiguous than "find any Save button on
    # the page".
    if sel and text_is_semantic:
        if has_scope:
            try:
                loc = scope.locator(sel).filter(has_text=text).first
                if loc.count() > 0:
                    return loc
            except Exception:
                pass
        try:
            loc = page.locator(sel).filter(has_text=text).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 4. Role + text ────────────────────────────────────────────
    if role and text_is_semantic:
        if has_scope:
            try:
                loc = scope.get_by_role(
                    role, name=text,  # type: ignore[arg-type]
                ).first
                if loc.count() > 0:
                    return loc
            except Exception:
                pass
        try:
            loc = page.get_by_role(role, name=text).first  # type: ignore[arg-type]
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 5. Tag → role inference + text ────────────────────────────
    inferred_role = None
    if tag == "button":
        inferred_role = "button"
    elif tag == "a":
        inferred_role = "link"
    elif tag in ("input", "textarea"):
        ttype = (target.get("type") or "").strip().lower()
        inferred_role = (
            "checkbox" if ttype == "checkbox"
            else "radio" if ttype == "radio"
            else "textbox"
        )
    if inferred_role and text_is_semantic:
        try:
            loc = page.get_by_role(
                inferred_role, name=text,  # type: ignore[arg-type]
            ).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 6. Visible text alone (only for non-generic strings) ─────
    # Search container-scoped first so we don't get cross-form text
    # collisions (e.g. two "Email" labels on the page).
    if text_is_semantic:
        if has_scope:
            try:
                loc = scope.get_by_text(text, exact=False).first
                if loc.count() > 0:
                    return loc
            except Exception:
                pass
        try:
            loc = page.get_by_text(text, exact=False).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 6b. ng-reflect-* attribute lookup ────────────────────────
    # Phase Z.3 — Angular's data-binding hints (``ng-reflect-X``)
    # are stable across class-hash churn and often carry semantic
    # value: router-link target, value binding, model name, etc.
    # Match on whichever recorded reflect attr has a value.
    ng_reflect = target.get("ng_reflect")
    if isinstance(ng_reflect, dict):
        for k, v in ng_reflect.items():
            if not k or not v:
                continue
            try:
                v_esc = str(v).replace('"', '\\"')
                attr_name = f"ng-reflect-{k}"
                attr_sel = f'[{attr_name}="{v_esc}"]'
                if has_scope:
                    loc = scope.locator(attr_sel).first
                    if loc.count() > 0:
                        return loc
                loc = page.locator(attr_sel).first
                if loc.count() > 0:
                    return loc
            except Exception:
                continue

    # ── 7. Placeholder match for inputs ──────────────────────────
    if placeholder:
        if has_scope:
            try:
                loc = scope.get_by_placeholder(placeholder).first
                if loc.count() > 0:
                    return loc
            except Exception:
                pass
        try:
            loc = page.get_by_placeholder(placeholder).first
            if loc.count() > 0:
                return loc
        except Exception:
            pass

    # ── 7b. Sibling-index disambiguation ─────────────────────────
    # Z.3 — when the recorded element was, say, "the 3rd input in
    # this form", and other strategies couldn't pin it down, use
    # the recorded position-within-parent. Only fires when we have
    # both a recorded sibling_index AND a scope to constrain to.
    sibling_index = target.get("sibling_index")
    if (
        has_scope
        and isinstance(sibling_index, int)
        and sibling_index >= 0
        and tag
    ):
        try:
            siblings = scope.locator(tag)
            if siblings.count() > sibling_index:
                return siblings.nth(sibling_index)
        except Exception:
            pass

    # ── 8. Bare selector ``.first`` (LAST resort) ────────────────
    # Only reached when nothing more specific resolved. Skip when
    # the selector is BUTTON[...] / DIV[...] legacy bracket syntax
    # (always wrong on Playwright).
    #
    # When NO semantic text was available (icons, SVGs, layout
    # wrappers, checkboxes with text="on"), we accept the bare
    # ``.first`` match — it lets Playwright's actionability check
    # run on the click and is usually correct for these cases (the
    # operator's recorded click landed on whatever .first matches
    # too, because there was no semantic anchor to differentiate).
    # When text WAS semantic but Strategy 3's text-filter found
    # nothing, returning bare ``.first`` would pick the wrong
    # element (the original sub-472 bug) — so we return None
    # instead and let the caller use coordinates.
    if sel and not sel.startswith("BUTTON[") and not sel.startswith("DIV["):
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                if not text_is_semantic:
                    return loc
                if not _looks_generic_selector(sel):
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
    self_heal_callback: Callable[
        [dict[str, Any], "Page", int, str], bool,
    ] | None = None,
    narrate_callback: Callable[
        [dict[str, Any], "Page", int, str], str,
    ] | None = None,
    run_id: int | None = None,
    step_id: int | None = None,
    screenshot_every: int = 5,
) -> ReplayResult:
    """Walk a ``user_actions`` recording deterministically.

    The page is assumed to be navigated to the recording's
    target_url. The first action's URL is checked against the
    current URL as a sanity-check; mismatch is logged but doesn't
    abort.

    ``self_heal_callback`` (Phase W.9): when a single action raises,
    the callback is invoked as ``cb(action, page, idx, error_str)``
    and given a chance to perform an equivalent action via a
    different strategy (typically a VL-assisted re-resolution).
    Return True to mark the action as healed (counted as executed,
    not failed); return False / raise to let it count as a failure.
    Recording stays the source of truth — agent only heals *this
    one action*, doesn't redo the submodule.

    ``run_id`` / ``step_id`` / ``screenshot_every`` (Phase Z.5):
    when both ids are provided, captures a screenshot of the page
    at trace start, after every Nth step (default 5), at the end,
    AND at every failure point. Saved under
    ``{screenshots_dir}/{run_id}/trace_step_{step_id}_{tag}.png``
    relative to the static mount, emitted as
    ``recording_replay_screenshot`` events the live presenter
    renders inline. First captured frame is returned as
    ``out.first_screenshot_relpath`` so the caller can set it on
    ``execution_steps.screenshot_path``.
    """
    t0 = time.monotonic()
    out = ReplayResult()

    actions = recording.get("actions") or []
    if not isinstance(actions, list):
        out.status = "failed"
        out.error_message = "recording.actions is not a list"
        return out

    # Phase Z.5 — screenshot helper. Captures the page to
    # ``{screenshots_dir}/{run_id}/trace_step_{step_id}_{tag}.png``
    # and emits a ``recording_replay_screenshot`` event with the
    # relative path so the live presenter can render a thumbnail.
    # Disabled when run_id / step_id aren't provided (legacy call
    # sites). All failures swallowed — screenshots are
    # nice-to-have telemetry, not load-bearing.
    screenshots_enabled = (
        run_id is not None and step_id is not None
    )
    screenshots_root = None
    if screenshots_enabled:
        try:
            from app.config import settings as _sx_settings  # noqa: PLC0415
            screenshots_root = _sx_settings.screenshots_dir
        except Exception:
            screenshots_enabled = False

    def _capture(tag: str, action_index: int | None = None) -> None:
        if not screenshots_enabled or screenshots_root is None:
            return
        try:
            rel = (
                f"{run_id}/trace_step_{step_id}_"
                f"{action_index if action_index is not None else 'na'}_"
                f"{tag}.png"
            )
            abs_path = screenshots_root / rel
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(abs_path), full_page=False)
            out.screenshot_relpaths.append(rel)
            if out.first_screenshot_relpath is None:
                out.first_screenshot_relpath = rel
            _emit(emit_event, "recording_replay_screenshot", {
                "submodule_id": submodule_id,
                "action_index": action_index,
                "tag": tag,
                "path": rel,
            })
        except Exception as exc:
            logger.debug(
                "trace screenshot failed (%s) on submodule %s "
                "action %s: %s",
                tag, submodule_id, action_index, exc,
            )

    _emit(emit_event, "recording_replay_started", {
        "submodule_id": submodule_id,
        "action_count": len(actions),
        "target_url": recording.get("target_url") or "",
    })
    _capture("start", None)

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
                    # Coordinate fallback for types — recorded x,y
                    # focuses the input, then we type via keyboard.
                    # The previous behaviour raised here; that was
                    # the dominant cause of cascading "type with no
                    # resolved target" failures when one earlier
                    # click missed and the subsequent inputs
                    # therefore weren't reachable by selector.
                    x = action.get("x")
                    y = action.get("y")
                    if isinstance(x, int) and isinstance(y, int):
                        page.mouse.click(x, y)
                        page.wait_for_timeout(120)
                        page.keyboard.press("Control+A")
                        page.keyboard.press("Delete")
                        page.keyboard.type(value, delay=15)
                    else:
                        raise RuntimeError(
                            f"type with no resolved target "
                            f"({target_text!r}) and no coords recorded",
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
            # Phase Z.5 — periodic + navigation-triggered screenshot.
            # Captures whenever (idx+1) divides screenshot_every,
            # AND whenever a navigation transition just landed (so
            # the report shows the new screen the operator reached).
            recorded_url_for_shot = (action.get("url") or "").strip()
            next_url_for_shot = ""
            if idx + 1 < len(actions):
                _nx = actions[idx + 1]
                if isinstance(_nx, dict):
                    next_url_for_shot = (_nx.get("url") or "").strip()
            triggered_nav = (
                bool(recorded_url_for_shot) and bool(next_url_for_shot)
                and recorded_url_for_shot != next_url_for_shot
            )
            if (
                screenshot_every > 0
                and ((idx + 1) % screenshot_every == 0 or triggered_nav)
            ):
                _capture("step", idx)
            # Phase AE — task-completion narration. Cheap-LLM-driven
            # past-tense summary of what was just accomplished, one
            # per action. Emitted as ``agent_task_completed`` so the
            # live presenter can render a checklist under each
            # submodule. Failures swallowed — narration is telemetry,
            # not load-bearing.
            if narrate_callback is not None:
                try:
                    task_narration = narrate_callback(
                        action, page, idx, description,
                    )
                except Exception as ncb_exc:
                    logger.debug(
                        "narrate_callback raised on idx %d: %s",
                        idx, ncb_exc,
                    )
                    task_narration = ""
                if task_narration:
                    _emit(emit_event, "agent_task_completed", {
                        "submodule_id": submodule_id,
                        "step_id": step_id,
                        "task_idx": idx,
                        "kind": kind,
                        "narration": task_narration,
                        "source": "trace",
                    })
            # Phase W.8 — navigation-aware settle between actions.
            #
            # If the recording shows the NEXT action happened on a
            # DIFFERENT URL than the current one, the action we just
            # performed triggered a navigation. Wait for the new
            # page to settle before the next action — otherwise the
            # next selector resolves on the OLD DOM and fails.
            #
            # We use ``domcontentloaded`` rather than ``networkidle``
            # because the latter blocks on long-polling XHRs (which
            # are common in admin apps) and would balloon the
            # per-step time.
            recorded_url = (action.get("url") or "").strip()
            next_url = ""
            if idx + 1 < len(actions):
                next_action = actions[idx + 1]
                if isinstance(next_action, dict):
                    next_url = (next_action.get("url") or "").strip()
            navigation_likely = (
                bool(recorded_url) and bool(next_url)
                and recorded_url != next_url
            )
            try:
                if navigation_likely:
                    try:
                        page.wait_for_load_state(
                            "domcontentloaded", timeout=4_000,
                        )
                    except Exception:
                        pass
                    page.wait_for_timeout(600)
                else:
                    page.wait_for_timeout(450)
            except Exception:
                pass
        except Exception as e:
            error_str = f"{type(e).__name__}: {str(e)[:200]}"
            # Phase W.9 — give the agent ONE shot to heal this
            # single action before counting it as a failure. The
            # callback receives the recorded action, the page, the
            # index, and the error string; it should perform the
            # equivalent action via its own strategy (typically
            # vision-assisted) and return True on success. Recording
            # stays canonical — we only let the agent fix the ONE
            # broken step, not redo the submodule.
            healed = False
            if self_heal_callback is not None:
                _emit(emit_event, "recording_replay_self_heal_attempting", {
                    "submodule_id": submodule_id,
                    "action_index": idx,
                    "kind": kind,
                    "description": description,
                    "error": error_str,
                })
                try:
                    healed = bool(
                        self_heal_callback(action, page, idx, error_str),
                    )
                except Exception as heal_exc:
                    logger.debug(
                        "self_heal_callback raised on idx %d: %s",
                        idx, heal_exc,
                    )
                    healed = False
                _emit(
                    emit_event,
                    "recording_replay_self_healed" if healed
                    else "recording_replay_self_heal_failed",
                    {
                        "submodule_id": submodule_id,
                        "action_index": idx,
                        "kind": kind,
                        "description": description,
                    },
                )
            if healed:
                out.actions_executed += 1
                # Settle briefly after a healed action too.
                try:
                    page.wait_for_timeout(450)
                except Exception:
                    pass
            else:
                out.actions_failed += 1
                out.failed_action_indices.append(idx)
                _emit(emit_event, "recording_replay_step_failed", {
                    "submodule_id": submodule_id,
                    "action_index": idx,
                    "kind": kind,
                    "target_text": target_text,
                    "description": description,
                    "error": error_str,
                })
                # Phase Z.5 — failure-frame capture. The screenshot
                # at the moment of failure is the most informative
                # piece of the report.
                _capture("fail", idx)
            # Continue on per-action failure — partial replay is
            # better than aborting the whole submodule.

    if out.status not in ("cancelled", "failed"):
        out.status = (
            "partial" if out.failed_action_indices else "completed"
        )
    _capture("end", len(actions) - 1 if actions else None)
    out.duration_s = round(time.monotonic() - t0, 2)
    _emit(emit_event, "recording_replay_completed", {
        "submodule_id": submodule_id,
        "status": out.status,
        "actions_executed": out.actions_executed,
        "actions_failed": out.actions_failed,
        "duration_s": out.duration_s,
    })
    return out
