"""Production-α.5 — Plan-scoped WorldState.

Carried across submodules within ONE run so submodule N can assert
preconditions set up by submodule N-1 (cart_count, logged_in_as,
current_url, etc.).

Storage is a JSON dict on ``agent_runs.world_state_json``. The
runtime mutates an in-process Python dict during the run and writes
back to the column at every submodule boundary so a crash mid-run
preserves the latest state in the report.

Conventions for keys
--------------------
The agent is allowed to invent any key it wants — this is intentional
flexibility for arbitrary apps. But these keys are RESERVED with
meanings the orchestrator + reporter understand:

- ``logged_in_as``   : str | None — username of the active session
- ``current_url``    : str — last known page URL
- ``cart_count``     : int — items in the active cart
- ``cart_items``     : list[str] — names of items in the active cart
- ``checkout_started`` : bool
- ``order_placed``   : bool
- ``last_search``    : str — most recent search query
- ``screens_visited`` : list[str] — paths walked this run

When the agent reads its prompt block, the orchestrator surfaces
these reserved keys first so the LLM can match them against the
goal's preconditions / postconditions cleanly.

Update behaviors
----------------
- Submodule's postconditions on success → merge into WorldState.
- Verify's signal-voting result → may update specific keys (e.g.
  "cart shows 1 item" → cart_count=1).
- Explicit ``set_world_state`` tool the agent can call (not added
  in this v1 — the prompt teaches it to use postconditions instead).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.models.agent_run import AgentRun

logger = logging.getLogger(__name__)


_RESERVED_KEYS = (
    # Phase E — structured identity / location keys. These are
    # auto-populated by ``record_auth_success`` and
    # ``record_current_page`` so the decomposer can read them as
    # guaranteed preconditions without re-checking the page.
    #
    #   auth_status        : "logged_in" | "logged_out" | "unknown"
    #   auth_identity      : dict (username, role, tenant, ...)
    #   current_page_path  : list[str] — e.g. ["Administration", "Roles"]
    #   current_url        : last observed URL (already used elsewhere)
    #
    # The agent's prompt builder lifts these into a dedicated
    # "KNOWN STATE" block — the decomposer doesn't need a screenshot
    # to know "I'm already logged in".
    "auth_status", "auth_identity", "current_page_path",
    # Legacy cart / e-commerce flow keys kept for back-compat.
    "logged_in_as", "current_url", "cart_count", "cart_items",
    "checkout_started", "order_placed", "last_search",
    "screens_visited",
    # Phase E — entities created during the run so subsequent
    # submodules can search for them (e.g. "the role created
    # 5 minutes ago"). Each entry: {kind, identity, created_at_url}.
    "entities_created",
)


# ── Phase E — structured state mutators ─────────────────────────


def record_auth_success(
    state: dict[str, Any],
    *,
    username: str,
    role: str | None = None,
    tenant: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Mark the agent as logged in. Called by ``auth_flow`` after a
    successful login submit + post-login screen detection. Idempotent
    — calling twice with the same username is a no-op."""
    state["auth_status"] = "logged_in"
    identity: dict[str, Any] = {"username": username}
    if role:
        identity["role"] = role
    if tenant:
        identity["tenant"] = tenant
    if extra:
        identity.update(extra)
    state["auth_identity"] = identity
    # Legacy key — kept in sync so existing prompts that read
    # ``logged_in_as`` keep working.
    state["logged_in_as"] = username


def record_current_page(
    state: dict[str, Any],
    *,
    url: str | None = None,
    breadcrumb: list[str] | None = None,
) -> None:
    """Update the agent's location tracking. Called from the agent
    loop after each settled observation (cheap; no LLM)."""
    if url:
        state["current_url"] = url
    if breadcrumb is not None:
        state["current_page_path"] = list(breadcrumb)


def record_entity_created(
    state: dict[str, Any],
    *,
    kind: str,
    identity: str,
    url: str | None = None,
) -> None:
    """Append to the entities_created log so subsequent submodules
    can search for what was just made (e.g. role name typed two
    submodules ago that needs to be assigned now)."""
    entries = state.get("entities_created")
    if not isinstance(entries, list):
        entries = []
    entries.append({
        "kind": kind,
        "identity": identity[:200],
        "created_at_url": url or state.get("current_url", ""),
    })
    # Cap the log at 50 to bound prompt size on long runs.
    state["entities_created"] = entries[-50:]


def load_world_state(run: "AgentRun") -> dict[str, Any]:
    """Read the current WorldState dict (or ``{}`` for legacy runs)."""
    raw = getattr(run, "world_state_json", None)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def save_world_state(
    db: "Session", run: "AgentRun", state: dict[str, Any],
) -> None:
    """Persist a copy of the in-process state to the run row.

    Runs at submodule boundaries; intentionally NOT every turn — that
    would thrash the JSON column for tiny updates with no commensurate
    benefit. The orchestrator's ``finally`` guard also flushes on run
    end so we don't lose state on crash.
    """
    try:
        run.world_state_json = dict(state)
        db.commit()
    except Exception as e:
        logger.warning(
            "WorldState save failed for run %s (non-fatal): %s",
            getattr(run, "id", "?"), e,
        )
        try:
            db.rollback()
        except Exception:
            pass


def format_for_prompt(state: dict[str, Any]) -> str:
    """Render WorldState as a compact prompt block.

    Reserved keys come first (the agent's prompt explicitly references
    them by name); user-defined keys come after. Empty state returns
    ``""`` so the caller can skip the block.
    """
    if not state:
        return ""
    lines: list[str] = []
    for k in _RESERVED_KEYS:
        if k in state:
            lines.append(f"  - {k}: {state[k]!r}")
    extras = sorted(set(state.keys()) - set(_RESERVED_KEYS))
    for k in extras:
        v = state[k]
        # Trim long values so the prompt stays bounded.
        sv = repr(v)
        if len(sv) > 120:
            sv = sv[:117] + "..."
        lines.append(f"  - {k}: {sv}")
    return "\n".join(lines)


def check_preconditions(
    state: dict[str, Any],
    preconditions: list[str],
) -> tuple[bool, list[str]]:
    """Evaluate preconditions against world state.

    Returns ``(all_satisfied, list_of_unsatisfied_strings)``. Empty
    preconditions → trivially satisfied. We don't try to interpret
    the FREE-TEXT precondition rigorously (that's an LLM-grade task
    handled by the agent's prompt context); this helper only checks
    the few RESERVED keys that map to common precondition phrasings:

    - "user is logged in" / "logged in" → state.get("logged_in_as")
    - "cart has at least one item" / "cart_count >= 1" → state.cart_count
    - "cart is empty" → state.cart_count == 0

    Anything else is reported as ``"unverified"`` — the agent reads
    the precondition text in its prompt and verifies it manually.
    """
    if not preconditions:
        return True, []

    unsatisfied: list[str] = []
    for cond in preconditions:
        text = (cond or "").lower()
        if not text:
            continue
        # Logged-in family.
        if (
            "logged in" in text
            or "is signed in" in text
            or "authenticated" in text
        ):
            if not state.get("logged_in_as"):
                unsatisfied.append(cond)
            continue
        # Cart family.
        if "cart" in text and "empty" in text:
            count = state.get("cart_count")
            if isinstance(count, int) and count != 0:
                unsatisfied.append(cond)
            continue
        if "cart" in text and (
            ">=" in text or "at least" in text
            or "one or more" in text or "non-empty" in text
            or "contains item" in text or "has item" in text
        ):
            count = state.get("cart_count")
            if not isinstance(count, int) or count < 1:
                unsatisfied.append(cond)
            continue
        # URL family.
        m = re.search(r"url contains ['\"]?([^'\"]+)['\"]?", text)
        if m:
            target = m.group(1).strip()
            cur = (state.get("current_url") or "").lower()
            if target not in cur:
                unsatisfied.append(cond)
            continue
        # Anything we couldn't classify is left to the agent —
        # it sees the condition text in its prompt + verifies
        # against the page.
    return len(unsatisfied) == 0, unsatisfied


def apply_postconditions(
    state: dict[str, Any],
    postconditions: list[str],
    *,
    current_url: str = "",
) -> None:
    """Mutate ``state`` to reflect postconditions the agent claimed
    were met. Same heuristic family as ``check_preconditions``.
    Symmetric design: the conditions a previous submodule asserts
    as MET become the next submodule's PRECONDITIONS that hold.
    """
    if not postconditions:
        return
    for cond in postconditions:
        text = (cond or "").lower()
        if not text:
            continue
        if "logged in" in text or "signed in" in text:
            # Don't invent a username — the auth flow sets this.
            if "logged_in_as" not in state:
                state["logged_in_as"] = "(post-login)"
            continue
        if "cart" in text and "empty" in text:
            state["cart_count"] = 0
            state["cart_items"] = []
            continue
        m_cnt = re.search(
            r"cart (?:has|contains|shows)\s+(\d+)\s*items?", text,
        )
        if m_cnt:
            state["cart_count"] = int(m_cnt.group(1))
            continue
        if (
            "cart" in text
            and (">=" in text or "at least" in text)
        ):
            cur = state.get("cart_count", 0)
            if not isinstance(cur, int) or cur < 1:
                state["cart_count"] = 1
            continue
        if "checkout" in text and "begun" in text:
            state["checkout_started"] = True
            continue
        if "order placed" in text or "purchase complete" in text:
            state["order_placed"] = True
            continue
    if current_url:
        state["current_url"] = current_url
