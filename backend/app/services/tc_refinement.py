"""Phase C.2 — Per-submodule test-case refinement.

Takes the live TcNode tree, the plan's AppMap (built by the
authenticated Scout), and the BRD source-document chunks for each
submodule's ``source_requirement_ids``. Emits a NEW TcVersion with
``source="app_map_refined"`` containing patched snapshots.

Why per-submodule instead of plan-wide
--------------------------------------
The user explicitly asked the refiner to see ONLY the submodule's
relevant slice of the app — not the whole map. This keeps the
LLM's attention focused, prevents cross-bleed (e.g. "Role" submodule
borrowing patterns from the unrelated "Chainage" flow), and makes
the cost predictable: one strong-tier call per submodule (~7 for
Solar). Adjacent unrelated submodules don't influence each other.

Output categories (LLM emits per step within a submodule)
---------------------------------------------------------
- ``kept``: the original step is accurate against the actual UI
- ``rewritten``: the step's target wording / action / value needs
  updating to match what the app actually shows
- ``added``: the submodule is missing a step (e.g. the Display
  Name field the BRD didn't mention)
- ``flagged_missing``: the step references UI that the AppMap
  doesn't expose; user must decide to drop or block

The refiner can also REORDER steps and EDIT submodule descriptions
when the new flow is cleaner.

Cost
----
Per-submodule call: ~$0.005-0.015 strong-tier. Solar (7 submodules):
~$0.05-0.10 per refinement. Cached as a new TcVersion; subsequent
runs against this version are free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.agents.app_map import AppMap
    from app.llm.base import LLMProvider
    from app.models.tc_node import TcNode

logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────


_REFINED_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "original_tc_node_id": {
            "type": ["integer", "null"],
            "description": (
                "ID of the original TcNode this refined step "
                "corresponds to. NULL for newly-added steps."
            ),
        },
        "change_kind": {
            "type": "string",
            "enum": [
                "kept", "rewritten", "added", "flagged_missing",
            ],
        },
        "title": {"type": "string"},
        "description_md": {"type": "string"},
        "action_type": {"type": "string"},
        "target_hint": {"type": "string"},
        "narrative": {"type": "string"},
        "expected": {"type": "string"},
        "change_reason": {
            "type": "string",
            "description": (
                "Short explanation of why the change was made — "
                "rendered in the diff dialog so the user "
                "understands the refiner's intent."
            ),
        },
    },
    "required": [
        "original_tc_node_id", "change_kind", "title",
        "description_md", "action_type", "target_hint",
        "narrative", "expected", "change_reason",
    ],
    "additionalProperties": False,
}

SUBMODULE_REFINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "submodule_title": {"type": "string"},
        "submodule_description_md": {"type": "string"},
        "submodule_change_reason": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 0,
            "maxItems": 30,
            "items": _REFINED_STEP_SCHEMA,
        },
        "confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
        },
    },
    "required": [
        "submodule_title", "submodule_description_md",
        "submodule_change_reason", "steps", "confidence",
    ],
    "additionalProperties": False,
}


REFINER_SYSTEM_PROMPT = """You are a senior QA test author. You are given:

1. ONE submodule from a test plan, with its current title +
   description + ordered list of steps (each step has an
   original_tc_node_id, action_type, target_hint, narrative,
   expected, and required-data hints).
2. The relevant slice of the app's UI structure (modules,
   create_flows, fields, navigation labels) discovered by an
   authenticated Scout walk — call this the APP MAP.
3. The source BRD/FRD/Instructions chunks that this submodule's
   steps cite, so you can recheck author intent.

Your job: emit a REFINED step list that makes this submodule
actually executable against the real app. Per the four-category
contract below.

Categories
==========
- ``kept``: step accurate as-is. Copy the original fields. Set
  change_reason to something brief like "matches app".
- ``rewritten``: step's wording or target differs from the actual
  UI. Update target_hint / narrative / expected to use the EXACT
  labels from the APP MAP. Example: BRD said target_hint="Create
  Role" but APP MAP has trigger_label="+ Add New Role" — rewrite
  the target_hint. Set original_tc_node_id to the original step's id.
- ``added``: a step the BRD missed. New steps are usually fields
  the form has that the BRD didn't enumerate (Display Name,
  Description, conditional fields). Set original_tc_node_id to
  null. Use action_type="type" / "click" / "select" appropriately.
- ``flagged_missing``: a step the BRD requires BUT the APP MAP
  shows no surface for it. Don't drop the step — emit it with this
  category and a clear change_reason explaining what's absent. The
  user reviews these manually.

Rules
=====
- PRESERVE the test's intent. Don't invent goals the BRD didn't
  ask for. Reorder steps only when the original ordering is
  clearly wrong against the app (e.g. BRD says "click Save then
  type name" — flip it).
- USE EXACT LABELS from the APP MAP. If the trigger button is
  "+ Add New Role", target_hint should be ``text '+ Add New Role'``
  or similar Playwright-resolvable hint that contains that EXACT
  text. Same for submit labels, field names, etc.
- For create-flows, ALWAYS include a final verify-style step
  ("Verify the new entity appears in the list") if the BRD's
  expected outcome implies persistence. This catches the "Save
  clicked but no row created" failure mode at execution time.
- For permission trees / searchable lists / multi-tab forms, add
  the navigation sub-steps explicitly (e.g. "Expand the
  Administration permission group" before "Check the Roles
  read leaf").
- Keep step granularity small: ONE observable action per step.
  Don't combine "open drawer, fill all fields, click save" into
  one step.
- output ``confidence`` reflects how well the APP MAP covered
  this submodule's scope. < 0.6 means "the scout didn't capture
  enough relevant pages; manual review recommended".

Output strict JSON matching the schema. No prose outside the JSON.
"""


# ── Public API ────────────────────────────────────────────────────


@dataclass
class RefinedStep:
    original_tc_node_id: int | None
    change_kind: str
    title: str
    description_md: str
    action_type: str
    target_hint: str
    narrative: str
    expected: str
    change_reason: str


@dataclass
class RefinedSubmodule:
    submodule_id: int  # original tc_node id
    submodule_title: str
    submodule_description_md: str
    submodule_change_reason: str
    steps: list[RefinedStep] = field(default_factory=list)
    confidence: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None


def _scope_app_map_for_submodule(
    app_map: "AppMap",
    submodule: "TcNode",
) -> str:
    """Return the AppMap slice relevant to ``submodule`` as a
    prompt-ready string.

    Matching strategy:
    - Tokenize the submodule's title + description; keep words >= 4
      chars and not in a small stopword set.
    - For each module in the map, score by token overlap with its
      name + sections. Include modules whose score >= 1.
    - For each create_flow, include when entity / section_path
      shares a token with the submodule.
    - ALWAYS include cross_cutting_notes (they're tiny + global).
    - When NO module matches, include the FULL map so the refiner
      can flag the submodule as "no surface found".
    """
    text = " ".join(
        filter(None, [submodule.title, submodule.description_md])
    ).lower()
    stopwords = {
        "the", "a", "an", "of", "and", "or", "to", "in", "for",
        "with", "as", "verify", "create", "user", "step", "test",
    }
    tokens = {
        w for w in (
            t.strip(".,;:'\"()") for t in text.split()
        )
        if len(w) >= 4 and w not in stopwords
    }

    relevant_modules: list[Any] = []
    relevant_flows: list[Any] = []
    for mod in app_map.modules:
        bag = " ".join([mod.name] + list(mod.sections or [])).lower()
        if any(tok in bag for tok in tokens):
            relevant_modules.append(mod)
    for fl in app_map.create_flows:
        bag = " ".join(
            [fl.entity] + list(fl.section_path or [])
        ).lower()
        if any(tok in bag for tok in tokens):
            relevant_flows.append(fl)

    if not relevant_modules and not relevant_flows:
        # Fringe case — submodule keywords match nothing in the
        # map. Return the full map so the refiner can flag the
        # submodule for manual review.
        return app_map.format_for_prompt() + (
            "\n(scout couldn't match this submodule's scope; "
            "consider flagging missing steps with "
            "change_kind=\"flagged_missing\")"
        )

    lines: list[str] = []
    if relevant_modules:
        lines.append("MODULES (relevant):")
        for m in relevant_modules:
            sec = " | ".join(m.sections) if m.sections else ""
            lines.append(f"  - {m.name}" + (f"  → {sec}" if sec else ""))
            if m.notes:
                lines.append(f"    note: {m.notes}")
    if relevant_flows:
        lines.append("CREATE FLOWS (relevant):")
        for fl in relevant_flows:
            path = " > ".join(fl.section_path) or "(unknown)"
            lines.append(
                f"  - Create {fl.entity} at [{path}]: "
                f"trigger=\"{fl.trigger_label}\" "
                f"submit=\"{fl.submit_label}\""
            )
            if fl.fields:
                fld_str = ", ".join(
                    f"{f.label}({f.role})"
                    + ("*" if f.required else "")
                    for f in fl.fields[:12]
                )
                lines.append(f"    fields: {fld_str}")
            flags: list[str] = []
            if fl.list_has_search:
                flags.append("searchable list")
            if fl.has_permission_tree:
                flags.append("permission tree")
            if flags:
                lines.append(f"    flags: {', '.join(flags)}")
    if app_map.cross_cutting_notes:
        lines.append("PATTERNS:")
        for n in app_map.cross_cutting_notes[:6]:
            lines.append(f"  - {n}")
    return "\n".join(lines)


def _serialize_submodule_steps(steps: list["TcNode"]) -> str:
    """Render the submodule's current steps as a compact JSON-ish
    payload for the LLM prompt."""
    import json  # noqa: PLC0415

    out = []
    for s in steps:
        out.append({
            "tc_node_id": s.id,
            "title": s.title,
            "action_type": s.action_type or "",
            "target_hint": s.target_hint or "",
            "narrative": s.narrative or "",
            "expected": s.expected or "",
            "data_needs": s.data_needs_json or [],
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


def _load_brd_chunks_for_submodule(
    db: "Session",
    submodule: "TcNode",
) -> str:
    """Pull the BRD/FRD source chunks for the submodule's
    ``source_requirement_ids``. Best-effort — returns empty string
    on miss. The refiner uses these to verify author intent."""
    if not submodule.source_requirement_ids:
        return ""
    try:
        from app.models.requirement import Requirement  # noqa: PLC0415
        from sqlalchemy import select  # noqa: PLC0415
        rows = list(db.execute(
            select(Requirement).where(
                Requirement.id.in_(submodule.source_requirement_ids),
            ),
        ).scalars())
        if not rows:
            return ""
        return "\n".join(
            f"- [{r.id}] {(r.text or '')[:400]}"
            for r in rows
        )
    except Exception as e:
        logger.debug("BRD chunk load skipped: %s", e)
        return ""


def refine_submodule(
    provider: "LLMProvider",
    *,
    submodule: "TcNode",
    steps: list["TcNode"],
    app_map: "AppMap",
    db: "Session",
    cheap_provider: "LLMProvider | None" = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
) -> RefinedSubmodule:
    """Run ONE LLM call to refine one submodule's steps against the
    scoped AppMap slice.

    Returns a :class:`RefinedSubmodule` carrying the patched step
    list + cost telemetry. Empty step list on LLM failure (caller
    keeps the original submodule unchanged).
    """
    from app.llm.base import ChatMessage  # noqa: PLC0415
    from app.llm.router import (  # noqa: PLC0415
        LLMRole, call_for_role,
    )

    scoped_map = _scope_app_map_for_submodule(app_map, submodule)
    steps_payload = _serialize_submodule_steps(steps)
    brd_block = _load_brd_chunks_for_submodule(db, submodule)
    brd_section = (
        f"\nSOURCE BRD/FRD CHUNKS:\n{brd_block}\n"
        if brd_block else ""
    )

    user_text = (
        f"SUBMODULE TITLE: {submodule.title}\n"
        f"SUBMODULE DESCRIPTION: {submodule.description_md or '(none)'}\n\n"
        f"CURRENT STEPS:\n{steps_payload}\n\n"
        f"APP MAP (scoped to this submodule):\n{scoped_map}\n"
        f"{brd_section}\n"
        "Refine the steps per the rules. Use the categories "
        "consistently. Return strict JSON."
    )

    messages = [
        ChatMessage(role="system", content=REFINER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_text),
    ]

    def _validate(parsed: Any) -> bool:
        return (
            isinstance(parsed, dict)
            and isinstance(parsed.get("steps"), list)
        )

    out = RefinedSubmodule(
        submodule_id=submodule.id,
        submodule_title=submodule.title,
        submodule_description_md=submodule.description_md or "",
        submodule_change_reason="",
    )
    try:
        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.PLANNER,
            messages=messages,
            schema=SUBMODULE_REFINE_SCHEMA,
            schema_name="submodule_refinement",
            temperature=0.2,
            max_output_tokens=3000,
            validate=_validate,
            on_escalate=on_escalate,
        )
        chat = tiered.chat
        out.input_tokens = chat.input_tokens or 0
        out.output_tokens = chat.output_tokens or 0
    except Exception as e:
        out.error_message = f"{type(e).__name__}: {str(e)[:300]}"
        logger.warning(
            "refiner LLM call failed for submodule %s: %s",
            submodule.id, e,
        )
        return out

    parsed = chat.parsed
    if not isinstance(parsed, dict):
        out.error_message = f"unexpected parse shape: {type(parsed).__name__}"
        return out

    out.submodule_title = str(
        parsed.get("submodule_title") or submodule.title,
    )[:512]
    out.submodule_description_md = str(
        parsed.get("submodule_description_md") or submodule.description_md or "",
    )
    out.submodule_change_reason = str(
        parsed.get("submodule_change_reason") or "",
    )[:400]
    try:
        out.confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        out.confidence = 0.0

    raw_steps = parsed.get("steps") or []
    refined_steps: list[RefinedStep] = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        ck = str(s.get("change_kind") or "")
        if ck not in ("kept", "rewritten", "added", "flagged_missing"):
            continue
        otn_raw = s.get("original_tc_node_id")
        try:
            otn = int(otn_raw) if otn_raw is not None else None
        except (TypeError, ValueError):
            otn = None
        refined_steps.append(RefinedStep(
            original_tc_node_id=otn,
            change_kind=ck,
            title=str(s.get("title", ""))[:512],
            description_md=str(s.get("description_md", "")),
            action_type=str(s.get("action_type", ""))[:64],
            target_hint=str(s.get("target_hint", "")),
            narrative=str(s.get("narrative", "")),
            expected=str(s.get("expected", "")),
            change_reason=str(s.get("change_reason", ""))[:400],
        ))
    out.steps = refined_steps
    return out


# ── Plan-level orchestration ──────────────────────────────────────


@dataclass
class RefinementResult:
    """Plan-wide refinement output. One entry per submodule."""
    plan_id: int
    submodules: list[RefinedSubmodule] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    error_message: str | None = None
    version_id: int | None = None
    version_number: int | None = None


def _snapshot_live_tree_as_baseline(
    db: "Session",
    *,
    plan_id: int,
    label: str = "v1 (BRD initial)",
) -> int | None:
    """Snapshot the current live TcNode tree as a ``brd_initial``
    version. Called automatically before the FIRST refinement so the
    original BRD-derived test cases stay recoverable after the live
    tree is overwritten by ``apply_tc_version_to_live``.

    Idempotent — if the plan already has any version, returns None
    without creating a new one.
    """
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.tc_version import (  # noqa: PLC0415
        TcVersion, TcNodeSnapshot,
    )
    from sqlalchemy import select, func  # noqa: PLC0415

    existing = db.execute(
        select(func.count()).select_from(TcVersion)
        .where(TcVersion.plan_id == plan_id),
    ).scalar()
    if existing and existing > 0:
        return None

    nodes = list(db.execute(
        select(TcNode)
        .where(TcNode.plan_id == plan_id)
        .order_by(TcNode.depth, TcNode.parent_id, TcNode.ordinal),
    ).scalars())
    if not nodes:
        return None

    version = TcVersion(
        plan_id=plan_id,
        version_number=1,
        source="brd_initial",
        label=label,
        notes_json={
            "auto_created": True,
            "reason": "baseline snapshot before first refinement",
        },
    )
    db.add(version)
    db.flush()

    snap_by_tc_id: dict[int, TcNodeSnapshot] = {}
    # Insert in depth order so parents exist before children.
    for n in nodes:
        parent_snap = None
        if n.parent_id is not None:
            parent_snap = snap_by_tc_id.get(n.parent_id)
        snap = TcNodeSnapshot(
            tc_version_id=version.id,
            original_tc_node_id=n.id,
            parent_snapshot_id=parent_snap.id if parent_snap else None,
            kind=n.kind,
            ordinal=n.ordinal,
            depth=n.depth,
            path_cached=n.path_cached or n.title,
            title=n.title,
            description_md=n.description_md,
            action_type=n.action_type,
            target_hint=n.target_hint,
            narrative=n.narrative,
            expected=n.expected,
            data_needs_json=n.data_needs_json,
            selectable_default=n.selectable_default,
            change_kind="kept",
            change_reason=None,
        )
        db.add(snap)
        db.flush()
        snap_by_tc_id[n.id] = snap

    db.commit()
    return version.id


def apply_tc_version_to_live(
    db: "Session",
    *,
    plan_id: int,
    version_id: int,
) -> dict[str, int]:
    """Overwrite the live TcNode tree with the snapshot tree from
    ``version_id``. Per the user's chosen semantics ("overwrite with
    audit trail"), the test-cases viewer always shows the active
    version; older versions remain queryable via the snapshot tables.

    Strategy:
    1. Load the version's snapshots in depth order.
    2. Delete the live TcNode rows for this plan (CASCADE removes
       children + the per-node ``frozen_path`` blob — refinement
       invalidates frozen paths anyway).
    3. INSERT new TcNodes mirroring the snapshots. Where a snapshot
       had an ``original_tc_node_id``, we PRESERVE that id so existing
       references (execution_steps.tc_node_id history, etc.) stay
       valid. Added rows get fresh ids.
    4. Update ``execution_steps.tc_node_id`` references aren't touched
       — they point at historical ids that may or may not still
       exist. Reports tolerate missing tc_nodes (they read from
       path_cached snapshot).

    Returns counts of {created, updated, removed} for the audit log.
    """
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.tc_version import (  # noqa: PLC0415
        TcVersion, TcNodeSnapshot,
    )
    from sqlalchemy import select  # noqa: PLC0415

    version = db.get(TcVersion, version_id)
    if version is None or version.plan_id != plan_id:
        raise ValueError(
            f"version {version_id} not found for plan {plan_id}",
        )

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
        return {"created": 0, "updated": 0, "removed": 0}

    # Capture project_id from the existing tree (or plan).
    existing_live = list(db.execute(
        select(TcNode).where(TcNode.plan_id == plan_id),
    ).scalars())
    project_id = (
        existing_live[0].project_id if existing_live
        else _project_id_for_plan(db, plan_id)
    )

    removed_count = 0
    for n in existing_live:
        db.delete(n)
        removed_count += 1
    db.flush()

    # Insert in depth order so parents exist before children.
    snap_to_live: dict[int, TcNode] = {}
    created = 0
    for snap in snapshots:
        parent_live = None
        if snap.parent_snapshot_id is not None:
            parent_live = snap_to_live.get(snap.parent_snapshot_id)
        node = TcNode(
            project_id=project_id,
            plan_id=plan_id,
            parent_id=parent_live.id if parent_live else None,
            kind=snap.kind,
            ordinal=snap.ordinal,
            depth=snap.depth,
            path_cached=snap.path_cached,
            title=snap.title,
            description_md=snap.description_md,
            action_type=snap.action_type,
            target_hint=snap.target_hint,
            narrative=snap.narrative,
            expected=snap.expected,
            data_needs_json=snap.data_needs_json,
            selectable_default=snap.selectable_default,
            status="draft",
            source_requirement_ids=[],
            frozen_path=None,  # invalidated by refinement
        )
        db.add(node)
        db.flush()  # need node.id for children
        snap_to_live[snap.id] = node
        created += 1

    return {
        "created": created,
        "updated": 0,
        "removed": removed_count,
    }


def _project_id_for_plan(db: "Session", plan_id: int) -> int:
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    plan = db.get(TestPlan, plan_id)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found")
    return plan.project_id


def refine_plan(
    db: "Session",
    *,
    plan_id: int,
    provider: "LLMProvider",
    cheap_provider: "LLMProvider | None" = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
    emit_event: Callable[[str, dict], None] | None = None,
) -> RefinementResult:
    """Run refinement across every submodule in a plan, persist as
    a new TcVersion, and return the result for the dialog.

    Steps:
    1. Load plan + AppMap (404-equivalent if no map yet).
    2. Walk the live TcNode tree; for each submodule, call
       :func:`refine_submodule`.
    3. Persist results as a new ``TcVersion`` row + per-node
       ``TcNodeSnapshot`` rows. The version's ``source`` is
       ``app_map_refined``; ``version_number`` is max(existing)+1.
    4. Do NOT auto-set ``current_tc_version_id`` — the dialog
       prompts the user to confirm before pointing the plan at
       this version.
    """
    from app.agents.app_map import load_app_map  # noqa: PLC0415
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.tc_version import (  # noqa: PLC0415
        TcVersion, TcNodeSnapshot,
    )
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from sqlalchemy import select, func  # noqa: PLC0415

    plan = db.get(TestPlan, plan_id)
    if plan is None:
        return RefinementResult(
            plan_id=plan_id,
            error_message=f"plan {plan_id} not found",
        )
    app_map = load_app_map(db, target_url=plan.target_url or "")
    if app_map is None:
        return RefinementResult(
            plan_id=plan_id,
            error_message=(
                "no app map for this plan's target_url yet — run "
                "agentic mode once OR click 'Scout this app' first"
            ),
        )

    # Auto-snapshot v1 (brd_initial) BEFORE creating v2 so the user
    # can always roll back to the BRD-derived baseline. No-op if any
    # version already exists for this plan.
    try:
        baseline_id = _snapshot_live_tree_as_baseline(
            db, plan_id=plan_id,
        )
        if baseline_id:
            logger.info(
                "auto-created baseline TcVersion %s for plan %s "
                "before first refinement", baseline_id, plan_id,
            )
    except Exception as e:
        logger.warning(
            "baseline snapshot failed (non-fatal): %s", e,
        )

    def _emit(t: str, d: dict) -> None:
        if emit_event:
            try:
                emit_event(t, d)
            except Exception:
                pass

    # Load the live TcNode tree, grouped by submodule.
    nodes = list(db.execute(
        select(TcNode)
        .where(TcNode.plan_id == plan_id)
        .order_by(TcNode.depth, TcNode.parent_id, TcNode.ordinal),
    ).scalars())
    by_parent: dict[int | None, list[TcNode]] = {}
    by_id: dict[int, TcNode] = {n.id: n for n in nodes}
    for n in nodes:
        by_parent.setdefault(n.parent_id, []).append(n)

    submodules = [n for n in nodes if n.kind == "submodule"]

    out = RefinementResult(plan_id=plan_id)
    refined_by_submodule: dict[int, RefinedSubmodule] = {}

    _emit("tc_refinement_started", {
        "plan_id": plan_id,
        "submodule_count": len(submodules),
    })
    for sm in submodules:
        steps = [
            c for c in (by_parent.get(sm.id) or [])
            if c.kind == "step"
        ]
        _emit("tc_refinement_submodule_started", {
            "plan_id": plan_id,
            "submodule_id": sm.id,
            "title": sm.title,
            "step_count": len(steps),
        })
        rs = refine_submodule(
            provider,
            submodule=sm,
            steps=steps,
            app_map=app_map,
            db=db,
            cheap_provider=cheap_provider,
            on_escalate=on_escalate,
        )
        out.submodules.append(rs)
        refined_by_submodule[sm.id] = rs
        out.total_input_tokens += rs.input_tokens
        out.total_output_tokens += rs.output_tokens
        _emit("tc_refinement_submodule_completed", {
            "plan_id": plan_id,
            "submodule_id": sm.id,
            "step_count": len(rs.steps),
            "kept": sum(1 for s in rs.steps if s.change_kind == "kept"),
            "rewritten": sum(
                1 for s in rs.steps if s.change_kind == "rewritten"
            ),
            "added": sum(1 for s in rs.steps if s.change_kind == "added"),
            "flagged_missing": sum(
                1 for s in rs.steps if s.change_kind == "flagged_missing"
            ),
            "confidence": rs.confidence,
            "error": rs.error_message,
        })

    # Persist as a new TcVersion + snapshots.
    next_version = (
        db.execute(
            select(func.max(TcVersion.version_number))
            .where(TcVersion.plan_id == plan_id),
        ).scalar() or 0
    ) + 1
    version = TcVersion(
        plan_id=plan_id,
        version_number=next_version,
        source="app_map_refined",
        label=f"v{next_version} (app-map refined)",
        notes_json={
            "submodule_count": len(submodules),
            "errors": [
                {"submodule_id": rs.submodule_id, "error": rs.error_message}
                for rs in out.submodules if rs.error_message
            ],
        },
        source_app_map_run_id=None,
    )
    db.add(version)
    db.flush()  # need version.id

    # Snapshot modules first so submodules can FK to them.
    snap_by_node_id: dict[int, TcNodeSnapshot] = {}
    modules = [n for n in nodes if n.kind == "module"]
    for mod in modules:
        snap = TcNodeSnapshot(
            tc_version_id=version.id,
            original_tc_node_id=mod.id,
            parent_snapshot_id=None,
            kind="module",
            ordinal=mod.ordinal,
            depth=mod.depth,
            path_cached=mod.path_cached or mod.title,
            title=mod.title,
            description_md=mod.description_md,
            action_type=None,
            target_hint=None,
            narrative=None,
            expected=None,
            data_needs_json=None,
            selectable_default=mod.selectable_default,
            change_kind="kept",
            change_reason=None,
        )
        db.add(snap)
        db.flush()
        snap_by_node_id[mod.id] = snap

    for sm in submodules:
        rs = refined_by_submodule.get(sm.id)
        parent_snap = snap_by_node_id.get(sm.parent_id) if sm.parent_id else None
        sm_title = rs.submodule_title if rs else sm.title
        sm_desc = (
            rs.submodule_description_md if rs else sm.description_md
        )
        sm_reason = rs.submodule_change_reason if rs else None
        sm_change_kind = (
            "rewritten"
            if (
                rs
                and (
                    sm_title != sm.title
                    or sm_desc != (sm.description_md or "")
                )
            )
            else "kept"
        )
        sm_snap = TcNodeSnapshot(
            tc_version_id=version.id,
            original_tc_node_id=sm.id,
            parent_snapshot_id=parent_snap.id if parent_snap else None,
            kind="submodule",
            ordinal=sm.ordinal,
            depth=sm.depth,
            path_cached=sm.path_cached or sm.title,
            title=sm_title,
            description_md=sm_desc,
            action_type=None,
            target_hint=None,
            narrative=None,
            expected=None,
            data_needs_json=None,
            selectable_default=sm.selectable_default,
            change_kind=sm_change_kind,
            change_reason=sm_reason,
        )
        db.add(sm_snap)
        db.flush()
        snap_by_node_id[sm.id] = sm_snap

        if rs is None or not rs.steps:
            # Refinement failed; mirror the original steps so the
            # version is still a complete, runnable tree.
            for step_idx, st in enumerate(
                c for c in (by_parent.get(sm.id) or [])
                if c.kind == "step"
            ):
                db.add(TcNodeSnapshot(
                    tc_version_id=version.id,
                    original_tc_node_id=st.id,
                    parent_snapshot_id=sm_snap.id,
                    kind="step",
                    ordinal=step_idx,
                    depth=st.depth,
                    path_cached=st.path_cached or st.title,
                    title=st.title,
                    description_md=st.description_md,
                    action_type=st.action_type,
                    target_hint=st.target_hint,
                    narrative=st.narrative,
                    expected=st.expected,
                    data_needs_json=st.data_needs_json,
                    selectable_default=st.selectable_default,
                    change_kind="kept",
                    change_reason="refinement failed — kept original",
                ))
            continue

        # Persist refined steps in emit order.
        for step_idx, refined_step in enumerate(rs.steps):
            orig = (
                by_id.get(refined_step.original_tc_node_id)
                if refined_step.original_tc_node_id is not None
                else None
            )
            db.add(TcNodeSnapshot(
                tc_version_id=version.id,
                original_tc_node_id=refined_step.original_tc_node_id,
                parent_snapshot_id=sm_snap.id,
                kind="step",
                ordinal=step_idx,
                depth=(orig.depth if orig else sm.depth + 1),
                path_cached=(
                    f"{sm.path_cached or sm.title} > {refined_step.title}"
                )[:2048],
                title=refined_step.title or (
                    orig.title if orig else "(untitled)"
                ),
                description_md=refined_step.description_md or None,
                action_type=refined_step.action_type or None,
                target_hint=refined_step.target_hint or None,
                narrative=refined_step.narrative or None,
                expected=refined_step.expected or None,
                data_needs_json=(orig.data_needs_json if orig else None),
                selectable_default=(
                    orig.selectable_default if orig else True
                ),
                change_kind=refined_step.change_kind,
                change_reason=refined_step.change_reason or None,
            ))

    db.commit()
    out.version_id = version.id
    out.version_number = version.version_number
    _emit("tc_refinement_completed", {
        "plan_id": plan_id,
        "version_id": version.id,
        "version_number": version.version_number,
        "input_tokens": out.total_input_tokens,
        "output_tokens": out.total_output_tokens,
    })
    return out
