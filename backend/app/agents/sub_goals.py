"""Phase A — vision-driven sub-goal decomposer + replanner.

The agent's outer goal comes from the BRD ("Create a new role and
assign all permissions"). The BRD doesn't know the actual UI, so
naive step-by-step execution fails when the real screen has
different button labels, missing controls, extra fields, etc.

This module fixes that by asking a vision-language model:
*"given this goal AND a screenshot of the current screen, what
are the 3-7 sub-goals to reach the objective?"* The agent then
works through those sub-goals sequentially. When one fails, we
replan (ask the VL again with the failure context) up to a
per-plan budget before falling through to HITL.

Two public entry points:

- :func:`decompose_goal` — called ONCE at submodule start with the
  initial screen. Produces the working sub-goal list.
- :func:`replan_sub_goals` — called after a sub-goal fails. The
  VL gets the original goal, the current (post-failure) screen,
  the failed sub-goal description + reason, and the sub-goals
  already completed; it returns a fresh list of 1-5 sub-goals to
  reach the remaining objective.

Decomposer cost: 1 strong-tier vision call per submodule.
Replanner cost: 1 cheap-tier vision call per failed sub-goal,
                capped at ``test_plan.max_replans_per_submodule``.

Output schema: strict JSON, OpenAI-compatible (every field
required, no additionalProperties), validated before return. Bad
output → empty list → caller falls through to the legacy single-
goal turn loop (so this module is non-fatal).

Sub-goal contract
-----------------
Each sub-goal is **observable on the screen**. We forbid
intent-style descriptions like "have the intent to create a role"
in the system prompt — the agent needs a UI-level outcome it can
verify. Examples that pass / fail:

  GOOD: "Open the Create Role drawer by clicking the +Add New
         Role button in the top-right of the role list"
  BAD:  "Want to make a role"  (no observable outcome)
  GOOD: "Type 'QA Full Access Role' into the Name input"
  BAD:  "Provide role name"  (no specific value, no anchor)

Numbered references
-------------------
When the screenshot is annotated with SoM boxes, sub-goal
descriptions are encouraged to reference "box N" — the VL
understands the link and the agent's planner can resolve the
box to its (x, y, w, h) at action time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from app.agents.app_map import AppMap
    from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


SubGoalStatus = Literal[
    "pending", "in_progress", "done", "failed", "skipped",
]


@dataclass
class RuntimeSubGoal:
    """One executable unit inside a submodule's agent loop.

    Distinct from ``goal.SubGoal`` (which is the BRD-time text-
    derived hint). This is the VISION-derived runtime contract
    the agent actually executes against. Persisted into
    ``execution_steps.details_json["sub_goals"]`` so the report
    timeline can render per-sub-goal pass/fail/skip rows.
    """

    id: str                  # "sg1", "sg2", ... (assigned by decomposer)
    description: str         # the verb-first observable outcome
    success_criterion: str   # one short signal that proves it done
    max_turns: int = 6       # cap to prevent any single sub-goal hogging
    status: SubGoalStatus = "pending"
    reason: str | None = None  # populated on failed / skipped
    started_at_turn: int | None = None
    ended_at_turn: int | None = None
    # Audit trail — the screenshot used to plan this sub-goal,
    # base64-encoded. Optional (truncated in JSON dump). Useful
    # when debugging "why did the decomposer say to click this".
    planning_screenshot_ref: str | None = None
    # When replanning produced this sub-goal, the iteration number
    # (1 = first replan, 2 = second). 0 = produced by the initial
    # decompose_goal call. Surfaced in the report timeline.
    replan_iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "success_criterion": self.success_criterion,
            "max_turns": self.max_turns,
            "status": self.status,
            "reason": self.reason,
            "started_at_turn": self.started_at_turn,
            "ended_at_turn": self.ended_at_turn,
            "replan_iteration": self.replan_iteration,
        }


@dataclass
class DecompositionResult:
    """Return type of decompose_goal / replan_sub_goals.

    Always returns at least an empty list — failure modes:
    - LLM raises: empty list + ``error_message`` set
    - LLM returns invalid JSON: empty list + ``error_message`` set
    - LLM returns 0 sub-goals: empty list (treated as "no decomposition needed")
    """

    sub_goals: list[RuntimeSubGoal] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None


# ── JSON schema (strict mode) ─────────────────────────────────────


_SUBGOAL_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": (
                "Verb-first sentence describing ONE discrete, "
                "observable outcome (e.g. 'Click the +Add New "
                "Role button in the top-right')."
            ),
        },
        "success_criterion": {
            "type": "string",
            "description": (
                "One short signal that proves this sub-goal is "
                "done (e.g. 'Create Role drawer is visible')."
            ),
        },
        "max_turns": {
            "type": "integer",
            "minimum": 1,
            "maximum": 12,
            "description": (
                "Upper bound on agent turns this sub-goal should "
                "need. Use 2-4 for a single click/type, 6-10 for "
                "multi-field form fills."
            ),
        },
    },
    "required": ["description", "success_criterion", "max_turns"],
    "additionalProperties": False,
}


DECOMPOSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_goals": {
            "type": "array",
            "minItems": 0,
            "maxItems": 7,
            "items": _SUBGOAL_OBJECT_SCHEMA,
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Short explanation of how you broke down the "
                "goal — useful in the live feed."
            ),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["sub_goals", "reasoning", "confidence"],
    "additionalProperties": False,
}


# ── System prompts ────────────────────────────────────────────────


DECOMPOSER_SYSTEM_PROMPT = """You are a senior QA test planner. You receive:
- A high-level GOAL written in business terms (from the BRD).
- A SCREENSHOT of the application's CURRENT screen.
- Optionally, an APP MAP — a structured summary of the modules,
  navigation, and create-flows the system already learned about
  this app during an authenticated Scout pass. When present, use
  it as the source of truth for module names, button labels, form
  field names, and where to find things. If the APP MAP says the
  Create button is labelled "+ Add New Role", that's what your
  sub-goal text says — even if the BRD said "Create Role".
- Optionally, knowledge-base notes about this app from prior runs.

Your job: produce 3-7 ordered SUB-GOALS the agent will execute
sequentially to reach the goal, BASED ON WHAT YOU SEE on the screen
AND what the APP MAP tells you about the app's structure.

Rules
=====
1. Each sub-goal must be OBSERVABLE on the actual UI. Reference
   the controls you SEE, not what the BRD says. If the BRD says
   "click Create Role" but the screen shows "+ Add New Role",
   the sub-goal says "click the +Add New Role button".
2. Each sub-goal must produce ONE discrete advance toward the
   goal. EXCEPTION: a create-form drawer is one atomic unit — emit
   ONE sub-goal "Fill the <entity> form via fill_form and submit"
   that bundles every field (incl. permission_tree / paginated
   tables) into a single fill_form tool call. DO NOT split a form
   into per-field sub-goals; the runtime's fill_form routine fills
   atomically, retries on validation, and submits. Per-field
   splitting wastes turns and almost always runs out before Save.
3. INCLUDE form fields and controls the BRD missed. If the BRD
   says "enter role name" but the form has Name + Display Name
   + Description, add sub-goals for the visible fields the agent
   will actually need to fill.
4. Order matters. The agent walks through your list top-down.
5. ``max_turns`` should be realistic — single click = 2-3,
   multi-field form = 6-10. Total across all sub-goals should
   fit a 30-turn budget.
6. If the screen is unrelated to the goal (e.g. you see a login
   screen but the goal is "create a role"), return ONE sub-goal:
   "navigate to the section where this goal can be performed".
7. APP-MAP RULES (when APP MAP is provided):
   a. NAVIGATION: For goals that require a specific section ("create
      a role"), the FIRST sub-goal is opening the module + section
      path from the map. Example: section_path=["Administration",
      "Roles"] → first sub-goal: "Open the Administration menu and
      click Roles".
   b. EXACT LABELS: When the map gives a trigger_label (e.g.
      "+ Add New Role"), your sub-goal text quotes it verbatim. Do
      NOT paraphrase to "click Create" — the actual button is
      "+ Add New Role".
   c. FORM FIELDS — BUNDLE INTO ONE fill_form. When the map's
      create-flow lists fields, the sub-goal that handles the form
      is ONE fill_form call with all REQUIRED fields (and any
      compound widgets — permission_tree, paginated_resource_table —
      included as additional entries in the same fill_form payload).
      The sub-goal description must literally start with
      "fill_form: fill the <Entity> drawer with …" so the planner
      picks fill_form (not a sequence of click/type). Example for
      a Role drawer with name + display + permission tree:
        description: "fill_form: fill the Create Role drawer with
                     Name='QA Role X', Display Name='QA Role X',
                     Description='…', Permissions=only:Administration,
                     Management; then submit via Save."
        success_criterion: "Role 'QA Role X' appears in the role
                            list"
      DO NOT emit "type the Name", "type the Display Name", "click
      each permission" as separate sub-goals.
   d. (Removed — see rule 7i for the authoritative permission-tree
      rule: bundle into ONE fill_form with role_hint=permission_tree.
      Do NOT emit "expand and check each leaf" sub-goals; that
      contradicts rule 7c / 7i and runs out of turns.)
   e. SEARCHABLE LIST: When list_has_search=true, sub-goals that
      require finding an entity in the list say "type the entity
      name into the search/filter input" rather than "scroll until
      you find it".
   f. SUBMIT BUTTON: Use the map's submit_label verbatim ("Save",
      "Create", etc.). Don't say "click Submit" if the actual
      button says "Save".
   g. VERIFY AFTER CREATE: For any goal that creates an entity, ALWAYS
      append a final sub-goal: "Verify the new <entity> appears in
      the list by searching for its name". Without this, the agent
      can't prove the create succeeded — a real QA would never trust
      "I clicked Save" as evidence of creation.
   h. DROPDOWN NAVIGATION: When a module is marked ``[dropdown]`` in
      the MODULES list (e.g. "Administration [dropdown] → Roles |
      Users"), navigation is TWO clicks, not one. Emit a sub-goal
      "Click the Administration top-bar menu to open the dropdown",
      then a separate sub-goal "Click Roles in the open dropdown".
      Reusing a single "Go to Administration > Roles" sub-goal loses
      the ordering and the agent will misclick.
   i. PERMISSION TREE — DETAILED: When the matched create-flow has
      ``[permission tree]`` flag, your sub-goal that handles the
      tree uses the ``fill_form`` tool with role_hint=permission_tree
      and an explicit value matching one of:
         - "all"               (tick every leaf)
         - "none"               (untick everything)
         - "only:Users,Roles"   (tick only leaves containing those tokens)
         - "all_except:Audit"  (tick everything except matching tokens)
      Pick "all" when the goal says "grant all permissions" or
      similar; pick "only:..." when the goal names specific modules.
      DO NOT emit a sub-goal "click each parent then each child" —
      that's brittle; the routine handles expansion + leaf clicking
      atomically.
   j. PAGINATED RESOURCE TABLE: When the create-flow has
      ``[paginated resource table]`` flag, your sub-goal for the
      access-grant step uses ``fill_form`` with
      role_hint=paginated_resource_table and a value of:
         - "all:read,update"
              (tick the read + update masters for every row across pages)
         - "specific:CH-0001:read,update;CH-0002:read"
              (tick only the named rows with their per-row actions)
         - "none"
              (untick everything visible)
      The routine walks pagination for you — do not emit a "click
      Next page" sub-goal between rows.
   k. CONDITIONAL SECTION ORDERING: When the create-flow has
      ``[conditional sections]`` flag and a section is listed
      ``"appears after \"<field>\""``, the sub-goal that fills
      ``<field>`` MUST come BEFORE the sub-goal that uses the
      conditional section's fields. Example: Solar's user-create
      drawer hides the Resource Access Control table until a role
      is selected — fill Role first, THEN the access table.

8. ENTITY REUSE (when KNOWN STATE lists recently created entities):
   a) SAME-KIND reuse: When the goal is to ACT ON an entity (assign,
      edit, delete a <kind>) and KNOWN STATE's "recently created
      entities" shows the SAME kind exists:
      - DO NOT emit a "create the X" sub-goal — it already exists.
      - The FIRST sub-goal opens the X list and searches for the
        previously created <kind>=<identity> (quote the identity
        verbatim from KNOWN STATE).
   b) CROSS-KIND reuse: When the goal's primary action is to CREATE
      a NEW entity (e.g. "Create user U") BUT the form references
      another entity that exists in KNOWN STATE (e.g. the user-
      create form has a Role dropdown, and KNOWN STATE shows a
      role was just created):
      - Use the EXACT identity from KNOWN STATE as the value for
        that field in fill_form.
      - Example: KNOWN STATE has ``role='QA Auto Role-616023'``;
        goal is "Create user Alice with that role". fill_form
        payload includes:
          {"label": "Role", "value": "QA Auto Role-616023",
           "role_hint": "custom_combobox"}
        NOT a made-up role name. NOT "the previously created role".
        The exact identity string, verbatim.
   c) When KNOWN STATE shows no relevant entity, fall through to
      regular create-flow planning.
   This prevents duplicate-name conflicts (same-kind) and prevents
   the planner from inventing values for fields that should reference
   real, just-created records (cross-kind).

9. GOAL-MODE (no step hints in the prompt):
   When the input contains a GOAL + SUCCESS CRITERIA but NO step
   hints / no executable test-case steps, treat the goal as the
   AUTHORED INTENT and decompose freely from:
     - The current screen (what you SEE)
     - The APP MAP (what the app supports)
     - The WORLD STATE (what's already true)
   Do NOT invent step-by-step prescription. Sub-goals should be
   OUTCOMES the agent can verify on screen, in the smallest set
   that reaches the goal. Examples:
     GOAL: "Verify a role can be created with all permissions"
       sg1: Open the Roles section (via Administration dropdown)
       sg2: Open the Create Role drawer
       sg3: Fill the role form (use fill_form with permission_tree
            value="all"); submit
       sg4: Verify the new role appears in the list
   This mode is INTENT-DRIVEN: the user trusts the agent + the
   AppMap to figure out HOW. Test cases are concepts, not scripts.

Output
======
Return strict JSON matching the schema. ``confidence`` is your
self-assessment of how well the sub-goal list matches the
screen. < 0.6 means "I'm not sure what's on this screen, the
agent should be conservative".
"""


REPLANNER_SYSTEM_PROMPT = """You are the SAME QA planner from the decomposer step.
A sub-goal you planned earlier FAILED. Now you see:
- The ORIGINAL goal.
- The SUB-GOALS already completed (✓) and any pending (○).
- The SUB-GOAL that FAILED, with the failure reason.
- A SCREENSHOT of the page AT THE MOMENT OF FAILURE.

Your job: produce a fresh, ordered list of 1-5 sub-goals to
continue the original goal from this new starting point.

Rules
=====
1. Do NOT repeat sub-goals already marked ✓ — assume those
   completed. Pick up from where the failure occurred.
2. If the failure was "button not found" but you SEE a similar
   button with a different label, your first sub-goal should
   click THAT button.
3. If the failure was "form validation rejected the input", your
   first sub-goal should fix the input value based on the error
   message visible in the screenshot.
4. If the screen is clearly broken / the goal looks unreachable
   (auth wall, feature missing, server error), return ONE
   sub-goal: "give up — flag this test case as blocked".
5. ``confidence`` reflects how recoverable the situation looks.
   < 0.4 = "I don't see a path forward".

Output
======
Same strict JSON schema as the decomposer.
"""


# ── Public API ────────────────────────────────────────────────────


def decompose_goal(
    provider: "LLMProvider",
    *,
    goal_description: str,
    goal_success_criteria: list[str],
    screenshot_bytes: bytes | None,
    akb_block: str = "",
    app_map: "AppMap | None" = None,
    frozen_sub_goals_hint: list[dict[str, Any]] | None = None,
    world_state: dict[str, Any] | None = None,
    cheap_provider: "LLMProvider | None" = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
) -> DecompositionResult:
    """Initial vision-driven decomposition. Called ONCE per submodule.

    Routes to the STRONG provider directly (this is a planner-tier
    call — accuracy matters more than cost). ``cheap_provider`` is
    accepted for symmetry with the rest of the agent helpers but
    unused unless we add a cheap-first variant later.

    Phase A.5 — ``app_map`` is the structured mindmap from the
    authenticated Scout pass. When present, the decomposer's prompt
    includes it as ground-truth context: real button labels, real
    form fields, real section paths. The agent's resulting sub-goals
    are anchored to the actual UI instead of paraphrased from the
    BRD.
    """
    from app.llm.base import ChatMessage  # noqa: PLC0415

    crit_block = "\n".join(f"  - {c}" for c in goal_success_criteria) or "  (none specified)"
    akb_section = (
        f"\n\nAPP KNOWLEDGE BASE (from prior runs):\n{akb_block}\n"
        if akb_block.strip() else ""
    )
    app_map_section = ""
    if app_map is not None:
        try:
            app_map_section = (
                "\n\nAPP MAP (from authenticated Scout — use as "
                "ground truth for navigation, labels, form fields):\n"
                + app_map.format_for_prompt()
                + "\n"
            )
        except Exception:
            app_map_section = ""

    # Phase B Step 3 — frozen sub-goals as decomposer hints. When a
    # prior run on this submodule passed and got frozen, the saved
    # segments come back here. Tell the LLM to RE-EMIT the same
    # breakdown when the screen still supports it — replay then
    # walks the frozen steps deterministically. When the LLM
    # deviates, replay falls through to agentic execution.
    frozen_hint_section = ""
    if frozen_sub_goals_hint:
        try:
            lines = []
            for h in frozen_sub_goals_hint:
                if not isinstance(h, dict):
                    continue
                desc = str(h.get("description", "")).strip()
                crit = str(h.get("success_criterion", "")).strip()
                mt = h.get("max_turns", 6)
                if not desc:
                    continue
                lines.append(
                    f"  - {desc}\n"
                    f"    success_criterion: {crit}\n"
                    f"    max_turns: {mt}"
                )
            if lines:
                frozen_hint_section = (
                    "\n\nFROZEN SUB-GOALS (a prior run passed this "
                    "submodule with the breakdown below — re-emit "
                    "the SAME sub-goals if the current screen still "
                    "supports them, in the SAME order, with the "
                    "SAME success_criterion text. The runtime "
                    "deterministically replays the proven steps "
                    "whenever your output matches. If the screen "
                    "has changed, plan freshly):\n"
                    + "\n".join(lines) + "\n"
                )
        except Exception:
            frozen_hint_section = ""

    # Phase E — structured WorldState block. When set, the decomposer
    # sees guaranteed preconditions ("user is logged in as admin",
    # "current page is Administration > Roles", "Role 'QA-1' was
    # created 2 submodules ago") so it doesn't waste sub-goals
    # re-asserting state we already have.
    world_state_section = ""
    if world_state:
        ws_lines: list[str] = []
        if world_state.get("auth_status") == "logged_in":
            identity = world_state.get("auth_identity") or {}
            who = identity.get("username") if isinstance(
                identity, dict,
            ) else None
            if who:
                ws_lines.append(
                    f"  - already logged in as {who}"
                    + (
                        f" (role={identity.get('role')})"
                        if isinstance(identity, dict)
                        and identity.get("role") else ""
                    )
                )
            else:
                ws_lines.append("  - already logged in")
        cur_path = world_state.get("current_page_path")
        if isinstance(cur_path, list) and cur_path:
            ws_lines.append(
                f"  - current page: {' > '.join(str(p) for p in cur_path)}"
            )
        cur_url = world_state.get("current_url")
        if isinstance(cur_url, str) and cur_url:
            ws_lines.append(f"  - current URL: {cur_url}")
        ents = world_state.get("entities_created")
        if isinstance(ents, list) and ents:
            # Show the LAST 5 created entities — most useful for
            # "now assign that role" / "now search for that user"
            # sub-goals.
            tail = ents[-5:]
            lines = ", ".join(
                f"{e.get('kind','?')}={e.get('identity','?')!r}"
                for e in tail if isinstance(e, dict)
            )
            if lines:
                ws_lines.append(
                    f"  - recently created entities: {lines}"
                )
        if ws_lines:
            world_state_section = (
                "\n\nKNOWN STATE (guaranteed — don't re-verify):\n"
                + "\n".join(ws_lines) + "\n"
            )

    user_text = (
        f"GOAL:\n  {goal_description}\n\n"
        f"SUCCESS CRITERIA:\n{crit_block}\n"
        f"{world_state_section}"
        f"{app_map_section}"
        f"{frozen_hint_section}"
        f"{akb_section}\n"
        "Look at the attached screenshot. Decompose the goal into "
        "3-7 ordered sub-goals based on what you actually see on "
        "this page AND the APP MAP above (when present). When KNOWN "
        "STATE says you're already logged in or already on the "
        "right page, DON'T emit sub-goals to do those again."
    )

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=DECOMPOSER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=user_text,
            image=screenshot_bytes,
        ) if screenshot_bytes is not None
        else ChatMessage(role="user", content=user_text),
    ]
    return _call_planner(
        provider=provider,
        cheap_provider=cheap_provider,
        messages=messages,
        on_escalate=on_escalate,
        replan_iteration=0,
    )


def replan_sub_goals(
    provider: "LLMProvider",
    *,
    goal_description: str,
    completed_sub_goals: list[RuntimeSubGoal],
    failed_sub_goal: RuntimeSubGoal,
    failure_reason: str,
    screenshot_bytes: bytes | None,
    cheap_provider: "LLMProvider | None" = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
    replan_iteration: int = 1,
    app_map: "AppMap | None" = None,
    world_state: dict[str, Any] | None = None,
) -> DecompositionResult:
    """Replan after a sub-goal fails. Routes to CHEAP tier first.

    ``replan_iteration`` is recorded on each returned sub-goal so
    the report timeline shows which replan produced it. ``app_map``
    (when present) is included in the prompt so the replanner can
    pick alternative trigger labels / section paths from ground truth.
    """
    from app.llm.base import ChatMessage  # noqa: PLC0415

    done_block = (
        "\n".join(f"  ✓ {sg.description}" for sg in completed_sub_goals)
        or "  (none — first sub-goal failed)"
    )
    app_map_section = ""
    if app_map is not None:
        try:
            app_map_section = (
                "\n\nAPP MAP (ground truth — use exact labels / paths):\n"
                + app_map.format_for_prompt() + "\n"
            )
        except Exception:
            app_map_section = ""
    # Phase E — guaranteed-state block. The replanner gets the same
    # short KNOWN STATE summary as the decomposer so it doesn't
    # waste recovery sub-goals re-asserting login or already-known
    # current_page_path.
    ws_section = ""
    if world_state:
        ws_bits: list[str] = []
        if world_state.get("auth_status") == "logged_in":
            ident = world_state.get("auth_identity") or {}
            who = ident.get("username") if isinstance(ident, dict) else None
            ws_bits.append(
                f"logged in as {who}" if who else "logged in"
            )
        cur_path = world_state.get("current_page_path")
        if isinstance(cur_path, list) and cur_path:
            ws_bits.append(
                "current page: " + " > ".join(str(p) for p in cur_path)
            )
        cur_url = world_state.get("current_url")
        if isinstance(cur_url, str) and cur_url:
            ws_bits.append(f"current URL: {cur_url}")
        if ws_bits:
            ws_section = (
                "\nKNOWN STATE: " + "; ".join(ws_bits) + "\n"
            )
    user_text = (
        f"ORIGINAL GOAL:\n  {goal_description}\n\n"
        f"SUB-GOALS COMPLETED:\n{done_block}\n\n"
        f"FAILED SUB-GOAL:\n  {failed_sub_goal.description}\n"
        f"  success_criterion: {failed_sub_goal.success_criterion}\n"
        f"  failure reason: {failure_reason}\n"
        f"{ws_section}"
        f"{app_map_section}\n"
        "The screenshot shows the page AT THE MOMENT OF FAILURE. "
        "Plan 1-5 fresh sub-goals to continue from here. Skip "
        "anything already completed."
    )
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=REPLANNER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=user_text,
            image=screenshot_bytes,
        ) if screenshot_bytes is not None
        else ChatMessage(role="user", content=user_text),
    ]
    return _call_planner(
        provider=provider,
        cheap_provider=cheap_provider,
        messages=messages,
        on_escalate=on_escalate,
        replan_iteration=replan_iteration,
    )


# ── Internal: shared LLM call + parse ────────────────────────────


def _call_planner(
    *,
    provider: "LLMProvider",
    cheap_provider: "LLMProvider | None",
    messages: list[Any],
    on_escalate: Callable[[str, str, str], None] | None,
    replan_iteration: int,
) -> DecompositionResult:
    """Shared LLM round-trip + JSON parse for both decompose / replan."""
    try:
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        def _validate(parsed: Any) -> bool:
            if not isinstance(parsed, dict):
                return False
            sgs = parsed.get("sub_goals")
            return isinstance(sgs, list)

        # Replanning uses GOAL_VERIFIER role (cheap-first with escalation);
        # initial decomposition uses PLANNER (strong-only) since we need
        # accuracy on the first split.
        role = LLMRole.GOAL_VERIFIER if replan_iteration > 0 else LLMRole.PLANNER
        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=role,
            messages=messages,
            schema=DECOMPOSITION_SCHEMA,
            schema_name="sub_goal_decomposition",
            temperature=0.2,
            max_output_tokens=1200,
            validate=_validate,
            on_escalate=on_escalate,
        )
        result = tiered.chat
    except Exception as e:
        logger.warning(
            "sub-goal decomposition LLM call failed: %s: %s",
            type(e).__name__, e,
        )
        return DecompositionResult(
            error_message=f"{type(e).__name__}: {str(e)[:300]}",
        )

    parsed = result.parsed
    if not isinstance(parsed, dict):
        return DecompositionResult(
            input_tokens=result.input_tokens or 0,
            output_tokens=result.output_tokens or 0,
            error_message=f"unexpected parse shape: {type(parsed).__name__}",
        )

    raw_sgs = parsed.get("sub_goals") or []
    sub_goals: list[RuntimeSubGoal] = []
    for i, sg in enumerate(raw_sgs, start=1):
        if not isinstance(sg, dict):
            continue
        desc = str(sg.get("description", "")).strip()
        crit = str(sg.get("success_criterion", "")).strip()
        if not desc or not crit:
            continue
        try:
            max_turns = int(sg.get("max_turns", 6))
        except (TypeError, ValueError):
            max_turns = 6
        max_turns = max(1, min(12, max_turns))
        sub_goals.append(RuntimeSubGoal(
            id=f"sg{i}",
            description=desc[:500],
            success_criterion=crit[:300],
            max_turns=max_turns,
            replan_iteration=replan_iteration,
        ))

    return DecompositionResult(
        sub_goals=sub_goals,
        input_tokens=result.input_tokens or 0,
        output_tokens=result.output_tokens or 0,
    )


# ── Phase A.5 — create→verify guarantor ──────────────────────────


_CREATE_KEYWORDS: tuple[str, ...] = (
    "create", "add", "new ", "register", " a new ",
    "make a", "set up",
)
_VERIFY_KEYWORDS: tuple[str, ...] = (
    "verify", "confirm", "check it appears", "appears in",
    "shows up", "is listed", "is present", "find it",
)


def ensure_create_verify_pattern(
    sub_goals: list[RuntimeSubGoal],
    *,
    goal_description: str,
    app_map: "AppMap | None" = None,
) -> tuple[list[RuntimeSubGoal], bool]:
    """Hardcoded guarantor: append a "verify in list" sub-goal when
    the goal looks like a create-flow and the decomposer forgot to.

    Why this exists
    ---------------
    Rule 7g in the decomposer prompt asks the LLM to append a verify
    sub-goal for create flows. The LLM gets this right MOST of the
    time but drifts under prompt pressure (long goals, dense app
    maps). This deterministic post-processor catches the misses so
    every create flow ends with an explicit, observable confirmation
    that a real QA would do.

    Heuristic
    ---------
    Triggers when:
    1. The goal description contains a create-style keyword.
    2. The decomposer's sub-goal list does NOT already end with a
       verify-style sub-goal.
    3. Optionally the AppMap exposes a matching create_flow with
       ``list_has_search=True`` so we know the search-then-verify
       pattern is feasible.

    Returns ``(possibly-augmented list, appended?)``.
    """
    if not sub_goals:
        return sub_goals, False
    goal_l = (goal_description or "").lower()
    if not any(k in goal_l for k in _CREATE_KEYWORDS):
        return sub_goals, False

    # Already has a verify step? Scan the LAST two sub-goals for
    # verify wording.
    tail = sub_goals[-2:]
    for sg in tail:
        text = (sg.description + " " + sg.success_criterion).lower()
        if any(k in text for k in _VERIFY_KEYWORDS):
            return sub_goals, False

    # Pick the entity from a matching create_flow when available;
    # otherwise fall back to a generic "newly created entity" phrase.
    entity_phrase = "the newly created entity"
    search_phrase = "scroll through the list to find it"
    if app_map is not None:
        for kw in (
            "role", "user", "project", "chainage",
            "permission", "tenant", "client", "account",
        ):
            if kw in goal_l:
                flow = app_map.create_flow_for_entity(kw)
                if flow is not None:
                    entity_phrase = f"the newly created {flow.entity}"
                    if flow.list_has_search:
                        search_phrase = (
                            "type its name into the list's search/"
                            "filter input to find it"
                        )
                    break

    next_idx = len(sub_goals) + 1
    verify_sg = RuntimeSubGoal(
        id=f"sg{next_idx}_verify",
        description=(
            f"Verify {entity_phrase} appears in the list — "
            f"{search_phrase}, then confirm a row with the entered "
            "name is visible."
        ),
        success_criterion=(
            f"A row matching {entity_phrase}'s name is visible in "
            "the list/table."
        ),
        max_turns=4,
        replan_iteration=0,
    )
    return sub_goals + [verify_sg], True
