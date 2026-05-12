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
- Optionally, knowledge-base notes about this app from prior runs.

Your job: produce 3-7 ordered SUB-GOALS the agent will execute
sequentially to reach the goal, BASED ON WHAT YOU SEE on the screen.

Rules
=====
1. Each sub-goal must be OBSERVABLE on the actual UI. Reference
   the controls you SEE, not what the BRD says. If the BRD says
   "click Create Role" but the screen shows "+ Add New Role",
   the sub-goal says "click the +Add New Role button".
2. Each sub-goal must produce ONE discrete advance toward the
   goal. "Fill the role form" is too coarse — split into "type
   the name", "type the display name", "configure permissions",
   "click Save".
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
    cheap_provider: "LLMProvider | None" = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
) -> DecompositionResult:
    """Initial vision-driven decomposition. Called ONCE per submodule.

    Routes to the STRONG provider directly (this is a planner-tier
    call — accuracy matters more than cost). ``cheap_provider`` is
    accepted for symmetry with the rest of the agent helpers but
    unused unless we add a cheap-first variant later.
    """
    from app.llm.base import ChatMessage  # noqa: PLC0415

    crit_block = "\n".join(f"  - {c}" for c in goal_success_criteria) or "  (none specified)"
    akb_section = (
        f"\n\nAPP KNOWLEDGE BASE (from prior runs):\n{akb_block}\n"
        if akb_block.strip() else ""
    )
    user_text = (
        f"GOAL:\n  {goal_description}\n\n"
        f"SUCCESS CRITERIA:\n{crit_block}\n"
        f"{akb_section}\n"
        "Look at the attached screenshot. Decompose the goal into "
        "3-7 ordered sub-goals based on what you actually see on "
        "this page."
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
) -> DecompositionResult:
    """Replan after a sub-goal fails. Routes to CHEAP tier first.

    ``replan_iteration`` is recorded on each returned sub-goal so
    the report timeline shows which replan produced it.
    """
    from app.llm.base import ChatMessage  # noqa: PLC0415

    done_block = (
        "\n".join(f"  ✓ {sg.description}" for sg in completed_sub_goals)
        or "  (none — first sub-goal failed)"
    )
    user_text = (
        f"ORIGINAL GOAL:\n  {goal_description}\n\n"
        f"SUB-GOALS COMPLETED:\n{done_block}\n\n"
        f"FAILED SUB-GOAL:\n  {failed_sub_goal.description}\n"
        f"  success_criterion: {failed_sub_goal.success_criterion}\n"
        f"  failure reason: {failure_reason}\n\n"
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
