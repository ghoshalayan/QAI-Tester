"""Goal extraction — turn a submodule + its steps into a QA goal.

A goal carries everything the agent loop needs to act like a human QA:

- ``description``       — single-sentence intent ("verify a user can add
                          a product to cart").
- ``success_criteria``  — concrete things that must hold for the goal
                          to be marked complete (URL contains '/cart',
                          cart count is >= 1, toast says 'Added', etc.).
                          The agent self-checks against these and won't
                          declare success unless ≥ 1 is verified.
- ``hints``             — the original step list (title + action_type +
                          target_hint + narrative). Treated as guidance,
                          not a contract — the agent is free to deviate.

The extractor is a single LLM call per submodule, run once at the start
of an agentic run. Cheap (~500 input + 200 output tokens) and stored on
the run for replay / debugging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.llm.base import ChatMessage, LLMProvider
from app.models.tc_node import TcNode

logger = logging.getLogger(__name__)


@dataclass
class StepHint:
    """One step from the test case — fed to the agent as guidance."""

    ordinal: int
    title: str
    action_type: str | None
    target_hint: str | None
    narrative: str | None
    expected: str | None


@dataclass
class SubGoal:
    """One ordered sub-step the agent works through to achieve the goal.

    Decomposed at submodule start by the goal extractor. The agent
    picks ``id`` each turn (telling us which sub-goal it's progressing)
    and reports completion when the action verifiably advances past
    one. Status flips drive the live presenter's checkbox UI.
    """

    id: str          # short stable id, e.g. "sg1", "sg2"
    description: str
    status: str = "pending"  # pending | in_progress | done | failed | skipped
    completed_at_turn: int | None = None


@dataclass
class Goal:
    """The QA agent's mission for a single test case (submodule)."""

    submodule_id: int
    submodule_title: str
    path: str  # e.g. "Sign In > Successful sign-in with valid creds"
    description: str
    success_criteria: list[str] = field(default_factory=list)
    sub_goals: list[SubGoal] = field(default_factory=list)
    hints: list[StepHint] = field(default_factory=list)
    # Production-α.4 — richer goal shape. All four are list[str] of
    # short conditions / signals; empty lists fall through to the
    # legacy "infer from description" behavior.
    #
    # preconditions: state that must hold BEFORE the agent acts
    #   (e.g. "cart contains at least one item", "user is logged in").
    #   Asserted against WorldState at submodule start; mismatch
    #   fires an auto-dispute with kind=precondition_failed.
    # postconditions: state that must hold AFTER for the goal to
    #   count as passed (e.g. "cart shows zero items", "URL
    #   contains '/checkout'"). Used to update WorldState.
    # evidence_signals: observable cues that prove the post-
    #   conditions met. The agent's verify is N-of-M — claim done
    #   when ≥ threshold of signals match. Bounded.
    # alternative_paths: human-readable hints about other ways to
    #   reach the postconditions (used as prompt context when the
    #   primary flow is blocked).
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    evidence_signals: list[str] = field(default_factory=list)
    alternative_paths: list[str] = field(default_factory=list)
    # Telemetry for the cost meter
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "submodule_id": self.submodule_id,
            "submodule_title": self.submodule_title,
            "path": self.path,
            "description": self.description,
            "success_criteria": list(self.success_criteria),
            "sub_goals": [
                {
                    "id": sg.id,
                    "description": sg.description,
                    "status": sg.status,
                    "completed_at_turn": sg.completed_at_turn,
                }
                for sg in self.sub_goals
            ],
            "hints": [
                {
                    "ordinal": h.ordinal,
                    "title": h.title,
                    "action_type": h.action_type,
                    "target_hint": h.target_hint,
                    "narrative": h.narrative,
                    "expected": h.expected,
                }
                for h in self.hints
            ],
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "evidence_signals": list(self.evidence_signals),
            "alternative_paths": list(self.alternative_paths),
        }


GOAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
        },
        # Ordered sub-goals: 2-5 items the agent will work through
        # sequentially. Each is a verb-first sentence describing one
        # discrete page-level outcome. The schema requires the field
        # for OpenAI strict-mode compatibility; an empty array is
        # fine for trivial goals (the agent will still work, just
        # without the sub-goal UI breakdown).
        "sub_goals": {
            "type": "array",
            "items": {"type": "string"},
        },
        # Production-α.4 — pre/post + signals. All four required by
        # OpenAI strict mode; empty array means "no specific
        # conditions for this submodule, infer from description".
        "preconditions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "postconditions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_signals": {
            "type": "array",
            "items": {"type": "string"},
        },
        "alternative_paths": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "description", "success_criteria", "sub_goals",
        "preconditions", "postconditions", "evidence_signals",
        "alternative_paths",
    ],
    "additionalProperties": False,
}


GOAL_SYSTEM_PROMPT = """You are a senior QA tester reading a test case to extract its goal.

You'll see a submodule (one test case) and the literal steps inside it.
Your job: rewrite this as a goal-oriented mission a human tester could
execute even if some of the steps are slightly wrong or out of date.

Output SEVEN things:

1. description: ONE sentence, present tense, written from the user's
   perspective. Example: "User searches for a product and adds it to
   their cart." Don't paraphrase the steps — capture the INTENT.

2. success_criteria: 2-4 concrete, observable signals that prove the
   goal was achieved. Each one should be something a human could
   verify by looking at the page. Examples:
   - "Cart icon shows count >= 1"
   - "URL contains '/cart' or '/checkout'"
   - "A confirmation toast or banner appears"
   AVOID criteria that just restate the action ("clicked the button").
   PREFER criteria that describe the OUTCOME.

3. sub_goals: 2-5 ORDERED page-level outcomes the agent will work
   through sequentially. Each is a verb-first sentence describing ONE
   discrete advance toward the goal.

4. preconditions: 0-3 short factual statements about state the
   submodule REQUIRES before it can run. The agent asserts these
   against world state at submodule start; mismatch flags the test
   case as having a precondition failure (the prior submodule was
   supposed to set this up). Examples:
   - "cart contains at least one item"
   - "user is logged in as a customer"
   - "the URL is on the cart page"
   Empty array when the submodule is self-contained.

5. postconditions: 0-3 short factual statements about state that
   MUST hold for the goal to count as passed. Examples:
   - "cart contains zero items"
   - "URL contains '/checkout' or '/payment'"
   - "checkout flow has begun"
   Used to update world state when the agent claims completion.

6. evidence_signals: 2-5 SHORT observable cues the agent verifies
   to prove the postconditions. Multiple signals = multi-evidence
   voting (the agent claims pass when ≥ majority match — robust
   against single-token mismatches). Examples:
   - "Subtotal text is visible"
   - "Cart icon badge shows ≥ 1"
   - "Product name visible in cart list"
   - "Checkout button is enabled"
   These should be DISTINCT signals (different parts of the page),
   not synonyms of each other.

7. alternative_paths: 0-3 free-text hints describing other ways
   to reach the goal when the primary flow is blocked. Examples:
   - "Use the cart icon in the header instead of footer link"
   - "Mini-cart dropdown also has a checkout button"
   - "Empty cart hides 'Proceed to checkout' — re-add items first"
   Empty array when there's no obvious alternative.

Output JSON only. No commentary.
"""


def _format_hints_for_prompt(hints: list[StepHint]) -> str:
    if not hints:
        return "(no steps authored — derive the goal from the submodule title alone)"
    lines: list[str] = []
    for h in hints:
        action = h.action_type or "?"
        target = h.target_hint or "?"
        narr = (h.narrative or "").strip()
        lines.append(
            f"  {h.ordinal + 1}. [{action}] {h.title}\n"
            f"     target: {target}\n"
            f"     narrative: {narr or '(none)'}\n"
            f"     expected: {h.expected or '(none)'}",
        )
    return "\n".join(lines)


def extract_goal(
    provider: LLMProvider,
    submodule: TcNode,
    steps: list[TcNode],
    *,
    submodule_path: str | None = None,
) -> Goal:
    """One LLM call: turn a submodule + steps into a structured Goal.

    Args:
        provider: Configured LLM provider.
        submodule: The submodule node (kind='submodule').
        steps: Step nodes underneath this submodule, in ordinal order.
        submodule_path: Optional precomputed breadcrumb (e.g.
            "Sign In > Successful sign-in"). Falls back to
            ``submodule.path_cached`` or ``submodule.title``.

    Returns:
        A populated :class:`Goal`. The hints list mirrors ``steps`` in
        order, with snapshot fields (so later edits to the source nodes
        don't affect an in-progress run).

    Raises:
        RuntimeError: LLM call failed or returned malformed shape.
    """
    # Phase I.4 — goal-mode short-circuit. When the submodule has no
    # step children AND has authored description/postconditions/
    # evidence_signals, the goal is ALREADY explicit — we build the
    # Goal directly without an LLM call. The agent's decomposer then
    # plans from the goal + screenshot + AppMap, ignoring step-level
    # prescription entirely (the user's stated preference: "concept
    # from the test cases text, not the executional test cases").
    leaf_steps = [s for s in steps if s.kind == "step"]
    if not leaf_steps:
        description = (
            (submodule.description_md or "").strip()
            or (submodule.title or "").strip()
            or "Achieve the submodule's stated outcome on this page."
        )
        # Success criteria come from evidence_signals when present;
        # fall back to postconditions; final fallback synthesises one
        # from the title so the agent still has something to verify.
        signals = list(submodule.evidence_signals or [])
        if not signals:
            signals = list(submodule.postconditions or [])
        if not signals:
            signals = [
                "the action's primary effect is visible on the page "
                "(toast, badge update, list-row added, drawer closed)",
            ]
        return Goal(
            submodule_id=submodule.id,
            submodule_title=submodule.title or "",
            path=(
                submodule_path
                or submodule.path_cached
                or submodule.title
                or ""
            ),
            description=description,
            success_criteria=signals,
            hints=[],
            preconditions=list(submodule.preconditions or []),
            postconditions=list(submodule.postconditions or []),
            evidence_signals=list(submodule.evidence_signals or []),
            alternative_paths=list(submodule.alternative_paths or []),
            input_tokens=0,
            output_tokens=0,
        )

    hints = [
        StepHint(
            ordinal=s.ordinal,
            title=s.title or "",
            action_type=s.action_type,
            target_hint=s.target_hint,
            narrative=s.narrative,
            expected=s.expected,
        )
        for s in leaf_steps
    ]
    hints.sort(key=lambda h: h.ordinal)

    user_prompt = (
        f"SUBMODULE: {submodule.title}\n"
        f"PATH: {submodule_path or submodule.path_cached or submodule.title}\n"
        f"STEPS:\n{_format_hints_for_prompt(hints)}\n\n"
        "Extract the goal description + observable success criteria."
    )

    try:
        import time as _goal_time  # noqa: PLC0415
        _goal_t0 = _goal_time.monotonic()
        result = provider.chat_structured(
            messages=[
                ChatMessage(role="system", content=GOAL_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            schema=GOAL_SCHEMA,
            schema_name="qa_goal",
            temperature=0.2,
            max_output_tokens=512,
        )
        # Cost: goal extraction is strong-only (correctness-critical).
        # Role + model + duration surface in the call-log drill-in.
        try:
            from app.llm.cost_tracker import record_call  # noqa: PLC0415

            record_call(
                "strong", result.input_tokens, result.output_tokens,
                role="goal_extract",
                model=getattr(provider, "model", None),
                cached_input_tokens=getattr(
                    result, "cached_input_tokens", None,
                ),
                duration_ms=int(
                    (_goal_time.monotonic() - _goal_t0) * 1000,
                ),
            )
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(
            f"Goal extraction failed for submodule "
            f"{submodule.id}: {type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Goal LLM returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    description = str(parsed.get("description", "")).strip()
    if not description:
        # Empty description is useless — fall back to the submodule
        # title so the agent has SOMETHING to aim at instead of crashing.
        description = (
            f"Verify the test case '{submodule.title}' end to end."
        )
        logger.warning(
            "Goal LLM returned empty description for submodule %s; "
            "falling back to title-derived description",
            submodule.id,
        )

    raw_criteria = parsed.get("success_criteria") or []
    success_criteria = [
        str(c).strip() for c in raw_criteria if isinstance(c, str) and c.strip()
    ]

    raw_sub_goals = parsed.get("sub_goals") or []
    sub_goals: list[SubGoal] = []
    for i, sg_text in enumerate(raw_sub_goals):
        if not isinstance(sg_text, str) or not sg_text.strip():
            continue
        sub_goals.append(
            SubGoal(
                id=f"sg{i + 1}",
                description=sg_text.strip(),
                status="pending",
            ),
        )

    # α.4 — pre/post + signals + alt-paths. Extracted from the LLM
    # output OR (preferred) lifted directly from the TC node when
    # the test author authored them explicitly. TC node values
    # WIN over LLM output — the human's spec is authoritative.
    def _str_list(key: str) -> list[str]:
        raw = parsed.get(key) or []
        return [
            str(x).strip()
            for x in raw
            if isinstance(x, str) and x.strip()
        ]

    pre = list(submodule.preconditions or []) or _str_list("preconditions")
    post = list(submodule.postconditions or []) or _str_list("postconditions")
    signals = (
        list(submodule.evidence_signals or [])
        or _str_list("evidence_signals")
    )
    alts = (
        list(submodule.alternative_paths or [])
        or _str_list("alternative_paths")
    )

    return Goal(
        submodule_id=submodule.id,
        submodule_title=submodule.title or "",
        path=submodule_path or submodule.path_cached or submodule.title or "",
        description=description,
        success_criteria=success_criteria,
        sub_goals=sub_goals,
        hints=hints,
        preconditions=pre,
        postconditions=post,
        evidence_signals=signals,
        alternative_paths=alts,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
