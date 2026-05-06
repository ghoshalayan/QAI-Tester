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
class Goal:
    """The QA agent's mission for a single test case (submodule)."""

    submodule_id: int
    submodule_title: str
    path: str  # e.g. "Sign In > Successful sign-in with valid creds"
    description: str
    success_criteria: list[str] = field(default_factory=list)
    hints: list[StepHint] = field(default_factory=list)
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
        }


GOAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "success_criteria": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["description", "success_criteria"],
    "additionalProperties": False,
}


GOAL_SYSTEM_PROMPT = """You are a senior QA tester reading a test case to extract its goal.

You'll see a submodule (one test case) and the literal steps inside it.
Your job: rewrite this as a goal-oriented mission a human tester could
execute even if some of the steps are slightly wrong or out of date.

Output two things:

1. description: ONE sentence, present tense, written from the user's
   perspective. Example: "User searches for a product and adds it to
   their cart." Don't paraphrase the steps — capture the INTENT.

2. success_criteria: 2-4 concrete, observable signals that prove the
   goal was achieved. Each one should be something a human could
   verify by looking at the page. Examples:
   - "Cart icon shows count >= 1"
   - "URL contains '/cart' or '/checkout'"
   - "A confirmation toast or banner appears"
   - "The product name is visible inside the cart drawer"

   AVOID: criteria that just restate the action ("clicked the button").
   PREFER: criteria that describe the OUTCOME ("the page now shows X").

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
    hints = [
        StepHint(
            ordinal=s.ordinal,
            title=s.title or "",
            action_type=s.action_type,
            target_hint=s.target_hint,
            narrative=s.narrative,
            expected=s.expected,
        )
        for s in steps
        if s.kind == "step"
    ]
    hints.sort(key=lambda h: h.ordinal)

    user_prompt = (
        f"SUBMODULE: {submodule.title}\n"
        f"PATH: {submodule_path or submodule.path_cached or submodule.title}\n"
        f"STEPS:\n{_format_hints_for_prompt(hints)}\n\n"
        "Extract the goal description + observable success criteria."
    )

    try:
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

    return Goal(
        submodule_id=submodule.id,
        submodule_title=submodule.title or "",
        path=submodule_path or submodule.path_cached or submodule.title or "",
        description=description,
        success_criteria=success_criteria,
        hints=hints,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
