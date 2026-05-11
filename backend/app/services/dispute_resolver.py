"""Production-γ.1 — dispute resolution feedback loop.

When the agent flags a test step as wrong (Phase 11) and the user
later reviews + resolves the dispute, this module:
- writes the resolution outcome back to ``app_knowledge`` as a
  ``dispute_outcome`` chunk, so future runs on the same target get
  the rule for free;
- (optionally) patches the test case via the suggested fix the
  agent proposed, with the user's edit on top.

Resolution actions:
- ``accept``  — agent's claim was right, suggested_fix is the rule.
  Optionally apply the fix to the TC node so the test case
  self-improves over time.
- ``reject``  — agent was wrong. The reasoning + the user's
  override note get written so the agent doesn't repeat the
  spurious dispute.
- ``edit``    — accept with a rephrased rule. The user's text wins.

The AKB chunk is high-confidence (0.95) because it's user-confirmed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def resolve_dispute(
    db: "Session",
    *,
    project_id: int,
    run_id: int,
    step_id: int,
    action: str,           # "accept" | "reject" | "edit"
    user_note: str = "",
    apply_to_test_case: bool = False,
) -> dict:
    """Persist a dispute resolution + feed AKB.

    Reads the dispute payload off ``execution_steps.details_json``
    (the agent wrote it there at flag time, surfaced via report),
    plus the plan's ``target_url`` so the AKB chunk lands on the
    right pattern.
    """
    from app.models.execution_step import ExecutionStep  # noqa: PLC0415
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from app.services.akb import write_chunk  # noqa: PLC0415

    if action not in ("accept", "reject", "edit"):
        raise ValueError(
            f"Unknown dispute action {action!r}; expected "
            "accept / reject / edit",
        )

    step = db.get(ExecutionStep, step_id)
    if step is None or step.run_id != run_id:
        raise ValueError(
            f"Step {step_id} not found on run {run_id}",
        )
    if step.project_id != project_id:
        raise ValueError(
            f"Step {step_id} doesn't belong to project {project_id}",
        )

    details = step.details_json or {}
    agent_log = details.get("agent_log") or []
    dispute = None
    for t in agent_log:
        sl = t.get("search_log") if isinstance(t, dict) else None
        if isinstance(sl, dict) and sl.get("kind") == "test_case_dispute":
            dispute = sl
            break
    if dispute is None:
        raise ValueError(
            f"Step {step_id} has no dispute payload — nothing to resolve",
        )

    plan = db.get(TestPlan, step.plan_id) if step.plan_id else None
    target_url = (plan.target_url if plan else "") or ""

    # Build the AKB chunk content based on the action.
    issue_kind = str(dispute.get("issue_kind") or "")
    evidence = str(dispute.get("evidence") or "")
    suggested_fix = str(dispute.get("suggested_fix") or "")
    rule_body = (user_note or "").strip()
    if not rule_body:
        if action == "accept":
            rule_body = (
                f"Confirmed test-case issue ({issue_kind}): "
                f"{evidence[:240]}. Fix: {suggested_fix[:240]}"
            )
        elif action == "reject":
            rule_body = (
                f"Rejected agent dispute ({issue_kind}): the test "
                f"step is correct. Agent claimed: {evidence[:240]}. "
                f"Don't flag this pattern again."
            )
        else:  # edit
            rule_body = (
                f"Test-case rule ({issue_kind}): {suggested_fix[:240]}"
            )

    # Confidence: accept/edit are user-confirmed → 0.95; reject is
    # also confirmed (we want the agent to NOT repeat) → 0.90.
    confidence = 0.95 if action in ("accept", "edit") else 0.90
    chunk_id = write_chunk(
        db,
        target_url_pattern=target_url,
        kind="dispute_outcome",
        content=rule_body[:1800],
        tags=["dispute", action, issue_kind] if issue_kind else ["dispute", action],
        confidence=confidence,
        source_run_id=run_id,
    )

    # Optionally patch the TC node when the user accepts a
    # suggested_fix and asks us to apply it. We don't try to do
    # anything fancy here — we append an annotation to the
    # description_md so a human reviewer can see what was changed
    # and revert if needed. Auto-applying selectors / pre/post
    # changes happens via the AKB feed; the test case stays the
    # human's spec.
    if apply_to_test_case and action in ("accept", "edit") and step.tc_node_id:
        node = db.get(TcNode, step.tc_node_id)
        if node is not None:
            stamp = (
                f"\n\n> Auto-annotation from dispute resolution "
                f"(run {run_id}, step {step_id}): {issue_kind}. "
                f"Suggested fix: {suggested_fix or rule_body}"
            )
            node.description_md = (node.description_md or "") + stamp
            db.commit()

    logger.info(
        "Dispute resolved on run %s step %s: action=%s, target=%r, "
        "akb_chunk_id=%s",
        run_id, step_id, action, target_url, chunk_id,
    )
    return {
        "action": action,
        "akb_chunk_id": chunk_id,
        "target_url": target_url,
        "applied_to_tc": apply_to_test_case,
    }
