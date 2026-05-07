"""Agents — LLM-driven workflows.

- ``brd_to_frd`` (week 3) — synthesizes Functional Requirements from BRD chunks
- ``frd_to_tc``  (week 4) — generates the N-level test-case tree from FRDs
- ``execute``    (week 5) — Playwright execution with narration
- ``reporter``   (week 8) — final report aggregator

Each agent is a **pure function** that takes a DB session, an LLM provider,
and the inputs it needs; emits typed events via a callback; and returns a
typed result. The thin runtime in ``app.services.agent_run_service`` wraps
them in an ``AgentRun`` row, manages state transitions, bridges the event
callback to the SSE bus, and runs the call on FastAPI's BackgroundTasks
threadpool.
"""

from app.agents.brd_to_frd import (
    AgentCancelled,
    SynthesisResult,
    synthesize_frd,
)
from app.agents.execute import (
    ExecutionResult,
    execute_plan,
)
from app.agents.frd_to_tc import (
    TcSynthesisResult,
    synthesize_tc,
)
from app.agents.page_intel import (
    GOAL_VERIFICATION_SCHEMA,
    IMPROVISATION_SCHEMA,
    ON_TRACK_SCHEMA,
    RECOVERY_SCHEMA,
    SEARCH_ACTION_SCHEMA,
    GoalVerdict,
    GoalVerification,
    ImprovisationSuggestion,
    OnTrackCheck,
    RecoveryAction,
    RecoverySuggestion,
    SearchAction,
    SearchSuggestion,
    check_on_track,
    propose_improvisation,
    propose_recovery,
    propose_search_action,
    verify_goal_via_screenshot,
)

__all__ = [
    "AgentCancelled",
    "ExecutionResult",
    "GOAL_VERIFICATION_SCHEMA",
    "GoalVerdict",
    "GoalVerification",
    "IMPROVISATION_SCHEMA",
    "ImprovisationSuggestion",
    "ON_TRACK_SCHEMA",
    "OnTrackCheck",
    "RECOVERY_SCHEMA",
    "RecoveryAction",
    "RecoverySuggestion",
    "SEARCH_ACTION_SCHEMA",
    "SearchAction",
    "SearchSuggestion",
    "SynthesisResult",
    "TcSynthesisResult",
    "check_on_track",
    "execute_plan",
    "propose_improvisation",
    "propose_recovery",
    "propose_search_action",
    "synthesize_frd",
    "synthesize_tc",
    "verify_goal_via_screenshot",
]
