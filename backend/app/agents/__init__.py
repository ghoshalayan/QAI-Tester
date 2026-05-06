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
    IMPROVISATION_SCHEMA,
    RECOVERY_SCHEMA,
    ImprovisationSuggestion,
    RecoveryAction,
    RecoverySuggestion,
    propose_improvisation,
    propose_recovery,
)

__all__ = [
    "AgentCancelled",
    "ExecutionResult",
    "IMPROVISATION_SCHEMA",
    "ImprovisationSuggestion",
    "RECOVERY_SCHEMA",
    "RecoveryAction",
    "RecoverySuggestion",
    "SynthesisResult",
    "TcSynthesisResult",
    "execute_plan",
    "propose_improvisation",
    "propose_recovery",
    "synthesize_frd",
    "synthesize_tc",
]
