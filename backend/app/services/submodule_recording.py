"""Phase W — per-submodule user-action recording.

The operator manually clicks / types in a Playwright browser; every
action is captured and persisted to the submodule's
``tc_nodes.frozen_path`` JSON column with a discriminator
``recording_kind="user_actions"``. The agentic runner reads it back
and walks the recorded actions deterministically — the agent
contributes self-healing only when a recorded selector breaks (UI
moved, label changed, etc.).

Why ``frozen_path`` and not a new table
---------------------------------------
``frozen_path`` already exists on every TcNode; it was added for
Phase B's agent-derived deterministic replay. Reusing the column
with a different ``recording_kind`` discriminator (defined here)
avoids a migration and keeps "this submodule has a saved walk-
through" as ONE concept regardless of who created it.

Storage shape
-------------
Stored as a dict on ``TcNode.frozen_path`` for the submodule node::

    {
        "recording_kind": "user_actions",
        "schema_version": 1,
        "recorded_at": "2026-05-13T18:00:00Z",
        "target_url": "https://...",   // for replay sanity check
        "viewport": {"width": 1920, "height": 1040},
        "actions": [
            {"kind": "click", "x": 100, "y": 200,
             "target": {"tag": "button", "role": "button",
                        "text": "+ Add New Role", "id": "...",
                        "selector": "..."}},
            {"kind": "type", "value": "QA Role",
             "target": {...same...}},
            {"kind": "key", "key": "Enter"},
            {"kind": "navigate", "url": "https://.../roles/new"},
            ...
        ],
    }

The Phase B agent-derived ``frozen_path`` shape uses
``recording_kind="agent_freeze"`` (legacy paths without that key are
treated as agent_freeze for back-compat).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Phase Y.3 — per-recording structured log files ────────────────
#
# Every recording session writes a JSONL log to
# ``data/recordings_log/run-<id>.log.jsonl``. One line per event:
# init / set-active / push (with per-submodule counters) / finalize.
# Helps the operator debug "why was nothing captured" without
# reading the live SSE stream — the file persists past the run.


def _recording_log_path(run_id: int) -> Path:
    from app.config import settings  # noqa: PLC0415
    log_dir = Path(settings.data_dir) / "recordings_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"run-{run_id}.log.jsonl"


def _log_recording_event(run_id: int, kind: str, **fields: Any) -> None:
    """Append a structured event to the per-run JSONL log. Best-
    effort — write failures are swallowed."""
    try:
        record = {
            "t": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **fields,
        }
        path = _recording_log_path(run_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("recording log write failed for run %s: %s", run_id, e)


RECORDING_KIND_USER_ACTIONS = "user_actions"
RECORDING_KIND_AGENT_FREEZE = "agent_freeze"
RECORDING_SCHEMA_VERSION = 1


# ── In-memory per-submodule event buffer (Phase W') ──────────────
#
# A recording session covers a whole MODULE; chunks within it are
# tagged with the SUBMODULE the operator was working on at capture
# time. Buffer shape: ``_buffers[run_id] = {submodule_id: [events]}``.
# Active-submodule state per run: ``_active[run_id]`` is the
# submodule_id every incoming event currently attributes to (None
# until the operator picks one and clicks Start chunk on the live
# presenter).
#
# Events arriving while ``_active`` is None are DROPPED (the
# operator hasn't decided what to attribute them to). This is the
# "Recording paused" initial state.


_buffers: dict[int, dict[int, list[dict[str, Any]]]] = {}
_active: dict[int, int | None] = {}
_buffer_meta: dict[int, dict[str, Any]] = {}
_buffer_lock = threading.Lock()


def init_buffer(
    run_id: int,
    *,
    module_id: int,
    target_url: str,
    viewport: dict[str, int] | None = None,
) -> None:
    """Create the in-memory buffer for a recording run rooted at a
    module. The active submodule starts as None — events are dropped
    until the operator picks one via set_active_submodule."""
    with _buffer_lock:
        _buffers[run_id] = {}
        _active[run_id] = None
        _buffer_meta[run_id] = {
            "module_id": module_id,
            "target_url": target_url,
            "viewport": viewport or {"width": 1920, "height": 1040},
        }
    _log_recording_event(
        run_id, "init",
        module_id=module_id,
        target_url=target_url,
        viewport=viewport or {"width": 1920, "height": 1040},
    )


def set_active_submodule(run_id: int, submodule_id: int | None) -> bool:
    """Switch which submodule subsequent events attribute to.

    Passing ``None`` parks the recording (operator paused / between
    chunks). Returns True when the change applied, False when the
    run isn't initialised.
    """
    with _buffer_lock:
        if run_id not in _buffers:
            return False
        _active[run_id] = submodule_id
        if submodule_id is not None and submodule_id not in _buffers[run_id]:
            _buffers[run_id][submodule_id] = []
    _log_recording_event(
        run_id, "set_active_submodule",
        submodule_id=submodule_id,
    )
    return True


def get_active_submodule(run_id: int) -> int | None:
    with _buffer_lock:
        return _active.get(run_id)


def push_events(run_id: int, events: list[dict[str, Any]]) -> dict[str, int]:
    """Append events to the CURRENTLY-ACTIVE submodule's slot.

    Returns a small summary of ``{active_submodule_id, written,
    dropped, total_for_active}``. When ``active`` is None, all
    events are dropped — the operator has the browser open but
    hasn't committed to a submodule yet.
    """
    with _buffer_lock:
        if run_id not in _buffers:
            logger.warning(
                "recording events for un-initialized run %s — "
                "%d events dropped", run_id, len(events),
            )
            _log_recording_event(
                run_id, "push_events_no_buffer",
                dropped=len(events),
            )
            return {
                "active_submodule_id": 0,
                "written": 0,
                "dropped": len(events),
                "total_for_active": 0,
            }
        active = _active.get(run_id)
        if active is None or not events:
            summary = {
                "active_submodule_id": active or 0,
                "written": 0,
                "dropped": len(events) if active is None else 0,
                "total_for_active": (
                    len(_buffers[run_id].get(active, []))
                    if active is not None else 0
                ),
            }
            if active is None and events:
                _log_recording_event(
                    run_id, "push_events_no_active_chunk",
                    dropped=len(events),
                    sample_kinds=[
                        ev.get("kind") for ev in events[:5]
                    ],
                )
            return summary
        slot = _buffers[run_id].setdefault(active, [])
        slot.extend(events)
        summary = {
            "active_submodule_id": active,
            "written": len(events),
            "dropped": 0,
            "total_for_active": len(slot),
        }
        _log_recording_event(
            run_id, "push_events",
            active_submodule_id=active,
            written=len(events),
            total_for_active=len(slot),
            kinds=[ev.get("kind") for ev in events],
        )
        return summary


def buffer_state(run_id: int) -> dict[str, Any]:
    """Snapshot for the live presenter — counts per submodule + the
    currently-active one. Cheap; safe to poll."""
    with _buffer_lock:
        if run_id not in _buffers:
            return {
                "exists": False,
                "active_submodule_id": None,
                "per_submodule_counts": {},
            }
        return {
            "exists": True,
            "active_submodule_id": _active.get(run_id),
            "per_submodule_counts": {
                int(sm): len(events)
                for sm, events in _buffers[run_id].items()
            },
        }


def buffer_size(run_id: int) -> int:
    """Total events across all submodule chunks for a run."""
    with _buffer_lock:
        if run_id not in _buffers:
            return 0
        return sum(len(v) for v in _buffers[run_id].values())


def discard_buffer(run_id: int) -> None:
    """Drop a buffer without persisting (cancel / crash)."""
    with _buffer_lock:
        _buffers.pop(run_id, None)
        _active.pop(run_id, None)
        _buffer_meta.pop(run_id, None)
    _log_recording_event(run_id, "discard")


def finalize_to_submodule(
    db: "Session",
    *,
    run_id: int,
) -> dict[str, Any]:
    """Phase W' — flush EVERY non-empty submodule chunk to its
    respective ``tc_nodes.frozen_path``. Returns a per-submodule
    summary. Empty chunks (a submodule that was picked but had no
    captured events before Stop) are skipped.

    Function name kept for caller compatibility, but it now iterates
    chunks rather than writing to a single submodule.
    """
    from app.models.tc_node import TcNode  # noqa: PLC0415

    with _buffer_lock:
        per_sm = {
            int(sm): list(events)
            for sm, events in (_buffers.get(run_id) or {}).items()
        }
        meta = dict(_buffer_meta.get(run_id) or {})

    if not meta:
        return {
            "saved": False,
            "reason": "no buffer for this run (already stopped?)",
            "submodules": [],
            "event_count": 0,
        }

    saved_summaries: list[dict[str, Any]] = []
    total_events = 0
    recorded_at = datetime.now(timezone.utc).isoformat()
    target_url = meta.get("target_url") or ""
    viewport = meta.get("viewport") or {"width": 1920, "height": 1040}

    for submodule_id, events in per_sm.items():
        if not events:
            continue
        submodule = db.get(TcNode, submodule_id)
        if submodule is None or submodule.kind != "submodule":
            saved_summaries.append({
                "submodule_id": submodule_id,
                "saved": False,
                "reason": "submodule not found / wrong kind",
                "event_count": len(events),
            })
            continue
        payload: dict[str, Any] = {
            "recording_kind": RECORDING_KIND_USER_ACTIONS,
            "schema_version": RECORDING_SCHEMA_VERSION,
            "recorded_at": recorded_at,
            "target_url": target_url,
            "viewport": viewport,
            "actions": events,
        }
        submodule.frozen_path = payload
        total_events += len(events)
        saved_summaries.append({
            "submodule_id": submodule_id,
            "saved": True,
            "event_count": len(events),
        })
    try:
        db.commit()
    except Exception as e:
        logger.exception("recording finalize commit failed")
        try:
            db.rollback()
        except Exception:
            pass
        return {
            "saved": False,
            "reason": f"db write failed: {type(e).__name__}: {e}",
            "submodules": saved_summaries,
            "event_count": total_events,
        }

    # Drop the buffer post-commit.
    with _buffer_lock:
        _buffers.pop(run_id, None)
        _active.pop(run_id, None)
        _buffer_meta.pop(run_id, None)

    _log_recording_event(
        run_id, "finalize",
        saved=True,
        event_count=total_events,
        submodules=saved_summaries,
        module_id=meta.get("module_id"),
    )
    return {
        "saved": True,
        "submodules": saved_summaries,
        "event_count": total_events,
        "recorded_at": recorded_at,
        "module_id": meta.get("module_id"),
    }


def load_recording(node_frozen_path: dict | None) -> dict | None:
    """Return the user-actions recording payload, or None when the
    frozen_path is empty or carries an agent-freeze instead.
    """
    if not isinstance(node_frozen_path, dict):
        return None
    kind = str(node_frozen_path.get("recording_kind") or "")
    if kind == RECORDING_KIND_USER_ACTIONS:
        return node_frozen_path
    return None


# ── Stop-signal registry ─────────────────────────────────────────
#
# The recording task runs in a background thread and BLOCKS until
# the operator clicks "Stop recording." We signal stop via a per-
# run threading.Event.

_stop_events: dict[int, threading.Event] = {}
_stop_lock = threading.Lock()


def register_stop_event(run_id: int) -> threading.Event:
    with _stop_lock:
        ev = threading.Event()
        _stop_events[run_id] = ev
        return ev


def signal_stop(run_id: int) -> bool:
    """Returns True when a stop event was actually set."""
    with _stop_lock:
        ev = _stop_events.get(run_id)
    if ev is None:
        return False
    ev.set()
    return True


def drop_stop_event(run_id: int) -> None:
    with _stop_lock:
        _stop_events.pop(run_id, None)
