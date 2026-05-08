"use client";

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { create } from "zustand";

import { api, type InterventionRequest } from "@/lib/api";

/**
 * Per-run transient progress info pushed by ``phase`` / ``done`` events.
 * Lives outside the TanStack Query cache so per-batch ticks don't churn
 * the runs query (which only invalidates on started / completed / failed /
 * cancelled).
 */
export interface RunProgress {
  phase?: string;            // "validating" | "loading" | "calling_llm" | "persisting"
  message?: string;
  chunks_seen?: number;
  truncated?: boolean;
  candidates?: number;
  input_tokens?: number;
  output_tokens?: number;
  // FRD → TC module progress
  module_name?: string;
  current?: number;          // 1-indexed module being processed
  total?: number;            // total modules in scope
  nodes_added?: number;      // nodes added by the just-completed module
  modules_generated?: number;
  modules_skipped?: string[];
  nodes_total?: number;
}

interface ProgressStore {
  byRunId: Record<number, RunProgress>;
  merge: (runId: number, info: RunProgress) => void;
  clear: (runId: number) => void;
  clearAll: () => void;
}

function pruneUndefined<T extends object>(obj: T): Partial<T> {
  const out: Partial<T> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined) (out as Record<string, unknown>)[k] = v;
  }
  return out;
}

export const useAgentRunProgress = create<ProgressStore>((set) => ({
  byRunId: {},
  merge: (runId, info) =>
    set((s) => ({
      byRunId: {
        ...s.byRunId,
        [runId]: { ...(s.byRunId[runId] ?? {}), ...pruneUndefined(info) },
      },
    })),
  clear: (runId) =>
    set((s) => {
      if (!(runId in s.byRunId)) return s;
      const next = { ...s.byRunId };
      delete next[runId];
      return { byRunId: next };
    }),
  clearAll: () => set({ byRunId: {} }),
}));


/**
 * Recent live events captured per run for the side-by-side presenter
 * panel. Caps at ``MAX_EVENTS`` per run so the store stays bounded
 * across long runs. Newest event last.
 *
 * The presenter popup mounts after the run starts, so we replay
 * whatever's already buffered (via the EventSource ``Last-Event-ID``
 * mechanism on the bus) plus everything that arrives live.
 */
export interface LiveEvent {
  /** Monotonic counter the panel uses for keys + ordering. */
  seq: number;
  /** ISO timestamp captured client-side when the event was received. */
  at: string;
  /** SSE event name — ``step_started``, ``ai_assist_completed``, etc. */
  type: string;
  /** Original ``data`` payload from the bus. */
  data: Record<string, unknown>;
}

const MAX_EVENTS_PER_RUN = 60;

interface EventLogStore {
  byRunId: Record<number, LiveEvent[]>;
  _nextSeq: number;
  push: (runId: number, type: string, data: Record<string, unknown>) => void;
  clear: (runId: number) => void;
  clearAll: () => void;
}

export const useAgentEventLog = create<EventLogStore>((set) => ({
  byRunId: {},
  _nextSeq: 1,
  push: (runId, type, data) =>
    set((s) => {
      const seq = s._nextSeq;
      const existing = s.byRunId[runId] ?? [];
      const next = [
        ...existing,
        { seq, at: new Date().toISOString(), type, data },
      ];
      // Cap the buffer; drop oldest first.
      const trimmed =
        next.length > MAX_EVENTS_PER_RUN
          ? next.slice(next.length - MAX_EVENTS_PER_RUN)
          : next;
      return {
        _nextSeq: seq + 1,
        byRunId: { ...s.byRunId, [runId]: trimmed },
      };
    }),
  clear: (runId) =>
    set((s) => {
      if (!(runId in s.byRunId)) return s;
      const next = { ...s.byRunId };
      delete next[runId];
      return { byRunId: next };
    }),
  clearAll: () => set({ byRunId: {}, _nextSeq: 1 }),
}));


/**
 * Active HITL interventions — one per run that's currently blocked
 * waiting for the user. Set on ``needs_intervention``, cleared on
 * ``intervention_resolved`` / ``intervention_auto_applied`` or any
 * terminal event.
 */
interface InterventionStore {
  byRunId: Record<number, InterventionRequest>;
  set: (runId: number, req: InterventionRequest) => void;
  clear: (runId: number) => void;
  clearAll: () => void;
}

export const useActiveInterventions = create<InterventionStore>((set) => ({
  byRunId: {},
  set: (runId, req) =>
    set((s) => ({ byRunId: { ...s.byRunId, [runId]: req } })),
  clear: (runId) =>
    set((s) => {
      if (!(runId in s.byRunId)) return s;
      const next = { ...s.byRunId };
      delete next[runId];
      return { byRunId: next };
    }),
  clearAll: () => set({ byRunId: {} }),
}));

/**
 * Subscribe to live agent-run events for the entire project.
 *
 * Wire-up:
 * - ``started``                       → invalidate the runs list (status flipped)
 * - ``phase`` / ``done``              → merge transient progress (no list refetch)
 * - ``completed``/``failed``/``cancelled`` → clear progress + invalidate runs + requirements
 *
 * Browser ``EventSource`` auto-reconnects; ``Last-Event-ID`` is sent so the
 * server replays missed events from the bus's history window.
 */
export function useAgentRunsEvents(projectId: number) {
  const qc = useQueryClient();
  const merge = useAgentRunProgress((s) => s.merge);
  const clear = useAgentRunProgress((s) => s.clear);
  const setIntervention = useActiveInterventions((s) => s.set);
  const clearIntervention = useActiveInterventions((s) => s.clear);
  const pushEvent = useAgentEventLog((s) => s.push);

  useEffect(() => {
    if (Number.isNaN(projectId) || projectId <= 0) return;

    const es = new EventSource(api.agentRunsEventsUrl(projectId));

    const onStarted = () => {
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.invalidateQueries({ queryKey: ["agent-run", projectId] });
    };

    const onProgress = (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const d = payload.data ?? {};
        if (typeof d.run_id !== "number") return;
        // Merge any of the optional progress fields. Undefined fields are
        // intentionally preserved (existing values stay) by re-merging the
        // current snapshot — handled inside the store's merge.
        merge(d.run_id, {
          phase: d.phase,
          message: d.message,
          chunks_seen: d.chunks_seen,
          truncated: d.truncated,
          candidates: d.candidates,
          input_tokens: d.input_tokens,
          output_tokens: d.output_tokens,
          module_name: d.module_name,
          current: d.current,
          total: d.total,
          nodes_added: d.nodes_added,
          modules_generated: d.modules_generated,
          modules_skipped: d.modules_skipped,
          nodes_total: d.nodes_total,
        });
      } catch {
        /* malformed event — ignore */
      }
    };

    const onTerminal = (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const runId = payload.data?.run_id;
        if (typeof runId === "number") {
          clear(runId);
          clearIntervention(runId);
        }
      } catch {
        /* ignore */
      }
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      // Per-run detail query (singular) used by the run detail page
      qc.invalidateQueries({ queryKey: ["agent-run", projectId] });
      qc.invalidateQueries({ queryKey: ["requirements", projectId] });
      // Prefix-match: refetches every plan's tc-nodes query under this project
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId] });
      // Per-run step rows used by the execution timeline (week 5 step 10)
      qc.invalidateQueries({ queryKey: ["run-steps", projectId] });
    };

    // After (re)connect: refetch in case events fell off the bus's history
    const onOpen = () => {
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
    };

    // Per-step events from the execute agent — refetch the timeline rows
    // so the run detail page reflects live transitions row-by-row.
    const onStepEvent = () => {
      qc.invalidateQueries({ queryKey: ["run-steps", projectId] });
    };

    // Pause / resume — the orchestrator flips run.status itself, so we
    // just refetch the run rows to pick up the change. Same payload shape
    // as `started` (no progress fields), so the runs query is enough.
    const onPauseEvent = () => {
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.invalidateQueries({ queryKey: ["agent-run", projectId] });
    };

    // HITL intervention — the orchestrator emits ``needs_intervention``
    // when a step has burned its retry budget AND AI assist couldn't fix
    // it. Frontend stores the payload so the modal can pop. Resolved /
    // auto-applied / terminal all clear it.
    const onNeedsIntervention = (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const d = payload.data ?? {};
        if (typeof d.run_id !== "number") return;
        setIntervention(d.run_id, {
          step_id: d.step_id,
          ordinal: d.ordinal,
          total: d.total,
          title: d.title ?? "",
          action_type: d.action_type ?? null,
          target_hint: d.target_hint ?? null,
          error_message: d.error_message ?? null,
          ai_suggestion: d.ai_suggestion ?? null,
          screenshot_path: d.screenshot_path ?? null,
        });
        qc.invalidateQueries({ queryKey: ["run-steps", projectId] });
      } catch {
        /* ignore */
      }
    };

    const onInterventionCleared = (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const runId = payload.data?.run_id;
        if (typeof runId === "number") clearIntervention(runId);
      } catch {
        /* ignore */
      }
      qc.invalidateQueries({ queryKey: ["run-steps", projectId] });
    };

    // Mirror every SSE event into the live presenter's event-log store.
    // Wrapped per event type so the existing listeners keep their precise
    // invalidation semantics; this is purely additive.
    const teeToLog = (type: string) => (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const d = payload.data ?? {};
        const runId = d.run_id;
        if (typeof runId === "number") {
          pushEvent(runId, type, d);
        }
      } catch {
        /* malformed event — ignore */
      }
    };

    const PRESENTER_EVENTS = [
      "started",
      "phase",
      "step_started",
      "step_retry",
      "step_completed",
      "ai_improvise_started",
      "ai_improvise_completed",
      "ai_assist_started",
      "ai_assist_completed",
      // Phase C: agentic-mode events
      "agent_goal_extracting",
      "agent_goal_ready",
      "agent_thought",
      "agent_acted",
      // Phase A1: sub-goal decomposition + progress
      "sub_goal_progress",
      // Phase A4.1b: vision-guided target search
      "agent_searching",
      "agent_search_completed",
      // Coordinate-click last-resort fallback (Operator pattern)
      "coordinate_click_proposed",
      "coordinate_click_completed",
      // Phase A4.1a / A4.1c: vision verification + mid-flow check
      "agent_verifying",
      "agent_verified",
      "agent_on_track_check",
      // Phase E: path freezing + replay + self-healing
      "frozen_path_captured",
      "frozen_step_completed",
      "frozen_step_self_healing",
      "frozen_step_self_heal_completed",
      // Phase 1: provider tiering — fired when the cheap tier hands
      // off to strong on a low-confidence / invalid result.
      "llm_escalated",
      // Phase 14: smart candidate selection — DOM ↔ browser-use
      // bridge picks the right element among many matches (skips
      // sponsored ads, items missing prices, etc.).
      "smart_pick_started",
      "smart_pick_completed",
      // Phase 9: semantic verify escalation
      "semantic_verify_started",
      "semantic_verify_completed",
      // Phase 11: agent disputes a test step that's provably wrong
      "test_case_disputed",
      // Phase 10: popup/overlay classifier (required vs dismissable
      // vs ad)
      "popup_classified",
      // Phase 4: typed HITL prompts (auth flow)
      "hitl_prompt_opened",
      "hitl_prompt_answered",
      // HITL + control
      "needs_intervention",
      "intervention_resolved",
      "intervention_auto_applied",
      "fix_promoted",
      "paused",
      "resumed",
      "module_started",
      "module_completed",
      "done",
      "completed",
      "failed",
      "cancelled",
    ];
    for (const type of PRESENTER_EVENTS) {
      es.addEventListener(type, teeToLog(type) as EventListener);
    }

    es.addEventListener("started", onStarted);
    es.addEventListener("phase", onProgress as EventListener);
    es.addEventListener("done", onProgress as EventListener);
    es.addEventListener("module_started", onProgress as EventListener);
    es.addEventListener("module_completed", onProgress as EventListener);
    es.addEventListener("step_started", onStepEvent);
    es.addEventListener("step_completed", onStepEvent);
    es.addEventListener("paused", onPauseEvent);
    es.addEventListener("resumed", onPauseEvent);
    es.addEventListener("needs_intervention", onNeedsIntervention as EventListener);
    es.addEventListener("intervention_resolved", onInterventionCleared as EventListener);
    es.addEventListener("intervention_auto_applied", onInterventionCleared as EventListener);
    es.addEventListener("completed", onTerminal as EventListener);
    es.addEventListener("failed", onTerminal as EventListener);
    es.addEventListener("cancelled", onTerminal as EventListener);
    es.addEventListener("open", onOpen);

    return () => {
      es.close();
    };
  }, [projectId, qc, merge, clear, setIntervention, clearIntervention, pushEvent]);
}
