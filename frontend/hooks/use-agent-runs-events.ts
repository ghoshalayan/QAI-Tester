"use client";

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { create } from "zustand";

import { api } from "@/lib/api";

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
        if (typeof runId === "number") clear(runId);
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

    es.addEventListener("started", onStarted);
    es.addEventListener("phase", onProgress as EventListener);
    es.addEventListener("done", onProgress as EventListener);
    es.addEventListener("module_started", onProgress as EventListener);
    es.addEventListener("module_completed", onProgress as EventListener);
    es.addEventListener("step_started", onStepEvent);
    es.addEventListener("step_completed", onStepEvent);
    es.addEventListener("completed", onTerminal as EventListener);
    es.addEventListener("failed", onTerminal as EventListener);
    es.addEventListener("cancelled", onTerminal as EventListener);
    es.addEventListener("open", onOpen);

    return () => {
      es.close();
    };
  }, [projectId, qc, merge, clear]);
}
