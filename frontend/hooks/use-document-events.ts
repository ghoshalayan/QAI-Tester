"use client";

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { create } from "zustand";

import { api } from "@/lib/api";

/**
 * Per-doc transient progress info pushed by ``doc_progress`` events.
 * Lives outside the TanStack Query cache so we don't churn it on every
 * embedding-batch tick (the documents query only invalidates on
 * started / completed / failed).
 */
export interface DocProgress {
  phase: string; // "parsing" | "chunking" | "embedding"
  message: string;
  current?: number;
  total?: number;
}

interface ProgressStore {
  byDocId: Record<number, DocProgress>;
  set: (docId: number, info: DocProgress) => void;
  clear: (docId: number) => void;
  clearAll: () => void;
}

export const useDocumentProgress = create<ProgressStore>((set) => ({
  byDocId: {},
  set: (docId, info) =>
    set((s) => ({ byDocId: { ...s.byDocId, [docId]: info } })),
  clear: (docId) =>
    set((s) => {
      if (!(docId in s.byDocId)) return s;
      const next = { ...s.byDocId };
      delete next[docId];
      return { byDocId: next };
    }),
  clearAll: () => set({ byDocId: {} }),
}));

/**
 * Subscribe to live document-ingest events for a project.
 *
 * Wire-up:
 * - ``doc_started``    → invalidate the documents list
 * - ``doc_progress``   → update transient per-doc progress (no list refetch)
 * - ``doc_completed``  → clear progress + invalidate list
 * - ``doc_failed``     → clear progress + invalidate list
 *
 * The browser's ``EventSource`` auto-reconnects on transient failures and
 * sends ``Last-Event-ID`` so the server can replay missed events.
 */
export function useDocumentEvents(projectId: number) {
  const qc = useQueryClient();
  const setProgress = useDocumentProgress((s) => s.set);
  const clearProgress = useDocumentProgress((s) => s.clear);

  useEffect(() => {
    if (Number.isNaN(projectId) || projectId <= 0) return;

    const es = new EventSource(api.documentEventsUrl(projectId));

    const onStarted = () => {
      qc.invalidateQueries({ queryKey: ["documents", projectId] });
    };

    const onProgress = (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const d = payload.data ?? {};
        if (typeof d.doc_id !== "number") return;
        setProgress(d.doc_id, {
          phase: d.phase ?? "",
          message: d.message ?? "",
          current: d.current,
          total: d.total,
        });
      } catch {
        /* ignore malformed event */
      }
    };

    const onCompleted = (ev: MessageEvent) => {
      try {
        const payload = JSON.parse(ev.data);
        const docId = payload.data?.doc_id;
        if (typeof docId === "number") clearProgress(docId);
      } catch {
        /* ignore */
      }
      qc.invalidateQueries({ queryKey: ["documents", projectId] });
    };

    const onFailed = onCompleted; // same cache action

    // After (re)connect, force a fresh list — covers any events missed
    // during the reconnect gap that fell outside the bus's history window.
    const onOpen = () => {
      qc.invalidateQueries({ queryKey: ["documents", projectId] });
    };

    es.addEventListener("doc_started", onStarted);
    es.addEventListener("doc_progress", onProgress as EventListener);
    es.addEventListener("doc_completed", onCompleted as EventListener);
    es.addEventListener("doc_failed", onFailed as EventListener);
    es.addEventListener("open", onOpen);

    return () => {
      es.close();
    };
  }, [projectId, qc, setProgress, clearProgress]);
}
