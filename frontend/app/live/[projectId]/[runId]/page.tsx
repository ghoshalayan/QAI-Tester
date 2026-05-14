"use client";

/**
 * Live presenter — pops alongside the headed Chromium window so the
 * user can watch the agent's reasoning + actions without alt-tabbing
 * to the run-detail tab.
 *
 * Why a top-level route instead of nested under /projects/[id]/...?
 * Because the project layout adds tabs / breadcrumbs / chrome that
 * we don't want in a 580px-wide popup. The route lives at the app
 * root so it inherits only the bare RootLayout.
 *
 * Subscribes to the project's SSE bus for live events; the event-log
 * store buffers the recent ones so we can render a scrolling feed.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  Circle,
  CircleDashed,
  Eye,
  ExternalLink,
  Loader2,
  Pause,
  Play,
  Sparkles,
  Square,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  type AgentRunRead,
  type AgentStatus,
  type OpenPrompt,
} from "@/lib/api";
import { HitlPromptCard } from "@/components/hitl-prompt-card";
import { Button } from "@/components/ui/button";
import {
  useActiveInterventions,
  useAgentEventLog,
  useAgentRunProgress,
  useAgentRunsEvents,
  type LiveEvent,
} from "@/hooks/use-agent-runs-events";
import { cn } from "@/lib/utils";

const RUN_STATUS_BADGE: Record<AgentStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
  paused: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
  completed: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  failed: "bg-red-500/15 text-red-700 dark:text-red-300",
  cancelled: "bg-muted text-muted-foreground",
};

const TERMINAL_STATUSES: AgentStatus[] = [
  "completed",
  "failed",
  "cancelled",
];

// Module-level frozen sentinel — using `?? []` inside a Zustand selector
// would mint a new array every render, which `useSyncExternalStore`
// flags as a state change and triggers an infinite re-render loop.
// A stable reference makes the selector idempotent.
const EMPTY_EVENTS: readonly LiveEvent[] = Object.freeze([]);

export default function LivePresenterPage() {
  const params = useParams<{ projectId: string; runId: string }>();
  const projectId = Number(params.projectId);
  const runId = Number(params.runId);
  const qc = useQueryClient();

  // Subscribes the popup to its OWN SSE channel — the main app tab is
  // a separate process, so the popup needs an independent connection.
  useAgentRunsEvents(projectId);

  const events =
    useAgentEventLog((s) => s.byRunId[runId]) ?? EMPTY_EVENTS;
  const progress = useAgentRunProgress((s) => s.byRunId[runId]);
  const intervention = useActiveInterventions((s) => s.byRunId[runId]);

  const { data: run } = useQuery({
    queryKey: ["agent-run", projectId, runId],
    queryFn: () => api.getAgentRun(projectId, runId),
  });

  const { data: plan } = useQuery({
    queryKey: ["plan", projectId, run?.plan_id],
    queryFn: () => api.getPlan(projectId, run!.plan_id!),
    enabled: !!run?.plan_id,
  });

  const isTerminal = run ? TERMINAL_STATUSES.includes(run.status) : false;
  const isPaused = run?.status === "paused";
  const isRunning = run?.status === "running";

  const pauseMut = useMutation({
    mutationFn: () => api.pauseAgentRun(projectId, runId),
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Pause failed", { description: msg });
    },
  });
  const resumeMut = useMutation({
    mutationFn: () => api.resumeAgentRun(projectId, runId),
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Resume failed", { description: msg });
    },
  });
  const cancelMut = useMutation({
    mutationFn: () => api.cancelAgentRun(projectId, runId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent-run", projectId, runId] });
      toast.success("Cancel requested");
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Cancel failed", { description: msg });
    },
  });
  const forceCancelMut = useMutation({
    mutationFn: () => api.forceCancelAgentRun(projectId, runId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent-run", projectId, runId] });
      toast.success("Force-stopped", {
        description:
          "Run marked cancelled in the DB. The worker thread may run for a beat longer.",
      });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Force-stop failed", { description: msg });
    },
  });
  // Phase AA — operator force-pass (Ctrl+Shift+D). Resolves the run
  // as completed with every remaining test case marked passed.
  // Distinct from cancel: status=completed, not cancelled.
  const forcePassMut = useMutation({
    mutationFn: () => api.forcePassAgentRun(projectId, runId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agent-run", projectId, runId] });
      // Backend acknowledged. The runner may take a moment to
      // observe the flag at its next safe checkpoint, but the
      // outcome is locked in from this moment.
      toast.success("All test cases covered", {
        description:
          "Backend agent acknowledged — every test cases will be promoted to passed.",
      });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Force-pass failed", { description: msg });
    },
  });
  // Phase W — stop a reading session. Triggers the backend's
  // stop event so the browser closes and the captured actions
  // get persisted to the submodule's frozen_path.
  const stopRecordingMut = useMutation({
    mutationFn: () => api.stopRecording(projectId, runId),
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["agent-run", projectId, runId] });
      toast.success("Reading stopped", {
        description: `${resp.buffered_events} action${
          resp.buffered_events === 1 ? "" : "s"
        } captured — saving to the submodule.`,
      });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Stop failed", { description: msg });
    },
  });
  const isRecording =
    run?.kind === "record" &&
    (run?.status === "running" || run?.status === "queued");

  // Page title reflects status so the user can spot completion in the
  // taskbar/dock without focusing the popup.
  useEffect(() => {
    const status = run?.status ?? "…";
    document.title = `Run #${runId} · ${status}`;
  }, [runId, run?.status]);

  // Keep view scrolled to the latest event.
  useEffect(() => {
    const el = document.getElementById("__live-event-tail");
    if (el) el.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length]);

  // Phase AA — Ctrl+Shift+D shortcut. Single keypress, no
  // confirmation dialog — fires the force-pass mutation directly.
  //
  // NOTE on focus: keyboard events only reach this listener when
  // THIS window (the live presenter) has focus. If the operator is
  // watching the Playwright browser the agent drives, those
  // keypresses go to Chromium, not here. The visible "Mark Passed"
  // button in the Controls bar is the focus-independent path; this
  // shortcut is the keyboard convenience.
  //
  // Accepts Ctrl OR Cmd (meta) so Mac users get a working shortcut
  // without us needing to know the OS. Uses capture phase so it
  // fires before any other listener can swallow the event.
  // preventDefault suppresses Chrome's "Bookmark all tabs" default.
  const isInFlight =
    run?.status === "queued" ||
    run?.status === "running" ||
    run?.status === "paused";
  const isAgenticInFlight =
    isInFlight && run?.kind !== "record";

  useEffect(() => {
    if (!isAgenticInFlight) return;
    const onKey = (e: KeyboardEvent) => {
      if (!e.shiftKey) return;
      if (!e.ctrlKey && !e.metaKey) return;
      if (e.key !== "D" && e.key !== "d") return;
      if (e.repeat) return;  // ignore key-held auto-repeats
      e.preventDefault();
      e.stopPropagation();
      if (forcePassMut.isPending) return;
      toast.success("All test cases covered", {
        description:
          "Backend agent acknowledged — every test cases will be promoted to passed.",
      });
      forcePassMut.mutate();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [isAgenticInFlight, forcePassMut]);

  return (
    <div className="flex h-screen flex-col bg-background text-sm text-foreground">
      <Header
        run={run ?? null}
        planName={plan?.name ?? null}
        progress={progress?.message}
      />

      {isRecording && (
        <RecordingControls
          projectId={projectId}
          runId={runId}
          stopping={stopRecordingMut.isPending}
          onStop={() => stopRecordingMut.mutate()}
        />
      )}

      {/* Phase Y.2 — Pause/Stop/Force-stop bar is hidden for
          record-kind runs; the RecordingControls bar above already
          provides Start chunk / Stop reading. Two bars side-by-
          side were confusing the operator (the lower Stop only
          cancels the runner thread; the upper Stop reading saves
          the buffer). For record runs the RecordingControls'
          Stop reading IS the only correct way to end. */}
      {!isRecording && (
        <Controls
          isPaused={isPaused}
          isRunning={isRunning}
          isTerminal={isTerminal}
          pausePending={pauseMut.isPending}
          resumePending={resumeMut.isPending}
          cancelPending={cancelMut.isPending}
          forceCancelPending={forceCancelMut.isPending}
          forcePassPending={forcePassMut.isPending}
          onPause={() => pauseMut.mutate()}
          onResume={() => resumeMut.mutate()}
          onCancel={() => cancelMut.mutate()}
          onForceCancel={() => {
            if (
              window.confirm(
                "Force-stop will mark this run cancelled in the DB immediately. " +
                  "The worker thread may keep running for a few seconds. Continue?",
              )
            ) {
              forceCancelMut.mutate();
            }
          }}
          onForcePass={() => {
            // Same payload as the Ctrl+Shift+D shortcut. No confirm
            // dialog — the click itself is the explicit gesture.
            toast.success("All test cases covered", {
              description:
                "Backend agent acknowledged — every test cases will be promoted to passed.",
            });
            forcePassMut.mutate();
          }}
        />
      )}

      {intervention && (
        <InterventionBanner
          projectId={projectId}
          runId={runId}
          stepTitle={intervention.title}
        />
      )}

      {/* Phase 4 — typed HITL prompts. Mounted unconditionally; the
          component only renders when there's an open prompt for any
          step that's currently surfaced via the events. */}
      <HitlPromptArea
        projectId={projectId}
        runId={runId}
        events={events}
      />

      {/* Phase A.6 Step 5 — Scout progress panel. Auto-opens on
          ``app_map_scout_started``, ticks through ``auth_scout_page``,
          closes on ``app_map_built``. Gives the user visibility into
          what the agent is learning about the app before execution
          begins. */}
      <ScoutProgressPanel
        projectId={projectId}
        runId={runId}
        events={events}
      />

      {/* Phase A.6 Step 6 — Plan ↔ AppMap reconciliation. Renders
          above execution so the user sees which submodules will
          likely struggle BEFORE turns burn on them. */}
      <ReconciliationPanel events={events} />

      <CostMeter run={run ?? null} />

      <div className="flex-1 overflow-y-auto px-3 py-2">
        <EventStream events={events} />
        <div id="__live-event-tail" />
      </div>

      <Footer projectId={projectId} runId={runId} />
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────

function Header({
  run,
  planName,
  progress,
}: {
  run: AgentRunRead | null;
  planName: string | null;
  progress?: string;
}) {
  return (
    <div className="border-b px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <h1 className="text-base font-semibold">
          Run #{run?.id ?? "…"}
        </h1>
        {run && (
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
              RUN_STATUS_BADGE[run.status],
            )}
          >
            {run.status === "running" && (
              <Loader2 className="size-3 animate-spin" />
            )}
            {run.status}
          </span>
        )}
      </div>
      {planName && (
        <p className="mt-0.5 truncate text-xs text-muted-foreground">
          {planName}
        </p>
      )}
      {progress && (
        <p className="mt-1 text-xs text-muted-foreground">{progress}</p>
      )}
    </div>
  );
}

function Controls({
  isPaused,
  isRunning,
  isTerminal,
  pausePending,
  resumePending,
  cancelPending,
  forceCancelPending,
  forcePassPending,
  onPause,
  onResume,
  onCancel,
  onForceCancel,
  onForcePass,
}: {
  isPaused: boolean;
  isRunning: boolean;
  isTerminal: boolean;
  pausePending: boolean;
  resumePending: boolean;
  cancelPending: boolean;
  forceCancelPending: boolean;
  forcePassPending: boolean;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
  onForceCancel: () => void;
  onForcePass: () => void;
}) {
  if (isTerminal) return null;
  return (
    <div className="flex items-center gap-2 border-b bg-muted/30 px-3 py-1.5">
      {isPaused ? (
        <Button
          size="sm"
          variant="outline"
          onClick={onResume}
          disabled={resumePending}
        >
          <Play className="size-3.5" />
          Resume
        </Button>
      ) : (
        <Button
          size="sm"
          variant="outline"
          onClick={onPause}
          disabled={pausePending || !isRunning}
        >
          <Pause className="size-3.5" />
          Pause
        </Button>
      )}
      <Button
        size="sm"
        variant="outline"
        onClick={onCancel}
        disabled={cancelPending}
      >
        <Square className="size-3.5" />
        Stop
      </Button>
      {/* Phase K.1 — STRICT STOP. Writes the DB row to cancelled
          immediately; the worker thread may run for a beat longer
          but the system treats it as terminal from this click. */}
      <Button
        size="sm"
        variant="destructive"
        onClick={onForceCancel}
        disabled={forceCancelPending}
        title="Force-stop: marks the run cancelled in the DB immediately"
      >
        <Square className="size-3.5" />
        Force stop
      </Button>
      {/* Phase AA — Mark Passed. Focus-independent path for the
          Ctrl+Shift+D shortcut. Shipping a visible button means
          operators can end a run as success even when they're
          watching the Playwright browser (the popup window
          doesn't have keyboard focus). Emerald variant to set
          it apart from the rose stop buttons — passing is the
          happy path. */}
      <Button
        size="sm"
        onClick={onForcePass}
        disabled={forcePassPending}
        title="Mark passed (Ctrl+Shift+D): every remaining test case promoted to passed; run resolves as completed"
        className="ml-auto bg-emerald-600 text-white hover:bg-emerald-700"
      >
        <CheckCircle2 className="size-3.5" />
        Mark passed
        <kbd className="ml-1 hidden rounded border border-emerald-300/30 bg-emerald-700/40 px-1 py-0 text-[10px] font-medium md:inline-block">
          Ctrl+Shift+D
        </kbd>
      </Button>
    </div>
  );
}

/**
 * Phase 4 — typed HITL prompt area.
 *
 * Watches for ``hitl_prompt_opened`` SSE events and renders the
 * ``HitlPromptCard`` for the latest open prompt INLINE in the
 * popup so the user doesn't have to leave the live view to enter
 * an OTP / credential / captcha solve.
 *
 * The card disappears when:
 * - The user submits (provideIntervention success → prompt cleared
 *   server-side).
 * - The agent's wait timed out / was cancelled
 *   (``hitl_prompt_answered`` event with status="cancelled").
 *
 * On mount we also fetch ``GET /intervention/open`` so a popup
 * reload picks up an in-flight prompt.
 */
/**
 * Phase W' — recording controls bar.
 *
 * Lives at the top of the live presenter while a kind="record" run
 * is active. Provides:
 *   - searchable submodule combobox (filter the active module's
 *     children by typing)
 *   - "+ Add new submodule" inline creator
 *   - "Start chunk" button — commits the currently-selected submodule
 *     as the active capture target; subsequent events attribute to it
 *   - "Stop reading" button — ends the session, persists per-submodule
 *     chunks to their respective frozen_paths
 *
 * Styled to match the existing Pause/Stop Controls bar (same flex
 * layout, same button variants).
 */
function RecordingControls({
  projectId,
  runId,
  stopping,
  onStop,
}: {
  projectId: number;
  runId: number;
  stopping: boolean;
  onStop: () => void;
}) {
  const qc = useQueryClient();

  // Poll the buffer state every 2s so the operator sees the active
  // submodule + per-chunk counters update as events stream in.
  const { data: state } = useQuery({
    queryKey: ["recording-state", projectId, runId],
    queryFn: () => api.getRecordingState(projectId, runId),
    refetchInterval: 2000,
  });
  const moduleId = state?.module_id ?? 0;
  const activeSubmoduleId = state?.active_submodule_id ?? null;
  const perSubmoduleCounts = state?.per_submodule_counts ?? {};

  // Fetch the module's submodules (children) so we can filter +
  // pick from them. Reuses listTcNodes; we filter client-side.
  const { data: nodes } = useQuery({
    queryKey: ["recording-module-children", projectId, runId, moduleId],
    queryFn: async () => {
      const planId = (await api.getAgentRun(projectId, runId)).input_json
        ?.plan_id as number | undefined;
      if (!planId) return [];
      return api.listTcNodes(projectId, planId);
    },
    enabled: moduleId > 0,
  });
  const submodules = (nodes ?? [])
    .filter((n) => n.id === moduleId)
    .flatMap((m) => m.children ?? [])
    .filter((c) => c.kind === "submodule");

  const [query, setQuery] = useState("");
  const [pickedId, setPickedId] = useState<number | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [newTitle, setNewTitle] = useState("");

  // Keep pickedId in sync with activeSubmoduleId on first load.
  useEffect(() => {
    if (pickedId === null && activeSubmoduleId !== null) {
      setPickedId(activeSubmoduleId);
    }
  }, [activeSubmoduleId, pickedId]);

  const filtered = submodules.filter((sm) =>
    (sm.title || "").toLowerCase().includes(query.trim().toLowerCase()),
  );

  const startChunkMut = useMutation({
    mutationFn: (submoduleId: number) =>
      api.setActiveSubmodule(projectId, runId, submoduleId),
    onSuccess: (resp) => {
      qc.invalidateQueries({
        queryKey: ["recording-state", projectId, runId],
      });
      const sm = submodules.find((s) => s.id === resp.active_submodule_id);
      toast.success("Recording into " + (sm?.title ?? `#${resp.active_submodule_id}`));
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Could not switch chunk", { description: msg });
    },
  });

  // Pause-chunk: parks the recording WITHOUT closing the browser. The
  // submodule's events stay in the buffer; the operator can pick a
  // different submodule and Start chunk again. Stop reading remains
  // the only way to end the whole session.
  const pauseChunkMut = useMutation({
    mutationFn: () => api.setActiveSubmodule(projectId, runId, null),
    onSuccess: () => {
      qc.invalidateQueries({
        queryKey: ["recording-state", projectId, runId],
      });
      toast.message("Chunk paused", {
        description: "Captured events kept. Pick another submodule + Start chunk to continue.",
      });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Could not pause chunk", { description: msg });
    },
  });

  const findPlanIdFromState = () => {
    // The state endpoint doesn't return plan_id, but we can read it
    // from the agent-run query cache populated by the parent.
    const cached = qc.getQueryData<{ input_json?: { plan_id?: number } }>([
      "agent-run",
      projectId,
      runId,
    ]);
    return cached?.input_json?.plan_id ?? null;
  };

  const addSubmoduleMut = useMutation({
    mutationFn: async () => {
      const planId = findPlanIdFromState();
      if (!planId) throw new Error("plan id unavailable");
      return api.createTcNode(projectId, planId, {
        title: newTitle.trim(),
        kind: "submodule",
        parent_id: moduleId,
      });
    },
    onSuccess: (created) => {
      qc.invalidateQueries({
        queryKey: ["recording-module-children", projectId, runId, moduleId],
      });
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId] });
      toast.success("Submodule added", {
        description: `"${created.title}" — pick + Start chunk to record into it.`,
      });
      setPickedId(created.id);
      setQuery("");
      setNewTitle("");
      setShowAdd(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Could not add submodule", { description: msg });
    },
  });

  const totalChunked = Object.values(perSubmoduleCounts).reduce(
    (a, b) => a + b, 0,
  );

  // Phase Y.4 — quick way to inspect recorded actions on a
  // submodule. Opens a console.log with the full payload + a toast
  // summary; clicking the same button after the recording ends
  // surfaces the saved actions immediately for verification.
  const inspectMut = useMutation({
    mutationFn: async (submoduleId: number) => {
      const planId = findPlanIdFromState();
      if (!planId) throw new Error("plan id unavailable");
      return api.getNodeRecording(projectId, planId, submoduleId);
    },
    onSuccess: (data) => {
      if (!data.has_recording) {
        toast.message(`No saved actions yet for "${data.title}"`, {
          description: "Buffer is in-memory until you click Stop reading.",
        });
        return;
      }
      // Log for full inspection; toast gives a summary.
      // eslint-disable-next-line no-console
      console.log("[recording]", data.title, data);
      toast.success(`${data.title} — ${data.action_count} actions`, {
        description: "Full payload printed to the browser console.",
      });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Could not load recording", { description: msg });
    },
  });

  return (
    <div className="flex flex-wrap items-center gap-2 border-b bg-rose-50 px-3 py-2 text-sm dark:bg-rose-950/30">
      <div className="flex items-center gap-2">
        <span className="relative flex size-3">
          <span
            className={cn(
              "absolute inset-0 rounded-full bg-rose-400 opacity-75",
              activeSubmoduleId !== null && "animate-ping",
            )}
          />
          <span
            className={cn(
              "relative inline-flex size-3 rounded-full",
              activeSubmoduleId !== null ? "bg-rose-500" : "bg-gray-400",
            )}
          />
        </span>
        <span className="font-medium text-rose-700 dark:text-rose-300">
          {activeSubmoduleId !== null
            ? `Recording → ${
                submodules.find((s) => s.id === activeSubmoduleId)?.title
                  ?? `#${activeSubmoduleId}`
              }`
            : "Recording paused"}
        </span>
        <span className="text-xs text-rose-700/70 dark:text-rose-300/70">
          {totalChunked} captured ·{" "}
          {Object.keys(perSubmoduleCounts).length} submodule
          {Object.keys(perSubmoduleCounts).length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search submodules…"
            className="h-8 w-52 rounded-md border border-input bg-background px-2 text-xs"
          />
          {(query || filtered.length > 0) && (
            <div className="absolute right-0 top-full z-10 mt-1 max-h-48 w-64 overflow-y-auto rounded-md border bg-popover text-popover-foreground shadow-md">
              {filtered.length === 0 ? (
                <div className="px-2 py-1.5 text-xs text-muted-foreground">
                  No matches — use <strong>+ New</strong> to add.
                </div>
              ) : (
                filtered.map((sm) => {
                  const count = perSubmoduleCounts[String(sm.id)] ?? 0;
                  return (
                    <div
                      key={sm.id}
                      className={cn(
                        "flex w-full items-center justify-between gap-1 px-2 py-1.5 text-xs hover:bg-muted",
                        pickedId === sm.id && "bg-muted font-medium",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          setPickedId(sm.id);
                          setQuery(sm.title);
                        }}
                        className="flex-1 text-left"
                      >
                        {sm.title}
                        {count ? (
                          <span className="ml-2 text-[10px] text-rose-600">
                            ({count})
                          </span>
                        ) : null}
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          inspectMut.mutate(sm.id);
                        }}
                        className="rounded px-1 py-0.5 text-[10px] text-muted-foreground hover:bg-background hover:text-foreground"
                        title="Log the saved actions to the console"
                      >
                        View
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={() => setShowAdd((v) => !v)}
          className="rounded-md border border-input bg-background px-2 py-1 text-xs hover:bg-muted"
        >
          {showAdd ? "Cancel" : "+ New"}
        </button>
        <button
          type="button"
          onClick={() => pickedId !== null && startChunkMut.mutate(pickedId)}
          disabled={pickedId === null || startChunkMut.isPending}
          className="rounded-md border border-emerald-600 bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
          title="Attribute subsequent events to the selected submodule"
        >
          {startChunkMut.isPending ? "Switching…" : "▶ Start chunk"}
        </button>
        <button
          type="button"
          onClick={() => pauseChunkMut.mutate()}
          disabled={activeSubmoduleId === null || pauseChunkMut.isPending}
          className="rounded-md border border-amber-500 bg-amber-500 px-3 py-1 text-xs font-medium text-white hover:bg-amber-600 disabled:opacity-50"
          title="Pause attribution — captured events kept; pick another submodule next"
        >
          {pauseChunkMut.isPending ? "Pausing…" : "⏸ Pause chunk"}
        </button>
        <button
          type="button"
          onClick={onStop}
          disabled={stopping}
          className="rounded-md bg-rose-600 px-3 py-1 text-xs font-medium text-white hover:bg-rose-700 disabled:opacity-60"
          title="End the recording session and save all submodule chunks"
        >
          {stopping ? "Stopping…" : "⬛ Stop reading"}
        </button>
      </div>

      {showAdd && (
        <div className="flex w-full items-center gap-2 rounded-md border bg-background/60 p-2">
          <input
            type="text"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="New submodule title — e.g. Create Role"
            className="h-8 flex-1 rounded-md border border-input bg-background px-2 text-xs"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter" && newTitle.trim()) addSubmoduleMut.mutate();
            }}
          />
          <button
            type="button"
            onClick={() => addSubmoduleMut.mutate()}
            disabled={!newTitle.trim() || addSubmoduleMut.isPending}
            className="rounded-md border border-input bg-background px-2 py-1 text-xs hover:bg-muted disabled:opacity-50"
          >
            {addSubmoduleMut.isPending ? "Adding…" : "Add"}
          </button>
        </div>
      )}
    </div>
  );
}


function HitlPromptArea({
  projectId,
  runId,
  events,
}: {
  projectId: number;
  runId: number;
  events: readonly LiveEvent[];
}) {
  // Track the most recent (step_id, prompt) pair from events.
  const [openStepId, setOpenStepId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState<OpenPrompt | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Walk backwards through the events for the latest open/answered
    // pair. answered → close. opened → open.
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (ev.type === "hitl_prompt_answered") {
        setOpenStepId(null);
        setPrompt(null);
        return;
      }
      if (ev.type === "hitl_prompt_opened") {
        const stepId = (ev.data as { step_id?: number }).step_id;
        if (typeof stepId === "number") {
          setOpenStepId(stepId);
          // Pull the freshest prompt from the server (event payload
          // intentionally doesn't carry ``fields`` to keep the SSE
          // stream small).
          setLoading(true);
          api
            .getOpenPrompt(projectId, runId, stepId)
            .then((p) => {
              if (p.open) {
                setPrompt(p);
              } else {
                setPrompt(null);
                setOpenStepId(null);
              }
            })
            .catch(() => {
              setPrompt(null);
              setOpenStepId(null);
            })
            .finally(() => setLoading(false));
        }
        return;
      }
    }
  }, [events, projectId, runId]);

  if (!prompt || openStepId === null) {
    return null;
  }
  return (
    <div className="border-b px-3 py-2">
      {loading ? (
        <p className="text-xs text-muted-foreground">
          Loading prompt…
        </p>
      ) : (
        <HitlPromptCard
          projectId={projectId}
          runId={runId}
          stepId={openStepId}
          prompt={prompt}
          onSubmitted={() => {
            setPrompt(null);
            setOpenStepId(null);
          }}
        />
      )}
    </div>
  );
}


function InterventionBanner({
  projectId,
  runId,
  stepTitle,
}: {
  projectId: number;
  runId: number;
  stepTitle: string;
}) {
  return (
    <div className="border-b border-amber-500/30 bg-amber-500/10 px-3 py-2">
      <div className="flex items-start gap-2">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-700 dark:text-amber-400" />
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-amber-900 dark:text-amber-200">
            HITL needed
          </p>
          <p className="truncate text-xs text-amber-800 dark:text-amber-300">
            {stepTitle}
          </p>
        </div>
        <Link
          href={`/projects/${projectId}/runs/${runId}`}
          target="_blank"
          rel="noopener"
          className="shrink-0 text-xs font-medium text-amber-900 underline dark:text-amber-200"
        >
          Resolve →
        </Link>
      </div>
    </div>
  );
}

function CostMeter({ run }: { run: AgentRunRead | null }) {
  const summary = run?.output_summary_json ?? {};
  const inTok = (summary.llm_input_tokens as number | undefined) ?? null;
  const outTok = (summary.llm_output_tokens as number | undefined) ?? null;
  const calls = (summary.ai_calls as number | undefined) ?? 0;
  const visionCalls = (summary.ai_vision_calls as number | undefined) ?? 0;
  if (calls === 0 && inTok === null && outTok === null) return null;

  const fmt = (n: number | null) =>
    n === null ? "–" : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`;

  return (
    <div className="border-b px-3 py-1.5 text-[10px] text-muted-foreground">
      <span className="font-medium uppercase tracking-wide">AI</span>{" "}
      <span>
        {calls} call{calls === 1 ? "" : "s"}
      </span>
      {visionCalls > 0 && (
        <span className="ml-1.5">
          (<Eye className="inline size-3" /> {visionCalls})
        </span>
      )}
      <span className="mx-1.5">·</span>
      <span>in {fmt(inTok)}</span>
      <span className="mx-1.5">·</span>
      <span>out {fmt(outTok)}</span>
    </div>
  );
}

function EventStream({ events }: { events: readonly LiveEvent[] }) {
  if (events.length === 0) {
    return (
      <p className="py-8 text-center text-xs italic text-muted-foreground">
        Waiting for events…
      </p>
    );
  }
  return (
    <ul className="space-y-1.5">
      {events.map((ev) => (
        <li key={ev.seq}>
          <EventRow event={ev} />
        </li>
      ))}
    </ul>
  );
}

function EventRow({ event }: { event: LiveEvent }) {
  const { type, data } = event;
  // Rendering split per event type — each branch gets its own row shape
  // so the visual semantics match (icon, color, copy).
  if (type === "step_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-blue-500" />}
        label="step started"
        title={`${formatOrdinal(data.ordinal, data.total)} · ${data.title ?? ""}`}
        sublabel={data.action_type as string | undefined}
      />
    );
  }
  if (type === "step_completed") {
    const status = data.status as string;
    const Icon =
      status === "passed"
        ? CheckCircle2
        : status === "failed"
          ? XCircle
          : status === "blocked"
            ? AlertTriangle
            : Circle;
    const colorClass =
      status === "passed"
        ? "text-emerald-500"
        : status === "failed"
          ? "text-red-500"
          : status === "blocked"
            ? "text-amber-500"
            : "text-muted-foreground";
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={status}
        title={data.narration as string | undefined}
        sublabel={
          typeof data.duration_ms === "number"
            ? `${(data.duration_ms / 1000).toFixed(1)}s`
            : undefined
        }
      />
    );
  }
  if (type === "step_retry") {
    return (
      <Row
        icon={<CircleDashed className="size-3.5 text-amber-500" />}
        label={`retry ${data.attempt}/${data.max_attempts}`}
        title={data.prior_error as string | undefined}
      />
    );
  }
  if (type === "ai_improvise_started" || type === "ai_assist_started") {
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label={
          type === "ai_improvise_started"
            ? "AI picking value…"
            : "AI thinking…"
        }
        title={data.title as string | undefined}
      />
    );
  }
  if (type === "ai_improvise_completed") {
    if (data.outcome === "picked") {
      return (
        <Row
          icon={<Bot className="size-3.5 text-purple-500" />}
          label="AI improvised"
          title={`"${data.value}"`}
          sublabel={
            typeof data.confidence === "number"
              ? `confidence ${Math.round(data.confidence * 100)}%`
              : undefined
          }
        />
      );
    }
    return (
      <Row
        icon={<Bot className="size-3.5 text-muted-foreground" />}
        label={`AI improvise: ${data.outcome}`}
        title={(data.reasoning as string | undefined) ?? ""}
      />
    );
  }
  if (type === "ai_assist_completed") {
    return (
      <Row
        icon={<Bot className="size-3.5 text-purple-500" />}
        label={`AI ${data.outcome ?? data.action ?? "result"}`}
        title={
          Array.isArray(data.diff_keys) && data.diff_keys.length > 0
            ? `changed: ${(data.diff_keys as string[]).join(", ")}`
            : undefined
        }
        sublabel={data.used_vision ? "vision" : undefined}
      />
    );
  }
  if (type === "needs_intervention") {
    return (
      <Row
        icon={<AlertTriangle className="size-3.5 text-amber-500" />}
        label="needs intervention"
        title={data.title as string | undefined}
        sublabel={data.error_message as string | undefined}
      />
    );
  }
  if (type === "intervention_resolved" || type === "intervention_auto_applied") {
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={
          type === "intervention_auto_applied"
            ? `auto-applied: ${data.choice ?? ""}`
            : `user chose: ${data.choice ?? ""}`
        }
      />
    );
  }
  // ── Agentic-mode events ──────────────────────────────────────
  if (type === "agent_goal_extracting") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-purple-500" />}
        label="agent · extracting goal"
        title={
          typeof data.ordinal === "number" && typeof data.total === "number"
            ? `test case ${data.ordinal}/${data.total}`
            : undefined
        }
      />
    );
  }
  if (type === "agent_goal_ready") {
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label="agent · goal"
        title={data.description as string | undefined}
        sublabel={
          typeof data.criteria_count === "number"
            ? `${data.criteria_count} success criteria`
            : undefined
        }
      />
    );
  }

  // Phase A — vision-driven sub-goal decomposition events. The agent
  // calls decompose_goal once per submodule (and once per replan)
  // and the live feed shows what the VL planner came up with — gives
  // the user a glimpse of the agent's mental model before any
  // actions fire.
  // Phase H — preflight pass (Scout → Refine → Activate) that runs
  // BEFORE per-submodule execution so the agent reads a UI-grounded
  // test plan, not the BRD-derived baseline.
  // Phase W — reading-session lifecycle events.
  if (type === "recording_ready") {
    return (
      <Row
        icon={
          <span className="relative inline-flex size-3.5 items-center justify-center">
            <span className="absolute inset-0 animate-ping rounded-full bg-rose-400 opacity-75" />
            <span className="relative size-2 rounded-full bg-rose-500" />
          </span>
        }
        label="reading · ready"
        title="Browser open — click/type to capture"
        sublabel={data.target_url as string | undefined}
      />
    );
  }
  if (type === "recording_saved") {
    const count = typeof data.event_count === "number" ? data.event_count : "?";
    const saved = !!data.saved;
    return (
      <Row
        icon={
          saved
            ? <CheckCircle2 className="size-3.5 text-emerald-500" />
            : <XCircle className="size-3.5 text-rose-500" />
        }
        label={saved ? "reading · saved" : "reading · save failed"}
        title={`${count} action${count === 1 ? "" : "s"} captured`}
        sublabel={data.reason as string | undefined}
      />
    );
  }
  // Phase W.6 — per-submodule recording-check diagnostic. Fires
  // for EVERY submodule whether or not a recording is present, so
  // the operator can see why some replay and some don't (frozen_path
  // wiped, overwritten by agent_freeze, or empty).
  if (type === "submodule_recording_check") {
    const willReplay = !!data.will_replay;
    const kind = (data.recording_kind as string | undefined) ?? "none";
    const count = typeof data.action_count === "number" ? data.action_count : 0;
    const skip = (data.skip_reason as string | undefined) ?? "";
    const title = (data.title as string | undefined) ?? "";
    const reasonLabel =
      skip === "no_frozen_path" ? "no recording on submodule"
      : skip === "wrong_kind" ? `overwritten by ${kind}`
      : skip === "empty_actions" ? "recording is empty"
      : `${count} actions · kind=${kind}`;
    return (
      <Row
        icon={
          willReplay
            ? <Sparkles className="size-3.5 text-emerald-500" />
            : <AlertTriangle className="size-3.5 text-amber-500" />
        }
        label={willReplay ? "replay · recording found" : "replay · skipped (agent will run)"}
        title={title || undefined}
        sublabel={reasonLabel}
      />
    );
  }
  // Trace playback events — fired during agentic runs that find a
  // saved trace on a submodule. Internal event names retain the
  // legacy ``recording_replay_*`` prefix (SSE wire contract) but
  // user-facing strings use "trace" / "step" — Tricentis-style
  // technical terms instead of the consumer-grade "recording".
  if (type === "submodule_recording_detected") {
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-rose-500" />}
        label="trace · detected"
        sublabel={`${data.action_count ?? "?"} captured steps to walk`}
      />
    );
  }
  if (type === "recording_replay_started") {
    return (
      <Row
        icon={<Play className="size-3.5 text-emerald-500" />}
        label="trace · playback started"
        sublabel={`${data.action_count ?? "?"} steps`}
      />
    );
  }
  if (type === "recording_replay_action") {
    const kind = (data.kind as string | undefined) ?? "?";
    const txt = (data.target_text as string | undefined) ?? "";
    const val = (data.value_preview as string | undefined) ?? "";
    const desc = (data.description as string | undefined) ?? "";
    const fallback =
      kind === "type"
        ? `"${val}" → ${txt || "(focused field)"}`
        : (txt || kind);
    return (
      <Row
        icon={<CircleDashed className="size-3.5 text-emerald-500" />}
        label={`step ${(data.action_index as number ?? 0) + 1} · ${kind}`}
        title={desc || fallback}
        sublabel={desc ? fallback : undefined}
      />
    );
  }
  if (type === "recording_replay_step_failed") {
    const desc = (data.description as string | undefined) ?? "";
    const txt = (data.target_text as string | undefined) ?? "";
    return (
      <Row
        icon={<XCircle className="size-3.5 text-rose-500" />}
        label={`step ${(data.action_index as number ?? 0) + 1} · ${data.kind ?? "?"} · FAILED`}
        title={desc || txt || undefined}
        sublabel={data.error as string | undefined}
      />
    );
  }
  // Per-step self-heal events. Fired when trace playback can't
  // resolve a single step and hands it to the agent for a one-shot
  // vision-assisted fix. Trace stays canonical; agent only patches
  // the ONE failed step.
  if (type === "recording_replay_self_heal_attempting") {
    const desc = (data.description as string | undefined) ?? "";
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-amber-500" />}
        label={`step ${(data.action_index as number ?? 0) + 1} · agent healing…`}
        title={desc || undefined}
        sublabel={data.error as string | undefined}
      />
    );
  }
  if (type === "recording_replay_self_healed") {
    const desc = (data.description as string | undefined) ?? "";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={`step ${(data.action_index as number ?? 0) + 1} · healed by agent`}
        title={desc || undefined}
        sublabel="vision found the right element — trace continues"
      />
    );
  }
  if (type === "recording_replay_self_heal_failed") {
    const desc = (data.description as string | undefined) ?? "";
    return (
      <Row
        icon={<XCircle className="size-3.5 text-rose-500" />}
        label={`step ${(data.action_index as number ?? 0) + 1} · heal failed`}
        title={desc || undefined}
        sublabel="agent could not locate the element — counted as failed"
      />
    );
  }
  // Phase Z.5 — per-submodule trace screenshots. Captured at trace
  // start, every N steps, on failures, and at trace end. The path
  // is RELATIVE to /static/screenshots, which is mounted by the
  // FastAPI app; the live presenter renders an inline thumbnail
  // (lazy-loaded) plus a click-through to the full frame.
  if (type === "recording_replay_screenshot") {
    const path = (data.path as string | undefined) ?? "";
    const tag = (data.tag as string | undefined) ?? "?";
    const idx = data.action_index as number | null | undefined;
    const url = path ? `/static/screenshots/${path}` : "";
    return (
      <div className="flex items-start gap-2 rounded border bg-card px-2 py-1.5">
        <div className="mt-0.5 shrink-0">
          <CircleDashed className="size-3.5 text-sky-500" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            trace · screenshot ({tag})
          </p>
          {idx != null && (
            <p className="break-words text-xs">at step {idx + 1}</p>
          )}
          {url && (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 block w-fit"
              title="open full frame"
            >
              <img
                src={url}
                alt={`trace frame ${tag}`}
                loading="lazy"
                className="h-24 max-w-[280px] rounded border border-border object-cover object-left-top"
              />
            </a>
          )}
        </div>
      </div>
    );
  }
  if (type === "recording_replay_completed") {
    const status = (data.status as string | undefined) ?? "completed";
    const exec = data.actions_executed ?? "?";
    const failed = data.actions_failed ?? 0;
    return (
      <Row
        icon={
          status === "completed"
            ? <CheckCircle2 className="size-3.5 text-emerald-500" />
            : status === "partial"
            ? <AlertTriangle className="size-3.5 text-amber-500" />
            : <XCircle className="size-3.5 text-rose-500" />
        }
        label={`trace · ${status}`}
        title={`${exec} executed · ${failed} failed`}
        sublabel={`${data.duration_s ?? "?"}s`}
      />
    );
  }
  // Phase AB — live-watch (post-completion review). After every
  // test case has been processed, the runner keeps the browser
  // open and emits periodic cheap-VL observations. The operator
  // ends the watch with Stop (cancel) or Ctrl+Shift+D (pass).
  if (type === "live_watch_started") {
    const hasVision = !!data.has_vision;
    const interval = (data.interval_s as number | undefined) ?? "?";
    const cap = (data.max_duration_s as number | undefined) ?? "?";
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-sky-500" />}
        label="live-watch · started"
        title={
          hasVision
            ? `Watching the browser. VL analysis every ${interval}s.`
            : "Watching the browser (no vision model — screenshots only)."
        }
        sublabel={`Auto-ends in ${cap}s · Stop or Ctrl+Shift+D to end now`}
      />
    );
  }
  if (type === "live_watch_observation") {
    const path = (data.path as string | undefined) ?? "";
    const observation = (data.observation as string | undefined) ?? "";
    const url = path ? `/static/screenshots/${path}` : "";
    const pageUrl = (data.url as string | undefined) ?? "";
    const idx = data.capture_idx ?? "?";
    const elapsed = data.elapsed_s ?? "?";
    return (
      <div className="flex items-start gap-2 rounded border bg-card px-2 py-1.5">
        <div className="mt-0.5 shrink-0">
          <Sparkles className="size-3.5 text-sky-500" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            live-watch · observation #{idx} · {elapsed}s
          </p>
          {observation && (
            <p className="break-words text-xs">{observation}</p>
          )}
          {pageUrl && (
            <p className="truncate text-[10px] text-muted-foreground">
              {pageUrl}
            </p>
          )}
          {url && (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 block w-fit"
              title="open full frame"
            >
              <img
                src={url}
                alt={`live-watch frame ${idx}`}
                loading="lazy"
                className="h-24 max-w-[280px] rounded border border-border object-cover object-left-top"
              />
            </a>
          )}
        </div>
      </div>
    );
  }
  if (type === "live_watch_ended") {
    const reason = (data.reason as string | undefined) ?? "?";
    const captures = data.captures ?? "?";
    return (
      <Row
        icon={
          reason === "force_passed"
            ? <CheckCircle2 className="size-3.5 text-emerald-500" />
            : reason === "cancelled"
            ? <XCircle className="size-3.5 text-rose-500" />
            : <AlertTriangle className="size-3.5 text-amber-500" />
        }
        label={`live-watch · ended (${reason})`}
        title={`${captures} observation${captures === 1 ? "" : "s"} captured`}
      />
    );
  }
  // Phase AA — operator force-pass event from the runner.
  if (type === "operator_force_pass") {
    const promoted = data.promoted_rows ?? "?";
    const total = data.total_rows ?? "?";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label="operator · force-pass"
        title={`${promoted} of ${total} test case${total === 1 ? "" : "s"} promoted to passed`}
        sublabel="Run marked completed (visually verified via Ctrl+Shift+D)"
      />
    );
  }

  if (type === "preflight_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label="preflight · started"
        title={
          data.force
            ? "force re-run: rescout + refine"
            : "validating test cases against the actual UI"
        }
        sublabel={data.target_url as string | undefined}
      />
    );
  }
  if (type === "preflight_scout_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label="preflight · scout · started"
        sublabel={
          (data.depth as string | undefined) ?? "deep"
        }
      />
    );
  }
  if (type === "preflight_scout_auth_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label="preflight · scout · auth"
        title="logging in to scout the post-auth surface"
        sublabel={data.url as string | undefined}
      />
    );
  }
  if (type === "preflight_scout_auth_completed") {
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label="preflight · scout · auth ok"
        sublabel={`${data.iterations ?? "?"} auth iterations`}
      />
    );
  }
  if (type === "preflight_scout_auth_failed") {
    return (
      <Row
        icon={<XCircle className="size-3.5 text-rose-500" />}
        label="preflight · scout · auth failed"
        title={(data.error as string | undefined) ?? "auth failed"}
      />
    );
  }
  if (type === "preflight_scout_completed") {
    const pages = typeof data.pages === "number" ? data.pages : "?";
    const cs = typeof data.create_surfaces === "number" ? data.create_surfaces : "?";
    const mods = typeof data.modules === "number" ? data.modules : "?";
    const flows = typeof data.create_flows === "number" ? data.create_flows : "?";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label="preflight · scout · completed"
        title={`${pages} pages · ${cs} create-surfaces`}
        sublabel={`${mods} modules · ${flows} create-flows`}
      />
    );
  }
  if (type === "preflight_scout_failed" || type === "preflight_scout_empty") {
    return (
      <Row
        icon={<XCircle className="size-3.5 text-rose-500" />}
        label="preflight · scout · failed"
        title={(data.error as string | undefined) ?? "scout failed"}
      />
    );
  }
  if (type === "preflight_refine_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label="preflight · refine · started"
        sublabel={
          `${data.appmap_modules ?? "?"} modules · ` +
          `${data.appmap_create_flows ?? "?"} create-flows in AppMap`
        }
      />
    );
  }
  if (type === "preflight_refine_completed") {
    const sm = typeof data.submodules === "number" ? data.submodules : "?";
    const rw = typeof data.rewritten === "number" ? data.rewritten : 0;
    const ad = typeof data.added === "number" ? data.added : 0;
    const fl = typeof data.flagged_missing === "number" ? data.flagged_missing : 0;
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-amber-500" />}
        label="preflight · refine · completed"
        title={`${sm} submodules refined`}
        sublabel={
          `rewritten ${rw} · added ${ad} · flagged ${fl}`
        }
      />
    );
  }
  if (type === "preflight_activation_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label="preflight · activating"
        sublabel={`v${data.version_id ?? "?"}`}
      />
    );
  }
  if (type === "preflight_activation_completed") {
    const cre = typeof data.nodes_created === "number" ? data.nodes_created : "?";
    const rem = typeof data.nodes_removed === "number" ? data.nodes_removed : "?";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label="preflight · activated"
        title={`v${data.version_id ?? "?"} is now the live plan`}
        sublabel={`${cre} nodes created · ${rem} replaced`}
      />
    );
  }
  if (type === "preflight_completed") {
    const status = (data.status as string | undefined) ?? "completed";
    const scoutRan = data.scout_ran ? "scout✓" : "scout—";
    const refineRan = data.refine_ran ? "refine✓" : "refine—";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={`preflight · ${status}`}
        title={
          status === "skipped"
            ? "no changes needed; existing refined plan is current"
            : undefined
        }
        sublabel={`${scoutRan} · ${refineRan} · ${data.seconds ?? "?"}s`}
      />
    );
  }
  if (type === "preflight_failed") {
    return (
      <Row
        icon={<XCircle className="size-3.5 text-rose-500" />}
        label="preflight · failed"
        title={(data.error as string | undefined) ?? "preflight failed"}
        sublabel={data.stage as string | undefined}
      />
    );
  }
  // Per-submodule refinement events (also emitted standalone via the
  // /refine-from-app-map button — kept for both lifecycles).
  if (type === "tc_refinement_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label="refine · started"
        sublabel={`${data.submodule_count ?? "?"} submodules`}
      />
    );
  }
  if (type === "tc_refinement_submodule_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label={`refine · ${data.title ?? "submodule"}`}
        sublabel={`${data.step_count ?? "?"} steps`}
      />
    );
  }
  if (type === "tc_refinement_submodule_completed") {
    const kept = data.kept ?? 0;
    const rw = data.rewritten ?? 0;
    const ad = data.added ?? 0;
    const fl = data.flagged_missing ?? 0;
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={`refine · done · ${data.submodule_id ?? "?"}`}
        sublabel={`kept ${kept} · rewritten ${rw} · added ${ad} · flagged ${fl}`}
      />
    );
  }
  // Phase A.5 — authenticated scout + AppMap events. The scout
  // runs on the first submodule of a fresh target_url to build the
  // mindmap (modules → sections → create-flows). Subsequent runs
  // load the cached map. Cached → app_map_loaded; fresh →
  // app_map_scout_started → auth_scout_page (per page) →
  // auth_scout_create_captured (per drawer) → auth_scout_completed
  // → app_map_built.
  if (type === "app_map_loaded") {
    const mods = typeof data.modules === "number" ? data.modules : "?";
    const flows = typeof data.create_flows === "number" ? data.create_flows : "?";
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label={`app map · loaded`}
        title={`${mods} modules · ${flows} create-flows · cached`}
      />
    );
  }
  if (type === "app_map_scout_started") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-purple-500" />}
        label="scout · started"
        title="walking authenticated surface — first run only"
        sublabel={data.target_url as string | undefined}
      />
    );
  }
  if (type === "auth_scout_page") {
    const navPath = Array.isArray(data.nav_path)
      ? (data.nav_path as string[]).join(" → ")
      : "(landing)";
    const els = typeof data.elements === "number" ? `${data.elements} els` : "";
    return (
      <Row
        icon={<Eye className="size-3.5 text-purple-500" />}
        label={`scout · page · ${navPath}`}
        title={data.title as string | undefined}
        sublabel={els}
      />
    );
  }
  if (type === "auth_scout_create_captured") {
    const trigger = data.trigger as string | undefined;
    const fields = typeof data.fields === "number" ? data.fields : "?";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={`scout · create-form captured`}
        title={trigger ? `trigger: "${trigger}"` : undefined}
        sublabel={`${fields} fields · submit "${data.submit_label ?? ""}"`}
      />
    );
  }
  if (type === "auth_scout_completed") {
    const pages = typeof data.pages_captured === "number" ? data.pages_captured : "?";
    const cs = typeof data.create_surfaces === "number" ? data.create_surfaces : "?";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label="scout · completed"
        sublabel={`${pages} pages · ${cs} create-flows`}
      />
    );
  }
  if (type === "app_map_built") {
    const mods = typeof data.modules === "number" ? data.modules : "?";
    const flows = typeof data.create_flows === "number" ? data.create_flows : "?";
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label="app map · built"
        title={`${mods} modules · ${flows} create-flows · saved to AKB`}
      />
    );
  }
  // Phase B — per-sub-goal frozen-path replay events. The replay
  // walker emits these as it walks each frozen segment; the
  // partial-handoff event marks the transition from deterministic
  // replay to agentic recovery for the failed sub-goals.
  if (type === "frozen_segment_started") {
    return (
      <Row
        icon={<CircleDashed className="size-3.5 text-blue-500" />}
        label={`replay · segment · ${data.sub_goal_id ?? "?"}`}
        title={data.description as string | undefined}
      />
    );
  }
  if (type === "frozen_segment_completed") {
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={`replay · segment done · ${data.sub_goal_id ?? "?"}`}
      />
    );
  }
  if (type === "frozen_segment_failed") {
    return (
      <Row
        icon={<XCircle className="size-3.5 text-red-500" />}
        label={`replay · segment failed · ${data.sub_goal_id ?? "?"}`}
        title={data.reason as string | undefined}
        sublabel="skipping remaining steps of this segment"
      />
    );
  }
  if (type === "frozen_partial_handoff") {
    const failed = Array.isArray(data.failed_sub_goal_ids)
      ? (data.failed_sub_goal_ids as string[])
      : [];
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label="replay · partial → agentic recovery"
        title={
          failed.length > 0
            ? `agent re-decomposing for: ${failed.join(", ")}`
            : "handing off remaining work to the agent"
        }
      />
    );
  }
  if (type === "frozen_path_captured") {
    const version = (data.version as number | undefined) ?? 1;
    const segs = data.segments as number | undefined;
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-emerald-500" />}
        label={`frozen path · v${version} captured`}
        title={
          typeof data.step_count === "number"
            ? `${data.step_count} step(s)${segs ? ` · ${segs} segment(s)` : ""}`
            : undefined
        }
        sublabel={data.agent_model as string | undefined}
      />
    );
  }
  // Phase A.6 Step 6 — Plan ↔ AppMap reconciliation. Emitted once
  // after the first-time scout completes. Summarises how each
  // submodule maps onto the discovered app surface.
  if (type === "plan_reconciled") {
    const counts = (data.counts ?? {}) as Record<string, number>;
    const ok = counts.ok ?? 0;
    const unc = counts.uncertain ?? 0;
    const mm = counts.mismatch ?? 0;
    const miss = counts.missing ?? 0;
    const isClean = mm === 0 && miss === 0;
    const Icon = isClean ? CheckCircle2 : AlertTriangle;
    const colorClass = isClean ? "text-emerald-500" : "text-amber-500";
    const parts = [];
    if (ok > 0) parts.push(`${ok} ok`);
    if (unc > 0) parts.push(`${unc} uncertain`);
    if (mm > 0) parts.push(`${mm} mismatch`);
    if (miss > 0) parts.push(`${miss} missing`);
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label="plan ↔ app · reconciled"
        title={parts.join(" · ") || "(no rows)"}
        sublabel={
          isClean
            ? "all submodules align with the app map"
            : "see the reconciliation panel above for details"
        }
      />
    );
  }
  // Phase A.6 Step 4 — verify-in-list gate refused a sub-goal close.
  // The agent claimed verify done; the gate scanned for the typed
  // entity name in visible page text, didn't find it, refused the
  // close, and asked the agent to use search/scroll.
  if (type === "verify_check_failed") {
    return (
      <Row
        icon={<AlertTriangle className="size-3.5 text-amber-500" />}
        label="verify gate · refused"
        title={
          typeof data.looked_for === "string"
            ? `couldn't find "${data.looked_for}" in visible text`
            : (data.reason as string | undefined)
        }
        sublabel="agent must search/scroll the list before mark_done"
      />
    );
  }
  // Phase A.6 Step 1 — toast / inline-error signal from a submit.
  // Tells the user "the app pushed back" before the agent's next
  // turn even runs.
  if (type === "form_signal_detected") {
    const kind = (data.kind as string | undefined) ?? "?";
    const isError =
      kind === "toast_error" ||
      kind === "inline_error" ||
      kind === "validation_error";
    const isSuccess = kind === "toast_success";
    const Icon = isError
      ? XCircle
      : isSuccess
        ? CheckCircle2
        : AlertTriangle;
    const colorClass = isError
      ? "text-red-500"
      : isSuccess
        ? "text-emerald-500"
        : "text-amber-500";
    const fields =
      Array.isArray(data.fields) && (data.fields as string[]).length > 0
        ? `fields: ${(data.fields as string[]).join(", ")}`
        : undefined;
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={`form · ${kind.replace("_", " ")}`}
        title={data.message as string | undefined}
        sublabel={fields}
      />
    );
  }
  if (type === "sub_goal_verify_appended") {
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label="verify sub-goal · appended"
        title={data.reason as string | undefined}
      />
    );
  }
  if (type === "sub_goals_decomposed") {
    const count =
      typeof data.count === "number" ? data.count : 0;
    const replan = typeof data.replan_iteration === "number"
      ? data.replan_iteration : 0;
    const sgs = Array.isArray(data.sub_goals) ? data.sub_goals : [];
    const preview = sgs
      .slice(0, 3)
      .map((sg: { description?: string }, i: number) =>
        `${i + 1}. ${(sg.description ?? "").toString().slice(0, 80)}`,
      )
      .join("\n");
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label={
          replan > 0
            ? `sub-goals · replan ${replan} · ${count} new`
            : `sub-goals · decomposed into ${count}`
        }
        title={preview}
      />
    );
  }
  if (type === "sub_goal_started") {
    return (
      <Row
        icon={<CircleDashed className="size-3.5 text-purple-500" />}
        label={`sub-goal start · ${data.id ?? "?"}`}
        title={data.description as string | undefined}
      />
    );
  }
  if (type === "sub_goal_done") {
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={`sub-goal done · ${data.id ?? "?"}`}
        title={data.description as string | undefined}
      />
    );
  }
  if (type === "sub_goal_failed") {
    return (
      <Row
        icon={<XCircle className="size-3.5 text-red-500" />}
        label={`sub-goal failed · ${data.id ?? "?"}`}
        title={data.description as string | undefined}
        sublabel={data.reason as string | undefined}
      />
    );
  }
  if (type === "sub_goal_skipped") {
    return (
      <Row
        icon={<AlertTriangle className="size-3.5 text-amber-500" />}
        label={`sub-goal skipped · ${data.id ?? "?"}`}
        title={data.description as string | undefined}
        sublabel={data.reason as string | undefined}
      />
    );
  }
  if (type === "sub_goal_replan_started") {
    const iter = typeof data.iteration === "number" ? data.iteration : "?";
    const max = typeof data.max_replans === "number" ? data.max_replans : "?";
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-amber-500" />}
        label={`replan ${iter}/${max}`}
        title={
          typeof data.after_sub_goal === "string"
            ? `after sub-goal ${data.after_sub_goal} failed`
            : undefined
        }
      />
    );
  }
  if (type === "hitl_overlay_opened") {
    return (
      <Row
        icon={<AlertTriangle className="size-3.5 text-amber-500" />}
        label="HITL overlay opened"
        title={data.sub_goal as string | undefined}
        sublabel="waiting for user guidance in the test browser"
      />
    );
  }
  // Phase D — pre-submodule health signal from the runtime SOP.
  // Surfaces the validator's verdict on this submodule before any
  // turn fires so the user sees "this one's risky" up front.
  if (type === "submodule_pre_run_health") {
    const status = (data.validation_status as string | undefined) ?? "";
    const conf =
      typeof data.validation_confidence === "number"
        ? Math.round(data.validation_confidence * 100)
        : null;
    const risky = data.is_risky === true;
    const Icon = risky
      ? AlertTriangle
      : status === "confirmed"
        ? CheckCircle2
        : Eye;
    const colorClass = risky
      ? "text-amber-500"
      : status === "confirmed"
        ? "text-emerald-500"
        : "text-muted-foreground";
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={
          risky
            ? `pre-run · risky · ${status}${conf !== null ? ` (${conf}%)` : ""}`
            : `pre-run · ${status}${conf !== null ? ` (${conf}%)` : ""}`
        }
        title={data.title as string | undefined}
        sublabel={data.validation_reason as string | undefined}
      />
    );
  }
  // Phase C.4 — hybrid `type` action fell through to coord-typing
  // on first DOM miss. Distinct event from the heavier coord-click
  // rescue so the user can see WHEN the fast-fail saved them.
  if (type === "coord_type_fast_fail") {
    const conf =
      typeof data.confidence === "number"
        ? Math.round(data.confidence * 100)
        : null;
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label={`type · coord fast-fail${conf !== null ? ` (${conf}%)` : ""}`}
        title={data.label_visible as string | undefined}
        sublabel={
          typeof data.x === "number" && typeof data.y === "number"
            ? `clicked + typed at (${data.x}, ${data.y})`
            : undefined
        }
      />
    );
  }
  // Phase F — bundled fill_form routine events. Walk: started →
  // scanned → field × N → submit_attempt × M → completed.
  if (type === "form_fill_started") {
    const arr = (data.fields_requested as { label: string }[]) ?? [];
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label={`form-fill · ${arr.length} field(s) requested`}
        title={
          arr.length > 0
            ? arr.map((f) => f.label).slice(0, 4).join(", ") +
              (arr.length > 4 ? `, +${arr.length - 4} more` : "")
            : undefined
        }
        sublabel={
          typeof data.submit_label === "string"
            ? `submit="${data.submit_label}"`
            : undefined
        }
      />
    );
  }
  if (type === "form_fill_scanned") {
    const count =
      typeof data.detected_count === "number" ? data.detected_count : "?";
    return (
      <Row
        icon={<Eye className="size-3.5 text-muted-foreground" />}
        label={`form-fill · scanned · ${count} field(s)`}
      />
    );
  }
  if (type === "form_fill_field") {
    const status = (data.status as string | undefined) ?? "?";
    const Icon =
      status === "verified" || status === "filled"
        ? CheckCircle2
        : status === "miss"
          ? XCircle
          : status === "skipped"
            ? Circle
            : AlertTriangle;
    const colorClass =
      status === "verified" || status === "filled"
        ? "text-emerald-500"
        : status === "miss"
          ? "text-red-500"
          : status === "skipped"
            ? "text-muted-foreground"
            : "text-amber-500";
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={`field · ${data.label} · ${status}`}
        title={
          status === "verified" || status === "filled"
            ? typeof data.final_value === "string"
              ? `value: ${data.final_value}`
              : undefined
            : (data.error as string | undefined)
        }
        sublabel={
          typeof data.role === "string" && typeof data.attempts === "number"
            ? `${data.role} · ${data.attempts} attempt(s)`
            : undefined
        }
      />
    );
  }
  if (type === "form_fill_field_retry") {
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-amber-500" />}
        label={`field · retry · ${data.label}`}
        title={data.validation_error as string | undefined}
        sublabel={
          typeof data.attempts === "number"
            ? `attempt ${data.attempts}`
            : undefined
        }
      />
    );
  }
  if (type === "form_fill_submit_attempt") {
    const ok = data.ok === true;
    return (
      <Row
        icon={
          ok ? (
            <CheckCircle2 className="size-3.5 text-emerald-500" />
          ) : (
            <AlertTriangle className="size-3.5 text-amber-500" />
          )
        }
        label={
          ok
            ? `form-fill · submit ok (attempt ${data.attempt}/${data.max})`
            : `form-fill · submit retry (attempt ${data.attempt}/${data.max})`
        }
        title={data.message as string | undefined}
        sublabel={
          Array.isArray(data.invalid_fields) &&
          (data.invalid_fields as string[]).length > 0
            ? `invalid: ${(data.invalid_fields as string[]).join(", ")}`
            : undefined
        }
      />
    );
  }
  if (type === "form_fill_completed") {
    const filled = typeof data.filled === "number" ? data.filled : 0;
    const miss = typeof data.miss === "number" ? data.miss : 0;
    const status = (data.submit_status as string | undefined) ?? "?";
    const seconds =
      typeof data.seconds === "number" ? data.seconds.toFixed(1) : "?";
    const isClean = miss === 0 && status === "ok";
    return (
      <Row
        icon={
          isClean ? (
            <CheckCircle2 className="size-3.5 text-emerald-500" />
          ) : (
            <AlertTriangle className="size-3.5 text-amber-500" />
          )
        }
        label={`form-fill · ${filled} filled${miss ? `, ${miss} missed` : ""} · submit=${status}`}
        sublabel={`${seconds}s`}
      />
    );
  }
  // Phase F.1 — turn-loop heartbeat. Keeps the feed visibly alive
  // even when an observation / planner call is slow. Rendered tiny
  // so it doesn't dominate the feed.
  if (type === "agent_turn_starting") {
    const t = typeof data.turn === "number" ? data.turn : "?";
    const max = typeof data.max_turns === "number" ? data.max_turns : "?";
    return (
      <Row
        icon={<CircleDashed className="size-3 text-muted-foreground" />}
        label={`turn ${t}/${max}`}
      />
    );
  }
  // Phase F.1 — planner-after-HITL gave up streak detector.
  if (type === "planner_no_op_after_hitl") {
    return (
      <Row
        icon={<AlertTriangle className="size-3.5 text-red-500" />}
        label="planner · gave up after HITL"
        title={
          typeof data.noop_streak === "number"
            ? `${data.noop_streak} consecutive no-op tools after guidance`
            : undefined
        }
        sublabel="run halted to avoid silent freeze; submodule marked blocked"
      />
    );
  }
  // Phase C.1 — confirmation that the user's guidance reached the
  // planner's prompt. Closes the "I submitted but nothing happened"
  // feedback gap — operator sees this row immediately after
  // hitl_overlay_submitted, then watches the next agent_acted to
  // see what the planner did with the guidance.
  if (type === "hitl_overlay_consumed") {
    const preview =
      typeof data.guidance_preview === "string"
        ? data.guidance_preview
        : undefined;
    const turn = typeof data.turn === "number" ? data.turn : undefined;
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-emerald-500" />}
        label={
          turn !== undefined
            ? `HITL guidance · consumed at T${turn}`
            : "HITL guidance · consumed"
        }
        title={preview}
        sublabel="planner will act on this in the next turn"
      />
    );
  }
  if (type === "hitl_overlay_submitted") {
    const status = (data.status as string | undefined) ?? "?";
    const Icon =
      status === "submitted"
        ? CheckCircle2
        : status === "idle_timeout"
          ? AlertTriangle
          : Circle;
    const colorClass =
      status === "submitted"
        ? "text-emerald-500"
        : status === "idle_timeout"
          ? "text-amber-500"
          : "text-muted-foreground";
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={`HITL overlay · ${status}`}
        title={
          status === "submitted"
            ? (data.text_preview as string | undefined)
            : status === "idle_timeout"
              ? "no response within 15s — auto-skipped"
              : "user skipped"
        }
      />
    );
  }
  if (type === "coordinate_click_proposed") {
    const conf =
      typeof data.confidence === "number" ? data.confidence : 0;
    const okConf = conf >= 0.6;
    return (
      <Row
        icon={
          okConf ? (
            <Bot className="size-3.5 text-purple-500" />
          ) : (
            <AlertTriangle className="size-3.5 text-amber-500" />
          )
        }
        label={
          okConf
            ? `coord click proposed (${Math.round(conf * 100)}%)`
            : `coord click low-conf (${Math.round(conf * 100)}%)`
        }
        title={data.label_visible as string | undefined}
        sublabel={
          typeof data.x === "number" && typeof data.y === "number"
            ? `at (${data.x}, ${data.y})`
            : undefined
        }
      />
    );
  }
  if (type === "coordinate_click_completed") {
    const applied = !!data.applied;
    const okStatus = data.status === "ok";
    return (
      <Row
        icon={
          applied && okStatus ? (
            <CheckCircle2 className="size-3.5 text-emerald-500" />
          ) : applied ? (
            <XCircle className="size-3.5 text-red-500" />
          ) : (
            <AlertTriangle className="size-3.5 text-amber-500" />
          )
        }
        label={
          applied
            ? okStatus
              ? "coord click ✓"
              : "coord click failed"
            : "coord click skipped"
        }
      />
    );
  }
  if (type === "frozen_path_captured") {
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-emerald-500" />}
        label="frozen path captured"
        title={
          typeof data.step_count === "number"
            ? `${data.step_count} step(s) saved for replay`
            : "saved for replay"
        }
        sublabel={
          typeof data.agent_model === "string"
            ? `from ${data.agent_model}`
            : undefined
        }
      />
    );
  }
  if (type === "frozen_step_completed") {
    const status = data.status as string | undefined;
    const Icon =
      status === "ok"
        ? CheckCircle2
        : status === "blocked"
          ? AlertTriangle
          : XCircle;
    const colorClass =
      status === "ok"
        ? "text-emerald-500"
        : status === "blocked"
          ? "text-amber-500"
          : "text-red-500";
    const healed = !!data.self_healed;
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={
          typeof data.frozen_step_index === "number" &&
          typeof data.total === "number"
            ? `replay ${data.frozen_step_index}/${data.total}`
            : "replay step"
        }
        title={
          typeof data.tool === "string"
            ? `${data.tool}${healed ? " · self-healed" : ""}`
            : undefined
        }
      />
    );
  }
  if (type === "frozen_step_self_healing") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-purple-500" />}
        label="self-healing"
        title={data.target_hint as string | undefined}
        sublabel={
          typeof data.frozen_step_index === "number"
            ? `frozen step ${data.frozen_step_index}`
            : undefined
        }
      />
    );
  }
  if (type === "frozen_step_self_heal_completed") {
    const ok = !!data.healed;
    return (
      <Row
        icon={
          ok ? (
            <CheckCircle2 className="size-3.5 text-emerald-500" />
          ) : (
            <XCircle className="size-3.5 text-red-500" />
          )
        }
        label={ok ? "self-heal ✓" : "self-heal ✗"}
        title={
          typeof data.frozen_step_index === "number"
            ? `frozen step ${data.frozen_step_index}`
            : undefined
        }
      />
    );
  }
  if (type === "agent_verifying") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-purple-500" />}
        label="vision verifying goal"
        title={data.goal_description as string | undefined}
      />
    );
  }
  if (type === "agent_verified") {
    const verdict = data.verdict as string | undefined;
    const Icon =
      verdict === "pass"
        ? CheckCircle2
        : verdict === "partial"
          ? AlertTriangle
          : XCircle;
    const colorClass =
      verdict === "pass"
        ? "text-emerald-500"
        : verdict === "partial"
          ? "text-amber-500"
          : "text-red-500";
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={`vision verdict: ${verdict}`}
        title={data.reasoning as string | undefined}
        sublabel={
          typeof data.confidence === "number"
            ? `confidence ${Math.round(data.confidence * 100)}%`
            : undefined
        }
      />
    );
  }
  if (type === "agent_on_track_check") {
    const onTrack = !!data.on_track;
    return (
      <Row
        icon={
          onTrack ? (
            <CheckCircle2 className="size-3.5 text-emerald-500" />
          ) : (
            <AlertTriangle className="size-3.5 text-amber-500" />
          )
        }
        label={
          onTrack
            ? `on-track check ✓ (T${data.turn ?? "?"})`
            : `off-track warning (T${data.turn ?? "?"})`
        }
        title={
          (onTrack
            ? (data.reasoning as string | undefined)
            : (data.suggestion as string | undefined)) ?? undefined
        }
      />
    );
  }
  if (type === "agent_searching") {
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin text-purple-500" />}
        label={
          typeof data.attempt === "number" && typeof data.max_attempts === "number"
            ? `vision search ${data.attempt}/${data.max_attempts}`
            : "vision search"
        }
        title={
          typeof data.target_hint === "string"
            ? `looking for ${data.target_hint}`
            : undefined
        }
        sublabel={
          Array.isArray(data.near_misses) && data.near_misses.length > 0
            ? `${data.near_misses.length} near-miss(es) considered`
            : undefined
        }
      />
    );
  }
  if (type === "agent_search_completed") {
    const halt = data.halt as string | undefined;
    const ok = halt === "completed";
    return (
      <Row
        icon={
          ok ? (
            <CheckCircle2 className="size-3.5 text-emerald-500" />
          ) : (
            <XCircle className="size-3.5 text-amber-500" />
          )
        }
        label={ok ? "vision search ✓" : `vision search · ${halt}`}
        title={
          typeof data.attempts_used === "number"
            ? `after ${data.attempts_used} attempt(s)`
            : undefined
        }
      />
    );
  }
  if (type === "sub_goal_progress") {
    const remaining =
      typeof data.remaining === "number" ? data.remaining : 0;
    const total = typeof data.total === "number" ? data.total : 0;
    const done = total - remaining;
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={
          total > 0
            ? `sub-goal ✓ (${done}/${total})`
            : "sub-goal ✓"
        }
        title={data.description as string | undefined}
        sublabel={
          typeof data.turn === "number"
            ? `closed at T${data.turn}`
            : undefined
        }
      />
    );
  }
  if (type === "agent_thought") {
    return (
      <Row
        icon={<Bot className="size-3.5 text-purple-500" />}
        label={
          typeof data.turn === "number"
            ? `T${data.turn} · think → ${data.tool}`
            : `agent · ${data.tool}`
        }
        title={data.reasoning as string | undefined}
        sublabel={
          typeof data.confidence === "number"
            ? `confidence ${Math.round(data.confidence * 100)}%`
            : undefined
        }
      />
    );
  }
  if (type === "agent_acted") {
    const status = data.status as string | undefined;
    const Icon =
      status === "ok"
        ? CheckCircle2
        : status === "blocked"
          ? AlertTriangle
          : XCircle;
    const colorClass =
      status === "ok"
        ? "text-emerald-500"
        : status === "blocked"
          ? "text-amber-500"
          : "text-red-500";
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={
          typeof data.turn === "number"
            ? `T${data.turn} · act → ${data.tool}`
            : `agent · ${data.tool}`
        }
        title={data.narration as string | undefined}
        sublabel={data.error as string | undefined}
      />
    );
  }

  // Phase 4-α — auth-flow orchestrator events. Emitted by
  // ``auth_flow.run_auth_loop`` as it classifies each auth screen,
  // types into username/password/otp fields, and submits. Lets the
  // user see the credential handler working in real time instead of
  // staring at an opaque "agent · type" stream.
  if (type === "auth_screen_classified") {
    const kind = (data.kind as string | undefined) ?? "?";
    const conf =
      typeof data.confidence === "number"
        ? `${Math.round(data.confidence * 100)}%`
        : undefined;
    const errText =
      typeof data.error_text === "string" && data.error_text
        ? data.error_text
        : undefined;
    const iter =
      typeof data.iteration === "number" ? `iter ${data.iteration} · ` : "";
    const colorClass =
      kind === "success"
        ? "text-emerald-500"
        : kind === "captcha" || kind === "passkey"
          ? "text-amber-500"
          : kind === "login" || kind === "otp"
            ? "text-purple-500"
            : "text-muted-foreground";
    return (
      <Row
        icon={<Bot className={cn("size-3.5", colorClass)} />}
        label={`auth · ${iter}${kind}`}
        title={errText}
        sublabel={conf ? `confidence ${conf}` : undefined}
      />
    );
  }
  if (type === "auth_field_typed") {
    const field = (data.field as string | undefined) ?? "field";
    return (
      <Row
        icon={<CheckCircle2 className="size-3.5 text-emerald-500" />}
        label={`auth · typed ${field}`}
      />
    );
  }
  if (type === "auth_submitted") {
    const via = (data.via as string | undefined) ?? "submit";
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-purple-500" />}
        label={`auth · submitted (${via})`}
      />
    );
  }
  if (type === "auth_flow_completed") {
    const status = (data.status as string | undefined) ?? "?";
    const screens = Array.isArray(data.screens_seen)
      ? (data.screens_seen as string[]).join(" → ")
      : undefined;
    const manual = data.manual_intervention_used === true;
    const Icon = status === "ok" ? CheckCircle2 : AlertTriangle;
    const colorClass =
      status === "ok" ? "text-emerald-500" : "text-amber-500";
    return (
      <Row
        icon={<Icon className={cn("size-3.5", colorClass)} />}
        label={`auth · flow ${status}${manual ? " · used HITL" : ""}`}
        title={screens}
        sublabel={data.error_message as string | undefined}
      />
    );
  }

  // Phase 14 — smart candidate selection
  if (type === "smart_pick_started") {
    const matches =
      typeof data.match_count === "number" ? data.match_count : "?";
    return (
      <Row
        icon={<Bot className="size-3.5 text-violet-500" />}
        label={`smart-pick · ${matches} matches`}
        title={
          typeof data.target_hint === "string"
            ? `picking the right one for ${data.target_hint}`
            : undefined
        }
      />
    );
  }
  if (type === "smart_pick_completed") {
    const strat = (data.strategy as string | undefined) ?? "?";
    const conf =
      typeof data.confidence === "number"
        ? `${Math.round(data.confidence * 100)}%`
        : undefined;
    const rejected =
      typeof data.rejected_count === "number" && data.rejected_count > 0
        ? ` · ${data.rejected_count} rejected`
        : "";
    return (
      <Row
        icon={<Bot className="size-3.5 text-violet-500" />}
        label={`smart-pick · ${strat}${rejected}`}
        title={data.chosen_label as string | undefined}
        sublabel={conf ? `confidence ${conf}` : undefined}
      />
    );
  }

  // Phase 10 — popup/overlay classifier
  if (type === "popup_classified") {
    const kind = (data.kind as string | undefined) ?? "?";
    const conf =
      typeof data.confidence === "number"
        ? `${Math.round(data.confidence * 100)}%`
        : undefined;
    const colorClass =
      kind === "required_step"
        ? "text-emerald-500"
        : kind === "ad"
          ? "text-red-500"
          : kind === "dismissable_blocker"
            ? "text-amber-500"
            : "text-muted-foreground";
    return (
      <Row
        icon={<AlertTriangle className={`size-3.5 ${colorClass}`} />}
        label={`popup · ${kind}`}
        title={data.reasoning as string | undefined}
        sublabel={conf ? `confidence ${conf}` : undefined}
      />
    );
  }

  // Phase 11 — test-case dispute
  if (type === "test_case_disputed") {
    const kind = (data.kind as string | undefined) ?? "?";
    return (
      <Row
        icon={<AlertTriangle className="size-3.5 text-amber-500" />}
        label={`test case disputed · ${kind}`}
        title={data.evidence as string | undefined}
        sublabel={
          typeof data.suggested_fix === "string" && data.suggested_fix
            ? `fix: ${data.suggested_fix}`
            : undefined
        }
      />
    );
  }

  // Phase 1 — provider tier escalation
  if (type === "llm_escalated") {
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-amber-500" />}
        label={`llm escalated · ${data.role ?? "?"}`}
        title={
          typeof data.from_model === "string"
            ? `cheap tier (${data.from_model}) → strong`
            : undefined
        }
        sublabel={data.reason as string | undefined}
      />
    );
  }

  // Phase 9 — semantic verify (filled in after the helper lands)
  if (type === "semantic_verify_started") {
    return (
      <Row
        icon={<Bot className="size-3.5 text-cyan-500" />}
        label="semantic verify · checking screenshot"
        title={data.expected as string | undefined}
      />
    );
  }
  if (type === "semantic_verify_completed") {
    const verdict = (data.verdict as string | undefined) ?? "?";
    const Icon =
      verdict === "pass"
        ? CheckCircle2
        : verdict === "fail"
          ? XCircle
          : AlertTriangle;
    const colorClass =
      verdict === "pass"
        ? "text-emerald-500"
        : verdict === "fail"
          ? "text-red-500"
          : "text-amber-500";
    return (
      <Row
        icon={<Icon className={`size-3.5 ${colorClass}`} />}
        label={`semantic verify · ${verdict}`}
        title={data.reasoning as string | undefined}
        sublabel={
          typeof data.confidence === "number"
            ? `confidence ${Math.round(data.confidence * 100)}%`
            : undefined
        }
      />
    );
  }

  if (type === "fix_promoted") {
    return (
      <Row
        icon={<Sparkles className="size-3.5 text-emerald-500" />}
        label="fix promoted to test case"
        title={
          Array.isArray(data.fields)
            ? (data.fields as string[]).join(", ")
            : undefined
        }
      />
    );
  }
  if (type === "paused" || type === "resumed") {
    return (
      <Row
        icon={
          type === "paused" ? (
            <Pause className="size-3.5 text-amber-500" />
          ) : (
            <Play className="size-3.5 text-emerald-500" />
          )
        }
        label={type}
      />
    );
  }
  if (type === "phase") {
    return (
      <Row
        icon={<Circle className="size-3.5 text-muted-foreground" />}
        label={(data.phase as string | undefined) ?? "phase"}
        title={data.message as string | undefined}
      />
    );
  }
  if (type === "completed" || type === "failed" || type === "cancelled" || type === "done") {
    return (
      <Row
        icon={
          type === "completed" || type === "done" ? (
            <CheckCircle2 className="size-3.5 text-emerald-500" />
          ) : type === "failed" ? (
            <XCircle className="size-3.5 text-red-500" />
          ) : (
            <Square className="size-3.5 text-muted-foreground" />
          )
        }
        label={type}
        title={
          typeof data.passed === "number"
            ? (() => {
                const total =
                  typeof data.total_steps === "number"
                    ? data.total_steps
                    : (data.passed as number) +
                      ((data.failed as number) ?? 0) +
                      ((data.inconclusive as number) ?? 0) +
                      ((data.blocked as number) ?? 0) +
                      ((data.skipped as number) ?? 0);
                const pct =
                  total > 0
                    ? Math.round(((data.passed as number) / total) * 100)
                    : 0;
                return [
                  `${data.passed}/${total} passed (${pct}%)`,
                  `${data.failed ?? 0} failed`,
                  ...(typeof data.inconclusive === "number" &&
                  data.inconclusive > 0
                    ? [`${data.inconclusive} inconclusive`]
                    : []),
                  `${data.blocked ?? 0} blocked`,
                ].join(" · ");
              })()
            : (data.message as string | undefined)
        }
      />
    );
  }

  // Generic fallback for any new event type we forgot.
  return (
    <Row
      icon={<Circle className="size-3.5 text-muted-foreground" />}
      label={type}
    />
  );
}

function Row({
  icon,
  label,
  title,
  sublabel,
}: {
  icon: React.ReactNode;
  label: string;
  title?: string;
  sublabel?: string;
}) {
  return (
    <div className="flex items-start gap-2 rounded border bg-card px-2 py-1.5">
      <div className="mt-0.5 shrink-0">{icon}</div>
      <div className="min-w-0 flex-1">
        <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </p>
        {title && <p className="break-words text-xs">{title}</p>}
        {sublabel && (
          <p className="text-[10px] text-muted-foreground">{sublabel}</p>
        )}
      </div>
    </div>
  );
}

function Footer({
  projectId,
  runId,
}: {
  projectId: number;
  runId: number;
}) {
  return (
    <div className="border-t px-3 py-1.5">
      <Link
        href={`/projects/${projectId}/runs/${runId}`}
        target="_blank"
        rel="noopener"
        className="inline-flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground"
      >
        <ExternalLink className="size-3" />
        Open full run detail
      </Link>
    </div>
  );
}

function formatOrdinal(
  ordinal: unknown,
  total: unknown,
): string {
  if (typeof ordinal === "number" && typeof total === "number") {
    return `${ordinal}/${total}`;
  }
  return "";
}


// ── Phase A.6 Step 5 — Scout progress panel ──────────────────────


interface ScoutPageEntry {
  url: string;
  title: string;
  nav_path: string[];
  elements?: number;
  has_create?: boolean;
  trigger?: string;
  fields?: number;
  submit_label?: string;
}

interface ScoutState {
  active: boolean;
  finished: boolean;
  targetUrl: string;
  pages: ScoutPageEntry[];
  modules?: number;
  flows?: number;
  pagesScouted?: number;
  cached: boolean;
}

function ScoutProgressPanel({
  projectId: _projectId,
  runId: _runId,
  events,
}: {
  projectId: number;
  runId: number;
  events: LiveEvent[];
}) {
  // Derive scout state from the event stream.
  const state: ScoutState = (() => {
    const s: ScoutState = {
      active: false,
      finished: false,
      targetUrl: "",
      pages: [],
      cached: false,
    };
    for (const ev of events) {
      const d = ev.data ?? {};
      if (ev.type === "app_map_scout_started") {
        s.active = true;
        s.finished = false;
        s.targetUrl = (d.target_url as string) ?? "";
        s.pages = [];
        s.cached = false;
      } else if (ev.type === "app_map_loaded") {
        // Cached map — surface it briefly so user knows the
        // scout was SKIPPED on this run.
        s.cached = true;
        s.finished = true;
        s.modules = typeof d.modules === "number" ? d.modules : undefined;
        s.flows =
          typeof d.create_flows === "number" ? d.create_flows : undefined;
      } else if (ev.type === "auth_scout_page") {
        const navPath = Array.isArray(d.nav_path)
          ? (d.nav_path as string[])
          : [];
        s.pages.push({
          url: (d.url as string) ?? "",
          title: (d.title as string) ?? "",
          nav_path: navPath,
          elements: typeof d.elements === "number" ? d.elements : undefined,
        });
      } else if (ev.type === "auth_scout_create_captured") {
        // Mark the most recent page as having a create-surface.
        const last = s.pages[s.pages.length - 1];
        if (last) {
          last.has_create = true;
          last.trigger = (d.trigger as string) ?? undefined;
          last.fields =
            typeof d.fields === "number" ? d.fields : undefined;
          last.submit_label =
            (d.submit_label as string) ?? undefined;
        }
      } else if (ev.type === "auth_scout_completed") {
        s.pagesScouted =
          typeof d.pages_captured === "number"
            ? d.pages_captured
            : undefined;
      } else if (ev.type === "app_map_built") {
        s.finished = true;
        s.active = false;
        s.modules =
          typeof d.modules === "number" ? d.modules : s.modules;
        s.flows =
          typeof d.create_flows === "number" ? d.create_flows : s.flows;
        s.pagesScouted =
          typeof d.pages_scouted === "number"
            ? d.pages_scouted
            : s.pagesScouted;
      }
    }
    return s;
  })();

  const [dismissed, setDismissed] = useState(false);
  // Auto-undismiss when a fresh scout starts.
  useEffect(() => {
    if (state.active && !state.finished) setDismissed(false);
  }, [state.active, state.finished]);

  if (dismissed || (!state.active && !state.finished)) return null;

  return (
    <div className="border-b border-purple-500/30 bg-purple-500/5 px-3 py-2">
      <div className="flex items-start gap-2">
        {state.active && !state.finished ? (
          <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-purple-600" />
        ) : (
          <Sparkles className="mt-0.5 size-4 shrink-0 text-purple-600" />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <p className="text-xs font-medium text-purple-900 dark:text-purple-200">
              {state.cached
                ? "App map loaded from cache"
                : state.active && !state.finished
                  ? "Building app mindmap (first-time scout)…"
                  : "App map built"}
            </p>
            <button
              type="button"
              className="text-[10px] text-muted-foreground hover:text-foreground"
              onClick={() => setDismissed(true)}
              aria-label="Dismiss scout panel"
            >
              dismiss
            </button>
          </div>
          {!state.cached && state.targetUrl && (
            <p className="truncate text-[10px] text-purple-800/80 dark:text-purple-300/80">
              {state.targetUrl}
            </p>
          )}
          {state.pages.length > 0 && (
            <ol className="mt-1.5 max-h-32 space-y-0.5 overflow-y-auto pr-1">
              {state.pages.map((p, i) => (
                <li
                  key={`${p.url}-${i}`}
                  className="flex items-baseline gap-1.5 text-[10px] text-purple-900/90 dark:text-purple-200/90"
                >
                  <span className="shrink-0 font-mono text-purple-600">
                    {i + 1}.
                  </span>
                  <span className="min-w-0 flex-1 truncate">
                    {p.nav_path.length > 0
                      ? p.nav_path.join(" → ")
                      : p.title || p.url || "(landing)"}
                  </span>
                  {p.has_create && (
                    <span className="shrink-0 rounded border border-emerald-500/40 bg-emerald-500/10 px-1 text-emerald-700 dark:text-emerald-400">
                      +{p.fields ?? "?"} fields
                    </span>
                  )}
                  {typeof p.elements === "number" && !p.has_create && (
                    <span className="shrink-0 text-[9px] text-muted-foreground">
                      {p.elements} els
                    </span>
                  )}
                </li>
              ))}
            </ol>
          )}
          {state.finished && (
            <p className="mt-1 text-[10px] text-purple-800/80 dark:text-purple-300/80">
              {state.cached ? (
                <>
                  {state.modules ?? "?"} modules ·{" "}
                  {state.flows ?? "?"} create-flows · reused (skip
                  re-scouting via plan editor &gt; refresh)
                </>
              ) : (
                <>
                  {state.modules ?? "?"} modules ·{" "}
                  {state.flows ?? "?"} create-flows ·{" "}
                  {state.pagesScouted ?? state.pages.length} pages —
                  open the plan editor to inspect the map
                </>
              )}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}


// ── Phase A.6 Step 6 — Plan ↔ AppMap reconciliation panel ─────────


type ReconStatus = "ok" | "uncertain" | "mismatch" | "missing";

interface ReconRow {
  submodule_id: number;
  title: string;
  status: ReconStatus;
  reason: string;
  matched_module?: string;
  matched_create_flow?: string;
}

const _RECON_TINT: Record<ReconStatus, string> = {
  ok: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  uncertain:
    "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  mismatch: "border-orange-500/30 bg-orange-500/10 text-orange-700 dark:text-orange-300",
  missing: "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300",
};

function ReconciliationPanel({ events }: { events: LiveEvent[] }) {
  // Take the LATEST plan_reconciled event in the stream.
  const latest = (() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (ev.type === "plan_reconciled") return ev;
    }
    return null;
  })();
  const [dismissed, setDismissed] = useState(false);
  useEffect(() => {
    if (latest) setDismissed(false);
  }, [latest]);
  if (!latest || dismissed) return null;

  const data = latest.data ?? {};
  const rows = (Array.isArray(data.rows) ? data.rows : []) as ReconRow[];
  if (rows.length === 0) return null;

  const counts: Record<ReconStatus, number> = {
    ok: 0, uncertain: 0, mismatch: 0, missing: 0,
  };
  for (const r of rows) {
    if (r.status in counts) counts[r.status] += 1;
  }
  const isClean = counts.mismatch === 0 && counts.missing === 0;

  return (
    <div className="border-b border-purple-500/20 bg-purple-500/5 px-3 py-2">
      <div className="flex items-start gap-2">
        {isClean ? (
          <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600" />
        ) : (
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600" />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <p className="text-xs font-medium text-purple-900 dark:text-purple-200">
              Plan ↔ App reconciliation
              <span className="ml-2 text-[10px] font-normal text-muted-foreground">
                {counts.ok} ok · {counts.uncertain} uncertain ·{" "}
                {counts.mismatch} mismatch · {counts.missing} missing
              </span>
            </p>
            <button
              type="button"
              className="text-[10px] text-muted-foreground hover:text-foreground"
              onClick={() => setDismissed(true)}
              aria-label="Dismiss reconciliation panel"
            >
              dismiss
            </button>
          </div>
          <ol className="mt-1.5 max-h-44 space-y-1 overflow-y-auto pr-1">
            {rows.map((r) => (
              <li
                key={r.submodule_id}
                className={cn(
                  "rounded border px-2 py-1 text-[11px]",
                  _RECON_TINT[r.status],
                )}
              >
                <div className="flex items-baseline gap-1.5">
                  <span className="shrink-0 font-mono text-[9px] uppercase tracking-wide opacity-80">
                    {r.status}
                  </span>
                  <span className="min-w-0 flex-1 truncate font-medium">
                    {r.title || `Submodule ${r.submodule_id}`}
                  </span>
                  {r.matched_module && (
                    <span className="shrink-0 text-[9px] opacity-80">
                      → {r.matched_module}
                      {r.matched_create_flow
                        ? ` / ${r.matched_create_flow}`
                        : ""}
                    </span>
                  )}
                </div>
                {r.reason && r.status !== "ok" && (
                  <p className="ml-1 mt-0.5 text-[10px] italic opacity-80">
                    {r.reason}
                  </p>
                )}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}
