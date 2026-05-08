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

  return (
    <div className="flex h-screen flex-col bg-background text-sm text-foreground">
      <Header
        run={run ?? null}
        planName={plan?.name ?? null}
        progress={progress?.message}
      />

      <Controls
        isPaused={isPaused}
        isRunning={isRunning}
        isTerminal={isTerminal}
        pausePending={pauseMut.isPending}
        resumePending={resumeMut.isPending}
        cancelPending={cancelMut.isPending}
        onPause={() => pauseMut.mutate()}
        onResume={() => resumeMut.mutate()}
        onCancel={() => cancelMut.mutate()}
      />

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
  onPause,
  onResume,
  onCancel,
}: {
  isPaused: boolean;
  isRunning: boolean;
  isTerminal: boolean;
  pausePending: boolean;
  resumePending: boolean;
  cancelPending: boolean;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
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
