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

import { useEffect } from "react";
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
} from "@/lib/api";
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
            ? [
                `${data.passed} passed`,
                `${data.failed ?? 0} failed`,
                ...(typeof data.inconclusive === "number" &&
                data.inconclusive > 0
                  ? [`${data.inconclusive} inconclusive`]
                  : []),
                `${data.blocked ?? 0} blocked`,
              ].join(" · ")
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
