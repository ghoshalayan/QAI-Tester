"use client";

import { useState } from "react";
import type { ComponentType } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Bot,
  Camera,
  CheckCircle2,
  Circle,
  Hourglass,
  Image as ImageIcon,
  Loader2,
  MinusCircle,
  Sparkles,
  Target,
  XCircle,
} from "lucide-react";

import {
  api,
  EXECUTION_STEP_STATUS_LABELS,
  type ExecutionStepRead,
  type ExecutionStepStatus,
} from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

const STATUS_ICON: Record<
  ExecutionStepStatus,
  ComponentType<{ className?: string }>
> = {
  pending: Circle,
  running: Loader2,
  passed: CheckCircle2,
  failed: XCircle,
  blocked: AlertTriangle,
  skipped: MinusCircle,
  inconclusive: AlertTriangle,
};

const STATUS_COLOR: Record<ExecutionStepStatus, string> = {
  pending: "text-muted-foreground",
  running: "text-blue-600 dark:text-blue-400",
  passed: "text-green-600 dark:text-green-400",
  failed: "text-red-600 dark:text-red-400",
  blocked: "text-yellow-700 dark:text-yellow-400",
  skipped: "text-muted-foreground",
  inconclusive: "text-orange-600 dark:text-orange-400",
};

const STATUS_ROW_TINT: Record<ExecutionStepStatus, string> = {
  pending: "",
  running: "border-l-blue-500",
  passed: "border-l-green-500/60",
  failed: "border-l-red-500",
  blocked: "border-l-yellow-500",
  skipped: "border-l-muted-foreground/30",
  inconclusive: "border-l-orange-500",
};

interface Props {
  projectId: number;
  runId: number;
  /** Optional status filter; defaults to "all" (show every row). */
  statusFilter?: ExecutionStepStatus | "all";
}

export function ExecutionTimeline({
  projectId,
  runId,
  statusFilter = "all",
}: Props) {
  const { data: steps, isLoading } = useQuery({
    queryKey: ["run-steps", projectId, runId],
    queryFn: () => api.listRunSteps(projectId, runId),
    refetchInterval: (query) => {
      // While there's a row not yet terminal, poll lightly as a fallback
      // for missed SSE events. Stop polling once everything's terminal.
      const data = query.state.data as ExecutionStepRead[] | undefined;
      if (!data || data.length === 0) return false;
      const anyActive = data.some(
        (s) => s.status === "pending" || s.status === "running",
      );
      return anyActive ? 5_000 : false;
    },
  });

  const [zoomedStep, setZoomedStep] = useState<ExecutionStepRead | null>(null);

  const visibleSteps =
    statusFilter === "all" || !steps
      ? steps ?? []
      : steps.filter((s) => s.status === statusFilter);

  if (isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-16 w-full" />
      </div>
    );
  }

  if (!steps || steps.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
        <Hourglass className="mx-auto size-8 text-muted-foreground" />
        <p className="mt-3">
          No steps yet — the runner will populate this as it walks the tree.
        </p>
      </div>
    );
  }

  if (visibleSteps.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No steps match the <strong>{statusFilter}</strong> filter.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border bg-card">
      <ul className="divide-y">
        {visibleSteps.map((step) => (
          <li key={step.id}>
            <ExecutionStepRow
              step={step}
              onZoom={() => setZoomedStep(step)}
            />
          </li>
        ))}
      </ul>

      <ScreenshotLightbox
        step={zoomedStep}
        onClose={() => setZoomedStep(null)}
      />
    </div>
  );
}

function ExecutionStepRow({
  step,
  onZoom,
}: {
  step: ExecutionStepRead;
  onZoom: () => void;
}) {
  const Icon = STATUS_ICON[step.status];
  const isRunning = step.status === "running";

  return (
    <div
      className={cn(
        "flex items-start gap-3 border-l-4 px-4 py-3 text-sm transition-colors",
        STATUS_ROW_TINT[step.status],
      )}
    >
      <div className="flex w-6 shrink-0 justify-center pt-0.5">
        <span className="font-mono text-[10px] text-muted-foreground">
          {step.ordinal + 1}
        </span>
      </div>
      <div className="flex shrink-0 pt-0.5">
        <Icon
          className={cn(
            "size-4",
            STATUS_COLOR[step.status],
            isRunning && "animate-spin",
          )}
        />
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="font-medium">{step.title_snapshot}</span>
          {step.action_type_snapshot && (
            <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              {step.action_type_snapshot}
            </span>
          )}
          <span
            className={cn(
              "text-[10px] font-medium uppercase tracking-wide",
              STATUS_COLOR[step.status],
            )}
          >
            {EXECUTION_STEP_STATUS_LABELS[step.status]}
          </span>
          {step.duration_ms !== null && step.duration_ms > 0 && (
            <span className="text-[10px] text-muted-foreground">
              {formatDuration(step.duration_ms)}
            </span>
          )}
        </div>

        {step.narration && (
          <p className="mt-0.5 break-words text-xs text-muted-foreground">
            {step.narration}
          </p>
        )}

        {step.error_message && (
          <p className="mt-1 break-words rounded-md border border-red-500/30 bg-red-500/5 px-2 py-1 font-mono text-[11px] text-red-700 dark:text-red-400">
            {step.error_message}
          </p>
        )}

        <AiCorrectionSurface step={step} />
        <AgentTurnsSurface step={step} />

        {step.path_snapshot &&
          step.path_snapshot !== step.title_snapshot && (
            <p className="mt-1 truncate text-[10px] text-muted-foreground">
              {step.path_snapshot}
            </p>
          )}
      </div>

      {step.screenshot_path && (
        <button
          type="button"
          onClick={onZoom}
          className="group relative shrink-0 overflow-hidden rounded-md border transition-colors hover:border-primary/50"
          title="Click to enlarge"
          aria-label={`Open screenshot for step ${step.ordinal + 1}`}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={api.screenshotUrl(step.screenshot_path)}
            alt={`Screenshot of step ${step.ordinal + 1}`}
            className="h-14 w-24 object-cover transition-transform group-hover:scale-105"
            loading="lazy"
          />
          <span className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/0 text-white opacity-0 transition-all group-hover:bg-black/40 group-hover:opacity-100">
            <Camera className="size-4" />
          </span>
        </button>
      )}
    </div>
  );
}

/**
 * Inline "Why" surface — renders when a step has ``details_json.ai_correction``
 * (set by the orchestrator after auto-retry exhausted and AI assist
 * proposed a fix). Auto-expanded for failed/blocked rows so the user can
 * see what the agent thought; collapsed for passed rows where the AI
 * helped quietly.
 */
function AiCorrectionSurface({ step }: { step: ExecutionStepRead }) {
  const raw = step.details_json?.ai_correction;
  if (!raw || typeof raw !== "object") return null;

  const c = raw as Record<string, unknown>;
  const action = typeof c.action === "string" ? c.action : "";
  const reasoning = typeof c.reasoning === "string" ? c.reasoning : "";
  const usedVision = !!c.used_vision;
  const tokensIn = typeof c.tokens_in === "number" ? c.tokens_in : null;
  const tokensOut = typeof c.tokens_out === "number" ? c.tokens_out : null;
  const diffObj =
    c.diff && typeof c.diff === "object"
      ? (c.diff as Record<string, { old?: unknown; new?: unknown }>)
      : {};
  const diffEntries = Object.entries(diffObj);

  const helped = step.status === "passed";

  return (
    <details
      className={cn(
        "mt-1 rounded-md border px-2 py-1 text-[11px]",
        helped
          ? "border-green-500/30 bg-green-500/5"
          : "border-blue-500/30 bg-blue-500/5",
      )}
      open={!helped}
    >
      <summary className="flex cursor-pointer flex-wrap items-center gap-1.5 list-none">
        <Sparkles
          className={cn(
            "size-3",
            helped
              ? "text-green-700 dark:text-green-400"
              : "text-blue-700 dark:text-blue-400",
          )}
        />
        <span className="font-medium">{helped ? "AI helped" : "AI tried"}</span>
        {action && (
          <span className="rounded border px-1 py-0.5 font-mono text-[9px] text-muted-foreground">
            {action}
          </span>
        )}
        {usedVision && (
          <span
            className="inline-flex items-center gap-0.5 rounded border border-blue-500/30 bg-blue-500/10 px-1 py-0.5 text-[9px] font-medium text-blue-700 dark:text-blue-400"
            title="Vision pass — model saw the page screenshot"
          >
            <ImageIcon className="size-2.5" />
            vision
          </span>
        )}
        {(tokensIn !== null || tokensOut !== null) && (
          <span className="ml-auto text-[9px] text-muted-foreground">
            ↑{tokensIn ?? 0} ↓{tokensOut ?? 0}
          </span>
        )}
      </summary>
      <div className="mt-1.5 space-y-1">
        {reasoning && (
          <p className="break-words text-foreground/85">{reasoning}</p>
        )}
        {diffEntries.length > 0 && (
          <div className="space-y-0.5 font-mono text-[10px]">
            {diffEntries.map(([field, change]) => (
              <div key={field} className="break-all">
                <span className="text-muted-foreground">{field}:</span>{" "}
                <span className="text-red-500/70 line-through">
                  {String((change as { old?: unknown }).old ?? "")}
                </span>
                {" → "}
                <span className="text-green-700 dark:text-green-400">
                  {String((change as { new?: unknown }).new ?? "")}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}

/**
 * Agentic-mode surface — renders the goal, success criteria, and the
 * full per-turn agent log when ``details_json.mode === "agentic"``.
 *
 * Auto-expanded for non-passed rows (failed / inconclusive / blocked)
 * so users can immediately see what the agent tried and why it halted.
 * Collapsed for passed rows where the trail is mostly noise.
 */
function AgentTurnsSurface({ step }: { step: ExecutionStepRead }) {
  const details = step.details_json ?? {};
  if ((details as Record<string, unknown>).mode !== "agentic") return null;

  const goal = (details as Record<string, unknown>).goal as
    | {
        description?: string;
        success_criteria?: string[];
        sub_goals?: Array<{
          id: string;
          description: string;
          status: string;
          completed_at_turn: number | null;
        }>;
      }
    | undefined;
  const haltReason = (details as Record<string, unknown>).halt_reason as
    | string
    | undefined;
  const turnLog = (details as Record<string, unknown>).agent_log;
  const turns: Array<Record<string, unknown>> = Array.isArray(turnLog)
    ? (turnLog as Array<Record<string, unknown>>)
    : [];
  const subGoals = Array.isArray(goal?.sub_goals) ? goal.sub_goals : [];

  const passed = step.status === "passed";

  return (
    <details
      className={cn(
        "mt-1 rounded-md border px-2 py-1 text-[11px]",
        passed
          ? "border-purple-500/30 bg-purple-500/5"
          : "border-orange-500/40 bg-orange-500/5",
      )}
      open={!passed}
    >
      <summary className="flex cursor-pointer flex-wrap items-center gap-1.5 list-none">
        <Bot
          className={cn(
            "size-3",
            passed
              ? "text-purple-700 dark:text-purple-400"
              : "text-orange-700 dark:text-orange-400",
          )}
        />
        <span className="font-medium">Agentic run</span>
        <span className="text-muted-foreground">
          · {turns.length} turn{turns.length === 1 ? "" : "s"}
        </span>
        {haltReason && (
          <span className="rounded border px-1 py-0.5 font-mono text-[9px] text-muted-foreground">
            halt: {haltReason}
          </span>
        )}
      </summary>

      <div className="mt-1.5 space-y-2">
        {goal?.description && (
          <div className="flex items-start gap-1.5 rounded border bg-card p-1.5">
            <Target className="mt-0.5 size-3 shrink-0 text-purple-600" />
            <div className="min-w-0 flex-1">
              <p className="break-words font-medium">
                {goal.description}
              </p>
              {Array.isArray(goal.success_criteria) &&
                goal.success_criteria.length > 0 && (
                  <ul className="mt-1 ml-3 list-disc space-y-0.5 text-[10px] text-muted-foreground">
                    {goal.success_criteria.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                )}
            </div>
          </div>
        )}

        {subGoals.length > 0 && <SubGoalChecklist subGoals={subGoals} />}

        {turns.length === 0 ? (
          <p className="italic text-muted-foreground">
            No turns ran (the agent halted before its first action).
          </p>
        ) : (
          <ol className="space-y-1">
            {turns.map((t, i) => (
              <AgentTurnRow key={i} turn={t} />
            ))}
          </ol>
        )}
      </div>
    </details>
  );
}

const _SUB_GOAL_GLYPH: Record<string, string> = {
  pending: "☐",
  in_progress: "▶",
  done: "✓",
  failed: "✗",
  skipped: "⊘",
};
const _SUB_GOAL_TINT: Record<string, string> = {
  pending: "text-muted-foreground",
  in_progress: "text-blue-600 dark:text-blue-400",
  done: "text-emerald-600 dark:text-emerald-400",
  failed: "text-red-600 dark:text-red-400",
  skipped: "text-amber-600 dark:text-amber-400",
};

function SubGoalChecklist({
  subGoals,
}: {
  subGoals: Array<{
    id: string;
    description: string;
    status: string;
    completed_at_turn: number | null;
  }>;
}) {
  const done = subGoals.filter((sg) => sg.status === "done").length;
  const total = subGoals.length;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  return (
    <div className="rounded border bg-card p-1.5 text-[11px]">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Sub-goals
        </span>
        <span className="text-[10px] text-muted-foreground">
          {done}/{total} done · {pct}%
        </span>
      </div>
      <ol className="space-y-0.5">
        {subGoals.map((sg) => (
          <li key={sg.id} className="flex items-baseline gap-1.5">
            <span
              className={cn(
                "shrink-0 font-mono",
                _SUB_GOAL_TINT[sg.status] ?? "text-muted-foreground",
              )}
            >
              {_SUB_GOAL_GLYPH[sg.status] ?? "?"}
            </span>
            <span className="font-mono text-[9px] text-muted-foreground">
              [{sg.id}]
            </span>
            <span
              className={cn(
                "min-w-0 flex-1 break-words",
                sg.status === "done" &&
                  "text-muted-foreground line-through decoration-emerald-500/60",
              )}
            >
              {sg.description}
            </span>
            {sg.completed_at_turn !== null && (
              <span className="shrink-0 text-[9px] text-muted-foreground">
                T{sg.completed_at_turn}
              </span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function AgentTurnRow({ turn }: { turn: Record<string, unknown> }) {
  const tNum = typeof turn.turn === "number" ? turn.turn : null;
  const tool = typeof turn.tool === "string" ? turn.tool : "?";
  const status = typeof turn.status === "string" ? turn.status : "ok";
  const reasoning =
    typeof turn.reasoning === "string" ? turn.reasoning : "";
  const narration =
    typeof turn.narration === "string" ? turn.narration : "";
  const error =
    typeof turn.error_message === "string" ? turn.error_message : null;
  const args = (turn.args ?? {}) as Record<string, unknown>;
  const confidence =
    typeof turn.confidence === "number" ? turn.confidence : null;

  // Compact one-line repr of the args the tool actually used.
  const argSummary = Object.entries(args)
    .filter(([_, v]) => v !== "" && v !== 0 && v !== null && v !== undefined)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
    .join(" ")
    .slice(0, 200);

  const Icon =
    status === "ok"
      ? CheckCircle2
      : status === "blocked"
        ? AlertTriangle
        : status === "stop"
          ? Sparkles
          : XCircle;
  const colorClass =
    status === "ok"
      ? "text-emerald-600"
      : status === "blocked"
        ? "text-amber-600"
        : status === "stop"
          ? "text-purple-600"
          : "text-red-600";

  return (
    <li className="rounded border bg-card px-2 py-1.5">
      <div className="flex flex-wrap items-baseline gap-1.5">
        <span className="font-mono text-[10px] text-muted-foreground">
          T{tNum ?? "?"}
        </span>
        <Icon className={cn("size-3", colorClass)} />
        <span className="rounded border px-1 py-0.5 font-mono text-[10px]">
          {tool}
        </span>
        {argSummary && (
          <span className="break-all font-mono text-[10px] text-muted-foreground">
            {argSummary}
          </span>
        )}
        {confidence !== null && (
          <span className="ml-auto text-[9px] text-muted-foreground">
            {Math.round(confidence * 100)}%
          </span>
        )}
      </div>
      {reasoning && (
        <p className="mt-0.5 break-words text-[11px]">{reasoning}</p>
      )}
      {narration && narration !== reasoning && (
        <p className="mt-0.5 break-words text-[10px] text-muted-foreground">
          → {narration}
        </p>
      )}
      {error && (
        <p className="mt-0.5 break-words rounded border border-red-500/30 bg-red-500/5 px-1.5 py-0.5 font-mono text-[10px] text-red-700 dark:text-red-400">
          {error}
        </p>
      )}
    </li>
  );
}


function ScreenshotLightbox({
  step,
  onClose,
}: {
  step: ExecutionStepRead | null;
  onClose: () => void;
}) {
  return (
    <Dialog open={step !== null} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-5xl overflow-auto">
        <DialogHeader>
          <DialogTitle className="flex flex-wrap items-center gap-2 text-sm">
            <span>Step {step ? step.ordinal + 1 : ""}</span>
            <span className="font-normal text-muted-foreground">·</span>
            <span className="font-normal">{step?.title_snapshot}</span>
            {step?.action_type_snapshot && (
              <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                {step.action_type_snapshot}
              </span>
            )}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Step screenshot, narration, and any error message captured during
            execution.
          </DialogDescription>
        </DialogHeader>
        {step?.screenshot_path && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={api.screenshotUrl(step.screenshot_path)}
            alt={`Screenshot of step ${step.ordinal + 1}`}
            className="mx-auto rounded-md border"
          />
        )}
        {step?.narration && (
          <p className="mt-2 text-xs text-muted-foreground">{step.narration}</p>
        )}
        {step?.error_message && (
          <p className="mt-1 break-words rounded-md border border-red-500/30 bg-red-500/5 px-2 py-1 font-mono text-[11px] text-red-700 dark:text-red-400">
            {step.error_message}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
