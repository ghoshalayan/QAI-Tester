"use client";

import { useState } from "react";
import type { ComponentType } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Circle,
  Hourglass,
  Image as ImageIcon,
  Loader2,
  MinusCircle,
  Sparkles,
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
};

const STATUS_COLOR: Record<ExecutionStepStatus, string> = {
  pending: "text-muted-foreground",
  running: "text-blue-600 dark:text-blue-400",
  passed: "text-green-600 dark:text-green-400",
  failed: "text-red-600 dark:text-red-400",
  blocked: "text-yellow-700 dark:text-yellow-400",
  skipped: "text-muted-foreground",
};

const STATUS_ROW_TINT: Record<ExecutionStepStatus, string> = {
  pending: "",
  running: "border-l-blue-500",
  passed: "border-l-green-500/60",
  failed: "border-l-red-500",
  blocked: "border-l-yellow-500",
  skipped: "border-l-muted-foreground/30",
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
