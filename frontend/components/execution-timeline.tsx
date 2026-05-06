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
  Loader2,
  MinusCircle,
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
