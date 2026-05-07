"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, Pause, Play, Sparkles, X } from "lucide-react";
import { toast } from "sonner";

import {
  AGENT_STATUS_LABELS,
  api,
  ApiError,
  type AgentRunRead,
  type AgentStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useAgentRunProgress } from "@/hooks/use-agent-runs-events";
import { cn } from "@/lib/utils";

const STATUS_CLASSES: Record<AgentStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
  paused: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
  completed: "bg-green-500/10 text-green-700 dark:text-green-400",
  failed: "bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
};

const ACTIVE_STATUSES: AgentStatus[] = ["queued", "running", "paused"];

const KIND_LABELS: Record<string, string> = {
  brd_to_frd: "BRD → FRD synthesis",
  frd_to_tc: "FRD → Test Case synthesis",
  execute: "Execution",
  reporter: "Reporter",
};

const PHASE_LABELS: Record<string, string> = {
  validating: "Validating",
  loading: "Loading chunks",
  calling_llm: "Calling LLM",
  chunking: "Chunking",
  persisting: "Persisting",
  embedding: "Embedding",
};

export function RunProgressCard({
  projectId,
  run,
}: {
  projectId: number;
  run: AgentRunRead;
}) {
  const qc = useQueryClient();
  const progress = useAgentRunProgress((s) => s.byRunId[run.id]);
  const isActive = ACTIVE_STATUSES.includes(run.status);

  const cancelMutation = useMutation({
    mutationFn: () => api.cancelAgentRun(projectId, run.id),
    onSuccess: () => {
      toast.success("Cancel requested", {
        description:
          "Will take effect at the next step boundary.",
      });
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.invalidateQueries({ queryKey: ["agent-run", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Cancel failed", { description: msg });
    },
  });

  const pauseMutation = useMutation({
    mutationFn: () => api.pauseAgentRun(projectId, run.id),
    onSuccess: () => {
      toast.success("Pause requested", {
        description: "Run will halt after the current step finishes.",
      });
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.invalidateQueries({ queryKey: ["agent-run", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Pause failed", { description: msg });
    },
  });

  const resumeMutation = useMutation({
    mutationFn: () => api.resumeAgentRun(projectId, run.id),
    onSuccess: () => {
      toast.success("Resumed");
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.invalidateQueries({ queryKey: ["agent-run", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Resume failed", { description: msg });
    },
  });

  const isFrdToTc = run.kind === "frd_to_tc";
  const isExecute = run.kind === "execute";
  const isRunning = run.status === "running";
  const isPaused = run.status === "paused";
  const canPause = isExecute && isRunning;
  const canResume = isExecute && isPaused;

  const sourceDocsCount = Array.isArray(run.input_json?.source_document_ids)
    ? (run.input_json.source_document_ids as number[]).length
    : 0;
  const capChunks =
    typeof run.input_json?.cap_chunks === "number"
      ? (run.input_json.cap_chunks as number)
      : null;
  const planId =
    typeof run.input_json?.plan_id === "number"
      ? (run.input_json.plan_id as number)
      : null;
  const capPerModuleFrds =
    typeof run.input_json?.cap_per_module_frds === "number"
      ? (run.input_json.cap_per_module_frds as number)
      : null;
  const capPerModuleChunks =
    typeof run.input_json?.cap_per_module_chunks === "number"
      ? (run.input_json.cap_per_module_chunks as number)
      : null;
  const headless = !!run.input_json?.headless;
  const selectedStepIdsCount = Array.isArray(run.input_json?.selected_step_ids)
    ? (run.input_json.selected_step_ids as number[]).length
    : null;

  const generated =
    typeof run.output_summary_json?.generated === "number"
      ? (run.output_summary_json.generated as number)
      : null;
  const inTokens =
    typeof run.output_summary_json?.input_tokens === "number"
      ? (run.output_summary_json.input_tokens as number)
      : null;
  const outTokens =
    typeof run.output_summary_json?.output_tokens === "number"
      ? (run.output_summary_json.output_tokens as number)
      : null;
  const modulesGenerated =
    typeof run.output_summary_json?.modules_generated === "number"
      ? (run.output_summary_json.modules_generated as number)
      : null;
  const modulesSkippedArr = Array.isArray(
    run.output_summary_json?.modules_skipped,
  )
    ? (run.output_summary_json.modules_skipped as string[])
    : [];
  const nodesTotal =
    typeof run.output_summary_json?.nodes_total === "number"
      ? (run.output_summary_json.nodes_total as number)
      : null;
  const exTotalSteps =
    typeof run.output_summary_json?.total_steps === "number"
      ? (run.output_summary_json.total_steps as number)
      : null;
  const exPassed =
    typeof run.output_summary_json?.passed === "number"
      ? (run.output_summary_json.passed as number)
      : null;
  const exFailed =
    typeof run.output_summary_json?.failed === "number"
      ? (run.output_summary_json.failed as number)
      : null;
  const exInconclusive =
    typeof run.output_summary_json?.inconclusive === "number"
      ? (run.output_summary_json.inconclusive as number)
      : null;
  const exBlocked =
    typeof run.output_summary_json?.blocked === "number"
      ? (run.output_summary_json.blocked as number)
      : null;
  const exSkipped =
    typeof run.output_summary_json?.skipped === "number"
      ? (run.output_summary_json.skipped as number)
      : null;
  const exDurationMs =
    typeof run.output_summary_json?.duration_ms === "number"
      ? (run.output_summary_json.duration_ms as number)
      : null;

  return (
    <Card
      className={cn(
        "p-4 transition-colors",
        isActive ? "border-primary/40 bg-primary/5" : "",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Sparkles className="size-4 text-primary" />
            <h3 className="font-medium">
              {KIND_LABELS[run.kind] ?? run.kind}
            </h3>
            <span
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium",
                STATUS_CLASSES[run.status],
              )}
            >
              {isActive && <Loader2 className="size-3 animate-spin" />}
              {AGENT_STATUS_LABELS[run.status]}
            </span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Run #{run.id}
            {!isFrdToTc && !isExecute && sourceDocsCount > 0 && (
              <>
                {" · "}
                {sourceDocsCount} BRD{sourceDocsCount === 1 ? "" : "s"}
              </>
            )}
            {!isFrdToTc && !isExecute && capChunks !== null && (
              <> · cap {capChunks}</>
            )}
            {isFrdToTc && planId !== null && <> · plan #{planId}</>}
            {isFrdToTc && capPerModuleFrds !== null && (
              <>
                {" · "}
                {capPerModuleFrds} FRDs/mod
              </>
            )}
            {isFrdToTc && capPerModuleChunks !== null && (
              <> · {capPerModuleChunks} chunks/mod</>
            )}
            {isExecute && planId !== null && <> · plan #{planId}</>}
            {isExecute && (
              <> · {headless ? "headless" : "headed"}</>
            )}
            {isExecute && typeof run.input_json?.speed === "string" && (
              <> · {run.input_json.speed as string}</>
            )}
            {isExecute && selectedStepIdsCount !== null && (
              <>
                {" · "}
                {selectedStepIdsCount} step{selectedStepIdsCount === 1 ? "" : "s"} (override)
              </>
            )}
            {run.started_at && (
              <> · started {new Date(run.started_at).toLocaleTimeString()}</>
            )}
          </p>
        </div>
        {isActive && (
          <div className="flex shrink-0 items-center gap-1">
            {canPause && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  pauseMutation.mutate();
                }}
                disabled={pauseMutation.isPending}
                aria-label="Pause run"
                title="Halt at the next step boundary"
              >
                <Pause className="size-4" />
                Pause
              </Button>
            )}
            {canResume && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  resumeMutation.mutate();
                }}
                disabled={resumeMutation.isPending}
                aria-label="Resume run"
              >
                <Play className="size-4" />
                Resume
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                // Stop the click bubbling to any parent <Link> wrapper
                // (Runs tab wraps each card in a Link to the detail page).
                e.preventDefault();
                e.stopPropagation();
                cancelMutation.mutate();
              }}
              disabled={cancelMutation.isPending}
              aria-label="Cancel run"
            >
              <X className="size-4" />
              {isPaused ? "Stop" : "Cancel"}
            </Button>
          </div>
        )}
      </div>

      {/* Live phase + transient counts */}
      {isActive &&
        progress &&
        (progress.phase ||
          progress.message ||
          progress.module_name ||
          progress.current) && (
          <div className="mt-3 rounded-md bg-background/60 p-3 text-xs">
            <div className="flex flex-wrap items-baseline gap-2">
              {progress.phase && (
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  {PHASE_LABELS[progress.phase] ?? progress.phase}
                </span>
              )}
              {typeof progress.current === "number" &&
                typeof progress.total === "number" && (
                  <span className="font-mono text-[10px] text-muted-foreground">
                    Module {progress.current}/{progress.total}
                  </span>
                )}
              {progress.module_name && (
                <span className="font-medium text-foreground">
                  {progress.module_name}
                </span>
              )}
              {progress.message && !progress.module_name && (
                <span className="text-foreground">{progress.message}</span>
              )}
            </div>
            <div className="mt-1.5 flex flex-wrap gap-3 text-muted-foreground">
              {typeof progress.chunks_seen === "number" && (
                <span>
                  {progress.chunks_seen} chunks
                  {progress.truncated && (
                    <span className="ml-1 text-yellow-700 dark:text-yellow-400">
                      (truncated)
                    </span>
                  )}
                </span>
              )}
              {typeof progress.candidates === "number" && (
                <span>{progress.candidates} candidate FRDs</span>
              )}
              {typeof progress.nodes_added === "number" && (
                <span>+{progress.nodes_added} nodes</span>
              )}
              {(progress.input_tokens || progress.output_tokens) && (
                <span>
                  {progress.input_tokens ?? 0} in /{" "}
                  {progress.output_tokens ?? 0} out tokens
                </span>
              )}
            </div>
            {/* Module progress bar (frd_to_tc only) */}
            {typeof progress.current === "number" &&
              typeof progress.total === "number" &&
              progress.total > 0 && (
                <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-primary transition-all"
                    style={{
                      width: `${Math.min(
                        100,
                        (progress.current / progress.total) * 100,
                      )}%`,
                    }}
                  />
                </div>
              )}
          </div>
        )}

      {/* Completion summary */}
      {run.status === "completed" && (
        <div
          className={cn(
            "mt-3 rounded-md p-3 text-xs",
            isExecute && (exFailed ?? 0) > 0
              ? "bg-red-500/5 text-red-700 dark:text-red-400"
              // Inconclusive (but no real failures) → orange/amber tint;
              // these are usually test-case wording problems, not bugs.
              : isExecute && (exInconclusive ?? 0) > 0
                ? "bg-orange-500/5 text-orange-700 dark:text-orange-400"
                : "bg-green-500/5 text-green-700 dark:text-green-400",
          )}
        >
          {isExecute ? (
            <>
              {(exFailed ?? 0) === 0 &&
              (exBlocked ?? 0) === 0 &&
              (exInconclusive ?? 0) === 0
                ? "✓"
                : "•"}{" "}
              <strong>{exPassed ?? 0}</strong>/{exTotalSteps ?? 0} passed
              {(exFailed ?? 0) > 0 && (
                <>
                  {" · "}
                  <strong>{exFailed}</strong> failed
                </>
              )}
              {(exInconclusive ?? 0) > 0 && (
                <>
                  {" · "}
                  <strong>{exInconclusive}</strong> inconclusive
                </>
              )}
              {(exBlocked ?? 0) > 0 && (
                <>
                  {" · "}
                  <strong>{exBlocked}</strong> blocked
                </>
              )}
              {(exSkipped ?? 0) > 0 && (
                <>
                  {" · "}
                  {exSkipped} skipped
                </>
              )}
              {exDurationMs !== null && (
                <>
                  {" · "}
                  {(exDurationMs / 1000).toFixed(1)}s
                </>
              )}
            </>
          ) : isFrdToTc ? (
            <>
              ✓ Generated <strong>{modulesGenerated ?? 0}</strong> module
              {modulesGenerated === 1 ? "" : "s"}
              {nodesTotal !== null && (
                <>
                  {" · "}
                  <strong>{nodesTotal}</strong> node{nodesTotal === 1 ? "" : "s"}
                </>
              )}
              {modulesSkippedArr.length > 0 && (
                <>
                  {" · "}
                  <span className="text-yellow-700 dark:text-yellow-400">
                    {modulesSkippedArr.length} skipped
                  </span>
                </>
              )}
            </>
          ) : (
            <>
              ✓ Generated <strong>{generated ?? 0}</strong> FRD
              {generated === 1 ? "" : "s"}
            </>
          )}
          {!isExecute && inTokens !== null && outTokens !== null && (
            <>
              {" "}
              · {inTokens.toLocaleString()} in / {outTokens.toLocaleString()}{" "}
              out tokens
            </>
          )}
        </div>
      )}

      {/* Error / cancel summary */}
      {(run.status === "failed" || run.status === "cancelled") &&
        run.error_message && (
          <div className="mt-3 rounded-md border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-700 dark:text-red-400">
            <span className="font-medium">
              {run.status === "cancelled" ? "Cancelled" : "Failed"}:
            </span>{" "}
            <span className="break-words">{run.error_message}</span>
          </div>
        )}
    </Card>
  );
}
