"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  EXECUTION_STEP_STATUS_LABELS,
  type ExecutionStepStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { ExecutionTimeline } from "@/components/execution-timeline";
import { RunHeader } from "@/components/run-header";
import { RunProgressCard } from "@/components/run-progress-card";
import { Skeleton } from "@/components/ui/skeleton";
import { useAgentRunsEvents } from "@/hooks/use-agent-runs-events";
import { cn } from "@/lib/utils";

type FilterValue = ExecutionStepStatus | "all";

const STATUS_ORDER: ExecutionStepStatus[] = [
  "running",
  "passed",
  "failed",
  "blocked",
  "skipped",
  "pending",
];

const STATUS_CHIP_CLASS: Record<ExecutionStepStatus, string> = {
  pending: "bg-muted text-muted-foreground",
  running: "bg-blue-500/10 text-blue-700 dark:text-blue-400",
  passed: "bg-green-500/10 text-green-700 dark:text-green-400",
  failed: "bg-red-500/10 text-red-700 dark:text-red-400",
  blocked: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
  skipped: "bg-muted text-muted-foreground",
};

export default function RunDetailPage() {
  const params = useParams<{ id: string; runId: string }>();
  const projectId = Number(params.id);
  const runId = Number(params.runId);
  const router = useRouter();
  const qc = useQueryClient();

  const [filter, setFilter] = useState<FilterValue>("all");

  useAgentRunsEvents(projectId);

  const { data: run, isLoading: runLoading } = useQuery({
    queryKey: ["agent-run", projectId, runId],
    queryFn: () => api.getAgentRun(projectId, runId),
  });

  const { data: plan } = useQuery({
    queryKey: ["plan", projectId, run?.plan_id],
    queryFn: () => api.getPlan(projectId, run!.plan_id!),
    enabled: !!run?.plan_id,
  });

  // Same query key as <ExecutionTimeline> uses, so this is shared cache
  const { data: steps } = useQuery({
    queryKey: ["run-steps", projectId, runId],
    queryFn: () => api.listRunSteps(projectId, runId),
    enabled: run?.kind === "execute",
  });

  const counts = useMemo(() => {
    const c: Record<FilterValue, number> = {
      all: 0,
      pending: 0,
      running: 0,
      passed: 0,
      failed: 0,
      blocked: 0,
      skipped: 0,
    };
    for (const s of steps ?? []) {
      c.all += 1;
      c[s.status] += 1;
    }
    return c;
  }, [steps]);

  const failedTcNodeIds = useMemo(
    () =>
      (steps ?? [])
        .filter((s) => s.status === "failed" && s.tc_node_id !== null)
        .map((s) => s.tc_node_id as number),
    [steps],
  );

  const rerunFailed = useMutation({
    mutationFn: () => {
      if (!run?.plan_id) throw new Error("Run has no plan_id");
      const headless = !!run.input_json?.headless;
      return api.startExecute(projectId, {
        plan_id: run.plan_id,
        selected_step_ids: failedTcNodeIds,
        headless,
      });
    },
    onSuccess: (newRun) => {
      toast.success("Re-run queued", {
        description: `Run #${newRun.id} — ${failedTcNodeIds.length} step${failedTcNodeIds.length === 1 ? "" : "s"} from this run.`,
      });
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      router.push(`/projects/${projectId}/runs/${newRun.id}`);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Re-run failed to start", { description: msg });
    },
  });

  const isExecute = run?.kind === "execute";
  const showRerunButton =
    isExecute &&
    run?.status &&
    ["completed", "failed", "cancelled"].includes(run.status) &&
    failedTcNodeIds.length > 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Button asChild variant="ghost" size="sm">
          <Link href={`/projects/${projectId}/runs`}>
            <ArrowLeft className="size-4" /> All runs
          </Link>
        </Button>
        <h2 className="text-lg font-semibold">Run #{runId}</h2>
      </div>

      {runLoading ? (
        <Skeleton className="h-32 w-full" />
      ) : !run ? (
        <p className="text-sm text-destructive">Run not found.</p>
      ) : (
        <>
          <RunProgressCard projectId={projectId} run={run} />

          {isExecute && <RunHeader run={run} plan={plan ?? null} />}

          {isExecute && (
            <div className="flex flex-wrap items-center gap-3">
              <FilterChips
                counts={counts}
                value={filter}
                onChange={setFilter}
              />
              {showRerunButton && (
                <Button
                  size="sm"
                  variant="outline"
                  className="ml-auto"
                  onClick={() => rerunFailed.mutate()}
                  disabled={rerunFailed.isPending}
                >
                  <RefreshCw
                    className={cn(
                      "size-4",
                      rerunFailed.isPending && "animate-spin",
                    )}
                  />
                  Re-run {failedTcNodeIds.length} failed step
                  {failedTcNodeIds.length === 1 ? "" : "s"}
                </Button>
              )}
            </div>
          )}

          {isExecute && (
            <ExecutionTimeline
              projectId={projectId}
              runId={runId}
              statusFilter={filter}
            />
          )}
        </>
      )}
    </div>
  );
}

function FilterChips({
  counts,
  value,
  onChange,
}: {
  counts: Record<FilterValue, number>;
  value: FilterValue;
  onChange: (v: FilterValue) => void;
}) {
  // Show "all" + every status that has at least one row (keeps the strip tight
  // for runs with only a couple of statuses)
  const options: FilterValue[] = [
    "all",
    ...STATUS_ORDER.filter((s) => counts[s] > 0),
  ];
  if (options.length <= 1) return null; // nothing to filter

  return (
    <div className="flex flex-wrap gap-2 text-xs">
      {options.map((opt) => {
        const isActive = value === opt;
        const baseClass =
          "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-medium transition-colors";
        const colorClass =
          opt === "all" ? "" : STATUS_CHIP_CLASS[opt as ExecutionStepStatus];
        const activeRing = isActive
          ? "ring-2 ring-primary ring-offset-1 ring-offset-background"
          : "hover:opacity-80";
        return (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={cn(baseClass, colorClass, activeRing)}
          >
            <span>
              {opt === "all"
                ? "All"
                : EXECUTION_STEP_STATUS_LABELS[opt as ExecutionStepStatus]}
            </span>
            <span className="font-mono opacity-70">{counts[opt]}</span>
          </button>
        );
      })}
    </div>
  );
}
