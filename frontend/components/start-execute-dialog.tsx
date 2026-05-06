"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, EyeOff, Play } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  PLAN_STATUS_LABELS,
  type PlanReadCompact,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: number;
  /** Pre-selected plan id (when launched from the Test Cases tab); editable. */
  defaultPlanId?: number | null;
}

export function StartExecuteDialog({
  open,
  onOpenChange,
  projectId,
  defaultPlanId,
}: Props) {
  const qc = useQueryClient();
  const [planId, setPlanId] = useState<number | null>(defaultPlanId ?? null);
  const [headless, setHeadless] = useState(false);

  useEffect(() => {
    if (open) {
      setPlanId(defaultPlanId ?? null);
      setHeadless(false);
    }
  }, [open, defaultPlanId]);

  const { data: plans, isLoading: plansLoading } = useQuery({
    queryKey: ["plans", projectId],
    queryFn: () => api.listPlans(projectId),
    enabled: open,
  });

  // Default-pick the first plan once they load
  useEffect(() => {
    if (open && planId === null && plans && plans.length > 0) {
      setPlanId(plans[0].id);
    }
  }, [open, planId, plans]);

  const selectedPlan = useMemo(
    () => plans?.find((p) => p.id === planId) ?? null,
    [plans, planId],
  );

  const startMutation = useMutation({
    mutationFn: () =>
      api.startExecute(projectId, {
        plan_id: planId!,
        headless,
      }),
    onSuccess: (run) => {
      toast.success("Execution queued", {
        description: `Run #${run.id} — progress streams on the Runs tab.`,
      });
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Failed to start run", { description: msg });
    },
  });

  const canSubmit =
    planId !== null &&
    !!selectedPlan &&
    !!selectedPlan.target_url &&
    !startMutation.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Play className="size-5" /> Start an execution run
          </DialogTitle>
          <DialogDescription>
            Walks every selected step (tri-state checkboxes on the Test Cases
            tab) in DFS order, capturing a screenshot per step.
          </DialogDescription>
        </DialogHeader>

        {plansLoading ? (
          <Skeleton className="h-32 w-full" />
        ) : !plans || plans.length === 0 ? (
          <p className="text-sm text-destructive">
            No plans yet. Create a plan first on the Plans tab.
          </p>
        ) : (
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <label
                htmlFor="execute-plan-picker"
                className="text-sm font-medium"
              >
                Plan
              </label>
              <select
                id="execute-plan-picker"
                value={planId ?? ""}
                onChange={(e) => setPlanId(Number(e.target.value))}
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              >
                {plans.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} — {PLAN_STATUS_LABELS[p.status]}
                  </option>
                ))}
              </select>
            </div>

            {selectedPlan && (
              <PlanSummary plan={selectedPlan} />
            )}

            <div className="flex items-start gap-3 rounded-md border p-3">
              <button
                type="button"
                onClick={() => setHeadless(!headless)}
                className={cn(
                  "mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md border transition-colors",
                  headless
                    ? "border-input bg-muted text-muted-foreground"
                    : "border-primary/40 bg-primary/10 text-primary",
                )}
                aria-pressed={!headless}
                aria-label={
                  headless
                    ? "Switch to visible browser"
                    : "Switch to headless browser"
                }
                title={
                  headless
                    ? "Headless: faster, no window"
                    : "Headed: you can see the browser"
                }
              >
                {headless ? (
                  <EyeOff className="size-4" />
                ) : (
                  <Eye className="size-4" />
                )}
              </button>
              <div className="min-w-0 flex-1 text-sm">
                <p className="font-medium">
                  {headless ? "Headless" : "Headed"} Chromium
                </p>
                <p className="text-xs text-muted-foreground">
                  {headless
                    ? "Faster, no visible window. Switch to headed if a step blocks for HITL (week 6)."
                    : "A visible Chrome window opens; you can watch the run. Recommended for debugging."}
                </p>
              </div>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={startMutation.isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={() => startMutation.mutate()}
            disabled={!canSubmit}
          >
            <Play className="size-4" />
            {startMutation.isPending ? "Queueing…" : "Start run"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function PlanSummary({ plan }: { plan: PlanReadCompact }) {
  return (
    <div className="rounded-md border bg-muted/30 p-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        Target
      </p>
      <p className="break-all font-mono text-xs">
        {plan.target_url || (
          <span className="italic text-destructive">
            (none — set a target_url before running)
          </span>
        )}
      </p>
      {plan.scope.length > 0 && (
        <p className="mt-2 text-xs text-muted-foreground">
          Scope: {plan.scope.length} module
          {plan.scope.length === 1 ? "" : "s"}
        </p>
      )}
    </div>
  );
}
