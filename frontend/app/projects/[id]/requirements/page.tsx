"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { Check, ScrollText, Sparkles, X } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  REQUIREMENT_STATUS_LABELS,
  type AgentStatus,
  type BulkAction,
  type RequirementStatus,
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
import { ReviewCard } from "@/components/review-card";
import { RunProgressCard } from "@/components/run-progress-card";
import { SynthesizeFrdDialog } from "@/components/synthesize-frd-dialog";
import { useAgentRunsEvents } from "@/hooks/use-agent-runs-events";
import { cn } from "@/lib/utils";

const ACTIVE_RUN_STATUSES: AgentStatus[] = ["queued", "running", "paused"];

const STATUS_CLASSES: Record<RequirementStatus, string> = {
  proposed: "bg-muted text-muted-foreground",
  edited: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
  approved: "bg-green-500/10 text-green-700 dark:text-green-400",
  rejected: "bg-red-500/10 text-red-700 dark:text-red-400",
};

const ALL_STATUSES: RequirementStatus[] = [
  "proposed",
  "edited",
  "approved",
  "rejected",
];

type FilterValue = RequirementStatus | "all";

export default function RequirementsTabPage() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);
  const qc = useQueryClient();

  const [synthesizeOpen, setSynthesizeOpen] = useState(false);
  const [bulkRejectOpen, setBulkRejectOpen] = useState(false);
  const [filter, setFilter] = useState<FilterValue>("all");

  useAgentRunsEvents(projectId);

  const { data: runs } = useQuery({
    queryKey: ["agent-runs", projectId],
    queryFn: () => api.listAgentRuns(projectId),
  });

  const { data: reqs, isLoading } = useQuery({
    queryKey: ["requirements", projectId],
    queryFn: () => api.listRequirements(projectId),
  });

  const activeRuns = useMemo(
    () => (runs ?? []).filter((r) => ACTIVE_RUN_STATUSES.includes(r.status)),
    [runs],
  );

  const counts = useMemo(() => {
    const c: Record<FilterValue, number> = {
      all: 0,
      proposed: 0,
      edited: 0,
      approved: 0,
      rejected: 0,
    };
    if (!reqs) return c;
    for (const r of reqs) {
      c.all += 1;
      c[r.status] += 1;
    }
    return c;
  }, [reqs]);

  const filteredReqs = useMemo(() => {
    if (!reqs) return [];
    if (filter === "all") return reqs;
    return reqs.filter((r) => r.status === filter);
  }, [reqs, filter]);

  const bulkMutation = useMutation({
    mutationFn: (action: BulkAction) =>
      api.bulkUpdateRequirements(projectId, {
        filter_status: "proposed",
        action,
      }),
    onSuccess: (res) => {
      const verb =
        res.action === "approve"
          ? "approved"
          : res.action === "reject"
            ? "rejected"
            : "deleted";
      toast.success(
        `${res.affected} FRD${res.affected === 1 ? "" : "s"} ${verb}`,
      );
      qc.invalidateQueries({ queryKey: ["requirements", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Bulk update failed", { description: msg });
    },
  });

  const hasProposed = counts.proposed > 0;

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <p className="max-w-2xl text-sm text-muted-foreground">
          Functional requirements derived by the BRD→FRD agent. Review each
          one, then approve to make it available for test-case generation.
        </p>
        <Button size="sm" onClick={() => setSynthesizeOpen(true)}>
          <Sparkles className="size-4" /> Synthesize FRDs
        </Button>
      </div>

      <SynthesizeFrdDialog
        open={synthesizeOpen}
        onOpenChange={setSynthesizeOpen}
        projectId={projectId}
      />

      {activeRuns.length > 0 && (
        <div className="space-y-3">
          {activeRuns.map((run) => (
            <RunProgressCard
              key={run.id}
              projectId={projectId}
              run={run}
            />
          ))}
        </div>
      )}

      {/* Filter chips + bulk actions */}
      {counts.all > 0 && (
        <div className="flex flex-wrap items-center gap-3">
          <FilterChips
            counts={counts}
            value={filter}
            onChange={setFilter}
          />
          {hasProposed && (
            <div className="ml-auto flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => bulkMutation.mutate("approve")}
                disabled={bulkMutation.isPending}
              >
                <Check className="size-4" />
                Approve all {counts.proposed} proposed
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setBulkRejectOpen(true)}
                disabled={bulkMutation.isPending}
              >
                <X className="size-4" />
                Reject all proposed
              </Button>
            </div>
          )}
        </div>
      )}

      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      ) : !reqs || reqs.length === 0 ? (
        <EmptyStateWithTrigger
          onTriggerOpen={() => setSynthesizeOpen(true)}
        />
      ) : filteredReqs.length === 0 ? (
        <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
          No FRDs match the <strong>{filter}</strong> filter. Switch to
          <button
            type="button"
            onClick={() => setFilter("all")}
            className="ml-1 text-primary underline-offset-2 hover:underline"
          >
            All
          </button>
          {" "}to see everything.
        </div>
      ) : (
        <div className="space-y-3">
          {filteredReqs.map((r) => (
            <ReviewCard
              key={r.id}
              projectId={projectId}
              requirement={r}
            />
          ))}
        </div>
      )}

      <BulkRejectConfirmDialog
        open={bulkRejectOpen}
        onOpenChange={setBulkRejectOpen}
        proposedCount={counts.proposed}
        onConfirm={() => {
          bulkMutation.mutate("reject");
          setBulkRejectOpen(false);
        }}
        disabled={bulkMutation.isPending}
      />
    </div>
  );
}

// ── Filter chips ───────────────────────────────────────────────────


function FilterChips({
  counts,
  value,
  onChange,
}: {
  counts: Record<FilterValue, number>;
  value: FilterValue;
  onChange: (v: FilterValue) => void;
}) {
  const options: FilterValue[] = ["all", ...ALL_STATUSES];

  return (
    <div className="flex flex-wrap gap-2 text-xs">
      {options.map((opt) => {
        const isActive = value === opt;
        const baseClass =
          "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-medium transition-colors";
        const colorClass =
          opt === "all" ? "" : STATUS_CLASSES[opt as RequirementStatus];
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
                : REQUIREMENT_STATUS_LABELS[opt as RequirementStatus]}
            </span>
            <span className="font-mono opacity-70">{counts[opt]}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── Bulk-reject confirm dialog ─────────────────────────────────────


function BulkRejectConfirmDialog({
  open,
  onOpenChange,
  proposedCount,
  onConfirm,
  disabled,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  proposedCount: number;
  onConfirm: () => void;
  disabled: boolean;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reject all proposed FRDs?</DialogTitle>
          <DialogDescription>
            Marks all <strong>{proposedCount}</strong> proposed FRD
            {proposedCount === 1 ? "" : "s"} as rejected. They stay in the
            list (status = rejected) but won&apos;t be embedded into FAISS or
            used for test-case generation. Reversible by approving each
            individually.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={disabled}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            disabled={disabled}
          >
            Reject {proposedCount}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Empty state ────────────────────────────────────────────────────


function EmptyStateWithTrigger({
  onTriggerOpen,
}: {
  onTriggerOpen: () => void;
}) {
  return (
    <div className="rounded-lg border border-dashed p-12 text-center">
      <ScrollText className="mx-auto size-10 text-muted-foreground" />
      <h3 className="mt-4 font-semibold">No requirements yet</h3>
      <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">
        Upload a BRD on the Documents tab, then run the BRD→FRD synthesis
        agent. Generated FRDs will appear here as <em>proposed</em> for your
        review.
      </p>
      <Button className="mt-4" onClick={onTriggerOpen}>
        <Sparkles className="size-4" /> Synthesize FRDs
      </Button>
    </div>
  );
}
