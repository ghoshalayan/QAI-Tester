"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { api, ApiError } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: number;
  planId: number;
}

const DEFAULT_FRDS_CAP = 15;
const DEFAULT_CHUNKS_CAP = 10;
const MIN_FRDS = 1;
const MAX_FRDS = 50;
const MIN_CHUNKS = 0;
const MAX_CHUNKS = 50;

export function SynthesizeTcDialog({
  open,
  onOpenChange,
  projectId,
  planId,
}: Props) {
  const qc = useQueryClient();
  const [capFrds, setCapFrds] = useState(DEFAULT_FRDS_CAP);
  const [capChunks, setCapChunks] = useState(DEFAULT_CHUNKS_CAP);

  // Reset to defaults when dialog opens
  useEffect(() => {
    if (open) {
      setCapFrds(DEFAULT_FRDS_CAP);
      setCapChunks(DEFAULT_CHUNKS_CAP);
    }
  }, [open]);

  const { data: plan, isLoading: planLoading } = useQuery({
    queryKey: ["plan", projectId, planId],
    queryFn: () => api.getPlan(projectId, planId),
    enabled: open,
  });

  const { data: approvedFrds } = useQuery({
    queryKey: ["requirements", projectId, "approved"],
    queryFn: () => api.listRequirements(projectId, { status: "approved" }),
    enabled: open,
  });

  const startMutation = useMutation({
    mutationFn: () =>
      api.startFrdToTc(projectId, {
        plan_id: planId,
        cap_per_module_frds: capFrds,
        cap_per_module_chunks: capChunks,
      }),
    onSuccess: (run) => {
      toast.success("Synthesis run queued", {
        description: `Run #${run.id} — progress streams on the Test Cases tab.`,
      });
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, planId] });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Failed to start synthesis", { description: msg });
    },
  });

  const approvedFrdCount = approvedFrds?.length ?? 0;
  const hasFrds = approvedFrdCount > 0;
  const hasDescription = !!(plan?.description && plan.description.trim());
  const hasLinkedDocs = (plan?.linked_documents.length ?? 0) > 0;
  const hasAnySignal = hasFrds || hasDescription || hasLinkedDocs;

  const moduleCount = plan?.scope.length ?? 0;
  const effectiveModuleCount = moduleCount === 0 ? 1 : moduleCount;

  const canSubmit =
    !!plan &&
    hasAnySignal &&
    capFrds >= MIN_FRDS &&
    capFrds <= MAX_FRDS &&
    capChunks >= MIN_CHUNKS &&
    capChunks <= MAX_CHUNKS &&
    !startMutation.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="size-5" /> Generate test cases
          </DialogTitle>
          <DialogDescription>
            One LLM call per module. The agent retrieves relevant approved FRDs
            (and linked-doc chunks) for each, then produces a Module → Submodule
            → Step tree.
          </DialogDescription>
        </DialogHeader>

        {planLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : !plan ? (
          <p className="text-sm text-destructive">Plan not found.</p>
        ) : (
          <div className="space-y-4 py-2">
            {/* Plan summary */}
            <div className="rounded-md border bg-muted/30 p-3">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Plan
              </p>
              <p className="font-medium">{plan.name}</p>
              <p className="break-all text-xs text-muted-foreground">
                {plan.target_url}
              </p>
            </div>

            {/* Scope */}
            <div className="space-y-2">
              <Label>Scope</Label>
              {plan.scope.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {plan.scope.map((s) => (
                    <Badge key={s} variant="outline">
                      {s}
                    </Badge>
                  ))}
                </div>
              ) : (
                <p className="text-xs italic text-muted-foreground">
                  Empty — agent will produce one synthetic &quot;All test
                  cases&quot; module.
                </p>
              )}
              <p className="text-xs text-muted-foreground">
                Will trigger <strong>{effectiveModuleCount}</strong> LLM call
                {effectiveModuleCount === 1 ? "" : "s"}.
              </p>
            </div>

            {/* Signal sources */}
            <div className="rounded-md border p-3">
              <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
                Signal sources
              </p>
              <div className="space-y-1">
                <SignalRow
                  label={`Approved FRDs in project (${approvedFrdCount})`}
                  ok={hasFrds}
                />
                <SignalRow
                  label={
                    plan.description && plan.description.trim()
                      ? `Plan instructions (${plan.description.trim().length} chars)`
                      : "Plan instructions"
                  }
                  ok={hasDescription}
                />
                <SignalRow
                  label={`Linked documents (${plan.linked_documents.length})`}
                  ok={hasLinkedDocs}
                />
              </div>
            </div>

            {!hasAnySignal && (
              <div className="flex items-start gap-2 rounded-md border border-yellow-500/40 bg-yellow-500/5 p-3 text-xs text-yellow-700 dark:text-yellow-400">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <p>
                  This plan has no signal sources. Approve at least one FRD on
                  the Requirements tab, link a doc to this plan, or add
                  free-text instructions before running synthesis.
                </p>
              </div>
            )}

            {/* Caps */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="cap-frds">FRDs per module</Label>
                <Input
                  id="cap-frds"
                  type="number"
                  min={MIN_FRDS}
                  max={MAX_FRDS}
                  value={capFrds}
                  onChange={(e) => {
                    const n = parseInt(e.target.value, 10);
                    if (Number.isNaN(n)) return;
                    setCapFrds(Math.max(MIN_FRDS, Math.min(MAX_FRDS, n)));
                  }}
                />
                <p className="text-xs text-muted-foreground">
                  Top-K approved FRDs sent to the LLM per module.
                </p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="cap-chunks">Doc chunks per module</Label>
                <Input
                  id="cap-chunks"
                  type="number"
                  min={MIN_CHUNKS}
                  max={MAX_CHUNKS}
                  value={capChunks}
                  onChange={(e) => {
                    const n = parseInt(e.target.value, 10);
                    if (Number.isNaN(n)) return;
                    setCapChunks(
                      Math.max(MIN_CHUNKS, Math.min(MAX_CHUNKS, n)),
                    );
                  }}
                />
                <p className="text-xs text-muted-foreground">
                  Top-K chunks from linked docs (0 to skip).
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
            <Sparkles className="size-4" />
            {startMutation.isPending ? "Queueing…" : "Start synthesis"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function SignalRow({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {ok ? (
        <CheckCircle2 className="size-3.5 text-green-600 dark:text-green-400" />
      ) : (
        <span className="size-3.5 rounded-full border border-muted-foreground/30" />
      )}
      <span className={ok ? "text-foreground" : "text-muted-foreground"}>
        {label}
      </span>
    </div>
  );
}
