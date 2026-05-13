"use client";

/**
 * Phase C.2/C.3b — Per-submodule TC refinement dialog.
 *
 * Wired to the "Refine test cases from app map" button on the plan
 * editor. Flow:
 *
 *   1. User clicks the button.
 *   2. Dialog opens in "ready" state with a one-paragraph explainer
 *      + a single Refine button.
 *   3. On click, calls POST /plans/{id}/refine-from-app-map.
 *      Synchronous (10-30s for Solar) — dialog shows a spinner.
 *   4. On response, dialog switches to "review" state and fetches
 *      the new version's snapshot tree via
 *      GET /plans/{id}/tc-versions/{id}. Renders a per-submodule
 *      summary with kept/rewritten/added/flagged_missing counts +
 *      a "View changes" expandable per submodule.
 *   5. User clicks "Activate this version" → calls
 *      PUT /plans/{id}/tc-versions/{id}/activate → plan now points
 *      at the new version. Or "Discard" → version is kept in the
 *      list for later inspection but current pointer is unchanged.
 *
 * Refinement is never auto-triggered; only this dialog opens it.
 */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  Loader2,
  Sparkles,
  XCircle,
  AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  type TcRefinementResponse,
  type TcValidationResponse,
  type TcVersionDetail,
  type TcChangeKind,
  type ValidationStatus,
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

const CHANGE_TINT: Record<TcChangeKind, string> = {
  kept: "border-muted bg-muted/30 text-muted-foreground",
  rewritten:
    "border-blue-500/40 bg-blue-500/10 text-blue-700 dark:text-blue-400",
  added:
    "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  flagged_missing:
    "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
};

const CHANGE_LABEL: Record<TcChangeKind, string> = {
  kept: "kept",
  rewritten: "rewritten",
  added: "added",
  flagged_missing: "flagged",
};

export function RefineTcsDialog({
  projectId,
  planId,
  open,
  onOpenChange,
  hasAppMap,
}: {
  projectId: number;
  planId: number;
  open: boolean;
  onOpenChange: (next: boolean) => void;
  hasAppMap: boolean;
}) {
  const qc = useQueryClient();
  const [phase, setPhase] = useState<
    "ready" | "running" | "review" | "validating"
  >("ready");
  const [refinement, setRefinement] = useState<TcRefinementResponse | null>(
    null,
  );
  const [validation, setValidation] = useState<TcValidationResponse | null>(
    null,
  );

  // Reset when reopened.
  useEffect(() => {
    if (open) {
      setPhase("ready");
      setRefinement(null);
      setValidation(null);
    }
  }, [open]);

  const refineMutation = useMutation({
    mutationFn: () => api.refineFromAppMap(projectId, planId),
    onSuccess: (resp) => {
      setRefinement(resp);
      setPhase("review");
      qc.invalidateQueries({
        queryKey: ["tc-versions", projectId, planId],
      });
      toast.success(
        `Refined v${resp.version_number} created — ${resp.submodule_count} submodules`,
      );
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Refinement failed", { description: msg });
      setPhase("ready");
    },
  });

  const validateMutation = useMutation({
    mutationFn: (versionId: number) =>
      api.validateTcVersion(projectId, planId, versionId),
    onSuccess: (resp) => {
      setValidation(resp);
      setPhase("review");
      qc.invalidateQueries({
        queryKey: [
          "tc-version-detail", projectId, planId,
          refinement?.version_id,
        ],
      });
      if (resp.error_message) {
        toast.warning("Validation completed with issues", {
          description: resp.error_message,
        });
      } else {
        toast.success(
          `Validation: probed ${resp.total_probed} step(s) in ${resp.total_seconds}s`,
        );
      }
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Validation failed", { description: msg });
      setPhase("review");
    },
  });

  const activateMutation = useMutation({
    mutationFn: (versionId: number) =>
      api.activateTcVersion(projectId, planId, versionId),
    onSuccess: (resp) => {
      toast.success(
        `Activated v${resp.version_number} — live test cases updated`,
      );
      qc.invalidateQueries({ queryKey: ["plan", projectId, planId] });
      qc.invalidateQueries({
        queryKey: ["tc-versions", projectId, planId],
      });
      // The activate endpoint overwrites the live TcNode tree, so
      // invalidate the test-cases query so the viewer reloads the
      // refined tree without a manual refresh.
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, planId] });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Couldn't activate version", { description: msg });
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="size-5 text-primary" />
            Refine test cases from app map
          </DialogTitle>
          <DialogDescription>
            Per-submodule LLM refinement that aligns each test case with
            the actual app UI captured by the authenticated Scout pass.
            Each refinement run creates a new version — your live
            test-case tree is never overwritten.
          </DialogDescription>
        </DialogHeader>

        {phase === "ready" && (
          <div className="space-y-3 text-sm">
            {!hasAppMap && (
              <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-800 dark:text-amber-200">
                <strong>No app map yet.</strong> Run agentic mode once OR
                click &quot;Scout this app&quot; first so the refiner has
                a real-app structure to anchor against.
              </div>
            )}
            <p className="text-muted-foreground">
              The refiner sees each submodule&apos;s scoped slice of the
              app map (only the relevant modules + create flows) plus
              the source BRD/FRD chunks. Per step it emits one of:
            </p>
            <ul className="ml-4 list-disc space-y-1 text-xs text-muted-foreground">
              <li>
                <strong className="text-foreground">kept</strong> — step
                already matches the app
              </li>
              <li>
                <strong className="text-foreground">rewritten</strong> —
                target/wording updated to use the real UI labels
              </li>
              <li>
                <strong className="text-foreground">added</strong> — step
                the BRD missed (e.g. a required field the form has)
              </li>
              <li>
                <strong className="text-foreground">flagged_missing</strong>{" "}
                — step references UI the app map doesn&apos;t cover
              </li>
            </ul>
          </div>
        )}

        {phase === "running" && (
          <div className="flex flex-col items-center justify-center gap-3 py-12">
            <Loader2 className="size-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Refining per submodule… (~10-30s)
            </p>
          </div>
        )}

        {phase === "validating" && (
          <div className="flex flex-col items-center justify-center gap-3 py-12">
            <Loader2 className="size-8 animate-spin text-blue-500" />
            <p className="text-sm text-muted-foreground">
              Validating refined steps against the live UI… (~60-90s)
            </p>
            <p className="text-xs text-muted-foreground">
              Opens a headless browser, logs in via auth_flow, probes
              each target without dispatching actions.
            </p>
          </div>
        )}

        {phase === "review" && refinement && (
          <ReviewPanel
            projectId={projectId}
            planId={planId}
            refinement={refinement}
            validation={validation}
          />
        )}

        <DialogFooter>
          {phase === "ready" && (
            <>
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button
                onClick={() => {
                  setPhase("running");
                  refineMutation.mutate();
                }}
                disabled={!hasAppMap || refineMutation.isPending}
              >
                <Sparkles className="size-4" />
                Refine now
              </Button>
            </>
          )}
          {phase === "review" && refinement && (
            <>
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={activateMutation.isPending}
              >
                Keep current version
              </Button>
              {!validation && (
                <Button
                  variant="outline"
                  onClick={() => {
                    setPhase("validating");
                    validateMutation.mutate(refinement.version_id);
                  }}
                  disabled={validateMutation.isPending}
                >
                  <Sparkles className="size-4 text-blue-500" />
                  Validate against live UI
                </Button>
              )}
              <Button
                onClick={() =>
                  activateMutation.mutate(refinement.version_id)
                }
                disabled={activateMutation.isPending}
              >
                {activateMutation.isPending ? (
                  <>
                    <Loader2 className="size-4 animate-spin" />
                    Activating…
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="size-4" />
                    Activate v{refinement.version_number}
                  </>
                )}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Phase D validation badge tint + glyph ────────────────────────

const VALIDATION_TINT: Record<ValidationStatus, string> = {
  pending: "border-muted bg-muted/30 text-muted-foreground",
  confirmed:
    "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  partial:
    "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  unresolved:
    "border-orange-500/40 bg-orange-500/10 text-orange-700 dark:text-orange-400",
  unreachable:
    "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  skipped: "border-muted bg-muted/30 text-muted-foreground",
};

const VALIDATION_LABEL: Record<ValidationStatus, string> = {
  pending: "not validated",
  confirmed: "live ✓",
  partial: "live ~",
  unresolved: "not found",
  unreachable: "unreachable",
  skipped: "skipped",
};

function ReviewPanel({
  projectId,
  planId,
  refinement,
  validation,
}: {
  projectId: number;
  planId: number;
  refinement: TcRefinementResponse;
  validation: TcValidationResponse | null;
}) {
  // Fetch the full snapshot tree so the user can drill into specific
  // changes per submodule.
  const { data: detail, isLoading } = useQuery({
    queryKey: [
      "tc-version-detail", projectId, planId, refinement.version_id,
    ],
    queryFn: () =>
      api.getTcVersion(projectId, planId, refinement.version_id),
  });

  const totals = refinement.submodule_summaries.reduce(
    (acc, s) => ({
      kept: acc.kept + s.kept,
      rewritten: acc.rewritten + s.rewritten,
      added: acc.added + s.added,
      flagged_missing: acc.flagged_missing + s.flagged_missing,
    }),
    { kept: 0, rewritten: 0, added: 0, flagged_missing: 0 },
  );

  // Phase D — validation rollup
  const validationBySnap = new Map(
    (validation?.submodules ?? []).map((s) => [s.title, s]),
  );
  const validationTotals = (validation?.submodules ?? []).reduce(
    (acc, s) => ({
      confirmed: acc.confirmed + s.confirmed,
      partial: acc.partial + s.partial,
      unresolved: acc.unresolved + s.unresolved,
      unreachable: acc.unreachable + s.unreachable,
      skipped: acc.skipped + s.skipped,
    }),
    { confirmed: 0, partial: 0, unresolved: 0, unreachable: 0, skipped: 0 },
  );

  return (
    <div className="space-y-3">
      <div className="rounded-md border bg-card p-3 text-xs">
        <p className="text-sm font-medium">
          v{refinement.version_number} — refinement summary
        </p>
        <div className="mt-1.5 flex flex-wrap gap-2">
          <SummaryChip kind="kept" count={totals.kept} />
          <SummaryChip kind="rewritten" count={totals.rewritten} />
          <SummaryChip kind="added" count={totals.added} />
          <SummaryChip
            kind="flagged_missing"
            count={totals.flagged_missing}
          />
        </div>
        <p className="mt-1.5 text-[10px] text-muted-foreground">
          {refinement.submodule_count} submodules · cost{" "}
          {refinement.input_tokens.toLocaleString()} in /{" "}
          {refinement.output_tokens.toLocaleString()} out tokens
        </p>
      </div>

      {validation && (
        <div className="rounded-md border border-blue-500/30 bg-blue-500/5 p-3 text-xs">
          <p className="text-sm font-medium text-blue-900 dark:text-blue-200">
            Live-UI validation summary
          </p>
          <div className="mt-1.5 flex flex-wrap gap-2">
            <ValidationChip
              status="confirmed"
              count={validationTotals.confirmed}
            />
            <ValidationChip
              status="partial"
              count={validationTotals.partial}
            />
            <ValidationChip
              status="unresolved"
              count={validationTotals.unresolved}
            />
            <ValidationChip
              status="unreachable"
              count={validationTotals.unreachable}
            />
            <ValidationChip
              status="skipped"
              count={validationTotals.skipped}
            />
          </div>
          <p className="mt-1.5 text-[10px] text-blue-800/80 dark:text-blue-300/80">
            Probed {validation.total_probed} step(s) in{" "}
            {validation.total_seconds.toFixed(1)}s
            {validation.error_message ? (
              <>
                {" · "}
                <span className="text-red-700 dark:text-red-400">
                  {validation.error_message}
                </span>
              </>
            ) : null}
          </p>
        </div>
      )}

      <div className="max-h-[40vh] space-y-1.5 overflow-y-auto pr-1">
        {refinement.submodule_summaries.map((sm) => (
          <SubmoduleRow
            key={sm.submodule_id}
            summary={sm}
            validation={validationBySnap.get(sm.title) ?? null}
            detail={detail ?? null}
            isLoadingDetail={isLoading}
          />
        ))}
      </div>
    </div>
  );
}


function ValidationChip({
  status,
  count,
}: {
  status: ValidationStatus;
  count: number;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-medium",
        VALIDATION_TINT[status],
        count === 0 && "opacity-50",
      )}
    >
      {count} {VALIDATION_LABEL[status]}
    </span>
  );
}

function SummaryChip({
  kind,
  count,
}: {
  kind: TcChangeKind;
  count: number;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-2 py-0.5 text-[11px] font-medium",
        CHANGE_TINT[kind],
        count === 0 && "opacity-50",
      )}
    >
      {count} {CHANGE_LABEL[kind]}
    </span>
  );
}

function SubmoduleRow({
  summary,
  validation,
  detail,
  isLoadingDetail,
}: {
  summary: TcRefinementResponse["submodule_summaries"][number];
  validation: TcValidationResponse["submodules"][number] | null;
  detail: TcVersionDetail | null;
  isLoadingDetail: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const lowConfidence = summary.confidence < 0.6;

  // Find this submodule's snapshot + its children for the expanded view.
  const sm = detail?.snapshots.find(
    (s) => s.original_tc_node_id === summary.submodule_id,
  );
  const steps = sm
    ? (detail?.snapshots ?? []).filter(
        (s) => s.parent_snapshot_id === sm.id && s.kind === "step",
      ).sort((a, b) => a.ordinal - b.ordinal)
    : [];

  return (
    <div className="rounded border bg-card">
      <button
        type="button"
        className="flex w-full items-baseline gap-2 px-2 py-1.5 text-left text-xs hover:bg-muted/50"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="font-mono text-[9px] text-muted-foreground">
          {expanded ? "▼" : "▶"}
        </span>
        <span className="min-w-0 flex-1 truncate font-medium">
          {summary.title}
        </span>
        {summary.error ? (
          <span className="shrink-0 inline-flex items-center gap-1 text-red-600">
            <XCircle className="size-3" />
            error
          </span>
        ) : (
          <>
            <SummaryChip kind="kept" count={summary.kept} />
            <SummaryChip kind="rewritten" count={summary.rewritten} />
            <SummaryChip kind="added" count={summary.added} />
            {summary.flagged_missing > 0 && (
              <SummaryChip
                kind="flagged_missing"
                count={summary.flagged_missing}
              />
            )}
          </>
        )}
        {lowConfidence && !summary.error && (
          <span
            className="shrink-0 inline-flex items-center gap-1 text-amber-600"
            title="Refiner reported low confidence — scout may have missed this submodule's scope"
          >
            <AlertTriangle className="size-3" />
            {Math.round(summary.confidence * 100)}%
          </span>
        )}
        {validation && (
          <span
            className={cn(
              "shrink-0 rounded border px-1.5 py-0.5 text-[9px] font-medium",
              validation.confidence >= 0.8
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                : validation.confidence >= 0.5
                  ? "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400"
                  : "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
            )}
            title={
              `${validation.confirmed} confirmed · ` +
              `${validation.partial} partial · ` +
              `${validation.unresolved} unresolved · ` +
              `${validation.unreachable} unreachable`
            }
          >
            live {Math.round(validation.confidence * 100)}%
          </span>
        )}
      </button>
      {expanded && (
        <div className="border-t bg-muted/20 px-2 py-1.5">
          {summary.error ? (
            <p className="text-[11px] text-red-700 dark:text-red-400">
              {summary.error}
            </p>
          ) : isLoadingDetail ? (
            <Skeleton className="h-16 w-full" />
          ) : steps.length === 0 ? (
            <p className="text-[10px] italic text-muted-foreground">
              (no step snapshots returned)
            </p>
          ) : (
            <ol className="space-y-0.5 text-[11px]">
              {steps.map((step) => (
                <li
                  key={step.id}
                  className={cn(
                    "rounded border px-1.5 py-1",
                    CHANGE_TINT[step.change_kind],
                  )}
                >
                  <div className="flex items-baseline gap-1.5">
                    <span className="shrink-0 font-mono text-[9px] uppercase tracking-wide opacity-80">
                      {step.change_kind}
                    </span>
                    <span className="min-w-0 flex-1 truncate font-medium">
                      {step.title}
                    </span>
                    {step.validation_status &&
                      step.validation_status !== "pending" && (
                        <span
                          className={cn(
                            "shrink-0 rounded border px-1 text-[9px] font-medium",
                            VALIDATION_TINT[step.validation_status],
                          )}
                          title={step.validation_reason ?? ""}
                        >
                          {VALIDATION_LABEL[step.validation_status]}
                        </span>
                      )}
                    {step.action_type && (
                      <span className="shrink-0 text-[9px] opacity-70">
                        {step.action_type}
                      </span>
                    )}
                  </div>
                  {step.target_hint && (
                    <p className="ml-1 truncate text-[10px] opacity-80">
                      target:{" "}
                      <code className="rounded bg-background/50 px-1">
                        {step.target_hint}
                      </code>
                    </p>
                  )}
                  {step.change_reason &&
                    step.change_kind !== "kept" && (
                      <p className="ml-1 mt-0.5 text-[10px] italic opacity-80">
                        {step.change_reason}
                      </p>
                    )}
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}
