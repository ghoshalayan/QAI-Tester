"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, Check, RefreshCw, SkipForward, X } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  type InterventionChoice,
  type InterventionRequest,
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface Props {
  projectId: number;
  runId: number;
  intervention: InterventionRequest;
}

/**
 * HITL modal that pops when a step has burned its auto-retry budget AND
 * AI assist couldn't fix it. Stays open indefinitely (Q1: wait for user).
 *
 * Closes via the ``intervention_resolved`` SSE event clearing the
 * Zustand store entry the parent reads to decide whether to render us.
 * The Cancel button (X in dialog header) is intentionally absent — a
 * dismissed modal would orphan the orchestrator on its waiter. Use Stop
 * run instead.
 */
export function InterventionModal({
  projectId,
  runId,
  intervention,
}: Props) {
  // Pre-fill the override input with the AI's suggested target_hint
  // if any — saves a copy/paste on the common path.
  const aiSuggestedHint = useMemo(() => {
    const diff = intervention.ai_suggestion?.diff;
    if (!diff || typeof diff !== "object") return "";
    const t = diff.target_hint;
    if (!t || typeof t !== "object") return "";
    return String((t as { new?: unknown }).new ?? "");
  }, [intervention.ai_suggestion]);

  const [overrideHint, setOverrideHint] = useState(aiSuggestedHint);
  const [applyToSubmodule, setApplyToSubmodule] = useState(false);

  // When the modal swaps from one stuck step to another (rare —
  // sequential failures), reset the form rather than carrying over
  // the previous step's override text.
  useEffect(() => {
    setOverrideHint(aiSuggestedHint);
    setApplyToSubmodule(false);
  }, [aiSuggestedHint, intervention.step_id]);

  const submitMutation = useMutation({
    mutationFn: (choice: InterventionChoice) =>
      api.provideIntervention(projectId, runId, {
        step_id: intervention.step_id,
        choice,
        override_target_hint:
          choice === "use_suggestion" ? overrideHint || null : null,
        apply_to_submodule: applyToSubmodule,
      }),
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Failed to submit", { description: msg });
    },
    // No onSuccess: the modal closes when the SSE
    // intervention_resolved event clears the Zustand entry the parent
    // checks. That's authoritative — we don't want to close prematurely
    // on a 200 if the orchestrator hasn't actually applied the choice yet.
  });

  const aiSuggestion = intervention.ai_suggestion;
  const aiHasDiff =
    !!aiSuggestion &&
    aiSuggestion.action === "replace" &&
    !!aiSuggestion.diff &&
    Object.keys(aiSuggestion.diff).length > 0;

  // Don't allow accidental ESC dismissal — the modal must persist
  // until the user picks something. ``onOpenChange`` ignores
  // close-attempts.
  return (
    <Dialog
      open={true}
      onOpenChange={() => {
        /* swallow — only the buttons close this */
      }}
    >
      <DialogContent
        className="max-h-[90vh] max-w-2xl overflow-auto"
        onEscapeKeyDown={(e) => e.preventDefault()}
        onPointerDownOutside={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="size-5 text-yellow-600 dark:text-yellow-400" />
            Step needs help
          </DialogTitle>
          <DialogDescription>
            Step {intervention.ordinal}/{intervention.total} — auto-retry
            {aiSuggestion ? " and AI assist " : " "}
            couldn&apos;t recover. Pick what to do next.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          {/* Step info */}
          <div className="rounded-md border bg-muted/30 p-3">
            <div className="flex flex-wrap items-baseline gap-2">
              <span className="font-medium">{intervention.title}</span>
              {intervention.action_type && (
                <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                  {intervention.action_type}
                </span>
              )}
            </div>
            {intervention.target_hint && (
              <code className="mt-1 block break-all text-xs text-muted-foreground">
                target_hint: {intervention.target_hint}
              </code>
            )}
          </div>

          {/* Screenshot — page state at failure */}
          {intervention.screenshot_path && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={api.screenshotUrl(intervention.screenshot_path)}
              alt="Page state at failure"
              className="w-full rounded-md border"
              loading="lazy"
            />
          )}

          {/* Last error */}
          {intervention.error_message && (
            <div className="rounded-md border border-red-500/30 bg-red-500/5 px-3 py-2 font-mono text-[11px] text-red-700 dark:text-red-400">
              {intervention.error_message}
            </div>
          )}

          {/* AI suggestion */}
          {aiSuggestion && (
            <div className="rounded-md border border-blue-500/30 bg-blue-500/5 px-3 py-2 text-xs">
              <p className="font-medium text-blue-700 dark:text-blue-400">
                AI suggestion · action: {aiSuggestion.action} ·{" "}
                {(aiSuggestion.confidence * 100).toFixed(0)}% confidence
              </p>
              {aiSuggestion.reasoning && (
                <p className="mt-1 text-foreground">
                  {aiSuggestion.reasoning}
                </p>
              )}
              {aiHasDiff && (
                <div className="mt-2 space-y-1 font-mono text-[10px]">
                  {Object.entries(aiSuggestion.diff).map(([field, change]) => (
                    <div key={field}>
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
          )}

          {/* Override input */}
          <div className="space-y-1">
            <Label htmlFor="override-hint">
              Override target_hint (used by &quot;Use override&quot;)
            </Label>
            <Input
              id="override-hint"
              value={overrideHint}
              onChange={(e) => setOverrideHint(e.target.value)}
              placeholder={
                intervention.target_hint || "leave empty to keep original"
              }
              className="font-mono text-xs"
            />
            <p className="text-xs text-muted-foreground">
              {aiSuggestedHint
                ? "Pre-filled with the AI's suggested selector — edit before submitting if needed."
                : "Type a CSS selector or text marker (e.g. \"text 'Sign In'\")."}
            </p>
          </div>

          {/* apply_to_submodule */}
          <label
            className={cn(
              "flex cursor-pointer items-start gap-2 rounded-md border p-3 text-sm transition-colors",
              applyToSubmodule
                ? "border-primary/40 bg-primary/5"
                : "hover:bg-muted/40",
            )}
          >
            <input
              type="checkbox"
              checked={applyToSubmodule}
              onChange={(e) => setApplyToSubmodule(e.target.checked)}
              className="mt-0.5 size-4 shrink-0 cursor-pointer accent-primary"
            />
            <div className="min-w-0">
              <p className="font-medium">
                Auto-apply for the rest of this submodule
              </p>
              <p className="text-xs text-muted-foreground">
                Skip the modal on subsequent failures under the same
                submodule; apply this same choice automatically.
              </p>
            </div>
          </label>
        </div>

        <DialogFooter className="flex-wrap gap-2">
          <Button
            variant="outline"
            onClick={() => submitMutation.mutate("retry")}
            disabled={submitMutation.isPending}
          >
            <RefreshCw className="size-4" />
            Retry as-is
          </Button>
          <Button
            onClick={() => submitMutation.mutate("use_suggestion")}
            disabled={submitMutation.isPending || !overrideHint}
            title={
              !overrideHint
                ? "Type or accept an override target_hint to enable"
                : undefined
            }
          >
            <Check className="size-4" />
            Use override
          </Button>
          <Button
            variant="outline"
            onClick={() => submitMutation.mutate("skip")}
            disabled={submitMutation.isPending}
          >
            <SkipForward className="size-4" />
            Skip step
          </Button>
          <Button
            variant="destructive"
            onClick={() => submitMutation.mutate("stop")}
            disabled={submitMutation.isPending}
          >
            <X className="size-4" />
            Stop run
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
