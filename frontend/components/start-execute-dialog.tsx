"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Eye,
  EyeOff,
  Gauge,
  GitCommit,
  Play,
  Sparkles,
  Turtle,
  Zap,
} from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  PLAN_STATUS_LABELS,
  type PlanReadCompact,
} from "@/lib/api";

type Speed = "slow" | "normal" | "fast";

const SPEED_OPTIONS: {
  value: Speed;
  label: string;
  icon: typeof Turtle;
  hint: string;
}[] = [
  {
    value: "slow",
    label: "Slow",
    icon: Turtle,
    hint: "Visible cursor glide, slow typing, 8s settle. Heavy SPAs friendly.",
  },
  {
    value: "normal",
    label: "Normal",
    icon: Gauge,
    hint: "Balanced. 5s settle, smaller cursor glide.",
  },
  {
    value: "fast",
    label: "Fast",
    icon: Zap,
    hint: "No typing animation, 2s settle. May fail on lazy-loading sites.",
  },
];
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
  const [speed, setSpeed] = useState<Speed>("slow");
  const [autoAdjust, setAutoAdjust] = useState(false);
  const [promoteFixes, setPromoteFixes] = useState(false);

  useEffect(() => {
    if (open) {
      setPlanId(defaultPlanId ?? null);
      setHeadless(false);
      setSpeed("slow");
      setAutoAdjust(false);
      setPromoteFixes(false);
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
    mutationFn: () => {
      // Compute screen-aware tiling once so the browser and the live
      // panel share the same geometry contract:
      //   browser : (0, 0) → (availWidth - PANEL, availHeight)
      //   panel   : (availWidth - PANEL, 0) → (PANEL, availHeight)
      // Sent to the backend as window_x / y / width / height. Headless
      // runs and tiny screens fall through to backend defaults safely.
      const layout = computeWindowLayout();
      return api.startExecute(projectId, {
        plan_id: planId!,
        headless,
        speed,
        auto_adjust: autoAdjust,
        promote_fixes: promoteFixes,
        window_x: layout?.browser.x,
        window_y: layout?.browser.y,
        window_width: layout?.browser.w,
        window_height: layout?.browser.h,
      });
    },
    onSuccess: (run) => {
      // Open the live presenter pinned to the right of the screen, sized
      // to exactly fill the gap left by the (left-tiled) browser window.
      try {
        const layout = computeWindowLayout();
        if (layout) {
          const { panel } = layout;
          const features = `popup=yes,width=${panel.w},height=${panel.h},left=${panel.x},top=${panel.y},resizable=yes,scrollbars=yes`;
          const url = `/live/${projectId}/${run.id}`;
          const popup = window.open(url, `qai-live-run-${run.id}`, features);
          if (!popup) {
            toast.message("Popup blocked", {
              description:
                "Allow popups for this site to see the live agent panel beside the browser.",
            });
          }
        }
      } catch {
        /* if window.open throws (sandboxed), fall through silently */
      }

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

            {/* Speed knob — slow is the default for heavy-data sites where
                an over-eager click on a skeleton row beats the actual data.
                Keep this prominent so it isn't accidentally skipped. */}
            <SpeedPicker value={speed} onChange={setSpeed} />

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

            <ToggleRow
              icon={Sparkles}
              active={autoAdjust}
              activeTone="amber"
              onToggle={() => setAutoAdjust(!autoAdjust)}
              ariaLabel={autoAdjust ? "Disable auto-adjust" : "Enable auto-adjust"}
              title={autoAdjust ? "Auto-adjust on" : "Auto-adjust off"}
              hint={
                autoAdjust
                  ? "AI fixes silently when it can. HITL only fires if both text and vision passes still fail. Faster, but skips your review."
                  : "AI suggestions are proposed only — the HITL modal pre-fills with the suggestion and you approve / edit / reject. Recommended."
              }
            />

            <ToggleRow
              icon={GitCommit}
              active={promoteFixes}
              activeTone="emerald"
              onToggle={() => setPromoteFixes(!promoteFixes)}
              ariaLabel={
                promoteFixes
                  ? "Disable promote fixes"
                  : "Enable promote fixes"
              }
              title={
                promoteFixes ? "Promote fixes to test case" : "Don't promote fixes"
              }
              hint={
                promoteFixes
                  ? "When a fix produces a passing step (AI auto or HITL override), patch the source test case so the next run starts with it."
                  : "Fixes apply only to this run. The source test case is left as-is."
              }
            />
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

function ToggleRow({
  icon: Icon,
  active,
  onToggle,
  ariaLabel,
  title,
  hint,
  activeTone = "primary",
}: {
  icon: typeof Sparkles;
  active: boolean;
  onToggle: () => void;
  ariaLabel: string;
  title: string;
  hint: string;
  activeTone?: "primary" | "amber" | "emerald";
}) {
  const activeClass =
    activeTone === "amber"
      ? "border-amber-500/50 bg-amber-500/15 text-amber-700 dark:text-amber-300"
      : activeTone === "emerald"
        ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300"
        : "border-primary/40 bg-primary/10 text-primary";
  return (
    <div className="flex items-start gap-3 rounded-md border p-3">
      <button
        type="button"
        onClick={onToggle}
        className={cn(
          "mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md border transition-colors",
          active
            ? activeClass
            : "border-input bg-muted text-muted-foreground",
        )}
        aria-pressed={active}
        aria-label={ariaLabel}
      >
        <Icon className="size-4" />
      </button>
      <div className="min-w-0 flex-1 text-sm">
        <p className="font-medium">{title}</p>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </div>
    </div>
  );
}

/**
 * Compute side-by-side window geometry from the user's actual screen.
 * Splits ``screen.availWidth × availHeight`` into:
 *   - browser pane on the left (everything minus the panel)
 *   - panel popup on the right (PANEL_WIDTH wide, full height)
 * Returns ``null`` when there's no DOM (SSR) or the screen is too narrow
 * to usefully tile — in which case the caller skips the popup and the
 * backend uses its built-in defaults.
 */
const PANEL_WIDTH = 600;
const MIN_BROWSER_WIDTH = 720;
function computeWindowLayout() {
  if (typeof window === "undefined" || !window.screen) return null;
  const availW = window.screen.availWidth ?? window.screen.width ?? 0;
  const availH = window.screen.availHeight ?? window.screen.height ?? 0;
  if (availW < PANEL_WIDTH + MIN_BROWSER_WIDTH || availH < 500) return null;
  const browserW = availW - PANEL_WIDTH;
  return {
    browser: { x: 0, y: 0, w: browserW, h: availH },
    panel: { x: browserW, y: 0, w: PANEL_WIDTH, h: availH },
  };
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

function SpeedPicker({
  value,
  onChange,
}: {
  value: Speed;
  onChange: (next: Speed) => void;
}) {
  const active = SPEED_OPTIONS.find((o) => o.value === value);
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">Speed</label>
      <div className="grid grid-cols-3 gap-2">
        {SPEED_OPTIONS.map((opt) => {
          const Icon = opt.icon;
          const isActive = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              aria-pressed={isActive}
              className={cn(
                "flex flex-col items-center gap-1 rounded-md border px-3 py-2 text-xs font-medium transition-colors",
                isActive
                  ? "border-primary/50 bg-primary/10 text-primary"
                  : "hover:border-input hover:bg-muted/50 text-muted-foreground",
              )}
            >
              <Icon className="size-4" />
              {opt.label}
            </button>
          );
        })}
      </div>
      {active && (
        <p className="text-xs text-muted-foreground">{active.hint}</p>
      )}
    </div>
  );
}
