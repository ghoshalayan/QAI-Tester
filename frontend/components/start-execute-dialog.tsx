"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  Circle,
  Eye,
  EyeOff,
  Gauge,
  GitCommit,
  ListChecks,
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
import { Input } from "@/components/ui/input";
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
  // Phase M — default to agentic + vision_only. The "Start" button
  // works with just a plan picked; everything else lives behind an
  // Advanced disclosure (closed by default) for power-user overrides.
  const [mode, setMode] = useState<
    "scripted" | "agentic" | "replay" | "record"
  >("agentic");
  // Phase W' — when mode = "record", the operator picks which MODULE
  // to record into. Submodule attribution happens live on the
  // presenter (via Start-chunk button + searchable submodule combobox).
  const [recordModuleId, setRecordModuleId] = useState<number | null>(
    null,
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Phase 6 — within agentic mode, choose hybrid (DOM-first ladder)
  // or vision_only (VL+coords for every click/type, computer-use
  // pattern). hybrid is faster + cheaper on most apps; vision_only
  // works on apps DOM resolution can't reach.
  const [agentStrategy, setAgentStrategy] = useState<"hybrid" | "vision_only">(
    "hybrid",
  );
  // Phase H — preflight selector. "auto" runs Scout + refine before
  // the first submodule if the plan isn't already pinned to an
  // app_map_refined version. "force" re-scouts + re-refines from
  // scratch. "skip" disables preflight entirely (debugging only).
  const [preflight, setPreflight] = useState<"auto" | "force" | "skip">(
    "auto",
  );

  useEffect(() => {
    if (open) {
      setPlanId(defaultPlanId ?? null);
      setHeadless(false);
      setSpeed("slow");
      setAutoAdjust(false);
      setPromoteFixes(false);
      // Phase M — sensible defaults match the recommended path:
      // agentic + vision_only + preflight=auto. Power users open the
      // "Advanced options" disclosure to override.
      setMode("agentic");
      setAgentStrategy("vision_only");
      setPreflight("auto");
      setAdvancedOpen(false);
      setRecordModuleId(null);
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
      // Compute screen-aware tiling once so the headed browser and
      // the live panel share the same geometry contract:
      //   headed:   browser 60% × full height,  panel 40% × full height
      //   headless: no browser, panel fills the screen
      // Backend window_* fields are only sent when there's a headed
      // browser to position; headless runs let the backend skip the
      // launch flags entirely.
      const layout = computeWindowLayout(headless);
      const browserGeom = layout?.browser;
      // Phase W' — record-mode branch: hits the start-recording
      // endpoint with a MODULE id. Submodule attribution happens
      // live on the presenter after the browser opens.
      if (mode === "record") {
        if (recordModuleId == null) {
          throw new Error("Pick a module to record into first.");
        }
        return api.startRecording(projectId, {
          plan_id: planId!,
          module_id: recordModuleId,
        });
      }
      return api.startExecute(projectId, {
        plan_id: planId!,
        headless,
        speed,
        // The backend's execute pipeline only knows
        // scripted/agentic/replay; "record" never reaches here.
        mode: mode as "scripted" | "agentic" | "replay",
        agent_strategy: agentStrategy,
        preflight,
        auto_adjust: autoAdjust,
        promote_fixes: promoteFixes,
        window_x: browserGeom?.x,
        window_y: browserGeom?.y,
        window_width: browserGeom?.w,
        window_height: browserGeom?.h,
      });
    },
    onSuccess: (run) => {
      // Open the live presenter sized to match the layout used for the
      // browser. The popup features string is only a HINT — Chromium
      // sometimes ignores it when DPI scaling is on, when the popup
      // blocker is aggressive, or when it decides to open as a tab.
      // We back the hint up with explicit resizeTo() + moveTo() calls
      // after the popup loads, which reliably set window geometry.
      try {
        const layout = computeWindowLayout(headless);
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
          } else {
            // Belt-and-braces: explicitly position + size the popup
            // after open. Chrome ignores the features `left`/`width`
            // about half the time but reliably honors moveTo + resizeTo
            // on a popup window.
            const enforce = () => {
              try {
                popup.resizeTo(panel.w, panel.h);
                popup.moveTo(panel.x, panel.y);
              } catch {
                /* same-origin policy or sandboxed — best-effort only */
              }
            };
            // Fire once now, then again after a short delay — the
            // popup needs a moment to navigate to the URL before
            // resizeTo reliably sticks.
            enforce();
            setTimeout(enforce, 150);
            setTimeout(enforce, 600);
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
    !startMutation.isPending &&
    // Phase W' — record mode also requires a target module.
    (mode !== "record" || recordModuleId !== null);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl max-h-[90vh] overflow-y-auto">
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

            {/* Phase M — default-run summary. The recommended path is
                agentic + vision-only + preflight=auto, set as the
                defaults. The full pickers live under Advanced. */}
            <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
              <p className="font-medium text-foreground">
                Default: agentic · vision-only · auto preflight · slow
              </p>
              <p>
                Just click <span className="font-medium">Start run</span>.
                Open <span className="font-medium">Advanced options</span> below
                only if you need to deviate.
              </p>
            </div>

            {/* Phase M — Advanced options accordion. Collapsed by
                default; all the legacy pickers live here. */}
            <details
              open={advancedOpen}
              onToggle={(e) =>
                setAdvancedOpen((e.target as HTMLDetailsElement).open)
              }
              className="rounded-md border"
            >
              <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium">
                Advanced options
              </summary>
              <div className="space-y-4 border-t p-3">
                <ModePicker value={mode} onChange={setMode} />

                {mode === "record" && planId !== null && (
                  <RecordModulePicker
                    projectId={projectId}
                    planId={planId}
                    value={recordModuleId}
                    onChange={setRecordModuleId}
                  />
                )}

                {mode === "agentic" && (
                  <AgentStrategyPicker
                    value={agentStrategy}
                    onChange={setAgentStrategy}
                  />
                )}

                {mode === "agentic" && (
                  <PreflightPicker
                    value={preflight}
                    onChange={setPreflight}
                  />
                )}

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
                        ? "Faster, no visible window. Switch to headed if a step blocks for HITL."
                        : "A visible Chrome window opens; you can watch the run. Recommended."}
                    </p>
                  </div>
                </div>

                <ToggleRow
                  icon={Sparkles}
                  active={autoAdjust}
                  activeTone="amber"
                  onToggle={() => setAutoAdjust(!autoAdjust)}
                  ariaLabel={
                    autoAdjust ? "Disable auto-adjust" : "Enable auto-adjust"
                  }
                  title={autoAdjust ? "Auto-adjust on" : "Auto-adjust off"}
                  hint={
                    autoAdjust
                      ? "AI fixes silently when it can. HITL only fires if both text and vision passes still fail."
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
                    promoteFixes
                      ? "Promote fixes to test case"
                      : "Don't promote fixes"
                  }
                  hint={
                    promoteFixes
                      ? "When a fix produces a passing step (AI auto or HITL override), patch the source test case so the next run starts with it."
                      : "Fixes apply only to this run. The source test case is left as-is."
                  }
                />
              </div>
            </details>
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
 * Compute window geometry for the headed browser + live panel popup.
 *
 * Strategy: the headed Chromium fills the FULL screen width, and the
 * live panel popup sits ON TOP of its right edge as an overlay. This
 * mirrors Atlas/Comet/Operator and avoids the "tile two windows
 * flush" problem on Windows — Chrome's `--window-size` and
 * `popup.moveTo` don't land pixel-exact under DPI scaling, so any
 * tiled layout leaks a black seam between the two windows. Letting
 * the browser extend past the popup means there's never a gap, no
 * matter how Chrome rounds.
 *
 * Trade-off: the rightmost ~PANEL_W pixels of the page are occluded
 * by the live panel. This is fine for QA — the agent renders a
 * dedicated panel anyway, and most page content is on the left.
 *
 * Headless runs: panel takes the full screen, browser is null.
 *
 * Returns ``null`` when there's no DOM (SSR) or the screen is too
 * small — in which case the caller skips popup geometry and the
 * backend uses its defaults.
 */
// Live panel as a fraction of total screen width. ~13% gives ~250px
// on a 1920px display — enough for one event per row.
const PANEL_FRACTION = 0.13;
const PANEL_MIN_W = 240;
const PANEL_MAX_W = 360;
const MIN_USEFUL_HEIGHT = 500;
const MIN_USEFUL_WIDTH_HEADED = 1000;
function computeWindowLayout(headless: boolean) {
  if (typeof window === "undefined" || !window.screen) return null;
  const availW = window.screen.availWidth ?? window.screen.width ?? 0;
  const availH = window.screen.availHeight ?? window.screen.height ?? 0;
  if (availH < MIN_USEFUL_HEIGHT) return null;

  if (headless) {
    // No headed browser to overlay against — panel fills the screen.
    if (availW < 400) return null;
    return {
      browser: null as null | { x: number; y: number; w: number; h: number },
      panel: { x: 0, y: 0, w: availW, h: availH },
    };
  }

  // Headed: browser spans the full screen so it always extends past
  // wherever the popup lands. The popup overlays the right strip.
  if (availW < MIN_USEFUL_WIDTH_HEADED) return null;
  const panelW = Math.min(
    PANEL_MAX_W,
    Math.max(PANEL_MIN_W, Math.floor(availW * PANEL_FRACTION)),
  );
  const panelX = availW - panelW;
  return {
    browser: { x: 0, y: 0, w: availW, h: availH },
    panel: { x: panelX, y: 0, w: panelW, h: availH },
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

function AgentStrategyPicker({
  value,
  onChange,
}: {
  value: "hybrid" | "vision_only";
  onChange: (next: "hybrid" | "vision_only") => void;
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">Agent strategy</label>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => onChange("hybrid")}
          aria-pressed={value === "hybrid"}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            value === "hybrid"
              ? "border-primary/50 bg-primary/10 text-primary"
              : "hover:border-input hover:bg-muted/50",
          )}
        >
          <span className="flex items-center gap-1.5 text-sm font-medium">
            <Bot className="size-4" />
            Hybrid (DOM + vision)
          </span>
          <span className="text-[11px] text-muted-foreground">
            DOM-first ladder with vision rescue. Fast + cheap on most
            apps. Recommended.
          </span>
        </button>
        <button
          type="button"
          onClick={() => onChange("vision_only")}
          aria-pressed={value === "vision_only"}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            value === "vision_only"
              ? "border-primary/50 bg-primary/10 text-primary"
              : "hover:border-input hover:bg-muted/50",
          )}
        >
          <span className="flex items-center gap-1.5 text-sm font-medium">
            <Eye className="size-4" />
            Vision-only (computer use)
          </span>
          <span className="text-[11px] text-muted-foreground">
            Every click + type via VL pixel coords. Bypasses DOM
            entirely. ~3-5× tokens. Pick for SAP / heavy canvas /
            sealed-shadow-DOM apps.
          </span>
        </button>
      </div>
    </div>
  );
}


// Phase W — pick which submodule the recording attaches to.
// Lists all module → submodule rows under the plan so the operator
// sees the test-case hierarchy. Disabled rows = no module yet
// (recordings can't attach to a module-level node).
function RecordModulePicker({
  projectId,
  planId,
  value,
  onChange,
}: {
  projectId: number;
  planId: number;
  value: number | null;
  onChange: (next: number | null) => void;
}) {
  const qc = useQueryClient();
  const { data: nodes, isLoading } = useQuery({
    queryKey: ["tc-nodes", projectId, planId],
    queryFn: () => api.listTcNodes(projectId, planId),
  });
  // Top-level entries returned by listTcNodes are modules.
  const modules = (nodes ?? []).filter((n) => n.kind === "module");

  const [newTitle, setNewTitle] = useState("");
  const [showInline, setShowInline] = useState(false);

  const addMut = useMutation({
    mutationFn: () =>
      api.createTcNode(projectId, planId, {
        title: newTitle.trim(),
        kind: "module",
      }),
    onSuccess: (created) => {
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, planId] });
      toast.success("Module added", {
        description: `"${created.title}" ready to record into.`,
      });
      onChange(created.id);
      setNewTitle("");
      setShowInline(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Could not add module", { description: msg });
    },
  });

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">
        Read into module
      </label>
      {isLoading ? (
        <Skeleton className="h-9 w-full" />
      ) : modules.length === 0 && !showInline ? (
        <div className="space-y-2">
          <p className="text-xs text-amber-600">
            This plan has no modules yet.
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setShowInline(true)}
          >
            + Add a module
          </Button>
          <p className="text-[10px] text-muted-foreground">
            Or generate the full Module → Submodule → Step tree via{" "}
            <strong>Test Cases tab → Generate test cases</strong> (uses
            your BRD / FRD).
          </p>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <select
            value={value ?? ""}
            onChange={(e) =>
              onChange(e.target.value ? Number(e.target.value) : null)
            }
            className="h-9 flex-1 rounded-md border border-input bg-background px-2 text-sm"
          >
            <option value="">— pick a module —</option>
            {modules.map((m) => (
              <option key={m.id} value={m.id}>
                {m.title || `Module #${m.id}`}
                {m.children?.length
                  ? ` · ${m.children.length} submodule${
                      m.children.length === 1 ? "" : "s"
                    }`
                  : ""}
              </option>
            ))}
          </select>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setShowInline((v) => !v)}
            title="Add a new module"
          >
            {showInline ? "Cancel" : "+ New"}
          </Button>
        </div>
      )}
      {showInline && (
        <div className="flex items-center gap-2 rounded-md border bg-muted/30 p-2">
          <Input
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="e.g. Administration"
            className="h-8 flex-1 text-sm"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === "Enter" && newTitle.trim()) addMut.mutate();
            }}
          />
          <Button
            type="button"
            size="sm"
            onClick={() => addMut.mutate()}
            disabled={!newTitle.trim() || addMut.isPending}
          >
            {addMut.isPending ? "Adding…" : "Add"}
          </Button>
        </div>
      )}
      <p className="text-[11px] text-muted-foreground">
        The browser opens maximized so the bottom of the page isn't
        hidden by the taskbar. On the live presenter you'll pick a{" "}
        <strong>submodule</strong> from a searchable list and click{" "}
        <strong>Start chunk</strong>. Every click/type from that
        moment attributes to that submodule. Switch submodules any
        time. Click <strong>Stop reading</strong> when the whole
        module is recorded — every populated submodule chunk
        persists separately and replays automatically.
      </p>
    </div>
  );
}


function PreflightPicker({
  value,
  onChange,
}: {
  value: "auto" | "force" | "skip";
  onChange: (next: "auto" | "force" | "skip") => void;
}) {
  const options: Array<{
    key: "auto" | "force" | "skip";
    label: string;
    blurb: string;
  }> = [
    {
      key: "auto",
      label: "Auto",
      blurb:
        "Scout + refine the plan against the actual UI on first run; reuse cached refinement on subsequent runs.",
    },
    {
      key: "force",
      label: "Force re-scout",
      blurb:
        "Re-scout the app and re-refine the test cases from scratch. Use after UI changes or BRD edits.",
    },
    {
      key: "skip",
      label: "Skip",
      blurb:
        "Run with the live test-case tree as-is. Faster but the agent may misclick on labels that don't match the UI.",
    },
  ];
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">
        Preflight (validate test cases against UI)
      </label>
      <div className="grid grid-cols-3 gap-2">
        {options.map((opt) => (
          <button
            key={opt.key}
            type="button"
            onClick={() => onChange(opt.key)}
            aria-pressed={value === opt.key}
            className={cn(
              "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
              value === opt.key
                ? "border-primary/50 bg-primary/10 text-primary"
                : "hover:border-input hover:bg-muted/50",
            )}
          >
            <span className="flex items-center gap-1.5 text-sm font-medium">
              <Sparkles className="size-4" />
              {opt.label}
            </span>
            <span className="text-[11px] text-muted-foreground">
              {opt.blurb}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}


function ModePicker({
  value,
  onChange,
}: {
  value: "scripted" | "agentic" | "replay" | "record";
  onChange: (
    next: "scripted" | "agentic" | "replay" | "record",
  ) => void;
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">Run mode</label>
      <div className="grid grid-cols-4 gap-2">
        <button
          type="button"
          onClick={() => onChange("scripted")}
          aria-pressed={value === "scripted"}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            value === "scripted"
              ? "border-primary/50 bg-primary/10 text-primary"
              : "hover:border-input hover:bg-muted/50",
          )}
        >
          <span className="flex items-center gap-1.5 text-sm font-medium">
            <ListChecks className="size-4" />
            Scripted
          </span>
          <span className="text-[11px] text-muted-foreground">
            Rigid walker. AI patches only on failure. Cheapest.
          </span>
        </button>
        <button
          type="button"
          onClick={() => onChange("record")}
          aria-pressed={value === "record"}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            value === "record"
              ? "border-primary/50 bg-primary/10 text-primary"
              : "hover:border-input hover:bg-muted/50",
          )}
        >
          <span className="flex items-center gap-1.5 text-sm font-medium">
            <Circle className="size-4 fill-rose-500 text-rose-500" />
            Read
          </span>
          <span className="text-[11px] text-muted-foreground">
            Open a browser, click/type yourself. Saved per
            submodule; future agentic runs replay it.
          </span>
        </button>
        <button
          type="button"
          onClick={() => onChange("agentic")}
          aria-pressed={value === "agentic"}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            value === "agentic"
              ? "border-primary/50 bg-primary/10 text-primary"
              : "hover:border-input hover:bg-muted/50",
          )}
        >
          <span className="flex items-center gap-1.5 text-sm font-medium">
            <Bot className="size-4" />
            Agentic
          </span>
          <span className="text-[11px] text-muted-foreground">
            Goal-oriented loop. Discovers paths. Captures frozen
            paths on success. ~10× tokens.
          </span>
        </button>
        <button
          type="button"
          onClick={() => onChange("replay")}
          aria-pressed={value === "replay"}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            value === "replay"
              ? "border-primary/50 bg-primary/10 text-primary"
              : "hover:border-input hover:bg-muted/50",
          )}
        >
          <span className="flex items-center gap-1.5 text-sm font-medium">
            <Play className="size-4" />
            Replay
          </span>
          <span className="text-[11px] text-muted-foreground">
            Deterministic walk of frozen paths. Self-heals on
            broken selectors. ~5% of agentic cost.
          </span>
        </button>
      </div>
      {value === "agentic" && (
        <p className="rounded border border-amber-500/30 bg-amber-500/5 px-2 py-1 text-[10px] text-amber-700 dark:text-amber-300">
          Agentic mode runs at the test-case (submodule) level. Each
          submodule with selected steps becomes one goal the agent will
          verify. Loop guards halt at 30 turns / 5 min / 80k tokens.
          On a clean pass, the working tool sequence is frozen onto
          the test case so future runs can use Replay mode.
        </p>
      )}
      {value === "replay" && (
        <p className="rounded border border-emerald-500/30 bg-emerald-500/5 px-2 py-1 text-[10px] text-emerald-700 dark:text-emerald-300">
          Replay walks the frozen path captured the last time agentic
          mode passed cleanly on each test case. Zero LLM cost on the
          happy path; vision LLM only fires for self-healing if a
          frozen step's selector breaks. Submodules without a frozen
          path fall through to agentic.
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
