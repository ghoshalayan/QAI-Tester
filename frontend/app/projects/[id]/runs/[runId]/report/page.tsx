"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  Bot,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileSpreadsheet,
  Sparkles,
  Target,
} from "lucide-react";

import {
  api,
  EXECUTION_STEP_STATUS_LABELS,
  type ExecutionStepStatus,
  type ReportModuleRead,
  type ReportPlanSummary,
  type ReportStepRead,
  type ReportSubmoduleRead,
} from "@/lib/api";
import { buildRecommendations } from "@/lib/recommendations";
import { Button } from "@/components/ui/button";
import { HeroStats } from "@/components/report-charts/hero-stats";
import { ModulePassBars } from "@/components/report-charts/module-pass-bars";
import { RecommendationsPanel } from "@/components/report-charts/recommendations-panel";
import { Skeleton } from "@/components/ui/skeleton";
import { StatusDonut } from "@/components/report-charts/status-donut";
import { cn } from "@/lib/utils";

const STEP_STATUS_DOT: Record<ExecutionStepStatus, string> = {
  pending: "bg-muted",
  running: "bg-blue-500",
  passed: "bg-green-500",
  failed: "bg-red-500",
  blocked: "bg-yellow-500",
  skipped: "bg-muted-foreground/40",
  inconclusive: "bg-orange-500",
};

const STEP_STATUS_TEXT: Record<ExecutionStepStatus, string> = {
  pending: "text-muted-foreground",
  running: "text-blue-600 dark:text-blue-400",
  passed: "text-green-700 dark:text-green-400",
  failed: "text-red-700 dark:text-red-400",
  blocked: "text-yellow-700 dark:text-yellow-400",
  skipped: "text-muted-foreground",
  inconclusive: "text-orange-700 dark:text-orange-400",
};

export default function RunReportPage() {
  const params = useParams<{ id: string; runId: string }>();
  const projectId = Number(params.id);
  const runId = Number(params.runId);

  const { data: report, isLoading, isError, error } = useQuery({
    queryKey: ["run-report", projectId, runId],
    queryFn: () => api.getRunReport(projectId, runId),
  });

  const recommendations = useMemo(
    () => (report ? buildRecommendations(report) : []),
    [report],
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Button asChild variant="ghost" size="sm">
          <Link href={`/projects/${projectId}/runs/${runId}`}>
            <ArrowLeft className="size-4" /> Run #{runId}
          </Link>
        </Button>
        <h2 className="text-lg font-semibold">Run report</h2>
        <Button asChild size="sm" variant="outline" className="ml-auto">
          <a href={api.runReportXlsxUrl(projectId, runId)} download>
            <FileSpreadsheet className="size-4" /> Download Excel
          </a>
        </Button>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-32 w-full" />
          <div className="grid gap-3 md:grid-cols-2">
            <Skeleton className="h-72" />
            <Skeleton className="h-72" />
          </div>
          <Skeleton className="h-48 w-full" />
        </div>
      ) : isError ? (
        <div className="rounded-md border border-red-500/30 bg-red-500/5 p-4 text-sm text-red-700 dark:text-red-400">
          Failed to load report: {(error as Error)?.message}
        </div>
      ) : !report ? null : (
        <>
          <HeroStats run={report.run} />

          <div className="grid gap-4 md:grid-cols-5">
            <ChartCard title="Status distribution" className="md:col-span-2">
              <StatusDonut run={report.run} />
            </ChartCard>
            <ChartCard
              title="Pass rate by module"
              className="md:col-span-3"
            >
              <ModulePassBars modules={report.modules} />
            </ChartCard>
          </div>

          <RunCostCard runId={runId} />

          <RecommendationsPanel recommendations={recommendations} />

          {report.plan && <PlanCard plan={report.plan} />}

          {report.modules.length === 0 ? (
            <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
              No steps in this run yet.
            </div>
          ) : (
            <section className="space-y-3">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                  Per-module breakdown
                </h3>
                <span className="text-xs text-muted-foreground">
                  · {report.modules.length} module
                  {report.modules.length === 1 ? "" : "s"}
                </span>
              </div>
              <div className="space-y-4">
                {report.modules.map((mod, i) => (
                  <ModuleSection key={`${mod.title}-${i}`} module={mod} />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

// ── Layout helpers ────────────────────────────────────────────────


function RunCostCard({ runId }: { runId: number }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["run-cost", runId],
    queryFn: () => api.getRunCost(runId),
    // Refresh after the run completes — the cost columns are written
    // in the agent loop's ``finally`` block.
    staleTime: 10_000,
  });

  const fmtUsd = (n: number | null | undefined) =>
    n == null ? "$—" : `$${n.toFixed(4)}`;
  const fmtTokens = (n: number) => n.toLocaleString();

  if (isLoading) {
    return (
      <div className="rounded-xl border bg-card p-4">
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }
  if (isError || !data) {
    return null;
  }

  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          Cost
          {data.estimated_from_aggregate && (
            <span className="ml-2 rounded bg-muted px-1 py-0.5 text-[9px] normal-case tracking-normal text-muted-foreground">
              estimate (pre-tier-tracking run)
            </span>
          )}
        </h3>
        <div className="text-right">
          <div className="text-xl font-semibold tabular-nums">
            {data.total_cost_usd == null
              ? "$—"
              : `$${data.total_cost_usd.toFixed(4)}`}
          </div>
          {data.total_cost_usd == null && (
            <div className="text-[10px] text-muted-foreground">
              <Link
                href="/settings"
                className="underline hover:no-underline"
              >
                Configure pricing
              </Link>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <CostTierBlock
          title="Strong tier"
          model={data.strong_model}
          inputLine={data.lines.find(
            (l) => l.tier === "strong" && l.direction === "input",
          )}
          cachedInputLine={data.lines.find(
            (l) => l.tier === "strong" && l.direction === "input_cached",
          )}
          outputLine={data.lines.find(
            (l) => l.tier === "strong" && l.direction === "output",
          )}
          fmtUsd={fmtUsd}
          fmtTokens={fmtTokens}
        />
        <CostTierBlock
          title="Cheap tier"
          model={data.cheap_model}
          inputLine={data.lines.find(
            (l) => l.tier === "cheap" && l.direction === "input",
          )}
          cachedInputLine={data.lines.find(
            (l) => l.tier === "cheap" && l.direction === "input_cached",
          )}
          outputLine={data.lines.find(
            (l) => l.tier === "cheap" && l.direction === "output",
          )}
          fmtUsd={fmtUsd}
          fmtTokens={fmtTokens}
        />
      </div>

      <p className="mt-3 text-[10px] text-muted-foreground">
        Embeddings (AKB + BRD ingest + requirement embeds) use a local
        CPU model and are free — not included in this cost. Cached
        input rows show prompt tokens the provider billed at the
        cached rate (OpenAI auto-caches prompts ≥ 1024 tokens at
        ~50%). Hidden when there&apos;s no cache hit.
      </p>
    </div>
  );
}


function CostTierBlock({
  title,
  model,
  inputLine,
  cachedInputLine,
  outputLine,
  fmtUsd,
  fmtTokens,
}: {
  title: string;
  model: string | null;
  inputLine: { tokens: number; price_per_m: number | null; cost_usd: number | null } | undefined;
  cachedInputLine: { tokens: number; price_per_m: number | null; cost_usd: number | null } | undefined;
  outputLine: { tokens: number; price_per_m: number | null; cost_usd: number | null } | undefined;
  fmtUsd: (n: number | null | undefined) => string;
  fmtTokens: (n: number) => string;
}) {
  const cachedTok = cachedInputLine?.tokens ?? 0;
  const showCached = cachedTok > 0;
  return (
    <div className="rounded-md border bg-muted/20 p-3 text-xs">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-medium">{title}</span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {model ?? "—"}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div className="text-[10px] text-muted-foreground">direction</div>
        <div className="text-right text-[10px] text-muted-foreground">
          tokens
        </div>
        <div className="text-right text-[10px] text-muted-foreground">
          cost
        </div>

        <div>input</div>
        <div className="text-right font-mono">
          {fmtTokens(inputLine?.tokens ?? 0)}
        </div>
        <div className="text-right font-mono">
          {fmtUsd(inputLine?.cost_usd ?? null)}
        </div>

        {showCached && (
          <>
            <div className="text-muted-foreground">
              input (cached)
            </div>
            <div className="text-right font-mono text-muted-foreground">
              {fmtTokens(cachedTok)}
            </div>
            <div className="text-right font-mono text-muted-foreground">
              {fmtUsd(cachedInputLine?.cost_usd ?? null)}
            </div>
          </>
        )}

        <div>output</div>
        <div className="text-right font-mono">
          {fmtTokens(outputLine?.tokens ?? 0)}
        </div>
        <div className="text-right font-mono">
          {fmtUsd(outputLine?.cost_usd ?? null)}
        </div>
      </div>
    </div>
  );
}


function ChartCard({
  title,
  children,
  className,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("rounded-xl border bg-card p-4", className)}>
      <h3 className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {title}
      </h3>
      <div className="mt-2">{children}</div>
    </div>
  );
}

function PlanCard({ plan }: { plan: ReportPlanSummary }) {
  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Plan
        </span>
        <span className="font-medium">{plan.name}</span>
        {plan.target_url && (
          <a
            href={plan.target_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 break-all text-xs text-primary hover:underline"
          >
            {plan.target_url}
            <ExternalLink className="size-3" />
          </a>
        )}
      </div>
      {plan.scope.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {plan.scope.map((s) => (
            <span
              key={s}
              className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
            >
              {s}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Per-module section ────────────────────────────────────────────


function ModuleSection({ module: mod }: { module: ReportModuleRead }) {
  // Health-tinted left border so weak modules pop visually in the list
  const borderColor =
    mod.pass_pct >= 90
      ? "border-l-green-500/70"
      : mod.pass_pct >= 70
        ? "border-l-yellow-500/70"
        : "border-l-red-500/70";

  const headerGradient =
    mod.pass_pct >= 90
      ? "bg-gradient-to-r from-green-500/10 to-transparent"
      : mod.pass_pct >= 70
        ? "bg-gradient-to-r from-yellow-500/10 to-transparent"
        : "bg-gradient-to-r from-red-500/10 to-transparent";

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-l-4 bg-card",
        borderColor,
      )}
    >
      <div
        className={cn(
          "flex flex-wrap items-baseline gap-3 border-b px-4 py-3",
          headerGradient,
        )}
      >
        <span className="text-base font-semibold">{mod.title}</span>
        <span className="text-xs text-muted-foreground">
          {mod.passed}/{mod.total} passed
          {mod.failed > 0 && ` · ${mod.failed} failed`}
          {mod.inconclusive > 0 && ` · ${mod.inconclusive} inconclusive`}
          {mod.blocked > 0 && ` · ${mod.blocked} blocked`}
          {mod.skipped > 0 && ` · ${mod.skipped} skipped`}
        </span>
        <PctBadge pct={mod.pass_pct} kind="pass" />
        {mod.fail_pct > 0 && <PctBadge pct={mod.fail_pct} kind="fail" />}
        {/* Partial badge — surfaces when results are mixed within the
            module, so the user can spot "this feature is fragile, half
            its tests fail" at a glance. */}
        {mod.pass_pct > 0 && mod.fail_pct > 0 && (
          <span className="inline-flex items-center rounded-md border border-orange-500/30 bg-orange-500/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-orange-700 dark:text-orange-400">
            Partial
          </span>
        )}
      </div>
      <table className="w-full text-sm">
        <thead className="bg-muted/30 text-[10px] uppercase tracking-wide text-muted-foreground">
          <tr>
            <th className="px-4 py-2 text-left font-medium">Submodule</th>
            <th className="px-2 py-2 text-center font-medium">Steps</th>
            <th className="px-2 py-2 text-center font-medium">Pass</th>
            <th className="px-2 py-2 text-center font-medium">Fail</th>
            <th className="px-4 py-2 text-left font-medium">Issues</th>
          </tr>
        </thead>
        <tbody className="divide-y">
          {mod.submodules.map((sub, i) => (
            <SubmoduleRow key={`${sub.title}-${i}`} submodule={sub} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SubmoduleRow({ submodule: sub }: { submodule: ReportSubmoduleRead }) {
  // Auto-expand submodules that ran in agentic mode — the per-turn
  // log is the whole point of the agent, so we shouldn't make users
  // click to discover it.
  const hasAgenticRow = sub.steps.some((s) => s.mode === "agentic");
  const [expanded, setExpanded] = useState(
    sub.failed > 0 ||
      sub.blocked > 0 ||
      sub.inconclusive > 0 ||
      hasAgenticRow,
  );
  const Caret = expanded ? ChevronDown : ChevronRight;

  return (
    <>
      <tr
        className="cursor-pointer hover:bg-muted/30"
        onClick={() => setExpanded(!expanded)}
      >
        <td className="px-4 py-2">
          <div className="flex items-center gap-2">
            <Caret className="size-3.5 text-muted-foreground" />
            <span className="font-medium">{sub.title}</span>
          </div>
        </td>
        <td className="px-2 py-2 text-center text-xs text-muted-foreground">
          {sub.total}
        </td>
        <td className="px-2 py-2 text-center">
          <div className="inline-flex flex-col items-center gap-0.5">
            <PctBadge pct={sub.pass_pct} kind="pass" />
            <span className="text-[9px] text-muted-foreground">
              {sub.passed}/{sub.total}
            </span>
          </div>
        </td>
        <td className="px-2 py-2 text-center">
          {sub.fail_pct > 0 ? (
            <div className="inline-flex flex-col items-center gap-0.5">
              <PctBadge pct={sub.fail_pct} kind="fail" />
              <span className="text-[9px] text-muted-foreground">
                {sub.failed}/{sub.total}
              </span>
            </div>
          ) : sub.inconclusive > 0 ? (
            <span className="inline-flex items-center rounded-md border border-orange-500/30 bg-orange-500/10 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-orange-700 dark:text-orange-400">
              {sub.inconclusive}× unclear
            </span>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
          {/* Partial test case — some sub-steps passed, some failed.
              This is the most useful visual signal for the user:
              "this test case is half-working" deserves attention
              that "all-pass" or "all-fail" don't. */}
          {sub.pass_pct > 0 && sub.fail_pct > 0 && (
            <div className="mt-0.5">
              <span className="inline-flex items-center rounded-md border border-orange-500/30 bg-orange-500/10 px-1 py-0.5 font-mono text-[9px] font-semibold text-orange-700 dark:text-orange-400">
                Partial
              </span>
            </div>
          )}
        </td>
        <td className="px-4 py-2 text-xs">
          {sub.issues.length === 0 ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            <span className="line-clamp-2 break-words text-red-700/80 dark:text-red-400/80">
              {sub.issues[0]}
            </span>
          )}
        </td>
      </tr>
      {expanded && sub.steps.length > 0 && (
        <tr>
          <td colSpan={5} className="bg-muted/15 px-0 py-0">
            <ul className="divide-y border-y">
              {sub.steps.map((step) => (
                <StepRow key={step.id} step={step} />
              ))}
            </ul>
          </td>
        </tr>
      )}
    </>
  );
}

function StepRow({ step }: { step: ReportStepRead }) {
  const isAgentic = step.mode === "agentic";
  return (
    <li className="px-6 py-2 text-sm">
      <div className="flex items-start gap-3">
        <span className="mt-1.5 flex shrink-0">
          <span
            className={cn("size-2 rounded-full", STEP_STATUS_DOT[step.status])}
          />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="font-mono text-[10px] text-muted-foreground">
              {step.ordinal + 1}.
            </span>
            <span className="font-medium">{step.title}</span>
            {step.action_type && (
              <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                {step.action_type}
              </span>
            )}
            {isAgentic && (
              <span className="inline-flex items-center gap-0.5 rounded border border-purple-500/30 bg-purple-500/10 px-1 py-0.5 font-mono text-[10px] text-purple-700 dark:text-purple-400">
                agentic · {step.agent_log.length} turn
                {step.agent_log.length === 1 ? "" : "s"}
              </span>
            )}
            <span
              className={cn(
                "text-[10px] font-medium uppercase tracking-wide",
                STEP_STATUS_TEXT[step.status],
              )}
            >
              {EXECUTION_STEP_STATUS_LABELS[step.status]}
            </span>
            {step.halt_reason && (
              <span className="rounded border px-1 py-0.5 font-mono text-[9px] text-muted-foreground">
                halt: {step.halt_reason}
              </span>
            )}
            {step.duration_ms !== null && step.duration_ms > 0 && (
              <span className="text-[10px] text-muted-foreground">
                {formatDuration(step.duration_ms)}
              </span>
            )}
            {step.ai_helped && (
              <span
                className="inline-flex items-center gap-0.5 rounded border border-green-500/30 bg-green-500/10 px-1 py-0.5 text-[9px] font-medium text-green-700 dark:text-green-400"
                title="AI assist fixed this step"
              >
                <Sparkles className="size-2.5" /> AI
                {step.ai_used_vision && " · vision"}
              </span>
            )}
            {/* A4 visibility — show concrete interventions that ran */}
            {step.fuzzy_rescues > 0 && (
              <span
                className="inline-flex items-center gap-0.5 rounded border border-blue-500/30 bg-blue-500/10 px-1 py-0.5 text-[9px] font-medium text-blue-700 dark:text-blue-400"
                title={`Fuzzy selector matched ${step.fuzzy_rescues} time(s) — test-case wording was slightly off`}
              >
                fuzzy ×{step.fuzzy_rescues}
              </span>
            )}
            {step.vision_rescues > 0 && (
              <span
                className="inline-flex items-center gap-0.5 rounded border border-purple-500/30 bg-purple-500/10 px-1 py-0.5 text-[9px] font-medium text-purple-700 dark:text-purple-400"
                title={`Vision-guided target search recovered ${step.vision_rescues} time(s)`}
              >
                vision ×{step.vision_rescues}
              </span>
            )}
            {step.goal_verification?.verdict === "fail" && (
              <span
                className="inline-flex items-center gap-0.5 rounded border border-red-500/30 bg-red-500/10 px-1 py-0.5 text-[9px] font-medium text-red-700 dark:text-red-400"
                title="Vision check disagreed with the agent's claim"
              >
                vision ✗
              </span>
            )}
            {step.goal_verification?.verdict === "partial" && (
              <span
                className="inline-flex items-center gap-0.5 rounded border border-orange-500/30 bg-orange-500/10 px-1 py-0.5 text-[9px] font-medium text-orange-700 dark:text-orange-400"
                title="Vision check confirmed only some criteria"
              >
                vision ~
              </span>
            )}
            {step.goal_verification?.verdict === "pass" && (
              <span
                className="inline-flex items-center gap-0.5 rounded border border-emerald-500/30 bg-emerald-500/10 px-1 py-0.5 text-[9px] font-medium text-emerald-700 dark:text-emerald-400"
                title="Vision check confirmed all goal criteria"
              >
                vision ✓
              </span>
            )}
          </div>
          {step.narration && (
            <p className="mt-0.5 break-words text-[11px] text-muted-foreground">
              {step.narration}
            </p>
          )}
          {step.error_message && (
            <p className="mt-0.5 break-words text-[11px] text-red-700/80 dark:text-red-400/80">
              {step.error_message}
            </p>
          )}
        </div>
        {step.screenshot_path && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={api.screenshotUrl(step.screenshot_path)}
            alt={`step ${step.ordinal + 1} screenshot`}
            className="h-10 w-16 shrink-0 rounded border object-cover"
            loading="lazy"
          />
        )}
      </div>

      {isAgentic && (
        <ReportAgentDetail step={step} />
      )}
    </li>
  );
}

function ReportAgentDetail({ step }: { step: ReportStepRead }) {
  const passed = step.status === "passed";
  return (
    <details
      className={cn(
        "ml-6 mt-2 rounded-md border px-2 py-1.5 text-[11px]",
        passed
          ? "border-purple-500/30 bg-purple-500/5"
          : "border-orange-500/40 bg-orange-500/5",
      )}
      open={!passed}
    >
      <summary className="flex cursor-pointer items-center gap-1.5 list-none">
        <Bot
          className={cn(
            "size-3",
            passed
              ? "text-purple-700 dark:text-purple-400"
              : "text-orange-700 dark:text-orange-400",
          )}
        />
        <span className="font-medium">Agent reasoning</span>
        <span className="text-muted-foreground">
          ({step.agent_log.length} turn
          {step.agent_log.length === 1 ? "" : "s"})
        </span>
      </summary>

      <div className="mt-2 space-y-2">
        {step.goal_description && (
          <div className="flex items-start gap-1.5 rounded border bg-card p-1.5">
            <Target className="mt-0.5 size-3 shrink-0 text-purple-600" />
            <div className="min-w-0 flex-1">
              <p className="break-words font-medium">
                {step.goal_description}
              </p>
              {step.success_criteria.length > 0 && (
                <ul className="mt-1 ml-3 list-disc space-y-0.5 text-[10px] text-muted-foreground">
                  {step.success_criteria.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        )}

        {step.divergence_summary && (
          <div
            className={cn(
              "rounded border p-1.5 text-[11px]",
              step.divergence_category === "passed_clean"
                ? "border-emerald-500/30 bg-emerald-500/5"
                : step.divergence_category === "passed_with_help"
                  ? "border-blue-500/30 bg-blue-500/5"
                  : step.divergence_category === "test_case_outdated"
                    ? "border-amber-500/40 bg-amber-500/5"
                    : step.divergence_category === "feature_missing"
                      ? "border-red-500/30 bg-red-500/5"
                      : step.divergence_category === "infra_issue"
                        ? "border-yellow-500/30 bg-yellow-500/5"
                        : "border-muted-foreground/30 bg-muted/30",
            )}
          >
            <div className="mb-0.5 flex items-baseline gap-1.5">
              <span className="font-mono text-[9px] uppercase tracking-wide text-muted-foreground">
                {step.divergence_category?.replace(/_/g, " ")}
              </span>
            </div>
            <p className="break-words text-muted-foreground">
              {step.divergence_summary}
            </p>
          </div>
        )}

        {step.goal_verification && (
          <div
            className={cn(
              "rounded border p-1.5 text-[11px]",
              step.goal_verification.verdict === "pass"
                ? "border-emerald-500/30 bg-emerald-500/5"
                : step.goal_verification.verdict === "partial"
                  ? "border-orange-500/30 bg-orange-500/5"
                  : "border-red-500/30 bg-red-500/5",
            )}
          >
            <div className="mb-0.5 flex items-baseline gap-1.5">
              <span className="font-mono text-[9px] uppercase tracking-wide text-muted-foreground">
                vision verification · {step.goal_verification.verdict}
              </span>
              {typeof step.goal_verification.confidence === "number" && (
                <span className="text-[9px] text-muted-foreground">
                  ({Math.round(step.goal_verification.confidence * 100)}%)
                </span>
              )}
            </div>
            {step.goal_verification.reasoning && (
              <p className="break-words text-muted-foreground">
                {step.goal_verification.reasoning}
              </p>
            )}
            {step.goal_verification.criteria_missed.length > 0 && (
              <ul className="mt-1 ml-3 list-disc space-y-0.5 text-[10px] text-red-700/80 dark:text-red-400/80">
                {step.goal_verification.criteria_missed.map((c, i) => (
                  <li key={i}>missed: {c}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {step.sub_goals.length > 0 && (
          <SubGoalChecklist subGoals={step.sub_goals} />
        )}

        {step.agent_log.length === 0 ? (
          <p className="italic text-muted-foreground">
            No turns ran (agent halted before the first action).
          </p>
        ) : (
          <ol className="space-y-1">
            {step.agent_log.map((t) => (
              <ReportAgentTurnRow key={t.turn} turn={t} />
            ))}
          </ol>
        )}
      </div>
    </details>
  );
}

function ReportAgentTurnRow({
  turn,
}: {
  turn: ReportStepRead["agent_log"][number];
}) {
  const argSummary = Object.entries(turn.args)
    .filter(([, v]) => v !== "" && v !== 0 && v !== null && v !== undefined)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
    .join(" ")
    .slice(0, 200);

  const colorClass =
    turn.status === "ok"
      ? "text-emerald-600"
      : turn.status === "blocked"
        ? "text-amber-600"
        : turn.status === "stop"
          ? "text-purple-600"
          : "text-red-600";

  return (
    <li className="rounded border bg-card px-2 py-1.5">
      <div className="flex flex-wrap items-baseline gap-1.5">
        <span className="font-mono text-[10px] text-muted-foreground">
          T{turn.turn}
        </span>
        <span
          className={cn(
            "text-[10px] font-medium uppercase tracking-wide",
            colorClass,
          )}
        >
          {turn.status}
        </span>
        <span className="rounded border px-1 py-0.5 font-mono text-[10px]">
          {turn.tool}
        </span>
        {argSummary && (
          <span className="break-all font-mono text-[10px] text-muted-foreground">
            {argSummary}
          </span>
        )}
        {turn.confidence > 0 && (
          <span className="ml-auto text-[9px] text-muted-foreground">
            {Math.round(turn.confidence * 100)}%
          </span>
        )}
      </div>
      {turn.reasoning && (
        <p className="mt-0.5 break-words">{turn.reasoning}</p>
      )}
      {turn.narration && turn.narration !== turn.reasoning && (
        <p className="mt-0.5 break-words text-[10px] text-muted-foreground">
          → {turn.narration}
        </p>
      )}
      {turn.error_message && (
        <p className="mt-0.5 break-words rounded border border-red-500/30 bg-red-500/5 px-1.5 py-0.5 font-mono text-[10px] text-red-700 dark:text-red-400">
          {turn.error_message}
        </p>
      )}
    </li>
  );
}

// ── Sub-goal checklist ────────────────────────────────────────────


const SUB_GOAL_STATUS_GLYPH: Record<string, string> = {
  pending: "☐",
  in_progress: "▶",
  done: "✓",
  failed: "✗",
  skipped: "⊘",
};

const SUB_GOAL_STATUS_TINT: Record<string, string> = {
  pending: "text-muted-foreground",
  in_progress: "text-blue-600 dark:text-blue-400",
  done: "text-emerald-600 dark:text-emerald-400",
  failed: "text-red-600 dark:text-red-400",
  skipped: "text-amber-600 dark:text-amber-400",
};

function SubGoalChecklist({
  subGoals,
}: {
  subGoals: ReportStepRead["sub_goals"];
}) {
  const done = subGoals.filter((sg) => sg.status === "done").length;
  const total = subGoals.length;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

  return (
    <div className="rounded border bg-card p-1.5">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Sub-goals
        </span>
        <span className="text-[10px] text-muted-foreground">
          {done}/{total} done · {pct}%
        </span>
      </div>
      <ol className="space-y-0.5">
        {subGoals.map((sg) => {
          const replanBadge =
            typeof sg.replan_iteration === "number" &&
            sg.replan_iteration > 0;
          // Phase B — source badge: frozen vs. agentic vs. recovered.
          const sourceLabel =
            sg.source === "frozen"
              ? `frozen${typeof sg.frozen_step_count === "number"
                  ? ` · ${sg.frozen_step_count} steps`
                  : ""}`
              : sg.source === "frozen_then_agentic"
                ? "frozen → recovered"
                : sg.source === "agentic"
                  ? "agentic"
                  : null;
          const sourceTint =
            sg.source === "frozen"
              ? "border-blue-500/40 bg-blue-500/10 text-blue-700 dark:text-blue-400"
              : sg.source === "frozen_then_agentic"
                ? "border-purple-500/40 bg-purple-500/10 text-purple-700 dark:text-purple-400"
                : "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400";
          return (
            <li
              key={sg.id}
              className="text-[11px]"
            >
              <div className="flex items-baseline gap-1.5">
                <span
                  className={cn(
                    "shrink-0 font-mono",
                    SUB_GOAL_STATUS_TINT[sg.status] ??
                      "text-muted-foreground",
                  )}
                >
                  {SUB_GOAL_STATUS_GLYPH[sg.status] ?? "?"}
                </span>
                <span className="font-mono text-[9px] text-muted-foreground">
                  [{sg.id}]
                </span>
                {sourceLabel && (
                  <span
                    className={cn(
                      "shrink-0 rounded border px-1 text-[9px] font-medium",
                      sourceTint,
                    )}
                  >
                    {sourceLabel}
                  </span>
                )}
                {replanBadge && (
                  <span className="shrink-0 rounded border border-amber-500/40 bg-amber-500/10 px-1 text-[9px] font-medium text-amber-700 dark:text-amber-400">
                    replan {sg.replan_iteration}
                  </span>
                )}
                <span
                  className={cn(
                    "min-w-0 flex-1 break-words",
                    sg.status === "done" &&
                      "text-muted-foreground line-through decoration-emerald-500/60",
                  )}
                >
                  {sg.description}
                </span>
                {sg.completed_at_turn !== null && (
                  <span className="shrink-0 text-[9px] text-muted-foreground">
                    T{sg.completed_at_turn}
                  </span>
                )}
              </div>
              {sg.reason && (sg.status === "failed" || sg.status === "skipped") && (
                <p className="ml-5 mt-0.5 text-[10px] italic text-muted-foreground">
                  {sg.status === "failed" ? "failed: " : "skipped: "}
                  {sg.reason}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}


// ── Pill ──────────────────────────────────────────────────────────


function PctBadge({
  pct,
  kind,
}: {
  pct: number;
  kind: "pass" | "fail";
}) {
  const tint =
    kind === "pass"
      ? pct >= 90
        ? "bg-green-500/15 text-green-700 dark:text-green-400 border-green-500/30"
        : pct >= 70
          ? "bg-yellow-500/15 text-yellow-700 dark:text-yellow-400 border-yellow-500/30"
          : "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30"
      : pct > 30
        ? "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30"
        : pct > 10
          ? "bg-yellow-500/15 text-yellow-700 dark:text-yellow-400 border-yellow-500/30"
          : "bg-muted text-muted-foreground border-border";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-1.5 py-0.5 font-mono text-[11px] font-semibold",
        tint,
      )}
    >
      {pct.toFixed(0)}% {kind === "pass" ? "pass" : "fail"}
    </span>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}
