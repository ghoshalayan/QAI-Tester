"use client";

import {
  Activity,
  CheckCircle2,
  Sparkles,
  Timer,
  XCircle,
} from "lucide-react";

import type { ReportRunSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  run: ReportRunSummary;
}

/**
 * Big-number hero strip — first thing the eye lands on after the header.
 * Five cards with gradient backgrounds tinted by health.
 *
 * Pass-rate card is health-tinted (green/yellow/red gradient); the others
 * use a neutral card surface so the run's headline number dominates.
 */
export function HeroStats({ run }: Props) {
  const passColor = passColorClass(run.pass_pct);
  const passGradient = passGradientClass(run.pass_pct);

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {/* Pass rate — the headline */}
      <div
        className={cn(
          "relative overflow-hidden rounded-xl border p-4",
          passGradient,
        )}
      >
        <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
          <Activity className="size-3" />
          Pass rate
        </div>
        <div className={cn("mt-2 text-4xl font-bold tabular-nums", passColor)}>
          {run.pass_pct.toFixed(0)}%
        </div>
        <div className="mt-1 text-[11px] text-muted-foreground">
          {run.passed} of {run.total_steps} steps
        </div>
      </div>

      {/* Total steps */}
      <Stat
        icon={Activity}
        label="Total steps"
        value={run.total_steps.toString()}
        sub={subStepBreakdown(run)}
      />

      {/* Failed (always visible — even 0 is a positive signal) */}
      <Stat
        icon={run.failed > 0 ? XCircle : CheckCircle2}
        label="Failed"
        value={run.failed.toString()}
        valueClass={
          run.failed === 0
            ? "text-green-600 dark:text-green-400"
            : "text-red-600 dark:text-red-400"
        }
        sub={
          run.failed === 0
            ? "Clean run"
            : `${((run.failed / Math.max(run.total_steps, 1)) * 100).toFixed(0)}% failure`
        }
      />

      {/* Duration */}
      {run.duration_ms !== null && (
        <Stat
          icon={Timer}
          label="Duration"
          value={formatDuration(run.duration_ms)}
          sub={
            run.completed_at
              ? new Date(run.completed_at).toLocaleTimeString()
              : "in progress"
          }
        />
      )}

      {/* AI cost — only when at least one call fired */}
      {run.ai_calls > 0 && (
        <Stat
          icon={Sparkles}
          label="AI assist"
          value={`${run.ai_calls} call${run.ai_calls === 1 ? "" : "s"}`}
          valueClass="text-blue-600 dark:text-blue-400"
          sub={aiCostSub(run)}
        />
      )}
    </div>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
  valueClass,
  sub,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  valueClass?: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
        <Icon className="size-3" />
        {label}
      </div>
      <div
        className={cn(
          "mt-2 text-3xl font-bold tabular-nums",
          valueClass,
        )}
      >
        {value}
      </div>
      {sub && (
        <div className="mt-1 text-[11px] text-muted-foreground">{sub}</div>
      )}
    </div>
  );
}

function passColorClass(pct: number): string {
  if (pct >= 90) return "text-green-600 dark:text-green-400";
  if (pct >= 70) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

function passGradientClass(pct: number): string {
  if (pct >= 90)
    return "bg-gradient-to-br from-green-500/15 via-green-500/5 to-transparent border-green-500/30";
  if (pct >= 70)
    return "bg-gradient-to-br from-yellow-500/15 via-yellow-500/5 to-transparent border-yellow-500/30";
  return "bg-gradient-to-br from-red-500/15 via-red-500/5 to-transparent border-red-500/30";
}

function subStepBreakdown(run: ReportRunSummary): string {
  const parts: string[] = [];
  if (run.passed > 0) parts.push(`${run.passed} ok`);
  if (run.failed > 0) parts.push(`${run.failed} fail`);
  if (run.blocked > 0) parts.push(`${run.blocked} blocked`);
  if (run.skipped > 0) parts.push(`${run.skipped} skip`);
  return parts.join(" · ") || "—";
}

function aiCostSub(run: ReportRunSummary): string {
  const parts: string[] = [];
  if (run.ai_vision_calls > 0) {
    parts.push(`${run.ai_vision_calls} vision`);
  }
  if (run.llm_input_tokens !== null) {
    const total = (run.llm_input_tokens ?? 0) + (run.llm_output_tokens ?? 0);
    parts.push(`${total.toLocaleString()} tokens`);
  }
  return parts.join(" · ");
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}
