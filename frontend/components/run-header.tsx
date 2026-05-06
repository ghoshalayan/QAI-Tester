"use client";

import { Calendar, ExternalLink, Eye, EyeOff, Layers, Timer } from "lucide-react";

import type { AgentRunRead } from "@/lib/api";

/** Structural — accepts both ``PlanReadCompact`` and ``PlanReadDetail``. */
interface PlanLike {
  name: string;
  target_url: string;
}

interface Props {
  run: AgentRunRead;
  plan: PlanLike | null;
}

export function RunHeader({ run, plan }: Props) {
  const headless = !!run.input_json?.headless;
  const wallClockMs = computeWallClockMs(run);
  const stepDurationMs =
    typeof run.output_summary_json?.duration_ms === "number"
      ? (run.output_summary_json.duration_ms as number)
      : null;

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-start gap-x-6 gap-y-3">
        {/* Plan link */}
        {plan && (
          <Field icon={Layers} label="Plan">
            <span className="font-medium">{plan.name}</span>
            {plan.target_url && (
              <a
                href={plan.target_url}
                target="_blank"
                rel="noopener noreferrer"
                className="ml-1 inline-flex items-center gap-1 break-all text-xs text-primary hover:underline"
              >
                {prettyUrl(plan.target_url)}
                <ExternalLink className="size-3" />
              </a>
            )}
          </Field>
        )}

        {/* Browser mode */}
        <Field icon={headless ? EyeOff : Eye} label="Browser">
          <span className="font-medium">
            {headless ? "Headless" : "Headed"} Chromium
          </span>
        </Field>

        {/* Started */}
        {run.started_at && (
          <Field icon={Calendar} label="Started">
            <time
              dateTime={run.started_at}
              title={new Date(run.started_at).toLocaleString()}
            >
              {new Date(run.started_at).toLocaleTimeString()}
            </time>
          </Field>
        )}

        {/* Duration */}
        {(wallClockMs !== null || stepDurationMs !== null) && (
          <Field icon={Timer} label="Duration">
            {wallClockMs !== null && (
              <span className="font-medium">{formatMs(wallClockMs)}</span>
            )}
            {stepDurationMs !== null && wallClockMs !== null && (
              <span className="ml-1.5 text-xs text-muted-foreground">
                ({formatMs(stepDurationMs)} stepwork)
              </span>
            )}
          </Field>
        )}
      </div>
    </div>
  );
}

function Field({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof Calendar;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-w-0 space-y-0.5">
      <p className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        <Icon className="size-3" /> {label}
      </p>
      <p className="break-words text-sm">{children}</p>
    </div>
  );
}

function computeWallClockMs(run: AgentRunRead): number | null {
  if (!run.started_at) return null;
  const start = new Date(run.started_at).getTime();
  if (!Number.isFinite(start)) return null;
  if (run.completed_at) {
    const end = new Date(run.completed_at).getTime();
    if (!Number.isFinite(end)) return null;
    return Math.max(0, end - start);
  }
  // Still running — wall clock from start to now
  return Math.max(0, Date.now() - start);
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function prettyUrl(url: string): string {
  try {
    const u = new URL(url);
    return u.host + (u.pathname === "/" ? "" : u.pathname);
  } catch {
    return url;
  }
}
