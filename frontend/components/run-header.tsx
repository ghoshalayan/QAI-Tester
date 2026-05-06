"use client";

import {
  Calendar,
  ExternalLink,
  Eye,
  EyeOff,
  Gauge,
  Image as ImageIcon,
  Layers,
  Sparkles,
  Timer,
  Turtle,
  Zap,
} from "lucide-react";

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

const SPEED_LABEL: Record<string, string> = {
  slow: "Slow",
  normal: "Normal",
  fast: "Fast",
};

const SPEED_ICON: Record<string, typeof Turtle> = {
  slow: Turtle,
  normal: Gauge,
  fast: Zap,
};

export function RunHeader({ run, plan }: Props) {
  const headless = !!run.input_json?.headless;
  const speedRaw =
    typeof run.input_json?.speed === "string"
      ? (run.input_json.speed as string)
      : "slow";
  const SpeedIcon = SPEED_ICON[speedRaw] ?? Gauge;
  const wallClockMs = computeWallClockMs(run);
  const stepDurationMs =
    typeof run.output_summary_json?.duration_ms === "number"
      ? (run.output_summary_json.duration_ms as number)
      : null;

  const llmIn =
    typeof run.output_summary_json?.llm_input_tokens === "number"
      ? (run.output_summary_json.llm_input_tokens as number)
      : null;
  const llmOut =
    typeof run.output_summary_json?.llm_output_tokens === "number"
      ? (run.output_summary_json.llm_output_tokens as number)
      : null;
  const aiCalls =
    typeof run.output_summary_json?.ai_calls === "number"
      ? (run.output_summary_json.ai_calls as number)
      : 0;
  const aiVisionCalls =
    typeof run.output_summary_json?.ai_vision_calls === "number"
      ? (run.output_summary_json.ai_vision_calls as number)
      : 0;
  const showAiCost = aiCalls > 0 && (llmIn !== null || llmOut !== null);

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

        {/* Speed preset */}
        <Field icon={SpeedIcon} label="Speed">
          <span className="font-medium">
            {SPEED_LABEL[speedRaw] ?? speedRaw}
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

        {/* AI cost — only renders when at least one AI assist call fired. */}
        {showAiCost && (
          <Field icon={Sparkles} label="AI cost">
            <span className="font-medium">
              {aiCalls} call{aiCalls === 1 ? "" : "s"}
            </span>
            {(llmIn !== null || llmOut !== null) && (
              <span className="ml-1.5 text-xs text-muted-foreground">
                · {llmIn?.toLocaleString() ?? 0} in /{" "}
                {llmOut?.toLocaleString() ?? 0} out tokens
              </span>
            )}
            {aiVisionCalls > 0 && (
              <span
                className="ml-1.5 inline-flex items-center gap-0.5 rounded border border-blue-500/30 bg-blue-500/5 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 dark:text-blue-400"
                title={`${aiVisionCalls} vision call${aiVisionCalls === 1 ? "" : "s"} (text + screenshot)`}
              >
                <ImageIcon className="size-2.5" />
                vision×{aiVisionCalls}
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
