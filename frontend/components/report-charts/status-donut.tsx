"use client";

import { useMemo } from "react";
import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import type { ReportRunSummary } from "@/lib/api";

interface Props {
  run: ReportRunSummary;
}

interface Slice {
  key: "passed" | "failed" | "blocked" | "skipped";
  label: string;
  value: number;
  color: string;
}

const SLICE_COLORS: Record<Slice["key"], string> = {
  passed: "#22c55e",
  failed: "#ef4444",
  blocked: "#eab308",
  skipped: "#94a3b8",
};

const SLICE_LABELS: Record<Slice["key"], string> = {
  passed: "Passed",
  failed: "Failed",
  blocked: "Blocked",
  skipped: "Skipped",
};

/**
 * Pass/fail/blocked/skipped distribution as a donut. Centerpiece is the
 * pass percentage in big bold text — most users will look at the center
 * first, then the legend for the breakdown.
 */
export function StatusDonut({ run }: Props) {
  const slices = useMemo<Slice[]>(() => {
    return (Object.keys(SLICE_COLORS) as Slice["key"][])
      .map((key) => ({
        key,
        label: SLICE_LABELS[key],
        value: run[key],
        color: SLICE_COLORS[key],
      }))
      .filter((s) => s.value > 0);
  }, [run]);

  const passColor =
    run.pass_pct >= 90
      ? "text-green-600 dark:text-green-400"
      : run.pass_pct >= 70
        ? "text-yellow-600 dark:text-yellow-400"
        : "text-red-600 dark:text-red-400";

  if (run.total_steps === 0) {
    return (
      <div className="flex h-full min-h-[220px] items-center justify-center text-sm text-muted-foreground">
        No steps to chart
      </div>
    );
  }

  return (
    <div className="relative">
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={slices}
              dataKey="value"
              nameKey="label"
              cx="50%"
              cy="50%"
              innerRadius="62%"
              outerRadius="92%"
              paddingAngle={2}
              stroke="none"
              isAnimationActive
            >
              {slices.map((slice) => (
                <Cell key={slice.key} fill={slice.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "rgba(15,18,25,0.95)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 8,
                color: "#fff",
                fontSize: 12,
              }}
              formatter={(value, name) => {
                const n = typeof value === "number" ? value : Number(value) || 0;
                return [`${n} step${n === 1 ? "" : "s"}`, String(name)];
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Centerpiece — pass% reads at-a-glance */}
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-3xl font-bold ${passColor}`}>
          {run.pass_pct.toFixed(0)}%
        </span>
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Pass rate
        </span>
        <span className="mt-0.5 text-[10px] text-muted-foreground">
          {run.passed} / {run.total_steps}
        </span>
      </div>

      {/* Legend */}
      <ul className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        {slices.map((slice) => (
          <li
            key={slice.key}
            className="flex items-center justify-between gap-2"
          >
            <span className="flex items-center gap-2">
              <span
                aria-hidden
                className="size-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: slice.color }}
              />
              <span className="text-muted-foreground">{slice.label}</span>
            </span>
            <span className="font-mono font-medium">{slice.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
