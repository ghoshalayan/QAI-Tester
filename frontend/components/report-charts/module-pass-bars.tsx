"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ReportModuleRead } from "@/lib/api";

interface Props {
  modules: ReportModuleRead[];
}

/**
 * Horizontal bars: one per module, showing pass-rate %. Bar fill is
 * health-tinted so weak modules pop visually:
 *   - >=90%  green
 *   - >=70%  yellow
 *   - <70%   red
 *
 * Sorted ascending so the worst modules sit at the top — first thing
 * the user reads. Long module names truncate via the YAxis tick.
 */
export function ModulePassBars({ modules }: Props) {
  const data = useMemo(() => {
    return modules
      .filter((m) => m.total > 0)
      .map((m) => ({
        title: m.title,
        passPct: m.pass_pct,
        passed: m.passed,
        total: m.total,
        failed: m.failed,
        color: barColor(m.pass_pct),
      }))
      .sort((a, b) => a.passPct - b.passPct);
  }, [modules]);

  if (data.length === 0) {
    return (
      <div className="flex h-full min-h-[220px] items-center justify-center text-sm text-muted-foreground">
        No modules to chart
      </div>
    );
  }

  // ~32px per row + a bit of padding. Caps at 320px for very long lists.
  const height = Math.min(Math.max(data.length * 32 + 24, 96), 320);

  return (
    <div className="min-w-0" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 32, bottom: 4, left: 4 }}
        >
          <XAxis
            type="number"
            domain={[0, 100]}
            ticks={[0, 25, 50, 75, 100]}
            tickFormatter={(v) => `${v}%`}
            stroke="currentColor"
            tick={{ fill: "currentColor", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            className="text-muted-foreground"
          />
          <YAxis
            type="category"
            dataKey="title"
            stroke="currentColor"
            tick={{ fill: "currentColor", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={120}
            className="text-foreground"
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            contentStyle={{
              background: "rgba(15,18,25,0.95)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
            formatter={(_value, _name, item) => {
              const p = item.payload as (typeof data)[number];
              return [
                `${p.passed}/${p.total} passed (${p.passPct.toFixed(1)}%) · ${p.failed} failed`,
                p.title,
              ];
            }}
          />
          <Bar
            dataKey="passPct"
            radius={[0, 4, 4, 0]}
            barSize={18}
            isAnimationActive
          >
            {data.map((d, i) => (
              <Cell key={i} fill={d.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function barColor(pct: number): string {
  if (pct >= 90) return "#22c55e";  // green-500
  if (pct >= 70) return "#eab308";  // yellow-500
  return "#ef4444";                  // red-500
}
