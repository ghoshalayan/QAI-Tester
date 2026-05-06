"use client";

import {
  AlertTriangle,
  CheckCircle2,
  Info,
  Lightbulb,
  XOctagon,
} from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  type Recommendation,
  type RecommendationSeverity,
} from "@/lib/recommendations";
import { cn } from "@/lib/utils";

interface Props {
  recommendations: Recommendation[];
}

const SEVERITY_STYLES: Record<
  RecommendationSeverity,
  { container: string; icon: typeof Info; iconClass: string; label: string }
> = {
  urgent: {
    container:
      "border-red-500/40 bg-gradient-to-br from-red-500/10 via-red-500/5 to-transparent",
    icon: XOctagon,
    iconClass: "text-red-600 dark:text-red-400",
    label: "Urgent",
  },
  warn: {
    container:
      "border-yellow-500/40 bg-gradient-to-br from-yellow-500/10 via-yellow-500/5 to-transparent",
    icon: AlertTriangle,
    iconClass: "text-yellow-600 dark:text-yellow-400",
    label: "Review",
  },
  info: {
    container:
      "border-blue-500/40 bg-gradient-to-br from-blue-500/10 via-blue-500/5 to-transparent",
    icon: Info,
    iconClass: "text-blue-600 dark:text-blue-400",
    label: "Note",
  },
  good: {
    container:
      "border-green-500/40 bg-gradient-to-br from-green-500/10 via-green-500/5 to-transparent",
    icon: CheckCircle2,
    iconClass: "text-green-600 dark:text-green-400",
    label: "Healthy",
  },
};

/**
 * Severity-tinted cards listing the rule-engine output. Sorted by
 * severity (urgent → good) by the engine; we just render in that order.
 *
 * Renders nothing when the list is empty — keeps the report page clean
 * for runs that have nothing to flag.
 */
export function RecommendationsPanel({ recommendations }: Props) {
  if (recommendations.length === 0) return null;

  return (
    <section className="space-y-3">
      <div className="flex items-center gap-2">
        <Lightbulb className="size-4 text-yellow-600 dark:text-yellow-400" />
        <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          Recommendations · {recommendations.length}
        </h3>
      </div>
      <ul className="grid gap-3 md:grid-cols-2">
        {recommendations.map((rec) => {
          const styles = SEVERITY_STYLES[rec.severity];
          const Icon = styles.icon;
          return (
            <li
              key={rec.id}
              className={cn(
                "rounded-xl border p-4 transition-shadow hover:shadow-md",
                styles.container,
              )}
            >
              <div className="flex items-start gap-3">
                <Icon
                  className={cn("mt-0.5 size-4 shrink-0", styles.iconClass)}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span
                      className={cn(
                        "text-[10px] font-medium uppercase tracking-wider",
                        styles.iconClass,
                      )}
                    >
                      {styles.label}
                    </span>
                    <span className="break-words text-sm font-semibold">
                      {rec.title}
                    </span>
                  </div>
                  <p className="mt-1 break-words text-xs text-foreground/80">
                    {rec.body}
                  </p>
                  {rec.action && (
                    <Button asChild size="sm" variant="outline" className="mt-2">
                      <Link href={rec.action.href}>{rec.action.label}</Link>
                    </Button>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
