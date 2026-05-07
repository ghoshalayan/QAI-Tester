/**
 * Recommendations engine — turns a ReportRead into a prioritized list of
 * actionable insights the user should act on.
 *
 * Pure function (no React, no fetch) so it can be unit-tested cheaply
 * and called from anywhere. Severity scale matches the visual tinting
 * the panel uses:
 *
 *   - urgent  : red    (run failed badly; data quality compromised)
 *   - warn    : yellow (something to look at; doesn't block)
 *   - info    : blue   (FYI / observation)
 *   - good    : green  (positive recognition — keeps the panel from being
 *                       all-doom on healthy runs)
 *
 * Rules are intentionally simple and local — derived from one run's data
 * only. Cross-run patterns ("this selector has been flaky for 3 runs")
 * belong on the cross-run dashboard (futurescope).
 */

import type { ReportRead, ReportStepRead } from "@/lib/api";

export type RecommendationSeverity = "urgent" | "warn" | "info" | "good";

export interface Recommendation {
  id: string;                  // stable key for React lists
  severity: RecommendationSeverity;
  title: string;
  body: string;                // 1-2 sentences, plain text
  /** Optional inline action — frontend renders a button/link if present. */
  action?: { label: string; href: string };
}

const SLOW_STEP_THRESHOLD_MS = 8_000;
const LOW_PASS_RATE = 70;
const VERY_LOW_PASS_RATE = 40;

/**
 * Apply every rule to a report, returning a flat list ordered by
 * severity (urgent → warn → info → good) and within the same tier by
 * insertion order.
 */
export function buildRecommendations(report: ReportRead): Recommendation[] {
  const out: Recommendation[] = [];
  const r = report.run;

  // ── Run-level signals ───────────────────────────────────────────

  if (r.total_steps === 0) {
    out.push({
      id: "no-steps",
      severity: "info",
      title: "No steps were executed",
      body:
        "This run produced zero step rows. Confirm at least one step was " +
        "ticked on the Test Cases tab before starting the next run.",
    });
    return out; // short-circuit — nothing else to recommend
  }

  if (r.failed > 0 && r.pass_pct < VERY_LOW_PASS_RATE) {
    out.push({
      id: "run-very-low-pass",
      severity: "urgent",
      title: `Pass rate ${r.pass_pct}% — run needs investigation`,
      body:
        `${r.failed} of ${r.total_steps} steps failed. ` +
        "Check the recommendations below for the worst submodules, then " +
        "use Re-run failed steps after fixing the root cause.",
    });
  } else if (r.failed > 0 && r.pass_pct < LOW_PASS_RATE) {
    out.push({
      id: "run-low-pass",
      severity: "warn",
      title: `Pass rate ${r.pass_pct}%`,
      body:
        `${r.failed} step${r.failed === 1 ? "" : "s"} failed. ` +
        "Drill into the failing modules below before promoting this plan.",
    });
  }

  if (r.blocked > 0) {
    const word = r.blocked === 1 ? "step" : "steps";
    out.push({
      id: "blocked-on-hitl",
      severity: "warn",
      title: `${r.blocked} ${word} blocked on HITL`,
      body:
        "Steps with credentials/OTP data needs are paused until the user " +
        "intervenes. Configure plan credentials or wait at the modal next time.",
    });
  }

  if (r.skipped > 0) {
    out.push({
      id: "skipped-after-cancel",
      severity: "info",
      title: `${r.skipped} skipped`,
      body:
        "Steps were skipped because the run was cancelled or a parent " +
        "halted. Re-run to validate them once the failing path is fixed.",
    });
  }

  // ── Module-level signals ────────────────────────────────────────

  // Sort modules by ascending pass rate; surface the worst ones first.
  const weakModules = [...report.modules]
    .filter((m) => m.total > 0)
    .sort((a, b) => a.pass_pct - b.pass_pct);

  for (const m of weakModules) {
    if (m.pass_pct >= LOW_PASS_RATE) break; // sorted, so we're done
    const severity: RecommendationSeverity =
      m.pass_pct < VERY_LOW_PASS_RATE ? "urgent" : "warn";
    out.push({
      id: `module-low-${m.title}`,
      severity,
      title: `${m.title}: ${m.pass_pct}% pass`,
      body:
        `${m.failed} of ${m.total} steps failed in ${m.title}. ` +
        topIssueExcerpt(m) +
        " Open the module below to see the failing steps and AI suggestions.",
    });
  }

  // ── AI assist signals ───────────────────────────────────────────

  if (r.ai_calls > 0) {
    const visionShare =
      r.ai_calls > 0 ? Math.round((r.ai_vision_calls / r.ai_calls) * 100) : 0;
    if (r.ai_vision_calls > 0) {
      out.push({
        id: "ai-vision-usage",
        severity: "info",
        title: `Vision used on ${r.ai_vision_calls} of ${r.ai_calls} AI calls (${visionShare}%)`,
        body:
          "Vision escalation kicked in when the text-only AI suggestion " +
          "wasn't enough. Persistent vision usage usually means " +
          "target_hint selectors don't disambiguate well — consider " +
          "tightening them in the Test Cases tab.",
      });
    }

    const aiHelpedSteps = countAiHelped(report);
    if (aiHelpedSteps > 0) {
      out.push({
        id: "ai-helped-count",
        severity: "info",
        title: `AI fixed ${aiHelpedSteps} step${aiHelpedSteps === 1 ? "" : "s"} on the fly`,
        body:
          "These would have failed without AI assist. The selectors the " +
          "agent proposed are recorded on each step's diff — copy them " +
          "back into your TC tree if you want them permanent.",
      });
    }
  }

  // ── Performance signals ─────────────────────────────────────────

  const slowSteps = collectSlowSteps(report);
  if (slowSteps.length > 0) {
    const top = slowSteps[0];
    out.push({
      id: "slow-steps",
      severity: "info",
      title:
        `${slowSteps.length} slow step${slowSteps.length === 1 ? "" : "s"} ` +
        `(>${(SLOW_STEP_THRESHOLD_MS / 1000).toFixed(0)}s)`,
      body:
        `Slowest: "${top.title}" took ${formatMs(top.duration_ms ?? 0)}. ` +
        "Slow steps are typically waiting on network-idle settle on " +
        "heavy-data SPAs. Try Speed=Fast for quick smoke tests when you " +
        "don't need the careful settle window.",
    });
  }

  // ── Inconclusive surface — different recommendation than failed ──
  // Halted-before-verification usually means the test case wording
  // was unclear or the agent got stuck on a UI it couldn't read.
  // Telling the user to "fix the bug" would be wrong; they should
  // tighten the test case's success criteria first.
  if (r.inconclusive > 0) {
    out.push({
      id: "inconclusive-cases",
      severity: "warn",
      title: `${r.inconclusive} test case${
        r.inconclusive === 1 ? "" : "s"
      } inconclusive`,
      body:
        `The agent halted before verifying ${
          r.inconclusive === 1 ? "this goal" : "these goals"
        }. ` +
        "This is usually a TEST-CASE problem (vague success criteria, " +
        "missing precondition, ambiguous step) — not necessarily an app " +
        "bug. Open the timeline, read the agent's last few turns, and " +
        "tighten the success criteria or hint targets accordingly.",
    });
  }

  // ── Positive recognition ───────────────────────────────────────

  if (
    r.failed === 0 &&
    r.blocked === 0 &&
    r.inconclusive === 0 &&
    r.total_steps > 0
  ) {
    const adjective =
      r.total_steps >= 20 ? "comprehensive " : r.total_steps >= 10 ? "solid " : "";
    out.push({
      id: "all-pass",
      severity: "good",
      title: `Clean run — ${r.total_steps} ${adjective}steps passed`,
      body:
        "Consider promoting this plan to your regression baseline. The " +
        "Excel export captures the run shape if you want a snapshot.",
    });
  }

  return sortBySeverity(out);
}

// ── Helpers ─────────────────────────────────────────────────────────

function topIssueExcerpt(
  module: ReportRead["modules"][number],
): string {
  for (const sub of module.submodules) {
    if (sub.issues.length > 0) {
      return `Common issue: "${sub.issues[0].slice(0, 80)}".`;
    }
  }
  return "";
}

function countAiHelped(report: ReportRead): number {
  let n = 0;
  for (const m of report.modules) {
    for (const sub of m.submodules) {
      for (const step of sub.steps) {
        if (step.ai_helped) n += 1;
      }
    }
  }
  return n;
}

function collectSlowSteps(report: ReportRead): ReportStepRead[] {
  const out: ReportStepRead[] = [];
  for (const m of report.modules) {
    for (const sub of m.submodules) {
      for (const step of sub.steps) {
        if (
          step.duration_ms !== null &&
          step.duration_ms > SLOW_STEP_THRESHOLD_MS
        ) {
          out.push(step);
        }
      }
    }
  }
  out.sort((a, b) => (b.duration_ms ?? 0) - (a.duration_ms ?? 0));
  return out;
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

const SEVERITY_ORDER: Record<RecommendationSeverity, number> = {
  urgent: 0,
  warn: 1,
  info: 2,
  good: 3,
};

function sortBySeverity(list: Recommendation[]): Recommendation[] {
  // Stable sort: same severity preserves insertion order (which is
  // semantic — e.g. modules sorted by ascending pass rate).
  return [...list].sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity],
  );
}
