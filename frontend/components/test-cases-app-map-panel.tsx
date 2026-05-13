"use client";

/**
 * Test Cases tab — App Map + Refinement panel.
 *
 * Co-locates everything related to "what's being tested" with the
 * test-cases viewer:
 *   - "Scout this app" button (triggers β.1 recon)
 *   - "Refine test cases" button (triggers Phase C.2 refinement +
 *     opens the diff dialog)
 *   - "Refresh on next run" (clears the cached app map)
 *   - Collapsible app-map summary (modules + create-flows + patterns)
 *   - TC versions list with Activate buttons
 *   - "Use live tree" rollback link
 *
 * Was previously in the plan editor; per user UX request it lives
 * with the test cases where users naturally look for it.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronDown,
  ChevronRight,
  RefreshCw,
  Sparkles,
  Telescope,
  Loader2,
} from "lucide-react";
import { toast } from "sonner";

import { api, ApiError, type AppMapRead } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RefineTcsDialog } from "@/components/refine-tcs-dialog";
import { cn } from "@/lib/utils";

export function TestCasesAppMapPanel({
  projectId,
  planId,
}: {
  projectId: number;
  planId: number;
}) {
  const qc = useQueryClient();
  const [refineDialogOpen, setRefineDialogOpen] = useState(false);
  const [mapExpanded, setMapExpanded] = useState(false);

  // ── App map ─────────────────────────────────────────────────
  const {
    data: appMap,
    isLoading: appMapLoading,
  } = useQuery<AppMapRead | null>({
    queryKey: ["app-map", projectId, planId],
    queryFn: async () => {
      try {
        return await api.getAppMap(projectId, planId);
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }
    },
    enabled: !Number.isNaN(projectId) && !Number.isNaN(planId),
  });

  // ── Versions ────────────────────────────────────────────────
  const { data: versionsData } = useQuery({
    queryKey: ["tc-versions", projectId, planId],
    queryFn: () => api.listTcVersions(projectId, planId),
    enabled: !Number.isNaN(projectId) && !Number.isNaN(planId),
  });

  // ── Mutations ───────────────────────────────────────────────
  const scoutMutation = useMutation({
    mutationFn: () => api.scoutApp(projectId, planId),
    onSuccess: (resp) => {
      const pages = resp.pages_visited ?? 0;
      toast.success(
        `Scout complete — visited ${pages} page${pages === 1 ? "" : "s"}`,
        {
          description: resp.auth_surface
            ? `Auth wall at ${resp.auth_surface} (scout stops there; ` +
              `run agentic mode once to scout the authenticated surface)`
            : undefined,
        },
      );
      qc.invalidateQueries({ queryKey: ["app-map", projectId, planId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Scout failed", { description: msg });
    },
  });

  const clearMapMutation = useMutation({
    mutationFn: () => api.clearAppMap(projectId, planId),
    onSuccess: () => {
      toast.success(
        "App map cleared. Next agentic run will rebuild it after login.",
      );
      qc.invalidateQueries({ queryKey: ["app-map", projectId, planId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Couldn't clear app map", { description: msg });
    },
  });

  const activateMutation = useMutation({
    mutationFn: (versionId: number) =>
      api.activateTcVersion(projectId, planId, versionId),
    onSuccess: (resp) => {
      toast.success(
        resp.current_tc_version_id
          ? "Version activated — live test cases updated"
          : "Reverted to the live tree",
      );
      qc.invalidateQueries({ queryKey: ["plan", projectId, planId] });
      qc.invalidateQueries({
        queryKey: ["tc-versions", projectId, planId],
      });
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, planId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Couldn't activate version", { description: msg });
    },
  });

  const hasMap = !!appMap;
  const versionCount = versionsData?.versions.length ?? 0;

  return (
    <div className="space-y-3 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-primary" />
          <span className="text-sm font-medium">App map &amp; refinement</span>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => scoutMutation.mutate()}
            disabled={scoutMutation.isPending}
            title="Walk the public surface of the app + write recon notes to AKB. Auth wall stops the walker."
          >
            {scoutMutation.isPending ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Telescope className="size-3.5" />
            )}
            {scoutMutation.isPending ? "Scouting…" : "Scout this app"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setRefineDialogOpen(true)}
            disabled={!hasMap}
            title={
              hasMap
                ? "Refine the test cases against this app map (creates a new version)"
                : "Build the app map first by running agentic mode once OR clicking Scout"
            }
          >
            <Sparkles className="size-3.5" />
            Refine test cases
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => clearMapMutation.mutate()}
            disabled={clearMapMutation.isPending || !hasMap}
            title={
              hasMap
                ? "Delete the cached map so the next run rebuilds it"
                : "No map cached yet"
            }
          >
            <RefreshCw className="size-3.5" />
            {clearMapMutation.isPending ? "Clearing…" : "Refresh on next run"}
          </Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Scout walks the app + builds the mindmap. Refine rewrites test
        cases to match the real UI labels and adds steps the BRD
        missed. Each refinement creates a new version — the live test
        cases below show the active version.
      </p>

      <RefineTcsDialog
        projectId={projectId}
        planId={planId}
        open={refineDialogOpen}
        onOpenChange={setRefineDialogOpen}
        hasAppMap={hasMap}
      />

      {/* App map summary */}
      {appMapLoading && <Skeleton className="h-16 w-full" />}
      {!appMapLoading && !hasMap && (
        <div className="rounded-md border border-dashed bg-muted/30 p-3 text-xs text-muted-foreground">
          No app map yet. Click <strong>Scout this app</strong> for a
          pre-login walk OR run agentic mode once to capture the
          authenticated surface.
        </div>
      )}
      {!appMapLoading && appMap && (
        <div className="rounded-md border bg-muted/10 text-xs">
          <button
            type="button"
            onClick={() => setMapExpanded((v) => !v)}
            className="flex w-full items-baseline gap-2 px-3 py-2 text-left hover:bg-muted/40"
          >
            {mapExpanded ? (
              <ChevronDown className="size-3 text-muted-foreground" />
            ) : (
              <ChevronRight className="size-3 text-muted-foreground" />
            )}
            <span className="font-medium">App map</span>
            <span className="text-muted-foreground">
              {appMap.modules.length} module
              {appMap.modules.length === 1 ? "" : "s"} ·{" "}
              {appMap.create_flows.length} create-flow
              {appMap.create_flows.length === 1 ? "" : "s"} ·{" "}
              {appMap.pages_scouted} pages scouted ({appMap.scout_depth})
            </span>
          </button>
          {mapExpanded && (
            <div className="space-y-2 border-t bg-card p-3">
              {appMap.modules.length > 0 && (
                <div>
                  <p className="font-medium">Modules</p>
                  <ul className="ml-3 mt-0.5 list-disc space-y-0.5">
                    {appMap.modules.map((m) => (
                      <li key={m.name}>
                        <span className="font-medium">{m.name}</span>
                        {m.sections.length > 0 && (
                          <span className="text-muted-foreground">
                            {" → "}
                            {m.sections.join(" · ")}
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {appMap.create_flows.length > 0 && (
                <div>
                  <p className="font-medium">Create flows</p>
                  <ul className="ml-3 mt-0.5 list-disc space-y-1">
                    {appMap.create_flows.map((fl, i) => (
                      <li key={`${fl.entity}-${i}`}>
                        <span className="font-medium">{fl.entity}</span>
                        {fl.section_path.length > 0 && (
                          <span className="text-muted-foreground">
                            {" at "}
                            {fl.section_path.join(" > ")}
                          </span>
                        )}
                        <span className="text-muted-foreground">
                          {" — trigger "}
                          <code className="rounded bg-muted px-1">
                            {fl.trigger_label}
                          </code>
                          {", submit "}
                          <code className="rounded bg-muted px-1">
                            {fl.submit_label}
                          </code>
                        </span>
                        {fl.fields.length > 0 && (
                          <div className="ml-3 mt-0.5 text-[10px] text-muted-foreground">
                            fields:{" "}
                            {fl.fields
                              .map(
                                (f) =>
                                  `${f.label}(${f.role})${f.required ? "*" : ""}`,
                              )
                              .join(", ")}
                          </div>
                        )}
                        {(fl.list_has_search ||
                          fl.has_permission_tree) && (
                          <div className="ml-3 mt-0.5 flex gap-1 text-[10px]">
                            {fl.list_has_search && (
                              <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1 text-emerald-700 dark:text-emerald-400">
                                searchable list
                              </span>
                            )}
                            {fl.has_permission_tree && (
                              <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1 text-amber-700 dark:text-amber-400">
                                permission tree
                              </span>
                            )}
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {appMap.cross_cutting_notes.length > 0 && (
                <div>
                  <p className="font-medium">Patterns</p>
                  <ul className="ml-3 mt-0.5 list-disc space-y-0.5">
                    {appMap.cross_cutting_notes
                      .slice(0, 6)
                      .map((n, i) => (
                        <li key={i}>{n}</li>
                      ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Versions list */}
      {versionsData && versionCount > 0 && (
        <div className="rounded-md border bg-muted/10 p-2 text-xs">
          <p className="px-1 font-medium">
            Test-case versions{" "}
            <span className="text-muted-foreground">
              ({versionCount})
            </span>
          </p>
          <ul className="mt-1 space-y-0.5">
            {versionsData.versions.map((v) => {
              const active = versionsData.current_tc_version_id === v.id;
              return (
                <li
                  key={v.id}
                  className="flex items-baseline gap-2 px-1"
                >
                  <span
                    className={cn(
                      "shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px]",
                      active
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground",
                    )}
                  >
                    {active ? "active" : `v${v.version_number}`}
                  </span>
                  <span className="min-w-0 flex-1 truncate">
                    {v.label}
                  </span>
                  <span className="shrink-0 text-[10px] text-muted-foreground">
                    {v.created_at
                      ? new Date(v.created_at).toLocaleString()
                      : ""}
                  </span>
                  {!active && (
                    <button
                      type="button"
                      className="shrink-0 text-[10px] text-primary hover:underline disabled:opacity-50"
                      onClick={() => activateMutation.mutate(v.id)}
                      disabled={activateMutation.isPending}
                    >
                      Activate
                    </button>
                  )}
                </li>
              );
            })}
            {versionsData.current_tc_version_id !== null && (
              <li className="flex items-baseline gap-2 pt-1">
                <button
                  type="button"
                  className="text-[10px] text-muted-foreground hover:text-foreground hover:underline"
                  onClick={() => activateMutation.mutate(0)}
                  disabled={activateMutation.isPending}
                  title="Run against the live TcNode tree instead of a version"
                >
                  Use live tree (clear active version)
                </button>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
