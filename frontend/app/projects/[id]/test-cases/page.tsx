"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  Check,
  ClipboardList,
  FileJson,
  FileText,
  ListTree,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  PLAN_STATUS_LABELS,
  TC_NODE_STATUS_LABELS,
  type AgentStatus,
  type PlanReadCompact,
  type TcNodeStatus,
  type TcNodeTreeRead,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RunProgressCard } from "@/components/run-progress-card";
import { SynthesizeTcDialog } from "@/components/synthesize-tc-dialog";
import { TcDetailPanel } from "@/components/tc-detail-panel";
import { TcTree } from "@/components/tc-tree";
import { useAgentRunsEvents } from "@/hooks/use-agent-runs-events";
import { cn } from "@/lib/utils";

type FilterValue = TcNodeStatus | "all";
const ALL_STATUSES: TcNodeStatus[] = ["draft", "approved", "archived"];

const ACTIVE_RUN_STATUSES: AgentStatus[] = ["queued", "running", "paused"];

const STATUS_CLASSES: Record<TcNodeStatus, string> = {
  draft: "bg-muted text-muted-foreground",
  approved: "bg-green-500/10 text-green-700 dark:text-green-400",
  archived: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
};

export default function TestCasesTabPage() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);

  const qc = useQueryClient();

  const [selectedPlanId, setSelectedPlanId] = useState<number | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const [synthesizeOpen, setSynthesizeOpen] = useState(false);
  const [filter, setFilter] = useState<FilterValue>("all");

  useAgentRunsEvents(projectId);

  const { data: plans, isLoading: plansLoading } = useQuery({
    queryKey: ["plans", projectId],
    queryFn: () => api.listPlans(projectId),
  });

  // Default to the most-recently-updated plan
  useEffect(() => {
    if (selectedPlanId !== null) return;
    if (plans && plans.length > 0) setSelectedPlanId(plans[0].id);
  }, [plans, selectedPlanId]);

  const { data: tree, isLoading: treeLoading } = useQuery({
    queryKey: ["tc-nodes", projectId, selectedPlanId],
    queryFn: () => api.listTcNodes(projectId, selectedPlanId!),
    enabled: !!selectedPlanId,
  });

  const { data: runs } = useQuery({
    queryKey: ["agent-runs", projectId],
    queryFn: () => api.listAgentRuns(projectId),
  });

  // Show active frd_to_tc runs for the currently-selected plan
  const activeFrdToTcRuns = useMemo(() => {
    if (!runs || selectedPlanId === null) return [];
    return runs.filter((r) => {
      if (r.kind !== "frd_to_tc") return false;
      if (!ACTIVE_RUN_STATUSES.includes(r.status)) return false;
      const runPlanId = r.input_json?.plan_id;
      return typeof runPlanId === "number" && runPlanId === selectedPlanId;
    });
  }, [runs, selectedPlanId]);

  const counts = useMemo(() => countNodes(tree ?? []), [tree]);
  const selectedPlan = useMemo(
    () => plans?.find((p) => p.id === selectedPlanId) ?? null,
    [plans, selectedPlanId],
  );
  const selectedNode = useMemo(
    () => (tree && selectedNodeId !== null ? findNode(tree, selectedNodeId) : null),
    [tree, selectedNodeId],
  );

  const filteredTree = useMemo(
    () => (tree ? filterTreeByStatus(tree, filter) : []),
    [tree, filter],
  );

  // If the selection points at a node that no longer exists (deleted, or
  // user switched plans), clear it so the panel doesn't linger as stale.
  useEffect(() => {
    if (selectedNodeId !== null && tree && !selectedNode) {
      setSelectedNodeId(null);
    }
  }, [selectedNodeId, selectedNode, tree]);

  // Clear selection whenever the plan changes
  useEffect(() => {
    setSelectedNodeId(null);
  }, [selectedPlanId]);

  const bulkApproveDrafts = useMutation({
    mutationFn: () =>
      api.bulkUpdateTcNodes(projectId, selectedPlanId!, {
        filter_status: "draft",
        action: "approve",
      }),
    onSuccess: (res) => {
      toast.success(
        `${res.affected} node${res.affected === 1 ? "" : "s"} approved`,
      );
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, selectedPlanId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Bulk approve failed", { description: msg });
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <p className="max-w-2xl text-sm text-muted-foreground">
          Hierarchical test cases — Module → Submodule → Step — generated from
          this plan&apos;s scope, instructions, and approved FRDs. Drill into
          any step to see the source FRDs.
        </p>
        <Button
          size="sm"
          onClick={() => setSynthesizeOpen(true)}
          disabled={!selectedPlanId}
        >
          <Sparkles className="size-4" /> Generate test cases
        </Button>
      </div>

      {selectedPlanId && (
        <SynthesizeTcDialog
          open={synthesizeOpen}
          onOpenChange={setSynthesizeOpen}
          projectId={projectId}
          planId={selectedPlanId}
        />
      )}

      {activeFrdToTcRuns.length > 0 && (
        <div className="space-y-3">
          {activeFrdToTcRuns.map((run) => (
            <RunProgressCard
              key={run.id}
              projectId={projectId}
              run={run}
            />
          ))}
        </div>
      )}

      {plansLoading ? (
        <Skeleton className="h-12 w-full" />
      ) : !plans || plans.length === 0 ? (
        <NoPlansState projectId={projectId} />
      ) : (
        <>
          <PlanPicker
            plans={plans}
            selectedId={selectedPlanId}
            onSelect={setSelectedPlanId}
          />

          {selectedPlan && (
            <TreeArea
              projectId={projectId}
              plan={selectedPlan}
              tree={tree}
              filteredTree={filteredTree}
              isLoading={treeLoading}
              counts={counts}
              filter={filter}
              onFilterChange={setFilter}
              selectedNode={selectedNode}
              selectedNodeId={selectedNodeId}
              onSelectNode={(node) => setSelectedNodeId(node.id)}
              onCloseDetail={() => setSelectedNodeId(null)}
              onTriggerSynthesize={() => setSynthesizeOpen(true)}
              onBulkApproveDrafts={() => bulkApproveDrafts.mutate()}
              bulkApproveDisabled={bulkApproveDrafts.isPending}
            />
          )}
        </>
      )}
    </div>
  );
}

// ── Plan picker ───────────────────────────────────────────────────


function PlanPicker({
  plans,
  selectedId,
  onSelect,
}: {
  plans: PlanReadCompact[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="text-sm text-muted-foreground" htmlFor="plan-picker">
        Plan:
      </label>
      <select
        id="plan-picker"
        value={selectedId ?? ""}
        onChange={(e) => onSelect(Number(e.target.value))}
        className="h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
      >
        {plans.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name} — {PLAN_STATUS_LABELS[p.status]}
          </option>
        ))}
      </select>
    </div>
  );
}

// ── Tree area (shell) ─────────────────────────────────────────────


function TreeArea({
  projectId,
  plan,
  tree,
  filteredTree,
  isLoading,
  counts,
  filter,
  onFilterChange,
  selectedNode,
  selectedNodeId,
  onSelectNode,
  onCloseDetail,
  onTriggerSynthesize,
  onBulkApproveDrafts,
  bulkApproveDisabled,
}: {
  projectId: number;
  plan: PlanReadCompact;
  tree: TcNodeTreeRead[] | undefined;
  filteredTree: TcNodeTreeRead[];
  isLoading: boolean;
  counts: ReturnType<typeof countNodes>;
  filter: FilterValue;
  onFilterChange: (v: FilterValue) => void;
  selectedNode: TcNodeTreeRead | null;
  selectedNodeId: number | null;
  onSelectNode: (node: TcNodeTreeRead) => void;
  onCloseDetail: () => void;
  onTriggerSynthesize: () => void;
  onBulkApproveDrafts: () => void;
  bulkApproveDisabled: boolean;
}) {
  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-8 w-1/3" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  if (!tree || tree.length === 0) {
    return (
      <EmptyTreeState
        plan={plan}
        projectId={projectId}
        onTriggerSynthesize={onTriggerSynthesize}
      />
    );
  }

  const draftCount = counts.byStatus.draft;
  const filterCounts: Record<FilterValue, number> = {
    all: counts.total,
    draft: counts.byStatus.draft,
    approved: counts.byStatus.approved,
    archived: counts.byStatus.archived,
  };

  return (
    <div className="space-y-4">
      {/* Quick stats */}
      <div className="flex flex-wrap gap-2 text-xs">
        <span className="rounded-md border px-2.5 py-1 font-medium">
          {counts.total} total
        </span>
        <span className="rounded-md border px-2.5 py-1 font-medium">
          {counts.modules} modules · {counts.submodules} submodules ·{" "}
          {counts.steps} steps
        </span>
        {counts.steps > 0 && (
          <span
            className={cn(
              "rounded-md border px-2.5 py-1 font-medium",
              counts.selectedSteps === 0
                ? "bg-muted text-muted-foreground"
                : "bg-primary/10 text-primary",
            )}
            title="Steps selected for the next execution run"
          >
            {counts.selectedSteps}/{counts.steps} steps selected
          </span>
        )}
      </div>

      {/* Filter chips + bulk actions */}
      <div className="flex flex-wrap items-center gap-3">
        <FilterChips
          counts={filterCounts}
          value={filter}
          onChange={onFilterChange}
        />
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {/* Download selected — `selected_only=true` so the export
              matches what would actually run on the next execute. */}
          <span className="text-xs text-muted-foreground">
            Download selected:
          </span>
          <Button asChild size="sm" variant="outline">
            <a
              href={api.exportTcNodesUrl(projectId, plan.id, {
                format: "json",
                selectedOnly: true,
              })}
              download
            >
              <FileJson className="size-4" />
              JSON
            </a>
          </Button>
          <Button asChild size="sm" variant="outline">
            <a
              href={api.exportTcNodesUrl(projectId, plan.id, {
                format: "md",
                selectedOnly: true,
              })}
              download
            >
              <FileText className="size-4" />
              Markdown
            </a>
          </Button>
          {draftCount > 0 && (
            <Button
              size="sm"
              variant="outline"
              onClick={onBulkApproveDrafts}
              disabled={bulkApproveDisabled}
            >
              <Check className="size-4" />
              Approve all {draftCount} draft{draftCount === 1 ? "" : "s"}
            </Button>
          )}
        </div>
      </div>

      {filteredTree.length === 0 ? (
        <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
          No nodes match the <strong>{filter}</strong> filter. Switch to
          <button
            type="button"
            onClick={() => onFilterChange("all")}
            className="ml-1 text-primary underline-offset-2 hover:underline"
          >
            All
          </button>
          {" "}to see everything.
        </div>
      ) : (
        <div
          className={cn(
            "grid gap-4",
            selectedNode
              ? "lg:grid-cols-[minmax(0,1fr)_420px]"
              : "grid-cols-1",
          )}
        >
          <TcTree
            projectId={projectId}
            planId={plan.id}
            tree={filteredTree}
            onSelectNode={onSelectNode}
            selectedNodeId={selectedNodeId}
          />
          {selectedNode && (
            <TcDetailPanel
              projectId={projectId}
              node={selectedNode}
              onClose={onCloseDetail}
            />
          )}
        </div>
      )}
    </div>
  );
}

// ── Filter chips ───────────────────────────────────────────────────


function FilterChips({
  counts,
  value,
  onChange,
}: {
  counts: Record<FilterValue, number>;
  value: FilterValue;
  onChange: (v: FilterValue) => void;
}) {
  const options: FilterValue[] = ["all", ...ALL_STATUSES];

  return (
    <div className="flex flex-wrap gap-2 text-xs">
      {options.map((opt) => {
        const isActive = value === opt;
        const baseClass =
          "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-medium transition-colors";
        const colorClass =
          opt === "all" ? "" : STATUS_CLASSES[opt as TcNodeStatus];
        const activeRing = isActive
          ? "ring-2 ring-primary ring-offset-1 ring-offset-background"
          : "hover:opacity-80";
        return (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={cn(baseClass, colorClass, activeRing)}
          >
            <span>
              {opt === "all"
                ? "All"
                : TC_NODE_STATUS_LABELS[opt as TcNodeStatus]}
            </span>
            <span className="font-mono opacity-70">{counts[opt]}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── Empty states ──────────────────────────────────────────────────


function NoPlansState({ projectId }: { projectId: number }) {
  return (
    <div className="rounded-lg border border-dashed p-12 text-center">
      <ClipboardList className="mx-auto size-10 text-muted-foreground" />
      <h3 className="mt-4 font-semibold">No plans yet</h3>
      <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">
        Test cases are generated per-plan. Create a plan first — set the
        target URL, scope, and either link some BRDs/FRDs or write
        instructions.
      </p>
      <Button asChild className="mt-4">
        <Link href={`/projects/${projectId}/plans`}>
          <ClipboardList className="size-4" /> Create a plan
        </Link>
      </Button>
    </div>
  );
}

function EmptyTreeState({
  plan,
  projectId,
  onTriggerSynthesize,
}: {
  plan: PlanReadCompact;
  projectId: number;
  onTriggerSynthesize: () => void;
}) {
  const hasScope = plan.scope.length > 0;
  return (
    <div className="rounded-lg border border-dashed p-12 text-center">
      <ListTree className="mx-auto size-10 text-muted-foreground" />
      <h3 className="mt-4 font-semibold">No test cases yet for this plan</h3>
      <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">
        Run the FRD→TC agent to generate a Module → Submodule → Step tree
        from this plan&apos;s {hasScope ? "scope" : "(empty scope)"} and any
        approved FRDs in the project.
      </p>
      <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
        <Button onClick={onTriggerSynthesize}>
          <Sparkles className="size-4" /> Generate test cases
        </Button>
        <span className="text-xs text-muted-foreground">·</span>
        <Link
          href={`/projects/${projectId}/plans/${plan.id}`}
          className="text-xs text-primary hover:underline"
        >
          Edit plan
        </Link>
      </div>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────


function countNodes(tree: TcNodeTreeRead[]) {
  const result = {
    total: 0,
    modules: 0,
    submodules: 0,
    steps: 0,
    selectedSteps: 0,
    byStatus: { draft: 0, approved: 0, archived: 0 } as Record<
      TcNodeStatus,
      number
    >,
  };

  function walk(nodes: TcNodeTreeRead[]) {
    for (const n of nodes) {
      result.total += 1;
      result.byStatus[n.status] += 1;
      if (n.kind === "module") result.modules += 1;
      else if (n.kind === "submodule") result.submodules += 1;
      else if (n.kind === "step") {
        result.steps += 1;
        if (n.selectable_default) result.selectedSteps += 1;
      }
      walk(n.children);
    }
  }

  walk(tree);
  return result;
}

function findNode(
  tree: TcNodeTreeRead[],
  id: number,
): TcNodeTreeRead | null {
  for (const n of tree) {
    if (n.id === id) return n;
    const found = findNode(n.children, id);
    if (found) return found;
  }
  return null;
}

// Tree-aware filter: keeps a node visible when the node itself matches OR
// any descendant matches (so the parent path stays around for context).
function filterTreeByStatus(
  tree: TcNodeTreeRead[],
  filter: FilterValue,
): TcNodeTreeRead[] {
  if (filter === "all") return tree;
  const out: TcNodeTreeRead[] = [];
  for (const node of tree) {
    const filteredChildren = filterTreeByStatus(node.children, filter);
    if (node.status === filter || filteredChildren.length > 0) {
      out.push({ ...node, children: filteredChildren });
    }
  }
  return out;
}
