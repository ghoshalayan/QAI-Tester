"use client";

import { useEffect, useRef, useState } from "react";
import type { ComponentType } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  Check,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Folder,
  FolderTree,
  Trash2,
  Undo2,
} from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  TC_NODE_KIND_LABELS,
  TC_NODE_STATUS_LABELS,
  type TcNodeKind,
  type TcNodeStatus,
  type TcNodeTreeRead,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// ── Selection helpers ─────────────────────────────────────────────

type SelectionState = "all" | "none" | "partial";

function getSelectionState(node: TcNodeTreeRead): SelectionState {
  if (node.children.length === 0) {
    return node.selectable_default ? "all" : "none";
  }
  let allSelected = node.selectable_default;
  let anySelected = node.selectable_default;
  const stack: TcNodeTreeRead[] = [...node.children];
  while (stack.length > 0) {
    const cur = stack.pop()!;
    if (cur.selectable_default) anySelected = true;
    else allSelected = false;
    for (const c of cur.children) stack.push(c);
  }
  if (allSelected) return "all";
  if (anySelected) return "partial";
  return "none";
}

function collectSubtreeIds(node: TcNodeTreeRead): number[] {
  const ids: number[] = [];
  const stack: TcNodeTreeRead[] = [node];
  while (stack.length > 0) {
    const cur = stack.pop()!;
    ids.push(cur.id);
    for (const c of cur.children) stack.push(c);
  }
  return ids;
}

const STATUS_CLASSES: Record<TcNodeStatus, string> = {
  draft: "bg-muted text-muted-foreground",
  approved: "bg-green-500/10 text-green-700 dark:text-green-400",
  archived: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
};

const KIND_ICON: Record<TcNodeKind, ComponentType<{ className?: string }>> = {
  module: Folder,
  submodule: FolderTree,
  step: CircleDot,
};

interface ValidationOverlay {
  status: "pending" | "confirmed" | "partial" | "unresolved" | "unreachable" | "skipped";
  confidence: number | null;
  reason: string | null;
}

interface TreeProps {
  projectId: number;
  planId: number;
  tree: TcNodeTreeRead[];
  onSelectNode?: (node: TcNodeTreeRead) => void;
  selectedNodeId?: number | null;
  /** Phase D — validation overlay map keyed by tc_node.id. When
   * present, each tree row shows a confidence chip. */
  validationByNodeId?: Map<number, ValidationOverlay>;
}

export function TcTree({
  projectId,
  planId,
  tree,
  onSelectNode,
  selectedNodeId,
  validationByNodeId,
}: TreeProps) {
  return (
    <div className="overflow-hidden rounded-lg border bg-card">
      {tree.map((root, idx) => (
        <div key={root.id} className={cn(idx > 0 && "border-t")}>
          <TcTreeNode
            node={root}
            projectId={projectId}
            planId={planId}
            depth={0}
            onSelectNode={onSelectNode}
            selectedNodeId={selectedNodeId}
            validationByNodeId={validationByNodeId}
          />
        </div>
      ))}
    </div>
  );
}

interface NodeProps {
  node: TcNodeTreeRead;
  projectId: number;
  planId: number;
  depth: number;
  onSelectNode?: (node: TcNodeTreeRead) => void;
  selectedNodeId?: number | null;
  validationByNodeId?: Map<number, ValidationOverlay>;
}

function TcTreeNode({
  node,
  projectId,
  planId,
  depth,
  onSelectNode,
  selectedNodeId,
  validationByNodeId,
}: NodeProps) {
  const validation = validationByNodeId?.get(node.id);
  const qc = useQueryClient();
  // Default-expand modules and submodules; step nodes have no children
  const [expanded, setExpanded] = useState(node.kind !== "step");

  const hasChildren = node.children.length > 0;
  const Icon = KIND_ICON[node.kind];
  const isSelected = selectedNodeId === node.id;
  const selectionState = getSelectionState(node);

  const checkboxRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (checkboxRef.current) {
      checkboxRef.current.indeterminate = selectionState === "partial";
    }
  }, [selectionState]);

  const statusMutation = useMutation({
    mutationFn: (status: TcNodeStatus) =>
      api.updateTcNode(projectId, planId, node.id, { status }),
    onSuccess: (updated) => {
      toast.success(
        `${TC_NODE_KIND_LABELS[node.kind]} ${TC_NODE_STATUS_LABELS[updated.status].toLowerCase()}`,
      );
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, planId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Update failed", { description: msg });
    },
  });

  const selectMutation = useMutation({
    mutationFn: (action: "select" | "deselect") =>
      api.bulkUpdateTcNodes(projectId, planId, {
        node_ids: collectSubtreeIds(node),
        action,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, planId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Selection update failed", { description: msg });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteTcNode(projectId, planId, node.id),
    onSuccess: () => {
      toast.success(`${TC_NODE_KIND_LABELS[node.kind]} deleted`);
      qc.invalidateQueries({ queryKey: ["tc-nodes", projectId, planId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Delete failed", { description: msg });
    },
  });

  const onTitleActivate = () => onSelectNode?.(node);

  // Click cycles: any partial/none → all-selected; all → none
  const onCheckboxToggle = () => {
    if (selectMutation.isPending) return;
    selectMutation.mutate(selectionState === "all" ? "deselect" : "select");
  };

  return (
    <>
      <div
        className={cn(
          "group flex items-start gap-2 border-b border-transparent py-2 pr-2 text-sm transition-colors last:border-b-0",
          isSelected ? "bg-primary/5" : "hover:bg-muted/40",
        )}
        style={{ paddingLeft: 12 + depth * 20 }}
      >
        {/* Tri-state selection checkbox */}
        <input
          ref={checkboxRef}
          type="checkbox"
          checked={selectionState === "all"}
          onChange={onCheckboxToggle}
          onClick={(e) => e.stopPropagation()}
          disabled={selectMutation.isPending}
          aria-label={
            selectionState === "all"
              ? `Deselect ${TC_NODE_KIND_LABELS[node.kind].toLowerCase()} for execution`
              : `Select ${TC_NODE_KIND_LABELS[node.kind].toLowerCase()} for execution`
          }
          title={
            selectionState === "partial"
              ? "Some descendants selected — click to select all"
              : selectionState === "all"
                ? "Selected for execution"
                : "Not selected"
          }
          className="mt-1 size-3.5 shrink-0 cursor-pointer rounded border-input accent-primary disabled:cursor-not-allowed disabled:opacity-50"
        />

        {/* Expand caret */}
        {hasChildren ? (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-label={expanded ? "Collapse" : "Expand"}
            className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded hover:bg-muted"
          >
            {expanded ? (
              <ChevronDown className="size-3.5" />
            ) : (
              <ChevronRight className="size-3.5" />
            )}
          </button>
        ) : (
          <span className="size-5 shrink-0" aria-hidden />
        )}

        {/* Kind icon */}
        <Icon
          className={cn(
            "mt-1 size-4 shrink-0",
            node.kind === "step"
              ? "text-blue-600 dark:text-blue-400"
              : "text-muted-foreground",
          )}
        />

        {/* Title + meta */}
        <div
          role="button"
          tabIndex={0}
          onClick={onTitleActivate}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onTitleActivate();
            }
          }}
          className="min-w-0 flex-1 cursor-pointer text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="break-words font-medium">{node.title}</span>
            <span
              className={cn(
                "inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
                STATUS_CLASSES[node.status],
              )}
            >
              {TC_NODE_STATUS_LABELS[node.status]}
            </span>
            {node.kind === "step" && node.action_type && (
              <span className="rounded border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                {node.action_type}
              </span>
            )}
            {node.source_requirement_ids.length > 0 && (
              <span className="text-[10px] text-muted-foreground">
                {node.source_requirement_ids.length} FRD
                {node.source_requirement_ids.length === 1 ? "" : "s"}
              </span>
            )}
            {validation && (
              <span
                className={cn(
                  "inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium",
                  validation.status === "confirmed"
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                    : validation.status === "partial"
                      ? "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400"
                      : validation.status === "unresolved"
                        ? "border-orange-500/40 bg-orange-500/10 text-orange-700 dark:text-orange-400"
                        : validation.status === "unreachable"
                          ? "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400"
                          : "border-muted bg-muted/30 text-muted-foreground",
                )}
                title={validation.reason ?? ""}
              >
                live{" "}
                {validation.status === "confirmed"
                  ? "✓"
                  : validation.status === "partial"
                    ? "~"
                    : validation.status === "unresolved"
                      ? "?"
                      : validation.status === "unreachable"
                        ? "✗"
                        : "—"}
                {validation.confidence !== null && (
                  <span className="ml-1 font-mono opacity-80">
                    {Math.round(validation.confidence * 100)}%
                  </span>
                )}
              </span>
            )}
          </div>
          {node.kind === "step" && node.narrative && (
            <p className="mt-0.5 line-clamp-1 text-xs text-muted-foreground">
              {node.narrative}
            </p>
          )}
        </div>

        {/* Action buttons (revealed on hover/focus) */}
        <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
          {node.status !== "approved" ? (
            <Button
              size="sm"
              variant="ghost"
              className="h-7 px-2"
              onClick={(e) => {
                e.stopPropagation();
                statusMutation.mutate("approved");
              }}
              disabled={statusMutation.isPending}
              aria-label="Approve"
              title="Approve"
            >
              <Check className="size-3.5" />
            </Button>
          ) : (
            <Button
              size="sm"
              variant="ghost"
              className="h-7 px-2"
              onClick={(e) => {
                e.stopPropagation();
                statusMutation.mutate("draft");
              }}
              disabled={statusMutation.isPending}
              aria-label="Move back to draft"
              title="Move back to draft"
            >
              <Undo2 className="size-3.5" />
            </Button>
          )}
          {node.status !== "archived" && (
            <Button
              size="sm"
              variant="ghost"
              className="h-7 px-2"
              onClick={(e) => {
                e.stopPropagation();
                statusMutation.mutate("archived");
              }}
              disabled={statusMutation.isPending}
              aria-label="Archive"
              title="Archive"
            >
              <Archive className="size-3.5" />
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2 text-red-600 hover:text-red-600 dark:text-red-400 dark:hover:text-red-400"
            onClick={(e) => {
              e.stopPropagation();
              const childCount = node.children.length;
              const suffix =
                childCount > 0
                  ? ` and its ${childCount} child node${childCount === 1 ? "" : "s"}`
                  : "";
              if (
                window.confirm(
                  `Delete ${TC_NODE_KIND_LABELS[node.kind].toLowerCase()} "${node.title}"${suffix}?`,
                )
              ) {
                deleteMutation.mutate();
              }
            }}
            disabled={deleteMutation.isPending}
            aria-label="Delete"
            title="Delete"
          >
            <Trash2 className="size-3.5" />
          </Button>
        </div>
      </div>

      {/* Children */}
      {hasChildren && expanded && (
        <div>
          {node.children.map((child) => (
            <TcTreeNode
              key={child.id}
              node={child}
              projectId={projectId}
              planId={planId}
              depth={depth + 1}
              onSelectNode={onSelectNode}
              selectedNodeId={selectedNodeId}
              validationByNodeId={validationByNodeId}
            />
          ))}
        </div>
      )}
    </>
  );
}
