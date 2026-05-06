"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { Check, ExternalLink, Pencil, X } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  TC_NODE_KIND_LABELS,
  TC_NODE_STATUS_LABELS,
  type TcNodeStatus,
  type TcNodeTreeRead,
  type TcNodeUpdate,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const STATUS_CLASSES: Record<TcNodeStatus, string> = {
  draft: "bg-muted text-muted-foreground",
  approved: "bg-green-500/10 text-green-700 dark:text-green-400",
  archived: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
};

// Action types the executor knows about; matches the dispatcher in
// app/executor/actions.py.
const ACTION_TYPES = [
  "navigate",
  "click",
  "type",
  "select",
  "verify",
  "wait",
  "submit",
  "screenshot",
] as const;

interface Props {
  projectId: number;
  node: TcNodeTreeRead;
  onClose: () => void;
}

interface DraftFields {
  title: string;
  action_type: string;
  target_hint: string;
  narrative: string;
  expected: string;
  description_md: string;
}

function nodeToDraft(node: TcNodeTreeRead): DraftFields {
  return {
    title: node.title,
    action_type: node.action_type ?? "",
    target_hint: node.target_hint ?? "",
    narrative: node.narrative ?? "",
    expected: node.expected ?? "",
    description_md: node.description_md ?? "",
  };
}

export function TcDetailPanel({ projectId, node, onClose }: Props) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<DraftFields>(() => nodeToDraft(node));

  // When the parent swaps the selected node OR toggles editing back on,
  // re-seed the draft from the live node values so we never carry stale
  // inputs from a previous selection.
  useEffect(() => {
    setDraft(nodeToDraft(node));
    setEditing(false);
  }, [node.id]);

  const isStep = node.kind === "step";

  const { data: allReqs, isLoading: reqsLoading } = useQuery({
    queryKey: ["requirements", projectId],
    queryFn: () => api.listRequirements(projectId),
    enabled: node.source_requirement_ids.length > 0,
  });

  const sourceReqs = useMemo(() => {
    if (!allReqs) return [];
    const wanted = new Set(node.source_requirement_ids);
    const byId = new Map(allReqs.map((r) => [r.id, r]));
    return node.source_requirement_ids
      .map((id) => byId.get(id))
      .filter((r): r is NonNullable<typeof r> => !!r && wanted.has(r.id));
  }, [allReqs, node.source_requirement_ids]);

  const childCount = node.children.length;

  const saveMutation = useMutation({
    mutationFn: (payload: TcNodeUpdate) =>
      api.updateTcNode(projectId, node.plan_id, node.id, payload),
    onSuccess: () => {
      toast.success("Saved");
      qc.invalidateQueries({
        queryKey: ["tc-nodes", projectId, node.plan_id],
      });
      setEditing(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Save failed", { description: msg });
    },
  });

  const onSave = () => {
    // Build a partial PATCH — only include fields that actually changed,
    // so unset values stay null on the server.
    const payload: TcNodeUpdate = {};
    const original = nodeToDraft(node);

    if (draft.title.trim() !== original.title) {
      payload.title = draft.title.trim();
    }
    if (isStep) {
      if (draft.action_type !== original.action_type) {
        payload.action_type = draft.action_type;
      }
      if (draft.target_hint !== original.target_hint) {
        payload.target_hint = draft.target_hint;
      }
      if (draft.narrative !== original.narrative) {
        payload.narrative = draft.narrative;
      }
      if (draft.expected !== original.expected) {
        payload.expected = draft.expected;
      }
    }
    if (draft.description_md !== original.description_md) {
      payload.description_md = draft.description_md;
    }

    if (Object.keys(payload).length === 0) {
      // Nothing actually changed — exit edit mode silently
      setEditing(false);
      return;
    }
    saveMutation.mutate(payload);
  };

  const onCancel = () => {
    setDraft(nodeToDraft(node));
    setEditing(false);
  };

  return (
    <aside className="sticky top-4 max-h-[calc(100vh-2rem)] w-full overflow-y-auto rounded-lg border bg-card p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            {TC_NODE_KIND_LABELS[node.kind]} · #{node.id}
          </p>
          {editing ? (
            <Input
              value={draft.title}
              onChange={(e) =>
                setDraft({ ...draft, title: e.target.value })
              }
              className="mt-1"
              placeholder="Title"
            />
          ) : (
            <h3 className="mt-0.5 break-words text-base font-semibold">
              {node.title}
            </h3>
          )}
          <p className="mt-0.5 break-words text-xs text-muted-foreground">
            {node.path_cached}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {!editing && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setEditing(true)}
              aria-label="Edit"
              title="Edit this node"
            >
              <Pencil className="size-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            aria-label="Close detail panel"
          >
            <X className="size-4" />
          </Button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span
          className={cn(
            "inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
            STATUS_CLASSES[node.status],
          )}
        >
          {TC_NODE_STATUS_LABELS[node.status]}
        </span>
        {isStep && !editing && node.action_type && (
          <span className="rounded border px-2 py-0.5 font-mono text-[10px]">
            {node.action_type}
          </span>
        )}
        {childCount > 0 && (
          <span className="text-[10px] text-muted-foreground">
            {childCount} {childCount === 1 ? "child" : "children"}
          </span>
        )}
      </div>

      {/* Action type — editable for steps only */}
      {isStep && editing && (
        <Section label="Action type">
          <select
            value={draft.action_type}
            onChange={(e) =>
              setDraft({ ...draft, action_type: e.target.value })
            }
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="">(none)</option>
            {ACTION_TYPES.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </Section>
      )}

      {/* Description — editable for any kind */}
      {(node.description_md || editing) && (
        <Section label="Description">
          {editing ? (
            <Textarea
              value={draft.description_md}
              onChange={(e) =>
                setDraft({ ...draft, description_md: e.target.value })
              }
              rows={3}
              placeholder="Optional markdown description"
            />
          ) : (
            <p className="whitespace-pre-wrap text-sm">{node.description_md}</p>
          )}
        </Section>
      )}

      {/* Step-only fields */}
      {isStep && (
        <>
          {(node.narrative || editing) && (
            <Section label="Narrative">
              {editing ? (
                <Textarea
                  value={draft.narrative}
                  onChange={(e) =>
                    setDraft({ ...draft, narrative: e.target.value })
                  }
                  rows={3}
                  placeholder="What the user does in this step (e.g. Type 'admin' into the username field)"
                />
              ) : (
                <p className="whitespace-pre-wrap text-sm">{node.narrative}</p>
              )}
            </Section>
          )}
          {(node.target_hint || editing) && (
            <Section label="Target hint">
              {editing ? (
                <Input
                  value={draft.target_hint}
                  onChange={(e) =>
                    setDraft({ ...draft, target_hint: e.target.value })
                  }
                  className="font-mono text-xs"
                  placeholder='CSS selector, "text Sign In", or role=button[name=...]'
                />
              ) : (
                <code className="block break-all rounded bg-muted px-2 py-1 font-mono text-xs">
                  {node.target_hint}
                </code>
              )}
            </Section>
          )}
          {(node.expected || editing) && (
            <Section label="Expected">
              {editing ? (
                <Textarea
                  value={draft.expected}
                  onChange={(e) =>
                    setDraft({ ...draft, expected: e.target.value })
                  }
                  rows={2}
                  placeholder="What should be observable after the step"
                />
              ) : (
                <p className="whitespace-pre-wrap text-sm">{node.expected}</p>
              )}
            </Section>
          )}
          {!editing &&
            node.data_needs_json &&
            node.data_needs_json.length > 0 && (
              <Section label="Data needs">
                <ul className="space-y-1.5">
                  {node.data_needs_json.map((dn, i) => (
                    <li
                      key={`${dn.kind}-${i}`}
                      className="flex items-start gap-2 text-xs"
                    >
                      <span className="shrink-0 rounded border px-1.5 py-0.5 font-mono uppercase">
                        {dn.kind}
                      </span>
                      <span className="text-muted-foreground">{dn.notes}</span>
                    </li>
                  ))}
                </ul>
              </Section>
            )}
        </>
      )}

      {/* Edit-mode action row — sticky at the bottom of the panel */}
      {editing && (
        <div className="mt-4 flex flex-wrap gap-2 border-t pt-3">
          <Button
            size="sm"
            onClick={onSave}
            disabled={saveMutation.isPending}
          >
            <Check className="size-4" />
            {saveMutation.isPending ? "Saving…" : "Save"}
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={onCancel}
            disabled={saveMutation.isPending}
          >
            Cancel
          </Button>
          <p className="ml-auto self-center text-[10px] text-muted-foreground">
            Title rename rebuilds path_cached for the subtree
          </p>
        </div>
      )}

      {/* Source FRDs — read-only */}
      {!editing && node.source_requirement_ids.length > 0 && (
        <Section label={`Source FRDs (${node.source_requirement_ids.length})`}>
          {reqsLoading && !allReqs ? (
            <p className="text-xs text-muted-foreground">Loading…</p>
          ) : sourceReqs.length === 0 ? (
            <p className="text-xs italic text-muted-foreground">
              Source FRDs no longer present in this project.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {sourceReqs.map((req) => (
                <li key={req.id}>
                  <Link
                    href={`/projects/${projectId}/requirements`}
                    className="group flex items-start gap-2 rounded-md border p-2 text-xs transition-colors hover:border-primary/50 hover:bg-muted/40"
                    title={`View ${req.code} on the Requirements tab`}
                  >
                    <span className="font-mono font-semibold text-primary">
                      {req.code}
                    </span>
                    <span className="min-w-0 flex-1 break-words">
                      {req.title}
                    </span>
                    <ExternalLink className="size-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
                  </Link>
                </li>
              ))}
              {sourceReqs.length < node.source_requirement_ids.length && (
                <li className="text-[10px] italic text-muted-foreground">
                  {node.source_requirement_ids.length - sourceReqs.length} cited
                  FRD(s) no longer in the project.
                </li>
              )}
            </ul>
          )}
        </Section>
      )}
    </aside>
  );
}

function Section({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="mt-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}
