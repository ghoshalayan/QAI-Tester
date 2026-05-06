"use client";

import { useMemo } from "react";
import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { ExternalLink, X } from "lucide-react";

import {
  api,
  TC_NODE_KIND_LABELS,
  TC_NODE_STATUS_LABELS,
  type TcNodeStatus,
  type TcNodeTreeRead,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const STATUS_CLASSES: Record<TcNodeStatus, string> = {
  draft: "bg-muted text-muted-foreground",
  approved: "bg-green-500/10 text-green-700 dark:text-green-400",
  archived: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
};

interface Props {
  projectId: number;
  node: TcNodeTreeRead;
  onClose: () => void;
}

export function TcDetailPanel({ projectId, node, onClose }: Props) {
  // Re-use the Requirements tab's cache key so this is usually free
  const { data: allReqs, isLoading: reqsLoading } = useQuery({
    queryKey: ["requirements", projectId],
    queryFn: () => api.listRequirements(projectId),
    enabled: node.source_requirement_ids.length > 0,
  });

  const sourceReqs = useMemo(() => {
    if (!allReqs) return [];
    const wanted = new Set(node.source_requirement_ids);
    // Preserve the source order from the node so citations stay stable
    const byId = new Map(allReqs.map((r) => [r.id, r]));
    return node.source_requirement_ids
      .map((id) => byId.get(id))
      .filter((r): r is NonNullable<typeof r> => !!r && wanted.has(r.id));
  }, [allReqs, node.source_requirement_ids]);

  const childCount = node.children.length;

  return (
    <aside className="sticky top-4 max-h-[calc(100vh-2rem)] w-full overflow-y-auto rounded-lg border bg-card p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            {TC_NODE_KIND_LABELS[node.kind]} · #{node.id}
          </p>
          <h3 className="mt-0.5 break-words text-base font-semibold">
            {node.title}
          </h3>
          <p className="mt-0.5 break-words text-xs text-muted-foreground">
            {node.path_cached}
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          aria-label="Close detail panel"
        >
          <X className="size-4" />
        </Button>
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
        {node.kind === "step" && node.action_type && (
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

      {node.description_md && (
        <Section label="Description">
          <p className="whitespace-pre-wrap text-sm">{node.description_md}</p>
        </Section>
      )}

      {node.kind === "step" && (
        <>
          {node.narrative && (
            <Section label="Narrative">
              <p className="whitespace-pre-wrap text-sm">{node.narrative}</p>
            </Section>
          )}
          {node.target_hint && (
            <Section label="Target hint">
              <code className="block break-all rounded bg-muted px-2 py-1 font-mono text-xs">
                {node.target_hint}
              </code>
            </Section>
          )}
          {node.expected && (
            <Section label="Expected">
              <p className="whitespace-pre-wrap text-sm">{node.expected}</p>
            </Section>
          )}
          {node.data_needs_json && node.data_needs_json.length > 0 && (
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

      {node.source_requirement_ids.length > 0 && (
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
