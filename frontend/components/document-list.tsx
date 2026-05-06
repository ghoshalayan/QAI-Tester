"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Loader2, Trash2 } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  DOCUMENT_KIND_LABELS,
  type DocumentStatus,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useDocumentEvents,
  useDocumentProgress,
} from "@/hooks/use-document-events";
import { cn } from "@/lib/utils";

const STATUS_CLASSES: Record<DocumentStatus, string> = {
  pending: "bg-muted text-muted-foreground",
  parsing: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
  embedding: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
  parsed: "bg-green-500/10 text-green-600 dark:text-green-400",
  failed: "bg-red-500/10 text-red-600 dark:text-red-400",
};

const ACTIVE_STATUSES: DocumentStatus[] = ["pending", "parsing", "embedding"];

export function DocumentList() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);
  const qc = useQueryClient();

  // Subscribe to backend SSE for live ingest events. Invalidates the
  // documents query on started/completed/failed; updates per-doc transient
  // progress on `doc_progress` (no list refetch on every embedding batch).
  useDocumentEvents(projectId);
  const progressMap = useDocumentProgress((s) => s.byDocId);

  const { data: documents, isLoading } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => api.listDocuments(projectId),
  });

  const deleteMutation = useMutation({
    mutationFn: (docId: number) => api.deleteDocument(projectId, docId),
    onSuccess: () => {
      toast.success("Document deleted");
      qc.invalidateQueries({ queryKey: ["documents", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Delete failed", { description: msg });
    },
  });

  if (isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
        No documents yet. Upload a file or paste text above.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/30 text-left text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th className="p-3 font-medium">Filename</th>
            <th className="p-3 font-medium">Kind</th>
            <th className="p-3 font-medium">Source</th>
            <th className="p-3 font-medium">Status</th>
            <th className="p-3 text-right font-medium">Chunks</th>
            <th className="p-3 font-medium">Date</th>
            <th className="p-3" />
          </tr>
        </thead>
        <tbody>
          {documents.map((d) => {
            const isActive = ACTIVE_STATUSES.includes(d.status);
            const progress = progressMap[d.id];
            return (
              <tr
                key={d.id}
                className="border-b last:border-b-0 hover:bg-accent/30"
              >
                <td className="max-w-[280px] p-3">
                  <Link
                    href={`/projects/${projectId}/documents/${d.id}`}
                    className="line-clamp-1 font-medium hover:underline"
                  >
                    {d.filename}
                  </Link>
                  {isActive && progress?.message && (
                    <p className="mt-1 line-clamp-1 text-xs text-muted-foreground">
                      {progress.message}
                      {progress.current !== undefined &&
                      progress.total !== undefined
                        ? ` (${progress.current}/${progress.total})`
                        : null}
                    </p>
                  )}
                  {d.error_message && (
                    <p className="mt-1 line-clamp-2 text-xs text-red-600 dark:text-red-400">
                      {d.error_message}
                    </p>
                  )}
                </td>
                <td className="p-3">
                  <Badge variant="outline">
                    {DOCUMENT_KIND_LABELS[d.kind]}
                  </Badge>
                </td>
                <td className="p-3 text-xs text-muted-foreground">
                  {d.source_type}
                </td>
                <td className="p-3">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium",
                      STATUS_CLASSES[d.status],
                    )}
                  >
                    {isActive && <Loader2 className="size-3 animate-spin" />}
                    {d.status}
                  </span>
                </td>
                <td className="p-3 text-right tabular-nums text-muted-foreground">
                  {d.status === "parsed" ? d.chunk_count : "—"}
                </td>
                <td className="p-3 text-xs text-muted-foreground">
                  {new Date(d.created_at).toLocaleString()}
                </td>
                <td className="p-3 text-right">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      if (
                        window.confirm(
                          `Delete "${d.filename}"? This cannot be undone.`,
                        )
                      ) {
                        deleteMutation.mutate(d.id);
                      }
                    }}
                    disabled={deleteMutation.isPending}
                    aria-label={`Delete ${d.filename}`}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
