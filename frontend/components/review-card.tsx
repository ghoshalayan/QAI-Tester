"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  ChevronDown,
  ChevronUp,
  FileText,
  Pencil,
  Trash2,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast } from "sonner";

import {
  api,
  ApiError,
  REQUIREMENT_STATUS_LABELS,
  type RequirementDetail,
  type RequirementRead,
  type RequirementStatus,
  type SourceChunkRef,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ReviewEditDialog } from "@/components/review-edit-dialog";
import { cn } from "@/lib/utils";

const STATUS_CLASSES: Record<RequirementStatus, string> = {
  proposed: "bg-muted text-muted-foreground",
  edited: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
  approved: "bg-green-500/10 text-green-700 dark:text-green-400",
  rejected: "bg-red-500/10 text-red-700 dark:text-red-400",
};

interface Props {
  projectId: number;
  requirement: RequirementRead;
}

export function ReviewCard({ projectId, requirement: req }: Props) {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [editOpen, setEditOpen] = useState(false);

  // Fetch full detail (includes source_chunks) only when user expands
  const { data: detail, isLoading: detailLoading } = useQuery({
    queryKey: ["requirement-detail", projectId, req.id],
    queryFn: () => api.getRequirement(projectId, req.id),
    enabled: expanded,
    staleTime: 30_000,
  });

  const statusMutation = useMutation({
    mutationFn: (status: RequirementStatus) =>
      api.updateRequirement(projectId, req.id, { status }),
    onSuccess: (updated) => {
      const verb =
        updated.status === "approved"
          ? "approved"
          : updated.status === "rejected"
            ? "rejected"
            : updated.status;
      toast.success(`${updated.code} ${verb}`);
      qc.invalidateQueries({ queryKey: ["requirements", projectId] });
      qc.invalidateQueries({
        queryKey: ["requirement-detail", projectId, req.id],
      });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Update failed", { description: msg });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteRequirement(projectId, req.id),
    onSuccess: () => {
      toast.success(`${req.code} deleted`);
      qc.invalidateQueries({ queryKey: ["requirements", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Delete failed", { description: msg });
    },
  });

  const isApproved = req.status === "approved";
  const isRejected = req.status === "rejected";
  const isPending = statusMutation.isPending || deleteMutation.isPending;

  return (
    <>
      <Card
        className={cn(
          "transition-colors",
          isApproved && "border-green-500/30",
          isRejected && "opacity-60",
        )}
      >
        <div className="p-4">
          <div className="flex items-start justify-between gap-3">
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="flex min-w-0 flex-1 items-start gap-2 text-left"
              aria-expanded={expanded}
              aria-controls={`req-${req.id}-details`}
            >
              {expanded ? (
                <ChevronUp className="mt-1 size-4 shrink-0 text-muted-foreground" />
              ) : (
                <ChevronDown className="mt-1 size-4 shrink-0 text-muted-foreground" />
              )}
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-2">
                  <span className="font-mono text-xs text-muted-foreground">
                    {req.code}
                  </span>
                  <span className="font-medium">{req.title}</span>
                </div>
                {!expanded && (
                  <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                    {req.body_md}
                  </p>
                )}
              </div>
            </button>
            <div className="flex shrink-0 flex-col items-end gap-1">
              <span
                className={cn(
                  "rounded-full px-2.5 py-0.5 text-xs font-medium",
                  STATUS_CLASSES[req.status],
                )}
              >
                {REQUIREMENT_STATUS_LABELS[req.status]}
              </span>
              {req.confidence != null && (
                <ConfidenceLabel value={req.confidence} />
              )}
            </div>
          </div>

          {expanded && (
            <div
              id={`req-${req.id}-details`}
              className="mt-4 space-y-4 border-t pt-4"
            >
              <article className="markdown-body text-sm">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {req.body_md}
                </ReactMarkdown>
              </article>

              {req.rationale && (
                <div className="rounded-md bg-muted/30 p-3 text-xs">
                  <p className="mb-1 font-medium uppercase tracking-wide text-muted-foreground">
                    Why this FRD?
                  </p>
                  <p className="text-foreground">{req.rationale}</p>
                </div>
              )}

              <SourceChunksSection
                detail={detail}
                isLoading={detailLoading}
                fallbackCount={req.source_chunk_ids.length}
              />

              <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                {req.embedding_id != null && (
                  <span className="text-green-700 dark:text-green-400">
                    ✓ in FAISS
                  </span>
                )}
                <span>
                  Created {new Date(req.created_at).toLocaleString()}
                </span>
                {req.reviewed_at && (
                  <span>
                    Reviewed {new Date(req.reviewed_at).toLocaleString()}
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-2 border-t pt-3">
            <Button
              size="sm"
              variant={isApproved ? "default" : "outline"}
              onClick={() => statusMutation.mutate("approved")}
              disabled={isPending || isApproved}
            >
              <Check className="size-4" />
              {isApproved ? "Approved" : "Approve"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setEditOpen(true)}
              disabled={isPending}
            >
              <Pencil className="size-4" /> Edit
            </Button>
            <Button
              size="sm"
              variant={isRejected ? "destructive" : "outline"}
              onClick={() => statusMutation.mutate("rejected")}
              disabled={isPending || isRejected}
            >
              <X className="size-4" />
              {isRejected ? "Rejected" : "Reject"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                if (
                  window.confirm(
                    `Delete ${req.code} (${req.title.slice(0, 80)})? This cannot be undone.`,
                  )
                ) {
                  deleteMutation.mutate();
                }
              }}
              disabled={isPending}
              className="ml-auto"
              aria-label={`Delete ${req.code}`}
            >
              <Trash2 className="size-4" />
            </Button>
          </div>
        </div>
      </Card>

      <ReviewEditDialog
        open={editOpen}
        onOpenChange={setEditOpen}
        projectId={projectId}
        requirement={req}
      />
    </>
  );
}

function ConfidenceLabel({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const cls =
    pct >= 80
      ? "text-green-700 dark:text-green-400"
      : pct >= 50
        ? "text-yellow-700 dark:text-yellow-400"
        : "text-muted-foreground";
  return <span className={cn("text-xs", cls)}>{pct}% conf.</span>;
}

function SourceChunksSection({
  detail,
  isLoading,
  fallbackCount,
}: {
  detail: RequirementDetail | undefined;
  isLoading: boolean;
  fallbackCount: number;
}) {
  if (isLoading || !detail) {
    return (
      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Source BRD chunks ({fallbackCount})
        </p>
        <Skeleton className="h-16 w-full" />
      </div>
    );
  }

  if (detail.source_chunks.length === 0) {
    return (
      <p className="text-xs italic text-muted-foreground">
        No source chunks {fallbackCount > 0 && "(BRD likely deleted)"}.
      </p>
    );
  }

  return (
    <div>
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Source BRD chunks ({detail.source_chunks.length})
      </p>
      <div className="space-y-2">
        {detail.source_chunks.map((c) => (
          <SourceChunkCard key={c.chunk_id} chunk={c} />
        ))}
      </div>
    </div>
  );
}

function SourceChunkCard({ chunk }: { chunk: SourceChunkRef }) {
  return (
    <div className="rounded-md border bg-muted/30 p-3 text-xs">
      <div className="mb-1.5 flex flex-wrap items-center gap-2 text-muted-foreground">
        <FileText className="size-3 shrink-0" />
        <span className="font-medium">{chunk.document_filename}</span>
        {chunk.heading_path && (
          <>
            <span>·</span>
            <span className="truncate">{chunk.heading_path}</span>
          </>
        )}
        <span className="ml-auto shrink-0">{chunk.char_count} chars</span>
      </div>
      <pre className="whitespace-pre-wrap font-sans leading-relaxed text-foreground">
        {chunk.text}
      </pre>
    </div>
  );
}
