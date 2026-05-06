"use client";

import { useQuery } from "@tanstack/react-query";

import {
  api,
  DOCUMENT_KIND_LABELS,
  type DocumentRead,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface Props {
  projectId: number;
  selectedIds: number[];
  onChange: (next: number[]) => void;
}

/**
 * Multi-select checkbox list of project documents. Only ``parsed`` docs are
 * tickable (others have no chunks yet, so the agent can't use them).
 */
export function LinkedDocsPicker({
  projectId,
  selectedIds,
  onChange,
}: Props) {
  const { data: documents, isLoading } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => api.listDocuments(projectId),
  });

  const toggle = (id: number) => {
    if (selectedIds.includes(id)) {
      onChange(selectedIds.filter((d) => d !== id));
    } else {
      onChange([...selectedIds, id]);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <p className="text-sm italic text-muted-foreground">
        No documents in this project yet. Upload BRDs/FRDs/Instructions on the
        Documents tab to link them here. Plans run fine without docs — the
        instructions field below is enough.
      </p>
    );
  }

  return (
    <div className="space-y-1">
      {documents.map((d) => (
        <DocRow
          key={d.id}
          doc={d}
          checked={selectedIds.includes(d.id)}
          onToggle={() => toggle(d.id)}
        />
      ))}
    </div>
  );
}

function DocRow({
  doc,
  checked,
  onToggle,
}: {
  doc: DocumentRead;
  checked: boolean;
  onToggle: () => void;
}) {
  const isParsed = doc.status === "parsed";
  const label = `${doc.filename} (${DOCUMENT_KIND_LABELS[doc.kind]})`;

  return (
    <label
      className={cn(
        "flex items-center gap-3 rounded-md border p-2 transition-colors",
        checked && "border-primary/50 bg-accent/30",
        !isParsed && "opacity-60",
      )}
    >
      <input
        type="checkbox"
        className="size-4 cursor-pointer accent-primary"
        checked={checked}
        onChange={onToggle}
        disabled={!isParsed}
        aria-label={`Link ${label}`}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">{doc.filename}</span>
          <Badge variant="outline" className="text-[10px]">
            {DOCUMENT_KIND_LABELS[doc.kind]}
          </Badge>
        </div>
        {!isParsed && (
          <p className="text-xs text-muted-foreground">
            Status: <em>{doc.status}</em>
            {doc.status !== "failed" &&
              " — wait for ingest to finish to link this doc."}
          </p>
        )}
        {isParsed && (
          <p className="text-xs text-muted-foreground">
            {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"} ·{" "}
            {doc.char_count.toLocaleString()} chars
          </p>
        )}
      </div>
    </label>
  );
}
