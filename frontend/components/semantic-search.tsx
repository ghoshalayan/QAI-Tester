"use client";

import { type FormEvent, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Loader2, Search as SearchIcon } from "lucide-react";

import {
  api,
  DOCUMENT_KIND_LABELS,
  type DocumentKind,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

type KindFilter = DocumentKind | "ALL";

const FILTER_OPTIONS: KindFilter[] = ["ALL", "BRD", "FRD", "INSTRUCTIONS"];
const K_OPTIONS = [5, 10, 20];

export function SemanticSearch() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);

  const [draft, setDraft] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [kindFilter, setKindFilter] = useState<KindFilter>("ALL");
  const [k, setK] = useState(10);

  const { data, isFetching, isError, error } = useQuery({
    queryKey: ["search", projectId, submitted, kindFilter, k],
    queryFn: () =>
      api.searchDocuments(projectId, {
        query: submitted,
        k,
        kind: kindFilter === "ALL" ? undefined : kindFilter,
      }),
    enabled: !!submitted.trim(),
    staleTime: 60_000, // identical query within a minute is cached
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const q = draft.trim();
    if (q) setSubmitted(q);
  };

  const clear = () => {
    setDraft("");
    setSubmitted("");
  };

  return (
    <Card className="p-4">
      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <SearchIcon className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Semantic search across all chunks…"
              className="pl-9"
            />
          </div>
          <Button type="submit" disabled={!draft.trim() || isFetching}>
            {isFetching ? (
              <>
                <Loader2 className="size-4 animate-spin" /> Searching…
              </>
            ) : (
              "Search"
            )}
          </Button>
          {submitted && (
            <Button type="button" variant="ghost" onClick={clear}>
              Clear
            </Button>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          <span className="text-muted-foreground">Filter:</span>
          <div className="inline-flex rounded-md border bg-background p-0.5">
            {FILTER_OPTIONS.map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => setKindFilter(opt)}
                className={cn(
                  "rounded-sm px-2.5 py-0.5 font-medium transition-colors",
                  kindFilter === opt
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {opt === "ALL" ? "Any" : DOCUMENT_KIND_LABELS[opt]}
              </button>
            ))}
          </div>

          <span className="ml-2 text-muted-foreground">k =</span>
          <div className="inline-flex rounded-md border bg-background p-0.5">
            {K_OPTIONS.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setK(n)}
                className={cn(
                  "rounded-sm px-2.5 py-0.5 font-medium transition-colors",
                  k === n
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
      </form>

      {submitted && (
        <ResultsPanel
          projectId={projectId}
          isFetching={isFetching}
          isError={isError}
          errorMessage={(error as Error | null)?.message}
          hits={data?.hits ?? []}
        />
      )}
    </Card>
  );
}

function ResultsPanel({
  projectId,
  isFetching,
  isError,
  errorMessage,
  hits,
}: {
  projectId: number;
  isFetching: boolean;
  isError: boolean;
  errorMessage?: string;
  hits: Array<{
    chunk_id: number;
    document_id: number;
    document_kind: DocumentKind;
    document_filename: string;
    heading_path: string | null;
    anchor: string | null;
    text: string;
    score: number;
  }>;
}) {
  if (isError) {
    return (
      <p className="mt-4 text-sm text-destructive">
        Search failed: {errorMessage ?? "unknown error"}
      </p>
    );
  }
  if (isFetching && !hits.length) {
    return (
      <p className="mt-4 text-sm text-muted-foreground">Searching…</p>
    );
  }
  if (!hits.length) {
    return (
      <p className="mt-4 text-sm text-muted-foreground">
        No matching chunks. Try broader terms, or remove the kind filter.
      </p>
    );
  }
  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        {hits.length} result{hits.length === 1 ? "" : "s"}
      </p>
      {hits.map((h) => {
        const pct = Math.round(h.score * 100);
        return (
          <Link
            key={h.chunk_id}
            href={`/projects/${projectId}/documents/${h.document_id}`}
            className="block rounded-md border p-3 text-sm transition-colors hover:border-primary/50 hover:bg-accent/30"
          >
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
              <div className="min-w-0">
                <span className="font-medium">{h.document_filename}</span>
                <Badge variant="outline" className="ml-2">
                  {DOCUMENT_KIND_LABELS[h.document_kind]}
                </Badge>
                {h.heading_path && (
                  <span className="ml-2 text-xs text-muted-foreground">
                    · {h.heading_path}
                  </span>
                )}
              </div>
              <span
                className={cn(
                  "shrink-0 rounded-full px-2 py-0.5 text-xs font-medium tabular-nums",
                  pct >= 70
                    ? "bg-green-500/10 text-green-700 dark:text-green-400"
                    : pct >= 50
                      ? "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400"
                      : "bg-muted text-muted-foreground",
                )}
              >
                {pct}%
              </span>
            </div>
            <p className="line-clamp-3 text-xs leading-relaxed text-muted-foreground">
              {h.text}
            </p>
          </Link>
        );
      })}
    </div>
  );
}
