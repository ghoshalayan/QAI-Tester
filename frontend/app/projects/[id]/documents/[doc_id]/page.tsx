"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  AlertCircle,
  ArrowLeft,
  Code,
  FileText,
  Layers,
  Loader2,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  api,
  DOCUMENT_KIND_LABELS,
  type ChunkRead,
  type DocumentRead,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useDocumentEvents } from "@/hooks/use-document-events";
import { cn } from "@/lib/utils";

const VIEWS = [
  { id: "rendered", label: "Rendered", icon: FileText },
  { id: "chunks", label: "Chunks", icon: Layers },
  { id: "raw", label: "Raw Markdown", icon: Code },
] as const;

type ViewId = (typeof VIEWS)[number]["id"];

export default function DocumentDetailPage() {
  const params = useParams<{ id: string; doc_id: string }>();
  const projectId = Number(params.id);
  const docId = Number(params.doc_id);
  const [view, setView] = useState<ViewId>("rendered");

  // Live events flow into the documents query cache; we mirror the doc's
  // status by reading from there too via a dedicated query.
  useDocumentEvents(projectId);

  const { data: doc, isLoading: docLoading } = useQuery({
    queryKey: ["document", projectId, docId],
    queryFn: () => api.getDocument(projectId, docId),
    enabled: !Number.isNaN(projectId) && !Number.isNaN(docId),
    // Re-fetch on focus so a freshly-completed ingest reflects here too
    refetchOnWindowFocus: true,
  });

  const isParsed = doc?.status === "parsed";

  const { data: parsed } = useQuery({
    queryKey: ["document-parsed", projectId, docId],
    queryFn: () => api.getParsedMd(projectId, docId),
    enabled: isParsed,
  });

  const { data: chunks } = useQuery({
    queryKey: ["document-chunks", projectId, docId],
    queryFn: () => api.listChunks(projectId, docId),
    enabled: isParsed,
  });

  if (docLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-9 w-1/2" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (!doc) {
    return (
      <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
        Document not found.
        <div className="mt-3">
          <Link
            href={`/projects/${projectId}/documents`}
            className="inline-flex items-center gap-1 text-primary hover:underline"
          >
            <ArrowLeft className="size-3" /> Back to documents
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Link
        href={`/projects/${projectId}/documents`}
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3" /> Back to documents
      </Link>

      <DocumentHeader doc={doc} />

      {!isParsed ? (
        <NotParsedBanner doc={doc} />
      ) : (
        <>
          <nav className="border-b">
            <div className="flex gap-1">
              {VIEWS.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setView(id)}
                  className={cn(
                    "-mb-px flex items-center gap-2 border-b-2 px-4 py-2 text-sm font-medium transition-colors",
                    view === id
                      ? "border-primary text-foreground"
                      : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
                  )}
                >
                  <Icon className="size-4" />
                  {label}
                </button>
              ))}
            </div>
          </nav>

          {view === "rendered" && (
            <RenderedView markdown={parsed?.parsed_md ?? ""} />
          )}
          {view === "chunks" && <ChunksView chunks={chunks ?? []} />}
          {view === "raw" && <RawView markdown={parsed?.parsed_md ?? ""} />}
        </>
      )}
    </div>
  );
}

function DocumentHeader({ doc }: { doc: DocumentRead }) {
  return (
    <header className="space-y-2">
      <h1 className="text-2xl font-semibold tracking-tight">{doc.filename}</h1>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge variant="outline">{DOCUMENT_KIND_LABELS[doc.kind]}</Badge>
        <Badge variant="outline">{doc.source_type}</Badge>
        <span className="text-muted-foreground">·</span>
        <span className="text-muted-foreground">
          {doc.char_count.toLocaleString()} chars
        </span>
        <span className="text-muted-foreground">·</span>
        <span className="text-muted-foreground">
          {doc.chunk_count} chunks
        </span>
        <span className="text-muted-foreground">·</span>
        <span className="text-muted-foreground">
          Created {new Date(doc.created_at).toLocaleString()}
        </span>
      </div>
    </header>
  );
}

function NotParsedBanner({ doc }: { doc: DocumentRead }) {
  if (doc.status === "failed") {
    return (
      <Card className="border-destructive/40 bg-destructive/5 p-6">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 size-5 shrink-0 text-destructive" />
          <div className="min-w-0">
            <p className="font-medium">Ingest failed</p>
            <p className="mt-1 break-words text-sm text-muted-foreground">
              {doc.error_message ?? "Unknown error"}
            </p>
          </div>
        </div>
      </Card>
    );
  }
  return (
    <Card className="p-6">
      <div className="flex items-center gap-3">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
        <div>
          <p className="font-medium capitalize">{doc.status}…</p>
          <p className="text-sm text-muted-foreground">
            Rendered view appears once ingest completes.
          </p>
        </div>
      </div>
    </Card>
  );
}

function RenderedView({ markdown }: { markdown: string }) {
  if (!markdown) {
    return (
      <p className="text-sm text-muted-foreground">No content.</p>
    );
  }
  return (
    <article className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
    </article>
  );
}

function ChunksView({ chunks }: { chunks: ChunkRead[] }) {
  if (!chunks.length) {
    return (
      <p className="text-sm text-muted-foreground">No chunks emitted.</p>
    );
  }
  return (
    <div className="space-y-3">
      {chunks.map((c) => (
        <Card key={c.id} className="p-4">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <div className="min-w-0">
              <span className="text-xs font-medium text-muted-foreground">
                #{c.ordinal + 1}
              </span>
              <span className="ml-2 text-sm font-medium">
                {c.heading_path || (
                  <span className="text-muted-foreground italic">
                    (no heading)
                  </span>
                )}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {c.anchor && (
                <code className="rounded bg-muted px-1 py-0.5 font-mono">
                  {c.anchor}
                </code>
              )}
              <span>{c.char_count} chars</span>
            </div>
          </div>
          <pre className="whitespace-pre-wrap rounded-md bg-muted/40 p-3 text-xs leading-relaxed">
            {c.text}
          </pre>
        </Card>
      ))}
    </div>
  );
}

function RawView({ markdown }: { markdown: string }) {
  return (
    <pre className="overflow-x-auto rounded-md border bg-muted/40 p-4 text-xs leading-relaxed">
      {markdown}
    </pre>
  );
}
