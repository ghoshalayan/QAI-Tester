"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { Lightbulb } from "lucide-react";

import { DocumentList } from "@/components/document-list";
import { DocumentUploader } from "@/components/document-uploader";
import { SemanticSearch } from "@/components/semantic-search";

export default function DocumentsTabPage() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);

  return (
    <div className="space-y-6">
      <PlansCrossLink projectId={projectId} />
      <SemanticSearch />
      <DocumentUploader />
      <div>
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">
          Documents
        </h2>
        <DocumentList />
      </div>
    </div>
  );
}

/**
 * Tiny callout pointing first-time users to the Plans tab if they don't
 * actually have BRDs/FRDs to upload — they can run with just instructions
 * + credentials.
 */
function PlansCrossLink({ projectId }: { projectId: number }) {
  return (
    <div className="flex items-start gap-3 rounded-md border border-blue-500/30 bg-blue-500/5 p-3 text-sm">
      <Lightbulb className="mt-0.5 size-4 shrink-0 text-blue-600 dark:text-blue-400" />
      <div className="flex-1">
        <strong>Don&apos;t have a BRD/FRD?</strong>{" "}
        Skip docs and go straight to{" "}
        <Link
          href={`/projects/${projectId}/plans`}
          className="font-medium text-primary underline underline-offset-2"
        >
          Plans
        </Link>{" "}
        — a plan only needs a target URL + login credentials + a few module
        names + free-text instructions. No upload required.
      </div>
    </div>
  );
}
