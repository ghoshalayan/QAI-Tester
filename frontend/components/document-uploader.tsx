"use client";

import { useCallback, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { ClipboardPaste, FileText, Upload } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  DOCUMENT_KIND_LABELS,
  type DocumentKind,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

import { PasteDialog } from "./paste-dialog";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx", ".md", ".markdown"] as const;
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(",");

export function DocumentUploader() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);
  const qc = useQueryClient();

  const [kind, setKind] = useState<DocumentKind>("BRD");
  const [dragActive, setDragActive] = useState(false);
  const [pasteOpen, setPasteOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadDocument(projectId, kind, file),
    onSuccess: (doc) => {
      toast.success(`${doc.filename} queued`, {
        description: `${doc.kind} · ${doc.source_type}`,
      });
      qc.invalidateQueries({ queryKey: ["documents", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Upload failed", { description: msg });
    },
  });

  const handleFile = useCallback(
    (file: File) => {
      const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
      if (!ACCEPTED_EXTENSIONS.includes(ext as (typeof ACCEPTED_EXTENSIONS)[number])) {
        toast.error(`Unsupported file type: ${ext || "(no extension)"}`, {
          description: `Allowed: ${ACCEPTED_EXTENSIONS.join(", ")}`,
        });
        return;
      }
      uploadMutation.mutate(file);
    },
    [uploadMutation],
  );

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    Array.from(e.dataTransfer.files).forEach(handleFile);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    // Only clear if the cursor actually left the dropzone bounds
    const rect = e.currentTarget.getBoundingClientRect();
    const { clientX, clientY } = e;
    if (
      clientX < rect.left ||
      clientX > rect.right ||
      clientY < rect.top ||
      clientY > rect.bottom
    ) {
      setDragActive(false);
    }
  };

  return (
    <>
      <Card className="p-6">
        {/* Kind selector */}
        <div className="mb-2 flex flex-wrap items-center gap-3">
          <span className="text-sm text-muted-foreground">Document type:</span>
          <div className="inline-flex rounded-md border bg-background p-0.5">
            {(["BRD", "FRD", "INSTRUCTIONS"] as const).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setKind(k)}
                className={cn(
                  "rounded-sm px-3 py-1 text-xs font-medium transition-colors",
                  kind === k
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {DOCUMENT_KIND_LABELS[k]}
              </button>
            ))}
          </div>
        </div>
        <p className="mb-4 text-xs text-muted-foreground">
          {kind === "INSTRUCTIONS"
            ? "Direct test-case instructions — skips the BRD→FRD analysis and feeds straight into test-case generation."
            : kind === "BRD"
              ? "Business requirements — the agent will derive an FRD from this in week 3."
              : "Functional requirements — the agent will derive test cases from this in week 4."}
        </p>

        {/* Drop zone */}
        <div
          onDragEnter={handleDragOver}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "rounded-lg border-2 border-dashed p-12 text-center transition-colors",
            dragActive
              ? "border-primary bg-accent"
              : "border-input hover:border-muted-foreground/40",
          )}
        >
          <FileText className="mx-auto size-10 text-muted-foreground" />
          <p className="mt-3 text-sm text-muted-foreground">
            Drop a <strong>PDF</strong>, <strong>DOCX</strong>, or{" "}
            <strong>Markdown</strong> file
          </p>
          <p className="mt-1 text-xs text-muted-foreground">— or —</p>
          <div className="mt-4 flex justify-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadMutation.isPending}
            >
              <Upload className="size-4" />
              {uploadMutation.isPending ? "Uploading…" : "Choose file"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPasteOpen(true)}
            >
              <ClipboardPaste className="size-4" /> Paste text
            </Button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT_ATTR}
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
              // Reset so the same file can be re-selected
              e.target.value = "";
            }}
          />
        </div>
      </Card>

      <PasteDialog
        open={pasteOpen}
        onOpenChange={setPasteOpen}
        projectId={projectId}
        kind={kind}
      />
    </>
  );
}
