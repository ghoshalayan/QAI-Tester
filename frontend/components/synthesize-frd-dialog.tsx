"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Sparkles } from "lucide-react";
import { toast } from "sonner";

import { api, ApiError, type DocumentRead } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: number;
}

const DEFAULT_CAP = 50;
const MIN_CAP = 1;
const MAX_CAP = 200;

export function SynthesizeFrdDialog({ open, onOpenChange, projectId }: Props) {
  const qc = useQueryClient();
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>([]);
  const [capChunks, setCapChunks] = useState<number>(DEFAULT_CAP);

  // Fresh state every time the dialog opens
  useEffect(() => {
    if (open) {
      setSelectedDocIds([]);
      setCapChunks(DEFAULT_CAP);
    }
  }, [open]);

  const { data: documents, isLoading } = useQuery({
    queryKey: ["documents", projectId],
    queryFn: () => api.listDocuments(projectId),
    enabled: open,
  });

  const allBrds = (documents ?? []).filter((d) => d.kind === "BRD");
  const eligibleBrds = allBrds.filter((d) => d.status === "parsed");
  const ineligibleBrds = allBrds.filter((d) => d.status !== "parsed");

  const toggle = (id: number) => {
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id],
    );
  };

  const selectAllEligible = () => {
    setSelectedDocIds(eligibleBrds.map((d) => d.id));
  };

  const startMutation = useMutation({
    mutationFn: () =>
      api.startBrdToFrd(projectId, {
        source_document_ids: selectedDocIds,
        cap_chunks: capChunks,
      }),
    onSuccess: (run) => {
      toast.success("Synthesis run queued", {
        description: `Run #${run.id} — progress streams on the Requirements tab.`,
      });
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.invalidateQueries({ queryKey: ["requirements", projectId] });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Failed to start synthesis", { description: msg });
    },
  });

  const totalChunksSelected = eligibleBrds
    .filter((d) => selectedDocIds.includes(d.id))
    .reduce((sum, d) => sum + d.chunk_count, 0);
  const willTruncate = totalChunksSelected > capChunks;

  const canSubmit =
    selectedDocIds.length > 0 &&
    capChunks >= MIN_CAP &&
    capChunks <= MAX_CAP &&
    !startMutation.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="size-5" /> Synthesize FRDs
          </DialogTitle>
          <DialogDescription>
            Pick BRD documents the agent should read. It will derive functional
            requirements with traceability back to the BRD chunks that motivated
            each one. The run executes in the background; you can review and
            approve results as they arrive.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Source BRDs */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Source BRDs</Label>
              {eligibleBrds.length > 1 && (
                <button
                  type="button"
                  onClick={selectAllEligible}
                  className="text-xs text-primary hover:underline"
                >
                  Select all ({eligibleBrds.length})
                </button>
              )}
            </div>

            {isLoading ? (
              <p className="text-sm text-muted-foreground">
                Loading documents…
              </p>
            ) : eligibleBrds.length === 0 ? (
              <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                {allBrds.length === 0 ? (
                  <>
                    No BRD documents in this project. Upload one on the{" "}
                    <strong>Documents</strong> tab first — set its kind to
                    BRD.
                  </>
                ) : (
                  <>
                    BRDs are still ingesting. Wait until status is{" "}
                    <strong>parsed</strong> to use them.
                  </>
                )}
              </div>
            ) : (
              <div className="max-h-[280px] space-y-1 overflow-y-auto">
                {eligibleBrds.map((d) => (
                  <BrdRow
                    key={d.id}
                    doc={d}
                    checked={selectedDocIds.includes(d.id)}
                    onToggle={() => toggle(d.id)}
                  />
                ))}
                {ineligibleBrds.length > 0 && (
                  <p className="pt-2 text-xs text-muted-foreground">
                    {ineligibleBrds.length} BRD
                    {ineligibleBrds.length === 1 ? "" : "s"} not yet parsed —
                    not selectable.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Cap chunks */}
          <div className="space-y-2">
            <Label htmlFor="cap-chunks">
              Max chunks per run{" "}
              <span className="font-normal text-muted-foreground">
                (default {DEFAULT_CAP})
              </span>
            </Label>
            <div className="flex items-start gap-3">
              <Input
                id="cap-chunks"
                type="number"
                min={MIN_CAP}
                max={MAX_CAP}
                value={capChunks}
                onChange={(e) => {
                  const n = parseInt(e.target.value, 10);
                  if (Number.isNaN(n)) return;
                  setCapChunks(Math.max(MIN_CAP, Math.min(MAX_CAP, n)));
                }}
                className="w-28"
              />
              <p className="text-xs text-muted-foreground">
                Hard cap on how many BRD chunks the LLM sees in one call.
                Reduce for huge BRDs or to control cost; raise for thorough
                coverage. Range {MIN_CAP}–{MAX_CAP}.
              </p>
            </div>
          </div>

          {/* Selection summary */}
          {selectedDocIds.length > 0 && (
            <div
              className={cn(
                "rounded-md border p-3 text-xs",
                willTruncate
                  ? "border-yellow-500/40 bg-yellow-500/5 text-yellow-700 dark:text-yellow-400"
                  : "bg-muted/30 text-muted-foreground",
              )}
            >
              Selected <strong>{selectedDocIds.length}</strong> BRD
              {selectedDocIds.length === 1 ? "" : "s"} ·{" "}
              <strong>{totalChunksSelected}</strong> total chunks
              {willTruncate ? (
                <>
                  {" "}
                  · ⚠ will be truncated to first <strong>{capChunks}</strong>{" "}
                  chunks. Raise the cap to cover more.
                </>
              ) : (
                <> · all fit under the cap of {capChunks}.</>
              )}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={startMutation.isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={() => startMutation.mutate()}
            disabled={!canSubmit}
          >
            <Sparkles className="size-4" />
            {startMutation.isPending ? "Queueing…" : "Start synthesis"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function BrdRow({
  doc,
  checked,
  onToggle,
}: {
  doc: DocumentRead;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-3 rounded-md border p-2.5 transition-colors",
        checked
          ? "border-primary/50 bg-accent/30"
          : "hover:bg-accent/20",
      )}
    >
      <input
        type="checkbox"
        className="size-4 cursor-pointer accent-primary"
        checked={checked}
        onChange={onToggle}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium">{doc.filename}</p>
        <p className="text-xs text-muted-foreground">
          {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"} ·{" "}
          {doc.char_count.toLocaleString()} chars · {doc.source_type}
        </p>
      </div>
    </label>
  );
}
