"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api, ApiError, type AgentRunRead } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: number;
  run: AgentRunRead;
}

export function DeleteRunDialog({
  open,
  onOpenChange,
  projectId,
  run,
}: Props) {
  const qc = useQueryClient();

  const mutation = useMutation({
    mutationFn: () => api.deleteAgentRun(projectId, run.id),
    onSuccess: () => {
      toast.success(`Run #${run.id} deleted`);
      qc.invalidateQueries({ queryKey: ["agent-runs", projectId] });
      qc.removeQueries({ queryKey: ["agent-run", projectId, run.id] });
      qc.removeQueries({ queryKey: ["run-steps", projectId, run.id] });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Delete failed", { description: msg });
    },
  });

  const isActive =
    run.status === "queued" ||
    run.status === "running" ||
    run.status === "paused";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete run #{run.id}?</DialogTitle>
          <DialogDescription>
            {isActive ? (
              <>
                This run is still <strong>{run.status}</strong>. Cancel it
                first from the run-detail page; only terminal runs can be
                deleted.
              </>
            ) : (
              <>
                Permanently delete this run, all its step rows, and any
                screenshots on disk. This cannot be undone.
              </>
            )}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || isActive}
          >
            {mutation.isPending ? "Deleting…" : "Delete run"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
