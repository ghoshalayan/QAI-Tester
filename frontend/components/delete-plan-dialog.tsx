"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { api, ApiError, type PlanReadDetail } from "@/lib/api";
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
  plan: PlanReadDetail;
  /** Where to navigate after a successful delete. */
  redirectTo?: string;
}

export function DeletePlanDialog({
  open,
  onOpenChange,
  projectId,
  plan,
  redirectTo,
}: Props) {
  const router = useRouter();
  const qc = useQueryClient();

  const mutation = useMutation({
    mutationFn: () => api.deletePlan(projectId, plan.id),
    onSuccess: () => {
      toast.success(`Plan "${plan.name}" deleted`);
      qc.invalidateQueries({ queryKey: ["plans", projectId] });
      qc.removeQueries({ queryKey: ["plan", projectId, plan.id] });
      onOpenChange(false);
      if (redirectTo) router.push(redirectTo);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Delete failed", { description: msg });
    },
  });

  const credCount = plan.credentials.length;
  const docCount = plan.linked_documents.length;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete plan?</DialogTitle>
          <DialogDescription>
            Permanently delete <strong>{plan.name}</strong>. This removes{" "}
            {credCount} credential{credCount === 1 ? "" : "s"} and unlinks{" "}
            {docCount} document{docCount === 1 ? "" : "s"} (the docs themselves
            stay). This cannot be undone.
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
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "Deleting…" : "Delete plan"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
