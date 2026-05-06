"use client";

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { api, ApiError } from "@/lib/api";
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

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: number;
}

/**
 * Minimal create — just name + URL. Plan starts in `draft` status; the user
 * fills in scope, instructions, credentials, and document links from the
 * Plan editor page.
 */
export function PlanQuickCreateDialog({
  open,
  onOpenChange,
  projectId,
}: Props) {
  const router = useRouter();
  const qc = useQueryClient();

  const [name, setName] = useState("");
  const [targetUrl, setTargetUrl] = useState("");

  useEffect(() => {
    if (open) {
      setName("");
      setTargetUrl("");
    }
  }, [open]);

  const mutation = useMutation({
    mutationFn: () =>
      api.createPlan(projectId, {
        name: name.trim(),
        target_url: targetUrl.trim(),
      }),
    onSuccess: (plan) => {
      toast.success(`Plan "${plan.name}" created`);
      qc.invalidateQueries({ queryKey: ["plans", projectId] });
      onOpenChange(false);
      router.push(`/projects/${projectId}/plans/${plan.id}`);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Create failed", { description: msg });
    },
  });

  const canSubmit = !!name.trim() && !!targetUrl.trim();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New test plan</DialogTitle>
          <DialogDescription>
            Start with a name and a target URL. Scope, credentials, and linked
            documents fill in from the Plan editor — plans start in{" "}
            <strong>draft</strong> status.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="plan-name">Plan name</Label>
            <Input
              id="plan-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Auth + Dashboard smoke"
              autoFocus
              maxLength={255}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="plan-url">Target URL</Label>
            <Input
              id="plan-url"
              type="url"
              value={targetUrl}
              onChange={(e) => setTargetUrl(e.target.value)}
              placeholder="https://demo.example.com"
              maxLength={2048}
            />
            <p className="text-xs text-muted-foreground">
              The URL the agent will exercise during runs.
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={mutation.isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !canSubmit}
          >
            {mutation.isPending ? "Creating…" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
