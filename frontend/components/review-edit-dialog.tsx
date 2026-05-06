"use client";

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api, ApiError, type RequirementRead } from "@/lib/api";
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
import { Textarea } from "@/components/ui/textarea";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: number;
  requirement: RequirementRead;
}

/**
 * Edit a single FRD's title + body.
 *
 * Backend semantics: any title/body change auto-demotes status to ``edited``
 * (an approved FRD that's edited returns to review and is removed from FAISS).
 * Re-approve when ready.
 */
export function ReviewEditDialog({
  open,
  onOpenChange,
  projectId,
  requirement,
}: Props) {
  const qc = useQueryClient();
  const [title, setTitle] = useState("");
  const [bodyMd, setBodyMd] = useState("");

  // Hydrate every time the dialog opens
  useEffect(() => {
    if (!open) return;
    setTitle(requirement.title);
    setBodyMd(requirement.body_md);
  }, [open, requirement]);

  const mutation = useMutation({
    mutationFn: () =>
      api.updateRequirement(projectId, requirement.id, {
        title: title.trim(),
        body_md: bodyMd,
      }),
    onSuccess: (updated) => {
      const msg =
        requirement.status === "approved"
          ? "Status moved to 'edited' — re-approve to add to FAISS."
          : "Status set to 'edited' — approve when ready.";
      toast.success(`${updated.code} updated`, { description: msg });
      qc.invalidateQueries({ queryKey: ["requirements", projectId] });
      qc.invalidateQueries({
        queryKey: ["requirement-detail", projectId, requirement.id],
      });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const detail = e instanceof ApiError ? e.message : e.message;
      toast.error("Update failed", { description: detail });
    },
  });

  const dirty =
    title.trim() !== requirement.title.trim() ||
    bodyMd !== requirement.body_md;
  const canSubmit = !!title.trim() && !!bodyMd.trim() && dirty && !mutation.isPending;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            Edit{" "}
            <span className="font-mono text-base">{requirement.code}</span>
          </DialogTitle>
          <DialogDescription>
            Refine the agent&apos;s draft. Editing moves status to{" "}
            <strong>edited</strong> regardless of where it was — approve again
            when you&apos;re happy.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="edit-title">Title</Label>
            <Input
              id="edit-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={512}
              autoFocus
            />
            <p className="text-xs text-muted-foreground">
              {title.length} / 512 characters
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="edit-body">
              Body{" "}
              <span className="font-normal text-muted-foreground">
                (Markdown)
              </span>
            </Label>
            <Textarea
              id="edit-body"
              value={bodyMd}
              onChange={(e) => setBodyMd(e.target.value)}
              rows={12}
              className="font-mono text-xs"
            />
            <p className="text-xs text-muted-foreground">
              {bodyMd.length.toLocaleString()} characters
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
            disabled={!canSubmit}
          >
            {mutation.isPending ? "Saving…" : "Save changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
