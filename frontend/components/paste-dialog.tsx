"use client";

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api, ApiError, type DocumentKind } from "@/lib/api";
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
  kind: DocumentKind;
}

export function PasteDialog({ open, onOpenChange, projectId, kind }: Props) {
  const qc = useQueryClient();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  // Reset form whenever the dialog opens
  useEffect(() => {
    if (open) {
      setTitle("");
      setContent("");
    }
  }, [open]);

  const mutation = useMutation({
    mutationFn: () =>
      api.pasteDocument(projectId, {
        kind,
        title: title.trim() || undefined,
        content,
      }),
    onSuccess: (doc) => {
      toast.success(`${doc.filename} queued`);
      qc.invalidateQueries({ queryKey: ["documents", projectId] });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Paste failed", { description: msg });
    },
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Paste {kind} content</DialogTitle>
          <DialogDescription>
            Paste markdown, plain text, or anything in between. Will be
            normalized as canonical Markdown.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="paste-title">
              Title{" "}
              <span className="font-normal text-muted-foreground">
                (optional)
              </span>
            </Label>
            <Input
              id="paste-title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Auth Module Spec"
              autoFocus
              maxLength={255}
            />
            <p className="text-xs text-muted-foreground">
              Prepended as <code className="rounded bg-muted px-1">{"# Title"}</code>
              . Skip if your content already has a top-level heading.
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="paste-content">Content</Label>
            <Textarea
              id="paste-content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="## Section&#10;&#10;Content here..."
              rows={12}
              className="font-mono text-xs"
            />
            <p className="text-xs text-muted-foreground">
              {content.length.toLocaleString()} characters
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
            disabled={mutation.isPending || !content.trim()}
          >
            {mutation.isPending ? "Submitting…" : `Submit as ${kind}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
