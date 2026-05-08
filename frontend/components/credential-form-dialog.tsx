"use client";

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { api, ApiError, type CredentialRead } from "@/lib/api";
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
  planId: number;
  /** When provided, dialog is in edit mode. */
  credential?: CredentialRead;
}

export function CredentialFormDialog({
  open,
  onOpenChange,
  projectId,
  planId,
  credential,
}: Props) {
  const qc = useQueryClient();
  const isEdit = !!credential;

  const [label, setLabel] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  // Phase 3 — TOTP seed (base32 OR otpauth:// URI). Empty = none.
  const [totpSecret, setTotpSecret] = useState("");
  const [urlPattern, setUrlPattern] = useState("");
  const [usernameHint, setUsernameHint] = useState("");
  const [passwordHint, setPasswordHint] = useState("");
  const [notes, setNotes] = useState("");

  // Reset / hydrate whenever the dialog opens
  useEffect(() => {
    if (!open) return;
    setLabel(credential?.label ?? "");
    setUsername(credential?.username ?? "");
    setPassword(""); // never pre-populate
    setTotpSecret(""); // never pre-populate (server returns totp_set boolean only)
    setUrlPattern(credential?.url_pattern ?? "");
    setUsernameHint(credential?.username_selector_hint ?? "");
    setPasswordHint(credential?.password_selector_hint ?? "");
    setNotes(credential?.notes ?? "");
  }, [open, credential]);

  const mutation = useMutation({
    mutationFn: () => {
      if (isEdit && credential) {
        return api.updateCredential(projectId, planId, credential.id, {
          label: label.trim(),
          username: username.trim(),
          password: password || undefined, // empty → keep existing
          // Empty string = explicit "clear TOTP"; undefined = preserve.
          // We only send the field when the user typed something OR
          // explicitly cleared a previously-set seed via the toggle.
          totp_secret: totpSecret.trim() ? totpSecret.trim() : undefined,
          url_pattern: urlPattern || undefined,
          username_selector_hint: usernameHint || undefined,
          password_selector_hint: passwordHint || undefined,
          notes: notes || undefined,
        });
      }
      return api.createCredential(projectId, planId, {
        label: label.trim(),
        username: username.trim(),
        password,
        totp_secret: totpSecret.trim() || undefined,
        url_pattern: urlPattern || undefined,
        username_selector_hint: usernameHint || undefined,
        password_selector_hint: passwordHint || undefined,
        notes: notes || undefined,
      });
    },
    onSuccess: () => {
      toast.success(isEdit ? "Credential updated" : "Credential added");
      qc.invalidateQueries({ queryKey: ["plan", projectId, planId] });
      qc.invalidateQueries({ queryKey: ["plans", projectId] });
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error(isEdit ? "Update failed" : "Create failed", {
        description: msg,
      });
    },
  });

  const passwordOnFile = isEdit && credential?.password_set;
  const canSubmit =
    !!label.trim() &&
    !!username.trim() &&
    (isEdit ? true : !!password); // password required only on create

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>
            {isEdit ? "Edit credential" : "Add credential"}
          </DialogTitle>
          <DialogDescription>
            Stored in plaintext on this machine per the local-MVP policy. The
            agent will use the right credential based on{" "}
            <code className="rounded bg-muted px-1">URL pattern</code> when
            multiple are defined.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label htmlFor="cred-label">Label</Label>
              <Input
                id="cred-label"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="admin / user1 / readonly"
                autoFocus
                maxLength={64}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="cred-url-pattern">
                URL pattern{" "}
                <span className="font-normal text-muted-foreground">
                  (optional)
                </span>
              </Label>
              <Input
                id="cred-url-pattern"
                value={urlPattern}
                onChange={(e) => setUrlPattern(e.target.value)}
                placeholder="/admin (defaults to plan target)"
                maxLength={2048}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="cred-username">Username / email</Label>
            <Input
              id="cred-username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="admin@example.com"
              autoComplete="off"
              maxLength={512}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="cred-password">Password</Label>
            <Input
              id="cred-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={
                passwordOnFile
                  ? "•••••••••••••••• (leave empty to keep)"
                  : "Required"
              }
              autoComplete="new-password"
              maxLength={512}
            />
            {passwordOnFile && (
              <p className="text-xs text-muted-foreground">
                ✓ Password on file — leave empty to keep, or type a new one to
                replace.
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="cred-totp">
              TOTP secret (optional — auto-generates 2FA codes)
            </Label>
            <Input
              id="cred-totp"
              type="password"
              value={totpSecret}
              onChange={(e) => setTotpSecret(e.target.value)}
              placeholder={
                credential?.totp_set
                  ? "•••••••••••••••• (leave empty to keep)"
                  : "JBSWY3DPEHPK3PXP  or  otpauth://totp/..."
              }
              autoComplete="off"
              maxLength={512}
            />
            <p className="text-xs text-muted-foreground">
              Paste a base32 seed or full <code>otpauth://</code> URI from
              the QR code provisioning page. When set, the agent
              generates 2FA codes automatically and never prompts you for
              an OTP. Leave empty when the site uses SMS / email / push
              2FA — those will fall back to a HITL prompt during the run.
              <br />
              Stored encrypted at rest with the same vault key as the
              password — same security posture.
            </p>
            {credential?.totp_set && (
              <p className="text-xs text-emerald-600 dark:text-emerald-400">
                ✓ TOTP secret on file — leave empty to keep, or paste a new
                one to replace.
              </p>
            )}
          </div>

          <details className="group">
            <summary className="cursor-pointer text-xs font-medium text-muted-foreground hover:text-foreground">
              Advanced — selector hints + notes
            </summary>
            <div className="mt-3 space-y-3 rounded-md border bg-muted/20 p-3">
              <div className="space-y-2">
                <Label htmlFor="cred-user-hint" className="text-xs">
                  Username field selector hint
                </Label>
                <Input
                  id="cred-user-hint"
                  value={usernameHint}
                  onChange={(e) => setUsernameHint(e.target.value)}
                  placeholder="input[name='email']"
                  className="font-mono text-xs"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="cred-pass-hint" className="text-xs">
                  Password field selector hint
                </Label>
                <Input
                  id="cred-pass-hint"
                  value={passwordHint}
                  onChange={(e) => setPasswordHint(e.target.value)}
                  placeholder="input[type='password']"
                  className="font-mono text-xs"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="cred-notes" className="text-xs">
                  Notes
                </Label>
                <Textarea
                  id="cred-notes"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Anything the agent should know about this account"
                  rows={3}
                />
              </div>
            </div>
          </details>
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
            {mutation.isPending
              ? isEdit
                ? "Saving…"
                : "Adding…"
              : isEdit
                ? "Save"
                : "Add credential"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
