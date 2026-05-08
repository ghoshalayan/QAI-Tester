"use client";

/**
 * Phase 4 — typed HITL input card rendered inside the live presenter
 * popup. The agent's auth flow opens a typed prompt (OTP / credentials
 * / manual-solve resume) by:
 *   1. Calling ``open_typed_prompt`` on the backend service
 *   2. Calling ``request_intervention`` (which blocks)
 *   3. Emitting ``hitl_prompt_opened`` SSE event
 *
 * This component renders the form that submits via
 * ``api.provideIntervention``, which unblocks the agent's wait.
 *
 * UX rules:
 * - Password fields use ``type="password"`` and never appear in any
 *   event log payload.
 * - On submit, we send ``choice="provide_text"`` with ``text_kind``
 *   describing the value (so the auth flow knows how to use it).
 * - Multi-field prompts (request_credentials) submit BOTH fields in
 *   one POST: text_value=username, text_value_secondary=password.
 * - For await_manual_solve, the form is just a "I've solved it,
 *   continue" button — no input fields.
 */

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { CheckCircle2, KeyRound, ShieldCheck } from "lucide-react";

import { api, ApiError, type OpenPrompt } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface Props {
  projectId: number;
  runId: number;
  stepId: number;
  prompt: OpenPrompt;
  onSubmitted: () => void;
}

export function HitlPromptCard({
  projectId,
  runId,
  stepId,
  prompt,
  onSubmitted,
}: Props) {
  const [primary, setPrimary] = useState("");
  const [secondary, setSecondary] = useState("");

  const mutation = useMutation({
    mutationFn: () => {
      if (prompt.kind === "await_manual_solve") {
        return api.provideIntervention(projectId, runId, {
          step_id: stepId,
          choice: "manual_solved",
        });
      }
      if (prompt.kind === "request_credentials") {
        return api.provideIntervention(projectId, runId, {
          step_id: stepId,
          choice: "provide_text",
          text_kind: "username",
          text_value: primary,
          text_value_secondary: secondary,
        });
      }
      // request_text — single free-form input. Field name hints at
      // the kind (otp_code, captcha_text, free_text).
      const fieldName = prompt.fields?.[0]?.name ?? "free_text";
      return api.provideIntervention(projectId, runId, {
        step_id: stepId,
        choice: "provide_text",
        text_kind: fieldName,
        text_value: primary,
      });
    },
    onSuccess: () => {
      toast.success("Sent — agent resuming");
      setPrimary("");
      setSecondary("");
      onSubmitted();
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Couldn't deliver", { description: msg });
    },
  });

  const Icon =
    prompt.kind === "request_credentials"
      ? KeyRound
      : prompt.kind === "await_manual_solve"
        ? ShieldCheck
        : CheckCircle2;

  const title =
    prompt.kind === "request_credentials"
      ? "Sign in needed"
      : prompt.kind === "await_manual_solve"
        ? "Manual step required"
        : "Agent needs input";

  return (
    <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3">
      <div className="mb-2 flex items-center gap-2">
        <Icon className="size-4 text-amber-500" />
        <p className="text-sm font-medium">{title}</p>
      </div>
      <p className="mb-3 text-xs text-muted-foreground">
        {prompt.question || "Your input is needed to continue."}
      </p>

      {prompt.kind === "await_manual_solve" ? (
        <Button
          size="sm"
          className="w-full"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Sending…" : "I've solved it — continue"}
        </Button>
      ) : prompt.kind === "request_credentials" ? (
        <form
          className="space-y-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (primary && secondary) mutation.mutate();
          }}
        >
          <div className="space-y-1">
            <Label htmlFor="hitl-username" className="text-xs">
              {prompt.fields?.[0]?.label ?? "Username / email"}
            </Label>
            <Input
              id="hitl-username"
              value={primary}
              onChange={(e) => setPrimary(e.target.value)}
              autoComplete="username"
              autoFocus
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="hitl-password" className="text-xs">
              {prompt.fields?.[1]?.label ?? "Password"}
            </Label>
            <Input
              id="hitl-password"
              type="password"
              value={secondary}
              onChange={(e) => setSecondary(e.target.value)}
              autoComplete="current-password"
            />
          </div>
          <Button
            type="submit"
            size="sm"
            className="w-full"
            disabled={mutation.isPending || !primary || !secondary}
          >
            {mutation.isPending ? "Sending…" : "Submit"}
          </Button>
        </form>
      ) : (
        // request_text — single input. Type=password if the field
        // name suggests a secret (OTP codes are often shown but we
        // keep them masked for safety on shared screens).
        <form
          className="space-y-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (primary) mutation.mutate();
          }}
        >
          <div className="space-y-1">
            <Label htmlFor="hitl-input" className="text-xs">
              {prompt.fields?.[0]?.label ?? "Value"}
            </Label>
            <Input
              id="hitl-input"
              type={
                prompt.fields?.[0]?.type === "password"
                  ? "password"
                  : "text"
              }
              value={primary}
              onChange={(e) => setPrimary(e.target.value)}
              autoFocus
              autoComplete="off"
            />
          </div>
          <Button
            type="submit"
            size="sm"
            className="w-full"
            disabled={mutation.isPending || !primary}
          >
            {mutation.isPending ? "Sending…" : "Submit"}
          </Button>
        </form>
      )}
    </div>
  );
}
