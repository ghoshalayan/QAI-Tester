"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  KeyRound,
  Pencil,
  Play,
  Plus,
  Save,
  Trash2,
  Wand2,
} from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  PLAN_STATUS_LABELS,
  type CredentialRead,
  type PlanStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { CredentialFormDialog } from "@/components/credential-form-dialog";
import { DeletePlanDialog } from "@/components/delete-plan-dialog";
import { LinkedDocsPicker } from "@/components/linked-docs-picker";
import { ScopeEditor } from "@/components/scope-editor";
import { cn } from "@/lib/utils";

const STATUSES: PlanStatus[] = ["draft", "ready", "archived"];

export default function PlanEditorPage() {
  const params = useParams<{ id: string; plan_id: string }>();
  const projectId = Number(params.id);
  const planId = Number(params.plan_id);
  const qc = useQueryClient();

  const {
    data: plan,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["plan", projectId, planId],
    queryFn: () => api.getPlan(projectId, planId),
    enabled: !Number.isNaN(projectId) && !Number.isNaN(planId),
  });

  // ── Local form state for the basic-fields section ───────────
  const [name, setName] = useState("");
  const [targetUrl, setTargetUrl] = useState("");
  const [status, setStatus] = useState<PlanStatus>("draft");
  const [description, setDescription] = useState("");
  const [scope, setScope] = useState<string[]>([]);
  const [linkedDocIds, setLinkedDocIds] = useState<number[]>([]);
  const [maxReplans, setMaxReplans] = useState<number>(2);

  // Hydrate when the plan loads (or after a fresh re-fetch)
  useEffect(() => {
    if (!plan) return;
    setName(plan.name);
    setTargetUrl(plan.target_url);
    setStatus(plan.status);
    setDescription(plan.description ?? "");
    setScope(plan.scope);
    setLinkedDocIds(plan.linked_documents.map((d) => d.document_id));
    setMaxReplans(
      typeof plan.max_replans_per_submodule === "number"
        ? plan.max_replans_per_submodule
        : 2,
    );
  }, [plan]);

  // ── Save mutation (basic fields + scope + linked docs together) ──
  const saveMutation = useMutation({
    mutationFn: () =>
      api.updatePlan(projectId, planId, {
        name: name.trim(),
        target_url: targetUrl.trim(),
        status,
        description: description.trim(),
        scope,
        linked_document_ids: linkedDocIds,
        max_replans_per_submodule: maxReplans,
      }),
    onSuccess: () => {
      toast.success("Plan saved");
      qc.invalidateQueries({ queryKey: ["plan", projectId, planId] });
      qc.invalidateQueries({ queryKey: ["plans", projectId] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Save failed", { description: msg });
    },
  });

  const [credDialogOpen, setCredDialogOpen] = useState(false);
  const [editingCred, setEditingCred] = useState<CredentialRead | undefined>();
  const [deleteOpen, setDeleteOpen] = useState(false);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-9 w-1/3" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (isError || !plan) {
    return (
      <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
        Plan not found.
        <div className="mt-3">
          <Link
            href={`/projects/${projectId}/plans`}
            className="inline-flex items-center gap-1 text-primary hover:underline"
          >
            <ArrowLeft className="size-3" /> Back to plans
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Link
        href={`/projects/${projectId}/plans`}
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3" /> Back to plans
      </Link>

      <header className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-semibold tracking-tight">
            {plan.name}
          </h1>
          <p className="mt-1 break-all text-sm text-muted-foreground">
            {plan.target_url}
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button
            size="sm"
            disabled
            title="Wires up to the BRD→FRD→TC agent in week 4"
          >
            <Wand2 className="size-4" /> Generate test cases
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled
            title="Lands when execution agent comes online (week 5)"
          >
            <Play className="size-4" /> Run
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setDeleteOpen(true)}
          >
            <Trash2 className="size-4" /> Delete
          </Button>
        </div>
      </header>

      {/* Basic fields + scope + linked docs all save together */}
      <Card>
        <CardHeader>
          <CardTitle>Plan details</CardTitle>
          <CardDescription>
            Save commits name, target URL, status, instructions, scope, and the
            linked-doc set as one transaction.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="plan-name">Name</Label>
              <Input
                id="plan-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
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
                maxLength={2048}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label>Status</Label>
            <div className="inline-flex rounded-md border bg-background p-0.5">
              {STATUSES.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setStatus(s)}
                  className={cn(
                    "rounded-sm px-3 py-1 text-xs font-medium transition-colors",
                    status === s
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {PLAN_STATUS_LABELS[s]}
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              Mark as <strong>Ready</strong> once the plan has all the config
              the agent needs.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="plan-description">Instructions for the agent</Label>
            <Textarea
              id="plan-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={5}
              placeholder="e.g., Test login with valid + invalid credentials, then verify the dashboard renders 4 widgets. Pay special attention to error states."
              maxLength={10_000}
            />
            <p className="text-xs text-muted-foreground">
              Free-text guidance the agent reads before generating test cases.
              Especially important when no docs are linked.
            </p>
          </div>

          <div className="space-y-3">
            <div>
              <Label>Linked documents (optional)</Label>
              <p className="mt-1 text-xs text-muted-foreground">
                Tick BRD/FRD/Instructions docs the agent should reference.
                Leave empty for an instructions-only plan.
              </p>
            </div>
            <LinkedDocsPicker
              projectId={projectId}
              selectedIds={linkedDocIds}
              onChange={setLinkedDocIds}
            />
          </div>

          <div className="space-y-3">
            <div>
              <Label>Scope — modules to test</Label>
              <p className="mt-1 text-xs text-muted-foreground">
                Suggestions appear automatically from the headings of any docs
                you tick above. You can also type custom names.
              </p>
            </div>
            <ScopeEditor
              projectId={projectId}
              linkedDocumentIds={linkedDocIds}
              scope={scope}
              onChange={setScope}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="plan-max-replans">
              Max replans per submodule
            </Label>
            <Input
              id="plan-max-replans"
              type="number"
              min={0}
              max={5}
              value={maxReplans}
              onChange={(e) => {
                const v = Number.parseInt(e.target.value, 10);
                if (Number.isFinite(v)) {
                  setMaxReplans(Math.max(0, Math.min(5, v)));
                }
              }}
              className="max-w-[8rem]"
            />
            <p className="text-xs text-muted-foreground">
              When a sub-goal stalls, the vision planner gets up to this
              many attempts to re-decompose from the current screen
              before HITL is offered. Default <strong>2</strong>; range
              0–5. Set to 0 to disable replanning for this plan.
            </p>
          </div>

          <div className="flex justify-end pt-2">
            <Button
              onClick={() => saveMutation.mutate()}
              disabled={
                saveMutation.isPending ||
                !name.trim() ||
                !targetUrl.trim()
              }
            >
              <Save className="size-4" />
              {saveMutation.isPending ? "Saving…" : "Save changes"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Credentials section (sub-resource — saves independently) */}
      <Card>
        <CardHeader className="flex flex-row items-start justify-between gap-3 space-y-0">
          <div>
            <CardTitle className="flex items-center gap-2">
              <KeyRound className="size-5" /> Credentials
            </CardTitle>
            <CardDescription>
              ⚠ Stored in plaintext on disk per the local-MVP policy. The agent
              uses URL pattern to pick the right cred when multiple are
              defined. OTP / MFA codes are <strong>never stored</strong> — the
              agent will pause and ask each time.
            </CardDescription>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              setEditingCred(undefined);
              setCredDialogOpen(true);
            }}
          >
            <Plus className="size-4" /> Add credential
          </Button>
        </CardHeader>
        <CardContent>
          <CredentialsTable
            projectId={projectId}
            planId={planId}
            credentials={plan.credentials}
            onEdit={(c) => {
              setEditingCred(c);
              setCredDialogOpen(true);
            }}
          />
        </CardContent>
      </Card>

      <CredentialFormDialog
        open={credDialogOpen}
        onOpenChange={setCredDialogOpen}
        projectId={projectId}
        planId={planId}
        credential={editingCred}
      />
      <DeletePlanDialog
        open={deleteOpen}
        onOpenChange={setDeleteOpen}
        projectId={projectId}
        plan={plan}
        redirectTo={`/projects/${projectId}/plans`}
      />
    </div>
  );
}

function CredentialsTable({
  projectId,
  planId,
  credentials,
  onEdit,
}: {
  projectId: number;
  planId: number;
  credentials: CredentialRead[];
  onEdit: (c: CredentialRead) => void;
}) {
  const qc = useQueryClient();
  const deleteMutation = useMutation({
    mutationFn: (credId: number) =>
      api.deleteCredential(projectId, planId, credId),
    onSuccess: () => {
      toast.success("Credential removed");
      qc.invalidateQueries({ queryKey: ["plan", projectId, planId] });
      qc.invalidateQueries({ queryKey: ["plans", projectId] });
    },
    onError: (e: Error) =>
      toast.error("Delete failed", { description: e.message }),
  });

  if (credentials.length === 0) {
    return (
      <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
        No credentials yet. Add one if the target app needs login. Plans for
        public/anonymous URLs can leave this empty.
      </p>
    );
  }

  return (
    <div className="overflow-hidden rounded-md border">
      <table className="w-full text-sm">
        <thead className="border-b bg-muted/30 text-left text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th className="p-3 font-medium">Label</th>
            <th className="p-3 font-medium">Username</th>
            <th className="p-3 font-medium">Password</th>
            <th className="p-3 font-medium">URL pattern</th>
            <th className="p-3 font-medium">Notes</th>
            <th className="p-3" />
          </tr>
        </thead>
        <tbody>
          {credentials.map((c) => (
            <tr key={c.id} className="border-b last:border-b-0">
              <td className="p-3 font-medium">{c.label}</td>
              <td className="p-3 text-muted-foreground">{c.username}</td>
              <td className="p-3 font-mono text-xs text-muted-foreground">
                {c.password_set ? "••••••••" : "—"}
              </td>
              <td className="p-3 text-xs text-muted-foreground">
                {c.url_pattern || "(plan default)"}
              </td>
              <td className="max-w-[260px] p-3 text-xs text-muted-foreground">
                <span className="line-clamp-1">{c.notes || "—"}</span>
              </td>
              <td className="p-3 text-right">
                <div className="flex justify-end gap-1">
                  <Button
                    size="icon"
                    variant="ghost"
                    onClick={() => onEdit(c)}
                    aria-label={`Edit credential ${c.label}`}
                  >
                    <Pencil className="size-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    disabled={deleteMutation.isPending}
                    onClick={() => {
                      if (
                        window.confirm(
                          `Remove credential "${c.label}"? This cannot be undone.`,
                        )
                      ) {
                        deleteMutation.mutate(c.id);
                      }
                    }}
                    aria-label={`Delete credential ${c.label}`}
                  >
                    <Trash2 className="size-4" />
                  </Button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
