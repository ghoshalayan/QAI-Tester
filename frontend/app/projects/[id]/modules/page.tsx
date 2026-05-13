"use client";

/**
 * Phase E — Sub-flow modules library page.
 *
 * Project-scoped library of reusable named flows. Each row is a
 * frozen v2 segment bundle that was promoted from a passed submodule.
 * Other plans can import any module — the imported submodule arrives
 * with its frozen_path pre-populated so the next run replays
 * deterministically.
 */

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import {
  Boxes,
  Download,
  Pencil,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  type SubFlowModuleSummary,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

export default function ModulesTabPage() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);
  const qc = useQueryClient();

  const { data: modules, isLoading } = useQuery({
    queryKey: ["sub-flow-modules", projectId],
    queryFn: () => api.listSubFlowModules(projectId),
    enabled: !Number.isNaN(projectId),
  });

  const { data: plans } = useQuery({
    queryKey: ["plans", projectId],
    queryFn: () => api.listPlans(projectId),
    enabled: !Number.isNaN(projectId),
  });

  const [importTarget, setImportTarget] = useState<SubFlowModuleSummary | null>(
    null,
  );
  const [editTarget, setEditTarget] = useState<SubFlowModuleSummary | null>(
    null,
  );

  const deleteMutation = useMutation({
    mutationFn: (moduleId: number) =>
      api.deleteSubFlowModule(projectId, moduleId),
    onSuccess: () => {
      toast.success("Module deleted");
      qc.invalidateQueries({
        queryKey: ["sub-flow-modules", projectId],
      });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Couldn't delete module", { description: msg });
    },
  });

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Boxes className="size-5 text-primary" /> Sub-flow modules
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Reusable building blocks extracted from passing
            submodules. Promote a clean run to the library, then
            import it into other plans — the imported submodule
            arrives with its proven frozen path pre-populated.
          </p>
        </div>
      </div>

      {isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-14 w-full" />
          <Skeleton className="h-14 w-full" />
        </div>
      )}

      {!isLoading && (!modules || modules.length === 0) && (
        <div className="rounded-lg border border-dashed bg-muted/20 p-8 text-center text-sm text-muted-foreground">
          <p>No modules yet.</p>
          <p className="mt-2 text-xs">
            Run a plan agentically. When a submodule passes cleanly
            it gets a frozen v2 path — open the Test Cases tab,
            find the passing submodule, and click{" "}
            <strong>Save as module</strong>.
          </p>
        </div>
      )}

      {!isLoading && modules && modules.length > 0 && (
        <ul className="space-y-2">
          {modules.map((m) => (
            <li
              key={m.id}
              className="rounded-lg border bg-card p-3"
            >
              <div className="flex items-start gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="font-medium">{m.name}</span>
                    <span className="rounded border bg-muted/40 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                      v{m.frozen_path_version} · {m.segments} segment
                      {m.segments === 1 ? "" : "s"} · {m.steps} step
                      {m.steps === 1 ? "" : "s"}
                    </span>
                    {m.tags.length > 0 && (
                      <span className="text-[10px] text-muted-foreground">
                        {m.tags.map((t) => `#${t}`).join(" ")}
                      </span>
                    )}
                  </div>
                  {m.description && (
                    <p className="mt-1 text-xs text-muted-foreground">
                      {m.description}
                    </p>
                  )}
                  <div className="mt-1 flex flex-wrap gap-x-3 text-[10px] text-muted-foreground">
                    {m.target_url_pattern && (
                      <span>pattern: {m.target_url_pattern}</span>
                    )}
                    {m.source_plan_id && (
                      <span>from plan #{m.source_plan_id}</span>
                    )}
                    {m.source_run_id && (
                      <span>run #{m.source_run_id}</span>
                    )}
                    {m.updated_at && (
                      <span>
                        updated{" "}
                        {new Date(m.updated_at).toLocaleString()}
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex shrink-0 gap-1">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setImportTarget(m)}
                  >
                    <Download className="size-3.5" />
                    Import
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setEditTarget(m)}
                    aria-label="Edit module"
                  >
                    <Pencil className="size-3.5" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      if (
                        window.confirm(
                          `Delete module "${m.name}"? This can't be undone.`,
                        )
                      ) {
                        deleteMutation.mutate(m.id);
                      }
                    }}
                    aria-label="Delete module"
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      <ImportDialog
        projectId={projectId}
        plans={plans ?? []}
        module={importTarget}
        onOpenChange={(o) => !o && setImportTarget(null)}
      />
      <EditMetadataDialog
        projectId={projectId}
        module={editTarget}
        onOpenChange={(o) => !o && setEditTarget(null)}
      />
    </div>
  );
}


function ImportDialog({
  projectId,
  plans,
  module,
  onOpenChange,
}: {
  projectId: number;
  plans: { id: number; name: string; target_url: string }[];
  module: SubFlowModuleSummary | null;
  onOpenChange: (next: boolean) => void;
}) {
  const qc = useQueryClient();
  const [selectedPlanId, setSelectedPlanId] = useState<number | null>(null);

  // Suggest plans whose target_url matches the module's
  // target_url_pattern (best-effort substring).
  const matchingPlans = module?.target_url_pattern
    ? plans.filter((p) =>
        p.target_url
          .toLowerCase()
          .includes(module.target_url_pattern!.toLowerCase()),
      )
    : plans;

  const importMutation = useMutation({
    mutationFn: () =>
      api.importSubFlowModule(projectId, module!.id, {
        plan_id: selectedPlanId!,
      }),
    onSuccess: (resp) => {
      toast.success(
        `Imported ${module?.name} — new submodule with ${resp.steps_created} step(s)`,
      );
      qc.invalidateQueries({
        queryKey: ["tc-nodes", projectId, selectedPlanId],
      });
      onOpenChange(false);
      setSelectedPlanId(null);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Import failed", { description: msg });
    },
  });

  return (
    <Dialog open={!!module} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            Import &quot;{module?.name}&quot; into a plan
          </DialogTitle>
          <DialogDescription>
            Creates a new submodule in the selected plan with this
            module&apos;s proven frozen path. The next run on that
            plan will replay the steps deterministically.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <Label className="text-sm">Target plan</Label>
            <select
              className={cn(
                "mt-1 w-full rounded-md border bg-background px-2 py-1.5 text-sm",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              )}
              value={selectedPlanId ?? ""}
              onChange={(e) =>
                setSelectedPlanId(
                  e.target.value ? Number(e.target.value) : null,
                )
              }
            >
              <option value="">Pick a plan…</option>
              {matchingPlans.length > 0 && (
                <optgroup label="Suggested (target URL matches pattern)">
                  {matchingPlans.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name} — {p.target_url}
                    </option>
                  ))}
                </optgroup>
              )}
              <optgroup label="All plans">
                {plans.map((p) => (
                  <option key={`all-${p.id}`} value={p.id}>
                    {p.name} — {p.target_url}
                  </option>
                ))}
              </optgroup>
            </select>
          </div>
          {module?.target_url_pattern && (
            <p className="text-xs text-muted-foreground">
              Module pattern:{" "}
              <code className="rounded bg-muted px-1">
                {module.target_url_pattern}
              </code>
              {" "}— suggested plans match this substring.
            </p>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            onClick={() => importMutation.mutate()}
            disabled={!selectedPlanId || importMutation.isPending}
          >
            <Download className="size-4" />
            {importMutation.isPending ? "Importing…" : "Import"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}


function EditMetadataDialog({
  projectId,
  module,
  onOpenChange,
}: {
  projectId: number;
  module: SubFlowModuleSummary | null;
  onOpenChange: (next: boolean) => void;
}) {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [pattern, setPattern] = useState("");
  const [tags, setTags] = useState("");

  // Seed when opened.
  if (module && name === "" && description === "" && pattern === "" && tags === "") {
    setName(module.name);
    setDescription(module.description ?? "");
    setPattern(module.target_url_pattern ?? "");
    setTags(module.tags.join(", "));
  }

  const updateMutation = useMutation({
    mutationFn: () =>
      api.updateSubFlowModule(projectId, module!.id, {
        name: name.trim(),
        description: description.trim(),
        target_url_pattern: pattern.trim() || null,
        tags: tags
          .split(",")
          .map((t) => t.trim())
          .filter(Boolean),
      }),
    onSuccess: () => {
      toast.success("Module updated");
      qc.invalidateQueries({
        queryKey: ["sub-flow-modules", projectId],
      });
      setName("");
      setDescription("");
      setPattern("");
      setTags("");
      onOpenChange(false);
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Update failed", { description: msg });
    },
  });

  return (
    <Dialog
      open={!!module}
      onOpenChange={(o) => {
        if (!o) {
          setName("");
          setDescription("");
          setPattern("");
          setTags("");
        }
        onOpenChange(o);
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit module metadata</DialogTitle>
          <DialogDescription>
            Update the module&apos;s name, description, pattern, and
            tags. The captured steps and frozen path are not editable
            here — re-promote a new run if you need to refresh them.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div>
            <Label htmlFor="module-name">Name</Label>
            <Input
              id="module-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={255}
            />
          </div>
          <div>
            <Label htmlFor="module-desc">Description</Label>
            <Textarea
              id="module-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>
          <div>
            <Label htmlFor="module-pattern">
              Target URL pattern (substring match)
            </Label>
            <Input
              id="module-pattern"
              value={pattern}
              onChange={(e) => setPattern(e.target.value)}
              placeholder="e.g. solar.com"
            />
          </div>
          <div>
            <Label htmlFor="module-tags">
              Tags (comma-separated)
            </Label>
            <Input
              id="module-tags"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="auth, admin, create"
            />
          </div>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            onClick={() => updateMutation.mutate()}
            disabled={!name.trim() || updateMutation.isPending}
          >
            {updateMutation.isPending ? "Saving…" : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
