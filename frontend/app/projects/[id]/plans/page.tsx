"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { ClipboardList, ExternalLink, Plus } from "lucide-react";

import {
  api,
  PLAN_STATUS_LABELS,
  type PlanStatus,
} from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { PlanQuickCreateDialog } from "@/components/plan-quick-create-dialog";
import { cn } from "@/lib/utils";

const STATUS_CLASSES: Record<PlanStatus, string> = {
  draft: "bg-muted text-muted-foreground",
  ready: "bg-green-500/10 text-green-700 dark:text-green-400",
  archived: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
};

export default function PlansTabPage() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);
  const [createOpen, setCreateOpen] = useState(false);

  const { data: plans, isLoading } = useQuery({
    queryKey: ["plans", projectId],
    queryFn: () => api.listPlans(projectId),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <p className="text-sm text-muted-foreground">
          A <strong>plan</strong> bundles a target URL, login credentials, scope
          of modules, and free-text instructions — everything the agent needs
          to run a test session.
        </p>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="size-4" /> New plan
        </Button>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      ) : !plans || plans.length === 0 ? (
        <div className="rounded-lg border border-dashed p-12 text-center">
          <ClipboardList className="mx-auto size-10 text-muted-foreground" />
          <h3 className="mt-4 font-semibold">No plans yet</h3>
          <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">
            Create a plan to specify the target URL + credentials + scope. You
            can run a plan even with no BRD/FRD docs uploaded — the
            instructions field is enough.
          </p>
          <Button className="mt-4" onClick={() => setCreateOpen(true)}>
            <Plus className="size-4" /> New plan
          </Button>
          <p className="mx-auto mt-4 max-w-md text-xs text-muted-foreground">
            Want the agent to read requirements docs first?{" "}
            <Link
              href={`/projects/${projectId}/documents`}
              className="font-medium text-primary underline underline-offset-2"
            >
              Upload them in Documents
            </Link>
            , then link them when you create the plan.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {plans.map((plan) => (
            <Link
              key={plan.id}
              href={`/projects/${projectId}/plans/${plan.id}`}
              className="block rounded-xl outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <Card className="transition-colors hover:border-primary/50">
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <CardTitle className="line-clamp-1">
                        {plan.name}
                      </CardTitle>
                      <p className="mt-1 flex items-center gap-1 text-sm text-muted-foreground">
                        <ExternalLink className="size-3 shrink-0" />
                        <span className="line-clamp-1">{plan.target_url}</span>
                      </p>
                    </div>
                    <span
                      className={cn(
                        "shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium",
                        STATUS_CLASSES[plan.status],
                      )}
                    >
                      {PLAN_STATUS_LABELS[plan.status]}
                    </span>
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    {plan.scope.length > 0 ? (
                      <>
                        {plan.scope.slice(0, 5).map((s) => (
                          <Badge key={s} variant="outline">
                            {s}
                          </Badge>
                        ))}
                        {plan.scope.length > 5 && (
                          <span className="text-muted-foreground">
                            +{plan.scope.length - 5} more
                          </span>
                        )}
                      </>
                    ) : (
                      <span className="italic text-muted-foreground">
                        no scope
                      </span>
                    )}
                    <span className="ml-auto text-muted-foreground">
                      {plan.credential_count} cred
                      {plan.credential_count === 1 ? "" : "s"} ·{" "}
                      {plan.linked_document_count} doc
                      {plan.linked_document_count === 1 ? "" : "s"}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}

      <PlanQuickCreateDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        projectId={projectId}
      />
    </div>
  );
}
