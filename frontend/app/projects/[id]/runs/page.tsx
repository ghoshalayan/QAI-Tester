"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Play } from "lucide-react";

import { api, type AgentRunRead } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RunProgressCard } from "@/components/run-progress-card";
import { StartExecuteDialog } from "@/components/start-execute-dialog";
import { useAgentRunsEvents } from "@/hooks/use-agent-runs-events";

export default function RunsTabPage() {
  const params = useParams<{ id: string }>();
  const projectId = Number(params.id);
  const [startOpen, setStartOpen] = useState(false);

  useAgentRunsEvents(projectId);

  const { data: runs, isLoading } = useQuery({
    queryKey: ["agent-runs", projectId],
    queryFn: () => api.listAgentRuns(projectId),
  });

  const executeRuns = useMemo(
    () => (runs ?? []).filter((r) => r.kind === "execute"),
    [runs],
  );

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <p className="max-w-2xl text-sm text-muted-foreground">
          Execution runs walk a plan&apos;s selected test cases against the
          target URL. Pick which steps to run via the tri-state checkboxes
          on the Test Cases tab, then click <strong>Start run</strong>.
        </p>
        <Button size="sm" onClick={() => setStartOpen(true)}>
          <Play className="size-4" /> Start run
        </Button>
      </div>

      <StartExecuteDialog
        open={startOpen}
        onOpenChange={setStartOpen}
        projectId={projectId}
      />

      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-24 w-full" />
        </div>
      ) : executeRuns.length === 0 ? (
        <EmptyState onStart={() => setStartOpen(true)} />
      ) : (
        <div className="space-y-3">
          {executeRuns.map((run) => (
            <RunRow key={run.id} projectId={projectId} run={run} />
          ))}
        </div>
      )}
    </div>
  );
}

function RunRow({
  projectId,
  run,
}: {
  projectId: number;
  run: AgentRunRead;
}) {
  return (
    <Link
      href={`/projects/${projectId}/runs/${run.id}`}
      className="block rounded-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <RunProgressCard projectId={projectId} run={run} />
    </Link>
  );
}

function EmptyState({ onStart }: { onStart: () => void }) {
  return (
    <div className="rounded-lg border border-dashed p-12 text-center">
      <Play className="mx-auto size-10 text-muted-foreground" />
      <h3 className="mt-4 font-semibold">No runs yet</h3>
      <p className="mx-auto mt-1 max-w-md text-sm text-muted-foreground">
        Execute a plan to see live step-by-step progress, screenshots, and
        pass/fail counts. Steps you tick on the Test Cases tab are the ones
        that&apos;ll run.
      </p>
      <Button className="mt-4" onClick={onStart}>
        <Play className="size-4" /> Start run
      </Button>
    </div>
  );
}
