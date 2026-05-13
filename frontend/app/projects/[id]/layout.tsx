"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, usePathname } from "next/navigation";
import {
  ArrowLeft,
  Boxes,
  ClipboardList,
  FileText,
  ListTree,
  Pencil,
  Play,
  ScrollText,
  Trash2,
} from "lucide-react";

import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ProjectFormDialog } from "@/components/project-form-dialog";
import { DeleteProjectDialog } from "@/components/delete-project-dialog";
import { cn } from "@/lib/utils";

// Tab order mirrors the canonical workflow:
// Documents (upload BRD) → Requirements (synthesize + approve FRDs) →
// Plans (target URL + scope + credentials) → Test Cases (generate tree) →
// Runs (execute selected steps).
// The instructions-only path simply skips Requirements; route slugs are
// independent of order so nothing else needs changing if you reshuffle.
const TABS = [
  { slug: "documents", label: "Documents", icon: FileText },
  { slug: "requirements", label: "Requirements", icon: ScrollText },
  { slug: "plans", label: "Plans", icon: ClipboardList },
  { slug: "test-cases", label: "Test Cases", icon: ListTree },
  { slug: "modules", label: "Modules", icon: Boxes },
  { slug: "runs", label: "Runs", icon: Play },
] as const;

export default function ProjectLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const params = useParams<{ id: string }>();
  const pathname = usePathname();
  const projectId = Number(params.id);

  const [editOpen, setEditOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const {
    data: project,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => api.getProject(projectId),
    enabled: !Number.isNaN(projectId),
  });

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl space-y-4 p-8">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-9 w-1/3" />
        <Skeleton className="h-4 w-1/2" />
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }

  if (isError || !project) {
    return (
      <div className="mx-auto max-w-5xl p-8">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-3" /> Back to projects
        </Link>
        <p className="mt-4 text-sm">Project not found.</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl p-8">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3" /> Back to projects
      </Link>

      <header className="mb-6 flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-3xl font-semibold tracking-tight">
            {project.name}
          </h1>
          {project.description && (
            <p className="mt-2 text-sm text-muted-foreground">
              {project.description}
            </p>
          )}
          <div className="mt-3 text-xs text-muted-foreground">
            Project #{project.id} · Updated{" "}
            {new Date(project.updated_at).toLocaleDateString()}
          </div>
        </div>
        <div className="flex shrink-0 gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setEditOpen(true)}
          >
            <Pencil className="size-4" /> Edit
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setDeleteOpen(true)}
          >
            <Trash2 className="size-4" /> Delete
          </Button>
        </div>
      </header>

      <nav className="mb-8 border-b">
        <div className="flex gap-1">
          {TABS.map(({ slug, label, icon: Icon }) => {
            const href = `/projects/${projectId}/${slug}`;
            const active = pathname === href || pathname.startsWith(href + "/");
            return (
              <Link
                key={slug}
                href={href}
                className={cn(
                  "-mb-px flex items-center gap-2 border-b-2 px-4 py-2 text-sm font-medium transition-colors",
                  active
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
                )}
              >
                <Icon className="size-4" />
                {label}
              </Link>
            );
          })}
        </div>
      </nav>

      <div>{children}</div>

      <ProjectFormDialog
        open={editOpen}
        onOpenChange={setEditOpen}
        project={project}
      />
      <DeleteProjectDialog
        open={deleteOpen}
        onOpenChange={setDeleteOpen}
        project={project}
        redirectTo="/"
      />
    </div>
  );
}
