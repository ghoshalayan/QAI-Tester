import { redirect } from "next/navigation";

/**
 * /projects/{id} → /projects/{id}/documents
 *
 * Server-side redirect. The tabbed layout in ``layout.tsx`` is what actually
 * renders the project header + tab nav; this page just picks the default tab.
 */
export default async function ProjectIndexPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  redirect(`/projects/${id}/documents`);
}
