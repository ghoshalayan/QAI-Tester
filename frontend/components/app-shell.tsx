"use client";

/**
 * Root chrome decision: render the global Sidebar for the main app,
 * but suppress it for popup-style routes that should fill their
 * window completely (the live presenter at /live/[projectId]/[runId]).
 *
 * Kept as a small client component so the rest of the layout tree
 * can stay server-rendered.
 */

import { usePathname } from "next/navigation";

import { Sidebar } from "@/components/sidebar";


// Routes that should render full-bleed without the global sidebar
// or main scroll wrapper. These are popups / presenter windows that
// already manage their own viewport.
const CHROMELESS_PREFIXES = ["/live/"];


export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() ?? "";
  const chromeless = CHROMELESS_PREFIXES.some((p) =>
    pathname.startsWith(p),
  );

  if (chromeless) {
    // Full-bleed: no sidebar, no main scroll wrapper. The live
    // presenter renders ``flex h-screen`` itself.
    return <>{children}</>;
  }

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">{children}</main>
    </div>
  );
}
