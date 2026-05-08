import type { Metadata } from "next";

import "./globals.css";
import { Providers } from "./providers";
import { AppShell } from "@/components/app-shell";
import { FirstRunGate } from "@/components/first-run-gate";

export const metadata: Metadata = {
  title: "QAI Tester v2",
  description: "Agentic QA testing platform — local MVP",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <Providers>
          <FirstRunGate>
            <AppShell>{children}</AppShell>
          </FirstRunGate>
        </Providers>
      </body>
    </html>
  );
}
