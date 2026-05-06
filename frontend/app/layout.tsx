import type { Metadata } from "next";

import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "@/components/sidebar";
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
            <div className="flex h-screen w-full overflow-hidden">
              <Sidebar />
              <main className="flex-1 overflow-y-auto">{children}</main>
            </div>
          </FirstRunGate>
        </Providers>
      </body>
    </html>
  );
}
