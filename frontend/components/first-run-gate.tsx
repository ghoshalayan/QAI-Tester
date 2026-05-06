"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * Blocks the app behind a "configure your LLM provider" screen until
 * `app_settings.is_configured` flips true. The Settings page itself is
 * always reachable so the user can complete setup.
 *
 * Also catches "backend unreachable" — surfaced as a retry card.
 */
export function FirstRunGate({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
    staleTime: 60_000,
    retry: 1,
  });

  if (isLoading) {
    return (
      <div className="grid min-h-screen place-items-center">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="grid min-h-screen place-items-center p-8">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Backend unreachable</CardTitle>
            <CardDescription>
              Couldn&apos;t connect to the backend API. Make sure it&apos;s running
              on <code className="rounded bg-muted px-1 py-0.5">http://localhost:8000</code>.
            </CardDescription>
          </CardHeader>
          <CardFooter>
            <Button onClick={() => refetch()}>Retry</Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  // Allow Settings page through even when not configured — that's where the
  // user goes to fix it.
  if (data && !data.is_configured && pathname !== "/settings") {
    return (
      <div className="grid min-h-screen place-items-center p-8">
        <Card className="w-full max-w-lg">
          <CardHeader>
            <CardTitle>Welcome to QAI Tester</CardTitle>
            <CardDescription>
              To get started, configure your LLM provider. You can use{" "}
              <strong>Gemini</strong>, <strong>OpenAI</strong>, or any{" "}
              <strong>OpenAI-compatible</strong> endpoint such as Ollama, vLLM,
              LM Studio, or OpenRouter.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="ml-5 list-disc space-y-1 text-sm text-muted-foreground">
              <li>All BRD/FRD parsing and test-case generation goes through this provider.</li>
              <li>You can switch providers any time from Settings.</li>
              <li>API keys are stored locally — never sent to any third party except the provider you choose.</li>
            </ul>
          </CardContent>
          <CardFooter>
            <Button asChild>
              <Link href="/settings">Configure now →</Link>
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return <>{children}</>;
}
