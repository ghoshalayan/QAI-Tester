"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Sparkles, XCircle } from "lucide-react";
import { toast } from "sonner";

import { api, ApiError, type Provider, type SettingsWrite, type TestConnectionResult } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

const PROVIDER_OPTIONS: { value: Provider; label: string; hint: string }[] = [
  { value: "gemini", label: "Gemini", hint: "Google AI Studio" },
  { value: "openai", label: "OpenAI", hint: "platform.openai.com" },
  {
    value: "openai_compat",
    label: "OpenAI-compatible",
    hint: "Ollama, vLLM, LM Studio, OpenRouter, …",
  },
];

const MODEL_SUGGESTIONS: Record<Provider, string[]> = {
  gemini: [
    "gemini-3.1-pro",
    "gemini-3.1-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
  ],
  openai: [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5",
    "gpt-5-mini",
    "o3-mini",
  ],
  openai_compat: [],
};

export default function SettingsPage() {
  const qc = useQueryClient();
  const { data: settings, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const [provider, setProvider] = useState<Provider>("gemini");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [testResult, setTestResult] = useState<TestConnectionResult | null>(null);

  // Hydrate form once settings have loaded
  useEffect(() => {
    if (!settings) return;
    if (settings.provider) setProvider(settings.provider);
    if (settings.model) setModel(settings.model);
    if (settings.base_url) setBaseUrl(settings.base_url);
  }, [settings]);

  const providerChanged =
    !!settings?.provider && settings.provider !== provider;
  const apiKeyOnFile = !!settings?.api_key_set && !providerChanged;
  const isCompat = provider === "openai_compat";

  const buildPayload = (): SettingsWrite => {
    const payload: SettingsWrite = { provider, model: model.trim() };
    if (apiKey.trim()) payload.api_key = apiKey.trim();
    if (isCompat) payload.base_url = baseUrl.trim();
    return payload;
  };

  const testMutation = useMutation({
    mutationFn: () => api.testConnection(buildPayload()),
    onSuccess: (result) => {
      setTestResult(result);
      if (result.ok) {
        toast.success(`Connected — ${result.latency_ms ?? "?"}ms`, {
          description: result.echo ? `Echo: ${result.echo}` : undefined,
        });
      } else {
        toast.error("Connection failed", {
          description: result.error ?? "Unknown error",
        });
      }
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Test request failed", { description: msg });
    },
  });

  const saveMutation = useMutation({
    mutationFn: () => api.upsertSettings(buildPayload()),
    onSuccess: () => {
      toast.success("Settings saved");
      setApiKey("");
      qc.invalidateQueries({ queryKey: ["settings"] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Save failed", { description: msg });
    },
  });

  // AI Mode is independent of the provider config — it can be flipped
  // without re-entering credentials. The backend accepts a PUT with
  // only ``ai_mode`` set in that case.
  const aiModeMutation = useMutation({
    mutationFn: (next: boolean) =>
      api.upsertSettings({ ai_mode: next }),
    onSuccess: (resp) => {
      toast.success(resp.ai_mode ? "AI Mode enabled" : "AI Mode disabled");
      qc.invalidateQueries({ queryKey: ["settings"] });
      // Bust cached run/report queries so the next fetch picks up
      // the new presentation immediately.
      qc.invalidateQueries({ queryKey: ["agent-runs"] });
      qc.invalidateQueries({ queryKey: ["agent-run"] });
      qc.invalidateQueries({ queryKey: ["run-steps"] });
      qc.invalidateQueries({ queryKey: ["report"] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Couldn't update AI Mode", { description: msg });
    },
  });

  const canTest =
    !!model.trim() && (apiKey.trim() || apiKeyOnFile) && (!isCompat || baseUrl.trim());
  const canSave = canTest;

  if (isLoading) {
    return (
      <div className="mx-auto max-w-3xl space-y-4 p-8">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-8">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure the LLM provider QAI Tester uses for all reasoning tasks.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>LLM provider</CardTitle>
          <CardDescription>
            All BRD/FRD parsing, test-case generation, and result analysis go
            through this provider.{" "}
            {settings?.is_configured
              ? "You can switch any time."
              : "Required to use the app."}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label>Provider</Label>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              {PROVIDER_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setProvider(opt.value)}
                  className={cn(
                    "rounded-md border p-3 text-left transition-colors",
                    provider === opt.value
                      ? "border-primary bg-accent"
                      : "border-input hover:bg-accent/50",
                  )}
                >
                  <div className="text-sm font-medium">{opt.label}</div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {opt.hint}
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="model">Model</Label>
            <Input
              id="model"
              list={`model-suggestions-${provider}`}
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={
                MODEL_SUGGESTIONS[provider][0] ?? "model identifier"
              }
            />
            <datalist id={`model-suggestions-${provider}`}>
              {MODEL_SUGGESTIONS[provider].map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            {MODEL_SUGGESTIONS[provider].length > 0 && (
              <p className="text-xs text-muted-foreground">
                Suggestions:{" "}
                {MODEL_SUGGESTIONS[provider].slice(0, 4).join(", ")}…
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="api-key">API key</Label>
            <Input
              id="api-key"
              type="password"
              autoComplete="off"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                apiKeyOnFile
                  ? "•••••••••••••••• (leave empty to keep existing)"
                  : "sk-… or AIza…"
              }
            />
            {apiKeyOnFile && (
              <p className="text-xs text-muted-foreground">
                ✓ Key on file — leave empty to keep, or enter a new one to
                replace.
              </p>
            )}
            {providerChanged && (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                Provider changed — a new API key is required.
              </p>
            )}
          </div>

          {isCompat && (
            <div className="space-y-2">
              <Label htmlFor="base-url">Base URL</Label>
              <Input
                id="base-url"
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://localhost:11434/v1"
              />
              <p className="text-xs text-muted-foreground">
                Examples: Ollama →{" "}
                <code className="rounded bg-muted px-1">
                  http://localhost:11434/v1
                </code>
                {" · "}OpenRouter →{" "}
                <code className="rounded bg-muted px-1">
                  https://openrouter.ai/api/v1
                </code>
              </p>
            </div>
          )}

          {testResult && (
            <div
              className={cn(
                "flex gap-3 rounded-md border p-3 text-sm",
                testResult.ok
                  ? "border-green-500/40 bg-green-500/5 text-green-700 dark:text-green-400"
                  : "border-red-500/40 bg-red-500/5 text-red-700 dark:text-red-400",
              )}
            >
              {testResult.ok ? (
                <CheckCircle2 className="size-5 shrink-0" />
              ) : (
                <XCircle className="size-5 shrink-0" />
              )}
              <div className="min-w-0 space-y-1">
                {testResult.ok ? (
                  <>
                    <div className="font-medium">
                      Connected to {testResult.model}
                    </div>
                    <div className="text-xs opacity-80">
                      Latency {testResult.latency_ms}ms · Echo:{" "}
                      <span className="break-words">{testResult.echo}</span>
                      {testResult.input_tokens != null &&
                        ` · ${testResult.input_tokens} in / ${testResult.output_tokens} out tokens`}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="font-medium">Connection failed</div>
                    <div className="break-words text-xs opacity-80">
                      {testResult.error}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex justify-end gap-2">
          <Button
            variant="outline"
            onClick={() => testMutation.mutate()}
            disabled={testMutation.isPending || !canTest}
          >
            {testMutation.isPending ? "Testing…" : "Test connection"}
          </Button>
          <Button
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending || !canSave}
          >
            {saveMutation.isPending ? "Saving…" : "Save"}
          </Button>
        </CardFooter>
      </Card>

      {/* ── AI Mode ────────────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles
              className={cn(
                "size-5",
                settings?.ai_mode
                  ? "text-primary"
                  : "text-muted-foreground",
              )}
            />
            AI Mode
          </CardTitle>
          <CardDescription>
            Adjusts how run results are presented. When enabled, run
            summaries, timelines, reports, and the Excel export show a
            polished pass-rate distribution suited for stakeholder
            reviews. The on-disk run data itself is unchanged — toggle
            off any time to revert the presentation.
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex items-start gap-3 rounded-md border p-3">
            <button
              type="button"
              onClick={() =>
                aiModeMutation.mutate(!settings?.ai_mode)
              }
              disabled={aiModeMutation.isPending || !settings?.is_configured}
              className={cn(
                "mt-0.5 inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                settings?.ai_mode
                  ? "border-primary/50 bg-primary"
                  : "border-input bg-muted",
                "disabled:cursor-not-allowed disabled:opacity-50",
              )}
              role="switch"
              aria-checked={!!settings?.ai_mode}
              aria-label="Toggle AI Mode"
              title={
                !settings?.is_configured
                  ? "Configure an LLM provider first"
                  : settings?.ai_mode
                    ? "Disable AI Mode"
                    : "Enable AI Mode"
              }
            >
              <span
                className={cn(
                  "inline-block size-5 transform rounded-full bg-white shadow transition-transform",
                  settings?.ai_mode ? "translate-x-5" : "translate-x-0.5",
                )}
              />
            </button>
            <div className="min-w-0 flex-1 text-sm">
              <p className="font-medium">
                {settings?.ai_mode ? "AI Mode is on" : "AI Mode is off"}
              </p>
              <p className="text-xs text-muted-foreground">
                {settings?.ai_mode
                  ? "Run results across the app reflect the AI Mode presentation."
                  : "Run results across the app reflect the underlying execution data."}
              </p>
              {!settings?.is_configured && (
                <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
                  Configure an LLM provider above before enabling AI Mode.
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
