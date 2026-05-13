"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Sparkles, XCircle } from "lucide-react";
import { toast } from "sonner";

import {
  api,
  ApiError,
  type Provider,
  type RunCallLog,
  type RunCost,
  type Settings,
  type SettingsWrite,
  type TestConnectionResult,
} from "@/lib/api";
import { SearchableDropdown } from "@/components/searchable-dropdown";
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
    value: "openrouter",
    label: "OpenRouter",
    hint: "openrouter.ai — multi-model gateway (no base_url needed)",
  },
  {
    value: "openai_compat",
    label: "OpenAI-compatible",
    hint: "Ollama, vLLM, LM Studio, self-hosted; requires base_url",
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
  openrouter: [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-chat",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-pro",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "meta-llama/llama-4-405b-instruct",
    "qwen/qwen-3-vision-72b",
  ],
  openai_compat: [],
};

export default function SettingsPage() {
  const qc = useQueryClient();
  const { data: settings, isLoading } = useQuery({
    queryKey: ["settings"],
    queryFn: api.getSettings,
  });

  const [activeTab, setActiveTab] = useState<
    "models" | "tiers" | "cost" | "logs"
  >("models");

  const [provider, setProvider] = useState<Provider>("gemini");
  const [model, setModel] = useState("");
  const [cheapModel, setCheapModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [testResult, setTestResult] = useState<TestConnectionResult | null>(null);

  // Hydrate form once settings have loaded
  useEffect(() => {
    if (!settings) return;
    if (settings.provider) setProvider(settings.provider);
    if (settings.model) setModel(settings.model);
    if (settings.cheap_model) setCheapModel(settings.cheap_model);
    if (settings.base_url) setBaseUrl(settings.base_url);
  }, [settings]);

  const providerChanged =
    !!settings?.provider && settings.provider !== provider;
  const apiKeyOnFile = !!settings?.api_key_set && !providerChanged;
  // openai_compat REQUIRES a base_url (self-hosted endpoint).
  // openrouter is also OpenAI-API-compatible but the factory has its
  // base_url baked in, so the user doesn't have to supply one.
  const isCompat = provider === "openai_compat";
  const needsBaseUrl = isCompat;

  const buildPayload = (): SettingsWrite => {
    const payload: SettingsWrite = { provider, model: model.trim() };
    // Send cheap_model on every save (empty string clears tiering) so
    // the user can disable an existing tier by blanking the field.
    payload.cheap_model = cheapModel.trim();
    if (apiKey.trim()) payload.api_key = apiKey.trim();
    if (needsBaseUrl) payload.base_url = baseUrl.trim();
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

  // Phase A — Set-of-Mark default toggle. Same backend treatment as
  // AI Mode (flippable without re-supplying credentials).
  const somToggleMutation = useMutation({
    mutationFn: (next: boolean) =>
      api.upsertSettings({ som_enabled_default: next }),
    onSuccess: (resp) => {
      toast.success(
        resp.som_enabled_default
          ? "Set-of-Mark annotation enabled"
          : "Set-of-Mark annotation disabled",
      );
      qc.invalidateQueries({ queryKey: ["settings"] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Couldn't update SoM setting", { description: msg });
    },
  });

  const canTest =
    !!model.trim()
    && (apiKey.trim() || apiKeyOnFile)
    && (!needsBaseUrl || baseUrl.trim());
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
    <div className="mx-auto max-w-5xl space-y-6 p-8">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure provider models, per-tier token pricing, and review
          historical run costs.
        </p>
      </header>

      {/* Tab nav */}
      <div className="flex gap-1 border-b">
        {(
          [
            { id: "models", label: "Models Config" },
            { id: "tiers", label: "Tiers & Fallbacks" },
            { id: "cost", label: "Cost Settings" },
            { id: "logs", label: "Cost Logs" },
          ] as const
        ).map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "border-b-2 px-4 py-2 text-sm font-medium transition-colors",
              activeTab === tab.id
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "models" && (
        <>
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
            <Label htmlFor="model">Model (strong)</Label>
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
            <p className="text-xs text-muted-foreground">
              Used for the per-turn planner, action reasoning, and
              coordinate-click. Always the highest-precision tier.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="cheap-model">
              Cheap model (optional, escalation tier)
            </Label>
            <Input
              id="cheap-model"
              list={`model-suggestions-${provider}`}
              value={cheapModel}
              onChange={(e) => setCheapModel(e.target.value)}
              placeholder="leave blank to disable tiering"
            />
            <p className="text-xs text-muted-foreground">
              When set, the agent runs vision-search, on-track checks,
              goal verification, smart-pick, and semantic verify on
              this model first. If the cheap tier returns confidence
              below 0.7 or fails validation, the agent re-runs the
              call on the strong model above. Leave blank to send
              every call to the strong model (legacy behavior).
            </p>
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

      {/* ── Set-of-Mark annotation (Phase A) ───────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles
              className={cn(
                "size-5",
                settings?.som_enabled_default
                  ? "text-primary"
                  : "text-muted-foreground",
              )}
            />
            Set-of-Mark annotation
          </CardTitle>
          <CardDescription>
            When enabled (default), screenshots sent to vision-LLM
            helpers (smart-pick, on-track, goal-verify, vision-search,
            semantic-verify) get colored bounding boxes and numbered
            labels drawn on interactive elements before upload. The
            model can then refer to <em>&quot;box 5&quot;</em> instead
            of inventing pixel coordinates — published benchmarks show
            ~10–15% targeting accuracy improvement. Disable to send
            raw screenshots (legacy behavior).
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex items-start gap-3 rounded-md border p-3">
            <button
              type="button"
              onClick={() =>
                somToggleMutation.mutate(!settings?.som_enabled_default)
              }
              disabled={
                somToggleMutation.isPending || !settings?.is_configured
              }
              className={cn(
                "mt-0.5 inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                settings?.som_enabled_default
                  ? "border-primary/50 bg-primary"
                  : "border-input bg-muted",
                "disabled:cursor-not-allowed disabled:opacity-50",
              )}
              role="switch"
              aria-checked={!!settings?.som_enabled_default}
              aria-label="Toggle Set-of-Mark annotation"
              title={
                !settings?.is_configured
                  ? "Configure an LLM provider first"
                  : settings?.som_enabled_default
                    ? "Disable SoM annotation"
                    : "Enable SoM annotation"
              }
            >
              <span
                className={cn(
                  "inline-block size-5 transform rounded-full bg-white shadow transition-transform",
                  settings?.som_enabled_default
                    ? "translate-x-5"
                    : "translate-x-0.5",
                )}
              />
            </button>
            <div className="min-w-0 flex-1 text-sm">
              <p className="font-medium">
                {settings?.som_enabled_default
                  ? "Set-of-Mark is on"
                  : "Set-of-Mark is off"}
              </p>
              <p className="text-xs text-muted-foreground">
                {settings?.som_enabled_default
                  ? "VL screenshots include colored bounding boxes + numbered labels."
                  : "VL screenshots are sent raw — no annotation overlay."}
              </p>
              {!settings?.is_configured && (
                <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
                  Configure an LLM provider above before changing this.
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
        </>
      )}

      {activeTab === "tiers" && (
        <TiersAndFallbacksTab settings={settings ?? null} />
      )}
      {activeTab === "cost" && <CostSettingsTab settings={settings ?? null} />}
      {activeTab === "logs" && <CostLogsTab />}
    </div>
  );
}

// ── Tiers & Fallbacks tab ───────────────────────────────────────
//
// Per-tier provider configuration. Each non-strong tier can pick its
// own (provider, model, api_key, base_url). The factory falls back to
// the primary tier's credentials when fields are blank AND the
// providers match, so single-provider setups don't have to re-enter
// keys per tier.
function TiersAndFallbacksTab({
  settings,
}: {
  settings: Settings | null;
}) {
  const qc = useQueryClient();

  type TierKey = "cheap" | "fallback_strong" | "fallback_cheap";
  type TierState = {
    provider: Provider | "";
    model: string;
    apiKey: string;
    baseUrl: string;
  };
  const [state, setState] = useState<Record<TierKey, TierState>>({
    cheap: { provider: "", model: "", apiKey: "", baseUrl: "" },
    fallback_strong: { provider: "", model: "", apiKey: "", baseUrl: "" },
    fallback_cheap: { provider: "", model: "", apiKey: "", baseUrl: "" },
  });

  useEffect(() => {
    if (!settings) return;
    setState({
      cheap: {
        provider: (settings.cheap_provider ?? settings.provider) || "",
        model: settings.cheap_model ?? "",
        apiKey: "",
        baseUrl: settings.cheap_base_url ?? "",
      },
      fallback_strong: {
        provider: settings.fallback_strong_provider ?? "",
        model: settings.fallback_strong_model ?? "",
        apiKey: "",
        baseUrl: settings.fallback_strong_base_url ?? "",
      },
      fallback_cheap: {
        provider: settings.fallback_cheap_provider ?? "",
        model: settings.fallback_cheap_model ?? "",
        apiKey: "",
        baseUrl: settings.fallback_cheap_base_url ?? "",
      },
    });
  }, [settings]);

  const saveMutation = useMutation({
    mutationFn: () => {
      const payload: SettingsWrite = {};
      // Cheap (primary fallback tier — model goes on the existing
      // cheap_model field; provider / key / base_url use the new
      // *_provider / *_api_key / *_base_url fields).
      payload.cheap_model = state.cheap.model.trim();
      if (state.cheap.provider) payload.cheap_provider = state.cheap.provider;
      if (state.cheap.apiKey.trim()) payload.cheap_api_key = state.cheap.apiKey.trim();
      payload.cheap_base_url = state.cheap.baseUrl.trim();

      payload.fallback_strong_model = state.fallback_strong.model.trim();
      if (state.fallback_strong.provider)
        payload.fallback_strong_provider = state.fallback_strong.provider;
      if (state.fallback_strong.apiKey.trim())
        payload.fallback_strong_api_key = state.fallback_strong.apiKey.trim();
      payload.fallback_strong_base_url = state.fallback_strong.baseUrl.trim();

      payload.fallback_cheap_model = state.fallback_cheap.model.trim();
      if (state.fallback_cheap.provider)
        payload.fallback_cheap_provider = state.fallback_cheap.provider;
      if (state.fallback_cheap.apiKey.trim())
        payload.fallback_cheap_api_key = state.fallback_cheap.apiKey.trim();
      payload.fallback_cheap_base_url = state.fallback_cheap.baseUrl.trim();

      return api.upsertSettings(payload);
    },
    onSuccess: () => {
      toast.success("Tier configuration saved");
      qc.invalidateQueries({ queryKey: ["settings"] });
      setState((s) => ({
        cheap: { ...s.cheap, apiKey: "" },
        fallback_strong: { ...s.fallback_strong, apiKey: "" },
        fallback_cheap: { ...s.fallback_cheap, apiKey: "" },
      }));
    },
    onError: (e: Error) => {
      toast.error("Save failed", {
        description: e instanceof ApiError ? e.message : e.message,
      });
    },
  });

  if (!settings?.is_configured) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Per-tier providers</CardTitle>
          <CardDescription>
            Configure the primary (strong) provider in the{" "}
            <strong>Models Config</strong> tab first. Per-tier providers
            and fallbacks become available once a strong tier is saved.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Tiers & Fallbacks</CardTitle>
        <CardDescription>
          Each tier picks its own (provider, model). Blank fields fall
          back to the primary tier's credentials when the providers
          match. <strong>Cheap</strong> handles high-volume helper
          calls. <strong>Fallback strong</strong> /{" "}
          <strong>Fallback cheap</strong> fire only when the primary
          tier raises (rate limit, network, 5xx).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <PrimaryTierSummary settings={settings} />
        {(
          [
            {
              key: "cheap",
              title: "Cheap tier",
              blurb:
                "Vision helpers, smart-pick, on-track, verifier. Skipped when blank.",
            },
            {
              key: "fallback_strong",
              title: "Fallback for strong",
              blurb:
                "Used only when the strong tier's call raises. Different provider recommended.",
            },
            {
              key: "fallback_cheap",
              title: "Fallback for cheap",
              blurb:
                "Used only when the cheap tier's call raises. Optional.",
            },
          ] as const
        ).map((tier) => (
          <TierEditor
            key={tier.key}
            title={tier.title}
            blurb={tier.blurb}
            state={state[tier.key]}
            keySet={
              tier.key === "cheap"
                ? settings.cheap_api_key_set
                : tier.key === "fallback_strong"
                ? settings.fallback_strong_api_key_set
                : settings.fallback_cheap_api_key_set
            }
            onChange={(next) =>
              setState((s) => ({ ...s, [tier.key]: next }))
            }
          />
        ))}
      </CardContent>
      <CardFooter className="justify-end">
        <Button
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending}
        >
          {saveMutation.isPending ? "Saving…" : "Save tier configuration"}
        </Button>
      </CardFooter>
    </Card>
  );
}

function PrimaryTierSummary({ settings }: { settings: Settings }) {
  return (
    <div className="rounded-md border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
      <span className="font-medium text-foreground">Strong tier (primary):</span>{" "}
      {settings.provider} · <span className="font-mono">{settings.model}</span>
      {settings.base_url ? ` · ${settings.base_url}` : ""}
      {" — edit in the Models Config tab."}
    </div>
  );
}

function TierEditor({
  title,
  blurb,
  state,
  keySet,
  onChange,
}: {
  title: string;
  blurb: string;
  state: { provider: Provider | ""; model: string; apiKey: string; baseUrl: string };
  keySet: boolean;
  onChange: (
    next: { provider: Provider | ""; model: string; apiKey: string; baseUrl: string },
  ) => void;
}) {
  const needsBaseUrl = state.provider === "openai_compat";
  return (
    <div className="space-y-3 rounded-md border p-3">
      <div>
        <h4 className="text-sm font-medium">{title}</h4>
        <p className="text-xs text-muted-foreground">{blurb}</p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs">Provider</Label>
          <select
            value={state.provider}
            onChange={(e) =>
              onChange({ ...state, provider: e.target.value as Provider | "" })
            }
            className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm"
          >
            <option value="">(inherit primary)</option>
            {PROVIDER_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">Model</Label>
          <Input
            value={state.model}
            onChange={(e) => onChange({ ...state, model: e.target.value })}
            placeholder="(blank = tier disabled)"
            list={
              state.provider
                ? `models-${title.replace(/\s+/g, "-")}-${state.provider}`
                : undefined
            }
          />
          {state.provider && MODEL_SUGGESTIONS[state.provider as Provider]?.length > 0 && (
            <datalist
              id={`models-${title.replace(/\s+/g, "-")}-${state.provider}`}
            >
              {MODEL_SUGGESTIONS[state.provider as Provider].map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          )}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs">
            API key {keySet ? <span className="text-emerald-600">· saved</span> : null}
          </Label>
          <Input
            type="password"
            value={state.apiKey}
            onChange={(e) => onChange({ ...state, apiKey: e.target.value })}
            placeholder={
              keySet
                ? "(leave blank to keep)"
                : "(blank = inherit primary when provider matches)"
            }
            autoComplete="off"
          />
        </div>
        {needsBaseUrl && (
          <div className="space-y-1">
            <Label className="text-xs">Base URL</Label>
            <Input
              value={state.baseUrl}
              onChange={(e) =>
                onChange({ ...state, baseUrl: e.target.value })
              }
              placeholder="https://your-endpoint/v1"
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Cost Settings tab ────────────────────────────────────────────


function CostSettingsTab({ settings }: { settings: typeof settingsType | null }) {
  const qc = useQueryClient();
  const [strongIn, setStrongIn] = useState<string>("");
  const [strongOut, setStrongOut] = useState<string>("");
  const [cheapIn, setCheapIn] = useState<string>("");
  const [cheapOut, setCheapOut] = useState<string>("");
  // Cached-input rates — typically ~50% of regular input on OpenAI.
  const [strongCachedIn, setStrongCachedIn] = useState<string>("");
  const [cheapCachedIn, setCheapCachedIn] = useState<string>("");

  useEffect(() => {
    if (!settings) return;
    setStrongIn(
      settings.strong_input_price_per_m != null
        ? String(settings.strong_input_price_per_m)
        : "",
    );
    setStrongOut(
      settings.strong_output_price_per_m != null
        ? String(settings.strong_output_price_per_m)
        : "",
    );
    setCheapIn(
      settings.cheap_input_price_per_m != null
        ? String(settings.cheap_input_price_per_m)
        : "",
    );
    setCheapOut(
      settings.cheap_output_price_per_m != null
        ? String(settings.cheap_output_price_per_m)
        : "",
    );
    setStrongCachedIn(
      settings.strong_cached_input_price_per_m != null
        ? String(settings.strong_cached_input_price_per_m)
        : "",
    );
    setCheapCachedIn(
      settings.cheap_cached_input_price_per_m != null
        ? String(settings.cheap_cached_input_price_per_m)
        : "",
    );
  }, [settings]);

  const parseRate = (s: string): number => {
    const n = parseFloat(s);
    if (!Number.isFinite(n) || n < 0) return 0;
    return n;
  };

  const saveMutation = useMutation({
    mutationFn: () =>
      api.upsertSettings({
        strong_input_price_per_m: parseRate(strongIn),
        strong_output_price_per_m: parseRate(strongOut),
        cheap_input_price_per_m: parseRate(cheapIn),
        cheap_output_price_per_m: parseRate(cheapOut),
        strong_cached_input_price_per_m: parseRate(strongCachedIn),
        cheap_cached_input_price_per_m: parseRate(cheapCachedIn),
      }),
    onSuccess: () => {
      toast.success("Pricing saved");
      qc.invalidateQueries({ queryKey: ["settings"] });
      qc.invalidateQueries({ queryKey: ["cost-runs"] });
      qc.invalidateQueries({ queryKey: ["cost-aggregate"] });
    },
    onError: (e: Error) => {
      const msg = e instanceof ApiError ? e.message : e.message;
      toast.error("Save failed", { description: msg });
    },
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cost Settings</CardTitle>
        <CardDescription>
          USD per million tokens. Strong tier is your primary model
          ({settings?.model ?? "—"}); cheap tier is the escalation
          model ({settings?.cheap_model ?? "—"}). Embeddings are
          local (BGE on CPU) and free. Leave a field at 0 to mark
          it &quot;not configured&quot; (Cost views show $— for that
          line). Changing pricing re-costs historical runs at the
          new rate.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <PriceField
            label="Strong · input $/M"
            value={strongIn}
            onChange={setStrongIn}
            hint={settings?.model ?? "strong model"}
          />
          <PriceField
            label="Strong · output $/M"
            value={strongOut}
            onChange={setStrongOut}
            hint={settings?.model ?? "strong model"}
          />
          <PriceField
            label="Cheap · input $/M"
            value={cheapIn}
            onChange={setCheapIn}
            hint={settings?.cheap_model ?? "cheap model"}
          />
          <PriceField
            label="Cheap · output $/M"
            value={cheapOut}
            onChange={setCheapOut}
            hint={settings?.cheap_model ?? "cheap model"}
          />
        </div>

        <div className="border-t pt-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Cached input rates (optional)
          </p>
          <p className="mb-3 text-xs text-muted-foreground">
            Prompt caching gives a discount on repeated input tokens.
            OpenAI applies it automatically on prompts ≥ 1024 tokens
            at ~50% of the regular input rate; Gemini's{" "}
            <code>cached_content</code> API charges ~25%. Set these
            to your provider&apos;s cached rate so cost reflects
            real billing. Leave blank to bill the cached portion at
            the regular input rate (safe over-estimate).
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <PriceField
              label="Strong · cached input $/M"
              value={strongCachedIn}
              onChange={setStrongCachedIn}
              hint={
                settings?.model
                  ? `${settings.model} (try ~50% of input)`
                  : "strong model"
              }
            />
            <PriceField
              label="Cheap · cached input $/M"
              value={cheapCachedIn}
              onChange={setCheapCachedIn}
              hint={
                settings?.cheap_model
                  ? `${settings.cheap_model} (try ~50% of input)`
                  : "cheap model"
              }
            />
          </div>
        </div>
      </CardContent>
      <CardFooter className="justify-end">
        <Button
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending}
        >
          {saveMutation.isPending ? "Saving…" : "Save pricing"}
        </Button>
      </CardFooter>
    </Card>
  );
}


function PriceField({
  label,
  value,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  hint: string;
}) {
  return (
    <div className="space-y-1">
      <Label className="text-xs font-medium">{label}</Label>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">$</span>
        <Input
          type="number"
          min={0}
          step="0.01"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="0.00"
          className="font-mono"
        />
      </div>
      <p className="text-[10px] text-muted-foreground">{hint}</p>
    </div>
  );
}


// ── Cost Logs tab ────────────────────────────────────────────────


function CostLogsTab() {
  const { data: agg } = useQuery({
    queryKey: ["cost-aggregate"],
    queryFn: () => api.aggregateCost({ limit: 500 }),
  });
  const { data: runs } = useQuery({
    queryKey: ["cost-runs"],
    queryFn: () => api.listRunCosts({ limit: 200 }),
  });
  // Drill-in pickers — populated by the projects list + per-project
  // run list. The selected projectId filters the run dropdown; the
  // selected runId fetches the per-LLM-call table below.
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(
    null,
  );
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const { data: projects } = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });
  const { data: drillRuns } = useQuery({
    queryKey: ["drill-runs", selectedProjectId],
    queryFn: () =>
      selectedProjectId
        ? api.listAgentRuns(Number(selectedProjectId))
        : Promise.resolve([] as never),
    enabled: selectedProjectId != null,
  });
  const { data: calls } = useQuery({
    queryKey: ["call-log", selectedRunId],
    queryFn: () =>
      selectedRunId
        ? api.listRunCallLogs(Number(selectedRunId))
        : Promise.resolve(null),
    enabled: selectedRunId != null,
  });

  const fmtUsd = (n: number | null | undefined) =>
    n == null ? "$—" : `$${n.toFixed(4)}`;
  const fmtUsdShort = (n: number | null | undefined) =>
    n == null ? "$—" : `$${n.toFixed(2)}`;
  const fmtTokens = (n: number) => n.toLocaleString();

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Total spend</CardTitle>
          <CardDescription>
            Roll-up across the last 500 runs (execute + recon +
            generation). Embeddings excluded (local CPU, free).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Metric
              label="Runs"
              value={agg ? String(agg.run_count) : "—"}
            />
            <Metric
              label="Total cost"
              value={fmtUsdShort(agg?.total_cost_usd ?? null)}
              tone="primary"
            />
            <Metric
              label="Strong tokens"
              value={
                agg
                  ? fmtTokens(
                      agg.total_strong_input_tokens +
                        agg.total_strong_cached_input_tokens +
                        agg.total_strong_output_tokens,
                    )
                  : "—"
              }
            />
            <Metric
              label="Cheap tokens"
              value={
                agg
                  ? fmtTokens(
                      agg.total_cheap_input_tokens +
                        agg.total_cheap_cached_input_tokens +
                        agg.total_cheap_output_tokens,
                    )
                  : "—"
              }
            />
          </div>
          {agg && (
            agg.total_strong_cached_input_tokens > 0
              || agg.total_cheap_cached_input_tokens > 0
          ) && (
            <div className="mt-3 rounded-md bg-emerald-500/5 px-3 py-2 text-xs text-emerald-700 dark:text-emerald-400">
              Cache hits this period:{" "}
              <span className="font-mono font-semibold">
                {fmtTokens(
                  agg.total_strong_cached_input_tokens
                    + agg.total_cheap_cached_input_tokens,
                )}
              </span>{" "}
              cached input tokens billed at the cached rate.
            </div>
          )}
          {agg && Object.keys(agg.by_kind).length > 0 && (
            <div className="mt-4 space-y-1 border-t pt-3">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                By kind
              </p>
              <ul className="grid grid-cols-2 gap-1 text-sm sm:grid-cols-3">
                {Object.entries(agg.by_kind).map(([k, v]) => (
                  <li key={k} className="flex justify-between">
                    <span className="text-muted-foreground">{k}</span>
                    <span className="font-mono">{fmtUsdShort(v)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Per-run breakdown</CardTitle>
          <CardDescription>
            Most recent 200 runs. Runs flagged &quot;est.&quot; predate
            per-tier tracking and assume strong-tier for their
            aggregate tokens.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead className="border-b text-left text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-3 py-2">Run</th>
                <th className="px-3 py-2">Kind</th>
                <th className="px-3 py-2">Strong model</th>
                <th className="px-3 py-2">Cheap model</th>
                <th className="px-3 py-2 text-right">Strong tok</th>
                <th className="px-3 py-2 text-right">Cheap tok</th>
                <th
                  className="px-3 py-2 text-right"
                  title="Cached input tokens (subset of input, billed at the cached rate)"
                >
                  Cached
                </th>
                <th className="px-3 py-2 text-right">Cost</th>
              </tr>
            </thead>
            <tbody>
              {(runs?.runs ?? []).map((r: RunCost) => {
                // Strong/cheap totals include cached portions so the
                // table number matches the run-detail page's total.
                const strongTok =
                  (r.lines.find(
                    (ln) => ln.tier === "strong" && ln.direction === "input",
                  )?.tokens ?? 0) +
                  (r.lines.find(
                    (ln) => ln.tier === "strong"
                      && ln.direction === "input_cached",
                  )?.tokens ?? 0) +
                  (r.lines.find(
                    (ln) => ln.tier === "strong" && ln.direction === "output",
                  )?.tokens ?? 0);
                const cheapTok =
                  (r.lines.find(
                    (ln) => ln.tier === "cheap" && ln.direction === "input",
                  )?.tokens ?? 0) +
                  (r.lines.find(
                    (ln) => ln.tier === "cheap"
                      && ln.direction === "input_cached",
                  )?.tokens ?? 0) +
                  (r.lines.find(
                    (ln) => ln.tier === "cheap" && ln.direction === "output",
                  )?.tokens ?? 0);
                const cachedTok =
                  (r.lines.find(
                    (ln) => ln.tier === "strong"
                      && ln.direction === "input_cached",
                  )?.tokens ?? 0) +
                  (r.lines.find(
                    (ln) => ln.tier === "cheap"
                      && ln.direction === "input_cached",
                  )?.tokens ?? 0);
                return (
                  <tr key={r.run_id} className="border-b last:border-b-0">
                    <td className="px-3 py-2 font-mono">#{r.run_id}</td>
                    <td className="px-3 py-2">{r.kind}</td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {r.strong_model ?? "—"}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {r.cheap_model ?? "—"}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {fmtTokens(strongTok)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {fmtTokens(cheapTok)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-muted-foreground">
                      {cachedTok > 0 ? fmtTokens(cachedTok) : "—"}
                    </td>
                    <td className="px-3 py-2 text-right font-mono">
                      {fmtUsd(r.total_cost_usd)}
                      {r.estimated_from_aggregate && (
                        <span className="ml-1 rounded bg-muted px-1 text-[10px] uppercase">
                          est.
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {(!runs || runs.runs.length === 0) && (
                <tr>
                  <td
                    colSpan={8}
                    className="px-3 py-8 text-center text-muted-foreground"
                  >
                    No runs yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {/* ── Drill-in: per-LLM-call telemetry for one run ───────── */}
      <Card>
        <CardHeader>
          <CardTitle>Drill into a run</CardTitle>
          <CardDescription>
            Pick a project, then a run within it. Below shows every
            individual LLM call the run made — role, model, tier,
            tokens, duration, and per-call cost — with the sum at
            the bottom.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <SearchableDropdown
              label="Project"
              placeholder="Select project…"
              options={(projects ?? []).map((p) => ({
                value: String(p.id),
                label: p.name,
                hint: `#${p.id}`,
              }))}
              value={selectedProjectId}
              onChange={(v) => {
                setSelectedProjectId(v);
                setSelectedRunId(null);
              }}
            />
            <SearchableDropdown
              label="Run"
              placeholder={
                selectedProjectId
                  ? "Select run…"
                  : "Pick a project first"
              }
              options={(drillRuns ?? []).map((r) => ({
                value: String(r.id),
                label: `#${r.id} · ${r.kind} · ${r.status}`,
                hint: r.created_at
                  ? new Date(r.created_at).toLocaleString()
                  : undefined,
              }))}
              value={selectedRunId}
              onChange={setSelectedRunId}
              disabled={!selectedProjectId}
            />
          </div>

          {!selectedRunId ? (
            <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
              Select a project + run to load its call log.
            </p>
          ) : calls === undefined ? (
            <Skeleton className="h-48 w-full" />
          ) : calls === null || calls.call_count === 0 ? (
            <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
              No LLM calls recorded for this run. Pre-feature runs
              (before per-call tracking landed) have aggregate
              totals only — see the table above.
            </p>
          ) : (
            <CallLogTable data={calls} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}


function CallLogTable({ data }: { data: RunCallLog }) {
  const fmtUsd = (n: number | null | undefined) =>
    n == null ? "$—" : `$${n.toFixed(6)}`;
  const fmtTokens = (n: number) => n.toLocaleString();
  return (
    <div className="overflow-x-auto rounded-md border">
      <div className="flex flex-wrap items-baseline gap-4 border-b bg-muted/30 px-3 py-2 text-xs">
        <span>
          Run <span className="font-mono">#{data.run_id}</span> ·{" "}
          {data.kind} · <strong>{data.call_count}</strong> call
          {data.call_count === 1 ? "" : "s"}
        </span>
        {data.strong_model && (
          <span className="text-muted-foreground">
            strong: <span className="font-mono">{data.strong_model}</span>
          </span>
        )}
        {data.cheap_model && (
          <span className="text-muted-foreground">
            cheap: <span className="font-mono">{data.cheap_model}</span>
          </span>
        )}
      </div>
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase text-muted-foreground">
          <tr>
            <th className="px-3 py-2">#</th>
            <th className="px-3 py-2">Role</th>
            <th className="px-3 py-2">Tier</th>
            <th className="px-3 py-2">Model</th>
            <th className="px-3 py-2">Step</th>
            <th className="px-3 py-2 text-right">In tok</th>
            <th className="px-3 py-2 text-right" title="Cached subset of input tokens (billed at the cached rate)">
              Cached
            </th>
            <th className="px-3 py-2 text-right">Out tok</th>
            <th className="px-3 py-2 text-right">Duration</th>
            <th className="px-3 py-2 text-right">Cost</th>
          </tr>
        </thead>
        <tbody>
          {data.calls.map((c) => (
            <tr key={c.id} className="border-t">
              <td className="px-3 py-1.5 font-mono text-xs">
                {c.ordinal + 1}
              </td>
              <td className="px-3 py-1.5">
                <span className="font-medium">{c.role}</span>
                {c.escalated && (
                  <span
                    className="ml-1 rounded bg-amber-500/20 px-1 text-[9px] uppercase text-amber-700 dark:text-amber-400"
                    title="Escalated from cheap to strong on this call"
                  >
                    esc
                  </span>
                )}
              </td>
              <td className="px-3 py-1.5">
                <span
                  className={cn(
                    "rounded px-1.5 py-0.5 text-[10px] font-medium",
                    c.tier === "strong"
                      ? "bg-primary/15 text-primary"
                      : "bg-muted text-muted-foreground",
                  )}
                >
                  {c.tier}
                </span>
              </td>
              <td className="px-3 py-1.5 font-mono text-xs">
                {c.model ?? "—"}
              </td>
              <td className="px-3 py-1.5 text-xs">
                {c.step_title ?? (
                  <span className="text-muted-foreground">—</span>
                )}
              </td>
              <td className="px-3 py-1.5 text-right font-mono">
                {fmtTokens(c.input_tokens)}
              </td>
              <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                {c.cached_input_tokens > 0
                  ? fmtTokens(c.cached_input_tokens)
                  : "—"}
              </td>
              <td className="px-3 py-1.5 text-right font-mono">
                {fmtTokens(c.output_tokens)}
              </td>
              <td className="px-3 py-1.5 text-right font-mono text-xs">
                {c.duration_ms == null ? "—" : `${c.duration_ms}ms`}
              </td>
              <td className="px-3 py-1.5 text-right font-mono">
                {fmtUsd(c.total_cost_usd)}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot className="border-t bg-muted/30">
          <tr>
            <td colSpan={9} className="px-3 py-2 text-right font-medium">
              Sum
            </td>
            <td className="px-3 py-2 text-right font-mono font-semibold">
              {fmtUsd(data.sum_total_cost_usd)}
            </td>
          </tr>
          {data.sum_input_cost_usd != null && (
            <tr className="text-xs text-muted-foreground">
              <td colSpan={9} className="px-3 py-1 text-right">
                · input (regular)
              </td>
              <td className="px-3 py-1 text-right font-mono">
                {fmtUsd(data.sum_input_cost_usd)}
              </td>
            </tr>
          )}
          {data.sum_cached_input_cost_usd != null
            && (data.sum_cached_input_cost_usd > 0
              || data.calls.some((c) => c.cached_input_tokens > 0)) && (
            <tr className="text-xs text-muted-foreground">
              <td colSpan={9} className="px-3 py-1 text-right">
                · input (cached)
              </td>
              <td className="px-3 py-1 text-right font-mono">
                {fmtUsd(data.sum_cached_input_cost_usd)}
              </td>
            </tr>
          )}
          {data.sum_output_cost_usd != null && (
            <tr className="text-xs text-muted-foreground">
              <td colSpan={9} className="px-3 py-1 text-right">
                · output
              </td>
              <td className="px-3 py-1 text-right font-mono">
                {fmtUsd(data.sum_output_cost_usd)}
              </td>
            </tr>
          )}
        </tfoot>
      </table>
    </div>
  );
}


function Metric({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "primary";
}) {
  return (
    <div className="rounded-md border p-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          "mt-1 text-2xl font-semibold tabular-nums",
          tone === "primary" && "text-primary",
        )}
      >
        {value}
      </p>
    </div>
  );
}


// Helper alias so the CostSettingsTab type signature stays readable.
type settingsType = NonNullable<Awaited<ReturnType<typeof api.getSettings>>>;
