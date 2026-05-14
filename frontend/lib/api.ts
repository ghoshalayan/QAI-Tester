/**
 * Backend client + DTO types.
 *
 * One thin wrapper, no auto-retry, no caching — TanStack Query handles those
 * concerns at the call site via hooks.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? detail;
    } catch {
      /* ignore — non-JSON body */
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── Types ─────────────────────────────────────────────────────────

export type Provider =
  | "gemini"
  | "openai"
  | "openai_compat"
  | "openrouter";

export interface Settings {
  is_configured: boolean;
  provider: Provider | null;
  model: string | null;
  /** Phase 1 — cheap-tier model. When set, vision helpers (search,
   * on-track, goal-verify, smart-pick, semantic-verify) try this
   * model first and escalate to ``model`` on low confidence.
   * null/empty = no tiering. */
  cheap_model: string | null;
  base_url: string | null;
  api_key_set: boolean;
  /** AI Mode toggle. When true, run summaries and per-row statuses
   * are transformed at the API boundary to a deterministic 80-90%
   * pass-rate distribution. Real data is untouched. */
  ai_mode: boolean;
  /** Phase A — Set-of-Mark annotation on VL screenshots. When true
   * (default), screenshots sent to VL helpers get colored bounding
   * boxes + numbered labels drawn before upload. The VL refers to
   * "box 5" instead of inventing pixel coords — ~10-15% targeting
   * accuracy improvement in published benchmarks. */
  som_enabled_default: boolean;
  /** Cost tracking — USD per million tokens. null = not configured
   * (the cost surface shows ``$—`` for that tier/direction). */
  strong_input_price_per_m: number | null;
  strong_output_price_per_m: number | null;
  cheap_input_price_per_m: number | null;
  cheap_output_price_per_m: number | null;
  /** Cached-input rate. null → cost service falls back to the
   * regular input rate (over-bills the cached portion slightly,
   * never under-bills). Set to ~50% of regular input for OpenAI. */
  strong_cached_input_price_per_m: number | null;
  cheap_cached_input_price_per_m: number | null;

  /** Migration 0025 — per-tier provider config. Each non-strong tier
   * can pick its own (provider, model, api_key, base_url). NULL on
   * any field falls back to the primary (strong) tier's value when
   * the providers match — keeps single-provider setups working
   * without re-entering keys per tier. ``*_api_key`` is never echoed
   * to the client; the ``*_api_key_set`` flag reflects whether the
   * tier has a key configured. */
  cheap_provider: Provider | null;
  cheap_base_url: string | null;
  cheap_api_key_set: boolean;

  fallback_strong_provider: Provider | null;
  fallback_strong_model: string | null;
  fallback_strong_base_url: string | null;
  fallback_strong_api_key_set: boolean;
  fallback_strong_input_price_per_m: number | null;
  fallback_strong_output_price_per_m: number | null;
  fallback_strong_cached_input_price_per_m: number | null;

  fallback_cheap_provider: Provider | null;
  fallback_cheap_model: string | null;
  fallback_cheap_base_url: string | null;
  fallback_cheap_api_key_set: boolean;
  fallback_cheap_input_price_per_m: number | null;
  fallback_cheap_output_price_per_m: number | null;
  fallback_cheap_cached_input_price_per_m: number | null;

  updated_at: string | null;
}

export interface SettingsWrite {
  provider?: Provider;
  model?: string;
  /** Phase 1 — cheap-tier model. Empty string clears tiering. */
  cheap_model?: string;
  api_key?: string;
  base_url?: string;
  ai_mode?: boolean;
  som_enabled_default?: boolean;
  /** Cost — USD per million tokens. Send 0 to clear; >0 to set. */
  strong_input_price_per_m?: number;
  strong_output_price_per_m?: number;
  cheap_input_price_per_m?: number;
  cheap_output_price_per_m?: number;
  strong_cached_input_price_per_m?: number;
  cheap_cached_input_price_per_m?: number;

  /** Migration 0025 — per-tier provider fields. Send empty string ""
   * to clear a field (disables the tier when applied to ``*_model``);
   * omit to preserve. */
  cheap_provider?: Provider;
  cheap_api_key?: string;
  cheap_base_url?: string;

  fallback_strong_provider?: Provider;
  fallback_strong_model?: string;
  fallback_strong_api_key?: string;
  fallback_strong_base_url?: string;
  fallback_strong_input_price_per_m?: number;
  fallback_strong_output_price_per_m?: number;
  fallback_strong_cached_input_price_per_m?: number;

  fallback_cheap_provider?: Provider;
  fallback_cheap_model?: string;
  fallback_cheap_api_key?: string;
  fallback_cheap_base_url?: string;
  fallback_cheap_input_price_per_m?: number;
  fallback_cheap_output_price_per_m?: number;
  fallback_cheap_cached_input_price_per_m?: number;
}

/** One line in the cost breakdown — tier × direction.
 * ``input_cached`` is the prompt-cached subset, billed at the
 * cached rate (typically ~50% of regular input for OpenAI). */
export interface CostLine {
  tier: "strong" | "cheap";
  direction: "input" | "input_cached" | "output";
  tokens: number;
  price_per_m: number | null;
  cost_usd: number | null;
}

/** Per-run breakdown returned by /settings/cost/runs/{run_id}. */
export interface RunCost {
  run_id: number;
  kind: string;
  status?: string;
  project_id?: number;
  plan_id?: number | null;
  strong_model: string | null;
  cheap_model: string | null;
  estimated_from_aggregate: boolean;
  lines: CostLine[];
  total_cost_usd: number | null;
  created_at?: string | null;
}

/** Aggregate roll-up across runs for the Cost Logs dashboard. */
export interface AggregateCost {
  run_count: number;
  total_strong_input_tokens: number;
  total_strong_output_tokens: number;
  total_cheap_input_tokens: number;
  total_cheap_output_tokens: number;
  /** Cached portions — sum of input_cached tokens across all runs.
   * Helps the user gauge "how much of my input is being cached". */
  total_strong_cached_input_tokens: number;
  total_cheap_cached_input_tokens: number;
  total_cost_usd: number | null;
  by_kind: Record<string, number>;
}

/** Per-LLM-call row returned by the drill-in endpoint. */
export interface CallLogEntry {
  id: number;
  ordinal: number;
  step_id: number | null;
  step_title: string | null;
  role: string;
  tier: "strong" | "cheap";
  model: string | null;
  /** Total prompt tokens (regular + cached combined). */
  input_tokens: number;
  output_tokens: number;
  /** Cached subset of input_tokens — billed at the cached rate. */
  cached_input_tokens: number;
  /** Regular = input_tokens - cached_input_tokens (server-computed
   * so the UI doesn't have to). */
  regular_input_tokens: number;
  /** Regular-input cost (regular_input_tokens × regular rate). */
  input_cost_usd: number | null;
  /** Cached-input cost (cached_input_tokens × cached rate). */
  cached_input_cost_usd: number | null;
  output_cost_usd: number | null;
  total_cost_usd: number | null;
  escalated: boolean;
  duration_ms: number | null;
  created_at: string | null;
}

/** Response shape from GET /cost/runs/{run_id}/calls. */
export interface RunCallLog {
  run_id: number;
  kind: string;
  strong_model: string | null;
  cheap_model: string | null;
  call_count: number;
  calls: CallLogEntry[];
  sum_input_cost_usd: number | null;
  sum_cached_input_cost_usd: number | null;
  sum_output_cost_usd: number | null;
  sum_total_cost_usd: number | null;
}

export interface TestConnectionResult {
  ok: boolean;
  provider: string;
  model: string;
  base_url: string | null;
  echo: string | null;
  latency_ms: number | null;
  error: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
}

export interface Project {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectCreate {
  name: string;
  description?: string;
}

export interface ProjectUpdate {
  name?: string;
  description?: string;
}

// ── Documents ─────────────────────────────────────────────────────

export type DocumentKind = "BRD" | "FRD" | "INSTRUCTIONS";

/** Display labels for the three document kinds (for UI use). */
export const DOCUMENT_KIND_LABELS: Record<DocumentKind, string> = {
  BRD: "BRD",
  FRD: "FRD",
  INSTRUCTIONS: "Instructions",
};
export type DocumentStatus =
  | "pending"
  | "parsing"
  | "embedding"
  | "parsed"
  | "failed";
export type DocumentSourceType = "pdf" | "docx" | "md" | "paste";

export interface DocumentRead {
  id: number;
  project_id: number;
  kind: DocumentKind;
  source_type: DocumentSourceType;
  filename: string;
  status: DocumentStatus;
  error_message: string | null;
  char_count: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface PasteRequest {
  kind: DocumentKind;
  title?: string;
  content: string;
}

export interface DocumentParsed {
  document_id: number;
  parsed_md: string;
  char_count: number;
}

export interface ChunkRead {
  id: number;
  document_id: number;
  ordinal: number;
  heading_path: string | null;
  anchor: string | null;
  text: string;
  char_count: number;
}

export interface SearchRequest {
  query: string;
  k?: number;
  kind?: DocumentKind;
}

export interface SearchHit {
  chunk_id: number;
  document_id: number;
  document_kind: DocumentKind;
  document_filename: string;
  heading_path: string | null;
  anchor: string | null;
  text: string;
  score: number;
}

export interface SearchResponse {
  query: string;
  k: number;
  hits: SearchHit[];
}

// ── Test Plans ────────────────────────────────────────────────────

export type PlanStatus = "draft" | "ready" | "archived";

export const PLAN_STATUS_LABELS: Record<PlanStatus, string> = {
  draft: "Draft",
  ready: "Ready",
  archived: "Archived",
};

export interface CredentialRead {
  id: number;
  plan_id: number;
  label: string;
  username: string;
  password_set: boolean;
  /** Phase 3 — TOTP indicator. True when a TOTP seed is stored;
   * agent will auto-generate codes without HITL. */
  totp_set: boolean;
  url_pattern: string | null;
  username_selector_hint: string | null;
  password_selector_hint: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface CredentialCreate {
  label: string;
  username: string;
  password: string;
  /** Phase 3 — base32 seed OR ``otpauth://`` URI. Vault normalizes. */
  totp_secret?: string;
  url_pattern?: string;
  username_selector_hint?: string;
  password_selector_hint?: string;
  notes?: string;
}

export interface CredentialUpdate {
  label?: string;
  username?: string;
  /** Empty string or omitted → keep existing password. */
  password?: string;
  /** Empty string clears existing TOTP seed; null/undefined preserves. */
  totp_secret?: string;
  url_pattern?: string;
  username_selector_hint?: string;
  password_selector_hint?: string;
  notes?: string;
}

export interface LinkedDocSummary {
  document_id: number;
  filename: string;
  kind: DocumentKind;
  status: DocumentStatus;
  chunk_count: number;
}

export interface PlanCreate {
  name: string;
  target_url: string;
  description?: string;
  scope?: string[];
  status?: PlanStatus;
  linked_document_ids?: number[];
  credentials?: CredentialCreate[];
  /** Phase A — replan budget per submodule. Range 0-5; default 2.
   * 0 disables sub-goal replanning entirely. */
  max_replans_per_submodule?: number;
}

export interface PlanUpdate {
  name?: string;
  target_url?: string;
  description?: string;
  scope?: string[];
  status?: PlanStatus;
  /** When present, replaces the entire set of linked docs. */
  linked_document_ids?: number[];
  max_replans_per_submodule?: number;
}

export interface PlanReadCompact {
  id: number;
  project_id: number;
  name: string;
  target_url: string;
  scope: string[];
  status: PlanStatus;
  credential_count: number;
  linked_document_count: number;
  created_at: string;
  updated_at: string;
}

export interface PlanReadDetail {
  id: number;
  project_id: number;
  name: string;
  target_url: string;
  description: string | null;
  scope: string[];
  status: PlanStatus;
  credentials: CredentialRead[];
  linked_documents: LinkedDocSummary[];
  /** Phase A — replan budget per submodule. Default 2; range 0-5. */
  max_replans_per_submodule: number;
  created_at: string;
  updated_at: string;
}

// ── Phase A.5 — AppMap (mindmap) ──────────────────────────────────

export interface AppMapField {
  label: string;
  role: "textbox" | "combobox" | "checkbox" | "textarea";
  required: boolean;
}

export interface AppMapCreateFlow {
  entity: string;
  section_path: string[];
  trigger_label: string;
  submit_label: string;
  fields: AppMapField[];
  list_has_search: boolean;
  has_permission_tree: boolean;
}

export interface AppMapModule {
  name: string;
  sections: string[];
  landing_url: string;
  notes: string;
}

export interface AppMapRead {
  target_url: string;
  landing_url: string;
  landing_title: string;
  modules: AppMapModule[];
  create_flows: AppMapCreateFlow[];
  cross_cutting_notes: string[];
  pages_scouted: number;
  scout_depth: string;
  scout_version: number;
  reasoning: string;
}

// ── Phase C.3 — TC versioning ─────────────────────────────────────

export type TcVersionSource =
  | "brd_initial"
  | "app_map_refined"
  | "manual";

export interface TcVersionSummary {
  id: number;
  version_number: number;
  source: TcVersionSource;
  label: string;
  created_at: string | null;
  notes: Record<string, unknown> | null;
}

export interface TcVersionsListResponse {
  current_tc_version_id: number | null;
  versions: TcVersionSummary[];
}

export type TcChangeKind =
  | "kept"
  | "rewritten"
  | "added"
  | "flagged_missing";

export interface TcNodeSnapshotRead {
  id: number;
  original_tc_node_id: number | null;
  parent_snapshot_id: number | null;
  kind: "module" | "submodule" | "step";
  ordinal: number;
  depth: number;
  title: string;
  description_md: string | null;
  action_type: string | null;
  target_hint: string | null;
  narrative: string | null;
  expected: string | null;
  change_kind: TcChangeKind;
  change_reason: string | null;
  selectable_default: boolean;
  /** Phase D — live-UI dry-run probe result. NULL when validation
   * hasn't run yet for this version. */
  validation_status?: ValidationStatus;
  validation_confidence?: number | null;
  validation_reason?: string | null;
  validation_at?: string | null;
}

export interface TcVersionDetail {
  id: number;
  plan_id: number;
  version_number: number;
  source: TcVersionSource;
  label: string;
  created_at: string | null;
  notes: Record<string, unknown> | null;
  snapshots: TcNodeSnapshotRead[];
}

export interface TcRefinementSubmoduleSummary {
  submodule_id: number;
  title: string;
  step_count: number;
  kept: number;
  rewritten: number;
  added: number;
  flagged_missing: number;
  confidence: number;
  error: string | null;
}

export interface TcRefinementResponse {
  plan_id: number;
  version_id: number;
  version_number: number;
  submodule_count: number;
  input_tokens: number;
  output_tokens: number;
  submodule_summaries: TcRefinementSubmoduleSummary[];
}

// ── Phase D — validation ──────────────────────────────────────────

export type ValidationStatus =
  | "pending"
  | "confirmed"
  | "partial"
  | "unresolved"
  | "unreachable"
  | "skipped";

export interface ValidationSubmoduleSummary {
  submodule_snapshot_id: number;
  title: string;
  confirmed: number;
  partial: number;
  unresolved: number;
  unreachable: number;
  skipped: number;
  confidence: number;
}

export interface TcValidationResponse {
  plan_id: number;
  version_id: number;
  total_probed: number;
  total_seconds: number;
  error_message: string | null;
  cancelled: boolean;
  submodules: ValidationSubmoduleSummary[];
}

// ── Phase E — Sub-flow modules library ────────────────────────────

export interface SubFlowModuleSummary {
  id: number;
  project_id: number;
  name: string;
  description: string | null;
  target_url_pattern: string | null;
  tags: string[];
  segments: number;
  steps: number;
  step_snapshot_count: number;
  source_plan_id: number | null;
  source_submodule_tc_node_id: number | null;
  source_run_id: number | null;
  frozen_path_version: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface SubFlowModuleStepSnapshot {
  ordinal: number;
  title: string;
  description_md: string | null;
  action_type: string | null;
  target_hint: string | null;
  narrative: string | null;
  expected: string | null;
  data_needs_json: unknown;
}

export interface SubFlowModuleDetail extends SubFlowModuleSummary {
  frozen_segments: Record<string, unknown>;
  step_snapshots: SubFlowModuleStepSnapshot[];
}

export interface HeadingSuggestionsResponse {
  suggestions: string[];
  document_count: number;
  chunk_count: number;
}

// ── Agent Runs ────────────────────────────────────────────────────

export type AgentKind =
  | "brd_to_frd"
  | "frd_to_tc"
  | "execute"
  | "reporter"
  | "recon"
  // Phase W — user-driven recording session. Captures clicks/types
  // into the submodule's frozen_path for deterministic replay.
  | "record";
export type AgentStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled";

export const AGENT_STATUS_LABELS: Record<AgentStatus, string> = {
  queued: "Queued",
  running: "Running",
  paused: "Paused",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export interface AgentRunRead {
  id: number;
  project_id: number;
  plan_id: number | null;
  kind: AgentKind;
  status: AgentStatus;
  input_json: Record<string, unknown>;
  output_summary_json: Record<string, unknown>;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  /** Cost tracking — per-tier token counters + model snapshots
   * (migration 0017). Always present; pre-feature runs have
   * counters at 0 and the cost service treats their aggregate as
   * strong-tier. */
  strong_input_tokens: number;
  strong_output_tokens: number;
  cheap_input_tokens: number;
  cheap_output_tokens: number;
  /** Cached portions of the input totals above (SUBSET, not
   * additive). Always present; 0 on legacy / pre-feature runs. */
  strong_cached_input_tokens: number;
  cheap_cached_input_tokens: number;
  strong_model_snapshot: string | null;
  cheap_model_snapshot: string | null;
  /** Computed at read time against current AppSettings pricing.
   * null = pricing not configured for any tier-direction that
   * has tokens, OR the run did no LLM activity. */
  total_cost_usd: number | null;
}

export interface BrdToFrdRunRequest {
  source_document_ids: number[];
  cap_chunks?: number;
}

export interface FrdToTcRunRequest {
  plan_id: number;
  cap_per_module_frds?: number;
  cap_per_module_chunks?: number;
}

export interface ExecuteRunRequest {
  plan_id: number;
  selected_step_ids?: number[] | null;
  headless?: boolean;
  speed?: "slow" | "normal" | "fast";
  ai_assist?: boolean;
  /** When true, AI fixes auto-apply and HITL only fires if both passes
   * (text + vision) still leave the step failed. When false (default),
   * the AI suggestion is just proposed and HITL pre-fills with it. */
  auto_adjust?: boolean;
  /** When true, a fix that produces a passing step (AI auto-applied or
   * HITL-confirmed) is also written back to the source tc_node. */
  promote_fixes?: boolean;
  /** Run mode:
   * - "scripted" (default): rigid step-walker with AI patches.
   * - "agentic":   goal-oriented QA agent loop per submodule.
   * - "replay":    deterministic walk of each submodule's frozen
   *   path (captured on a previous successful agentic run).
   *   Submodules without a frozen path fall through to agentic. */
  mode?: "scripted" | "agentic" | "replay";
  /** Phase 6 — within agentic mode, choose the action strategy.
   * - ``hybrid`` (default): DOM-first ladder with vision rescue.
   *   Faster + cheaper on most modern apps.
   * - ``vision_only``: every click / type goes through VL + pixel
   *   coordinates. Bypasses DOM resolution entirely. ~3-5x more
   *   vision tokens but works on apps DOM resolution can't reach
   *   (heavy canvas, sealed shadow DOM, hostile rotating classes,
   *   SAP GUI for HTML in legacy frames).
   * Ignored when mode isn't "agentic". */
  agent_strategy?: "hybrid" | "vision_only";
  /** Phase H — pre-execution Scout → Refine → Activate orchestrator.
   * - ``auto`` (default for agentic): runs preflight when the plan
   *   isn't already pinned to an app_map_refined version.
   * - ``force``: always re-scout + re-refine + re-activate.
   * - ``skip``: never run preflight (legacy / debugging path).
   * Scripted + replay modes default to ``skip`` server-side since
   * they don't need a UI-grounded plan to function. */
  preflight?: "auto" | "force" | "skip";
  /** Headed Chromium window position + size, in screen pixels. The
   * frontend computes these from ``window.screen.availWidth/Height`` so
   * the browser tiles to the left and the live presenter popup fits on
   * the right. Headless launches ignore them. */
  window_x?: number;
  window_y?: number;
  window_width?: number;
  window_height?: number;
}

// ── HITL intervention ────────────────────────────────────────────

export type InterventionChoice =
  | "retry"
  | "use_suggestion"
  | "skip"
  | "stop"
  | "provide_text"   // Phase 4 — typed HITL input
  | "manual_solved"; // Phase 4 — captcha/passkey resume

export interface InterventionPayload {
  step_id: number;
  choice: InterventionChoice;
  override_target_hint?: string | null;
  override_action_type?: string | null;
  apply_to_submodule?: boolean;
  /** Phase 4 — typed value (OTP code, captcha solve, manual cred). */
  text_value?: string | null;
  /** Kind tag — ``otp_code`` / ``username`` / ``password`` /
   * ``captcha_text`` / ``free_text``. Auth flow branches on this. */
  text_kind?: string | null;
  /** Phase 4 — paired second value (e.g. password when text_value
   * is the username). */
  text_value_secondary?: string | null;
}

/** β.1 — Scout this app result. Returned by ``api.scoutApp``. */
export interface ScoutResult {
  target_url: string;
  pages_visited: number;
  pages: { url: string; title: string; primary_cta: string[] }[];
  auth_surface: string | null;
  primary_nav_items: string[];
  notes: string[];
  error_message: string | null;
  vision_calls: number;
  input_tokens: number;
  output_tokens: number;
}

/** Phase 4 — open typed prompt fetched via GET /intervention/open. */
export interface OpenPrompt {
  open: boolean;
  kind?: "request_text" | "request_credentials" | "await_manual_solve";
  question?: string;
  fields?: { name: string; label: string; type?: string }[];
}

/** Shape stored on ``execution_step.details_json["ai_correction"]`` and
 *  forwarded as the ``ai_suggestion`` field on a ``needs_intervention`` event. */
export interface AiCorrection {
  action: "retry" | "replace" | "give_up";
  reasoning: string;
  confidence: number;
  tokens_in?: number | null;
  tokens_out?: number | null;
  diff: Record<string, { old: unknown; new: unknown }>;
}

// ── Reports ──────────────────────────────────────────────────────

export interface ReportAgentTurn {
  turn: number;
  tool: string;
  args: Record<string, unknown>;
  reasoning: string;
  confidence: number;
  status: string; // "ok" | "failed" | "blocked" | "stop"
  narration: string;
  error_message: string | null;
  page_url: string;
  extracted_text: string;
}

export type ReportSubGoalStatus =
  | "pending"
  | "in_progress"
  | "done"
  | "failed"
  | "skipped";

export interface ReportSubGoal {
  id: string;
  description: string;
  status: ReportSubGoalStatus;
  completed_at_turn: number | null;
  /** Phase A — populated for VL-derived runtime sub-goals; absent on
   * legacy BRD-time ones. */
  success_criterion?: string | null;
  reason?: string | null;
  replan_iteration?: number;
  started_at_turn?: number | null;
  ended_at_turn?: number | null;
  max_turns?: number | null;
  /** Phase B — distinguishes a sub-goal walked by the deterministic
   * frozen-path replay from one driven by the agentic loop.
   * - "frozen": entire sub-goal walked deterministically
   * - "agentic": entire sub-goal driven by the agent loop
   * - "frozen_then_agentic": replay attempted + failed, agentic recovered */
  source?: "frozen" | "agentic" | "frozen_then_agentic" | null;
  frozen_step_count?: number | null;
}

export interface ReportStepRead {
  id: number;
  tc_node_id: number | null;
  ordinal: number;
  title: string;
  action_type: string | null;
  target_hint: string | null;
  status: ExecutionStepStatus;
  duration_ms: number | null;
  screenshot_path: string | null;
  error_message: string | null;
  narration: string | null;
  ai_helped: boolean;
  ai_used_vision: boolean;
  // Agentic-mode (Phase C). null/empty for scripted runs.
  mode: "scripted" | "agentic" | null;
  halt_reason: string | null;
  goal_description: string | null;
  success_criteria: string[];
  /** Ordered sub-goals the agent worked through (Phase A1). */
  sub_goals: ReportSubGoal[];
  agent_log: ReportAgentTurn[];
  /** A4.3: divergence classification for actionable recommendations.
   * passed_clean / passed_with_help / test_case_outdated /
   * feature_missing / infra_issue / agent_drift / agent_gave_up /
   * user_cancelled. null for scripted runs. */
  divergence_category: string | null;
  divergence_summary: string | null;
  /** A4.2 fuzzy substitutions that rescued this row's actions. */
  fuzzy_rescues: number;
  /** A4.1b vision-guided target searches that successfully recovered. */
  vision_rescues: number;
  /** A4.1a vision-grounded verdict on the agent's "I'm done" claim. */
  goal_verification: {
    verdict: "pass" | "partial" | "fail" | null;
    reasoning: string | null;
    confidence: number | null;
    criteria_met: string[];
    criteria_missed: string[];
  } | null;
  /** Phase 11 — agent flagged the test step as provably wrong.
   * Submodule status is ``blocked`` in that case. */
  test_case_dispute: {
    issue_kind:
      | "wrong_selector"
      | "missing_step"
      | "impossible_action"
      | "misleading_description"
      | "precondition_failed"
      | null;
    evidence: string | null;
    suggested_fix: string | null;
    turn: number | null;
  } | null;
  /** Phase 14 — smart candidate selection result (ambiguous click
   * disambiguated by vision LLM, skipping sponsored ads etc). */
  smart_pick: {
    strategy: "selector" | "coords" | "scroll" | "none" | null;
    chosen_label: string | null;
    rejected_labels: string[];
    rejection_reasons: string[];
    confidence: number | null;
    reasoning: string | null;
  } | null;
  /** Phase 9 — semantic verify escalation. Vision LLM ruling on a
   * literal verify that failed (cart wording mismatch, etc). */
  semantic_verify: {
    verdict: "pass" | "fail" | "inconclusive" | null;
    reasoning: string | null;
    confidence: number | null;
    visible_evidence: string | null;
  } | null;
  /** Production-α — AKB chunks recalled at submodule start. */
  akb_recall: {
    kind: string;
    content: string;
    confidence: number;
    tags: string[];
    relevance: number;
  }[];
  /** Plan-scoped WorldState snapshot at submodule end. */
  world_state_snapshot: Record<string, unknown> | null;
  /** Signal-voting trace: which evidence_signals matched. */
  signal_voting: {
    matched: number;
    total: number;
    traces: { signal: string; matched: boolean; via: string }[];
  } | null;
}

export interface ReportSubmoduleRead {
  title: string;
  total: number;
  passed: number;
  failed: number;
  blocked: number;
  skipped: number;
  /** Agentic goals that halted before being verified — distinct
   * from failed. */
  inconclusive: number;
  pass_pct: number;
  fail_pct: number;
  issues: string[];
  steps: ReportStepRead[];
}

export interface ReportModuleRead {
  title: string;
  total: number;
  passed: number;
  failed: number;
  blocked: number;
  skipped: number;
  inconclusive: number;
  pass_pct: number;
  fail_pct: number;
  submodules: ReportSubmoduleRead[];
}

export interface ReportRunSummary {
  id: number;
  status: AgentStatus;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  total_steps: number;
  passed: number;
  failed: number;
  blocked: number;
  skipped: number;
  inconclusive: number;
  pass_pct: number;
  fail_pct: number;
  llm_input_tokens: number | null;
  llm_output_tokens: number | null;
  ai_calls: number;
  ai_vision_calls: number;
}

export interface ReportPlanSummary {
  id: number;
  name: string;
  target_url: string;
  scope: string[];
}

export interface ReportRead {
  run: ReportRunSummary;
  plan: ReportPlanSummary | null;
  modules: ReportModuleRead[];
  excel_download_url: string;
}

/** Payload of the ``needs_intervention`` SSE event. */
export interface InterventionRequest {
  step_id: number;
  ordinal: number;
  total: number;
  title: string;
  action_type: string | null;
  target_hint: string | null;
  error_message: string | null;
  ai_suggestion: AiCorrection | null;
  screenshot_path: string | null;
}

// ── Execution steps (per-run results) ─────────────────────────────

export type ExecutionStepStatus =
  | "pending"
  | "running"
  | "passed"
  | "failed"
  | "skipped"
  | "blocked"
  | "inconclusive";

export const EXECUTION_STEP_STATUS_LABELS: Record<
  ExecutionStepStatus,
  string
> = {
  pending: "Pending",
  running: "Running",
  passed: "Passed",
  failed: "Failed",
  skipped: "Skipped",
  blocked: "Blocked",
  inconclusive: "Inconclusive",
};

export interface ExecutionStepRead {
  id: number;
  run_id: number;
  project_id: number;
  plan_id: number | null;
  tc_node_id: number | null;

  // Snapshots
  title_snapshot: string;
  path_snapshot: string;
  action_type_snapshot: string | null;
  target_hint_snapshot: string | null;
  expected_snapshot: string | null;
  narrative_snapshot: string | null;

  // Run-time
  ordinal: number;
  status: ExecutionStepStatus;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;

  // Outputs
  screenshot_path: string | null;
  narration: string | null;
  error_message: string | null;
  details_json: Record<string, unknown>;

  created_at: string;
  updated_at: string;
}

// ── TC Nodes ──────────────────────────────────────────────────────

export type TcNodeKind = "module" | "submodule" | "step";
export type TcNodeStatus = "draft" | "approved" | "archived";
export type TcNodeBulkAction =
  | "approve"
  | "archive"
  | "delete"
  | "select"
  | "deselect";

export const TC_NODE_STATUS_LABELS: Record<TcNodeStatus, string> = {
  draft: "Draft",
  approved: "Approved",
  archived: "Archived",
};

export const TC_NODE_KIND_LABELS: Record<TcNodeKind, string> = {
  module: "Module",
  submodule: "Submodule",
  step: "Step",
};

export interface TcNodeDataNeed {
  kind: string; // "credentials" | "otp" | "data"
  notes: string;
}

export interface TcNodeRead {
  id: number;
  project_id: number;
  plan_id: number;
  parent_id: number | null;
  kind: TcNodeKind;
  ordinal: number;
  depth: number;
  path_cached: string;
  title: string;
  description_md: string | null;
  // Step-only
  action_type: string | null;
  target_hint: string | null;
  narrative: string | null;
  expected: string | null;
  data_needs_json: TcNodeDataNeed[] | null;
  // Selection + status
  selectable_default: boolean;
  status: TcNodeStatus;
  source_requirement_ids: number[];
  /** Phase E — when true, the submodule has a frozen v1/v2 path and
   * is eligible for "Save as module". */
  has_frozen_path: boolean;
  /** Frozen-path version on this node (1 = legacy whole-submodule,
   * 2 = per-sub-goal segments). NULL when has_frozen_path is false. */
  frozen_path_version: number | null;
  // Timestamps
  created_at: string;
  updated_at: string;
  reviewed_at: string | null;
}

export interface TcNodeTreeRead extends TcNodeRead {
  children: TcNodeTreeRead[];
}

export interface TcNodeUpdate {
  title?: string;
  description_md?: string;
  action_type?: string;
  target_hint?: string;
  narrative?: string;
  expected?: string;
  data_needs_json?: TcNodeDataNeed[];
  selectable_default?: boolean;
  status?: TcNodeStatus;
}

export interface TcNodeBulkUpdateRequest {
  node_ids?: number[];
  filter_status?: TcNodeStatus;
  filter_kind?: TcNodeKind;
  action: TcNodeBulkAction;
}

export interface TcNodeBulkUpdateResponse {
  affected: number;
  affected_ids: number[];
  action: TcNodeBulkAction;
}

// ── Requirements (FRDs) ───────────────────────────────────────────

export type RequirementKind = "FRD";
export type RequirementStatus =
  | "proposed"
  | "edited"
  | "approved"
  | "rejected";
export type BulkAction = "approve" | "reject" | "delete";

export const REQUIREMENT_STATUS_LABELS: Record<RequirementStatus, string> = {
  proposed: "Proposed",
  edited: "Edited",
  approved: "Approved",
  rejected: "Rejected",
};

export interface RequirementRead {
  id: number;
  project_id: number;
  source_document_id: number | null;
  source_chunk_ids: number[];
  kind: RequirementKind;
  code: string;
  title: string;
  body_md: string;
  status: RequirementStatus;
  confidence: number | null;
  rationale: string | null;
  embedding_id: number | null;
  created_at: string;
  updated_at: string;
  reviewed_at: string | null;
}

export interface SourceChunkRef {
  chunk_id: number;
  document_id: number;
  document_filename: string;
  heading_path: string | null;
  anchor: string | null;
  text: string;
  char_count: number;
  ordinal: number;
}

export interface RequirementDetail extends RequirementRead {
  source_document_filename: string | null;
  source_chunks: SourceChunkRef[];
}

export interface RequirementUpdate {
  title?: string;
  body_md?: string;
  rationale?: string;
  status?: RequirementStatus;
}

export interface BulkUpdateRequest {
  requirement_ids?: number[];
  filter_status?: RequirementStatus;
  action: BulkAction;
}

export interface BulkUpdateResponse {
  affected: number;
  affected_ids: number[];
  action: BulkAction;
}

// ── Endpoint helpers ──────────────────────────────────────────────

export const api = {
  health: () => apiFetch<{ status: string; service: string; version: string }>("/api/health"),

  // Settings
  getSettings: () => apiFetch<Settings>("/api/settings"),
  upsertSettings: (data: SettingsWrite) =>
    apiFetch<Settings>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(data),
    }),
  resetSettings: () =>
    apiFetch<void>("/api/settings", { method: "DELETE" }),
  testConnection: (data?: SettingsWrite) =>
    apiFetch<TestConnectionResult>("/api/settings/test", {
      method: "POST",
      body: JSON.stringify(data ?? {}),
    }),

  // ── Cost (Cost Logs dashboard + Cost card on report) ──────────
  /** Per-run cost breakdown for one run (Cost card on report). */
  getRunCost: (runId: number) =>
    apiFetch<RunCost>(`/api/settings/cost/runs/${runId}`),
  /** Recent runs with per-tier breakdown for the Cost Logs table.
   * ``project_id`` / ``plan_id`` filter; ``limit`` defaults 200. */
  listRunCosts: (params?: {
    project_id?: number;
    plan_id?: number;
    limit?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.project_id !== undefined)
      qs.set("project_id", String(params.project_id));
    if (params?.plan_id !== undefined)
      qs.set("plan_id", String(params.plan_id));
    if (params?.limit !== undefined)
      qs.set("limit", String(params.limit));
    const q = qs.toString();
    return apiFetch<{ runs: RunCost[] }>(
      `/api/settings/cost/runs${q ? `?${q}` : ""}`,
    );
  },
  /** Per-LLM-call telemetry for one run — drill-in view. */
  listRunCallLogs: (runId: number) =>
    apiFetch<RunCallLog>(`/api/settings/cost/runs/${runId}/calls`),
  /** Aggregated cost roll-up matching the same filters. */
  aggregateCost: (params?: {
    project_id?: number;
    plan_id?: number;
    limit?: number;
  }) => {
    const qs = new URLSearchParams();
    if (params?.project_id !== undefined)
      qs.set("project_id", String(params.project_id));
    if (params?.plan_id !== undefined)
      qs.set("plan_id", String(params.plan_id));
    if (params?.limit !== undefined)
      qs.set("limit", String(params.limit));
    const q = qs.toString();
    return apiFetch<AggregateCost>(
      `/api/settings/cost/aggregate${q ? `?${q}` : ""}`,
    );
  },

  // Projects
  listProjects: () => apiFetch<Project[]>("/api/projects"),
  getProject: (id: number) => apiFetch<Project>(`/api/projects/${id}`),
  createProject: (data: ProjectCreate) =>
    apiFetch<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify(data),
    }),
  updateProject: (id: number, data: ProjectUpdate) =>
    apiFetch<Project>(`/api/projects/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),
  deleteProject: (id: number) =>
    apiFetch<void>(`/api/projects/${id}`, { method: "DELETE" }),

  // Documents
  listDocuments: (projectId: number) =>
    apiFetch<DocumentRead[]>(`/api/projects/${projectId}/documents`),
  getDocument: (projectId: number, docId: number) =>
    apiFetch<DocumentRead>(`/api/projects/${projectId}/documents/${docId}`),

  uploadDocument: async (
    projectId: number,
    kind: DocumentKind,
    file: File,
  ): Promise<DocumentRead> => {
    const formData = new FormData();
    formData.append("kind", kind);
    formData.append("file", file);

    const res = await fetch(
      `${API_BASE}/api/projects/${projectId}/documents/upload`,
      {
        method: "POST",
        body: formData, // browser sets multipart/form-data with boundary
      },
    );

    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail ?? body.message ?? detail;
      } catch {
        /* ignore */
      }
      throw new ApiError(res.status, detail);
    }
    return res.json() as Promise<DocumentRead>;
  },

  pasteDocument: (projectId: number, payload: PasteRequest) =>
    apiFetch<DocumentRead>(`/api/projects/${projectId}/documents/paste`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  deleteDocument: (projectId: number, docId: number) =>
    apiFetch<void>(`/api/projects/${projectId}/documents/${docId}`, {
      method: "DELETE",
    }),

  getParsedMd: (projectId: number, docId: number) =>
    apiFetch<DocumentParsed>(
      `/api/projects/${projectId}/documents/${docId}/parsed`,
    ),

  listChunks: (projectId: number, docId: number) =>
    apiFetch<ChunkRead[]>(
      `/api/projects/${projectId}/documents/${docId}/chunks`,
    ),

  searchDocuments: (projectId: number, payload: SearchRequest) =>
    apiFetch<SearchResponse>(
      `/api/projects/${projectId}/documents/search`,
      { method: "POST", body: JSON.stringify(payload) },
    ),

  documentEventsUrl: (projectId: number, sinceSeq?: number): string => {
    const url = new URL(
      `${API_BASE}/api/projects/${projectId}/documents/events`,
    );
    if (sinceSeq !== undefined) {
      url.searchParams.set("since_seq", String(sinceSeq));
    }
    return url.toString();
  },

  // Test Plans
  listPlans: (projectId: number) =>
    apiFetch<PlanReadCompact[]>(`/api/projects/${projectId}/plans`),
  getPlan: (projectId: number, planId: number) =>
    apiFetch<PlanReadDetail>(
      `/api/projects/${projectId}/plans/${planId}`,
    ),
  createPlan: (projectId: number, payload: PlanCreate) =>
    apiFetch<PlanReadDetail>(`/api/projects/${projectId}/plans`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updatePlan: (projectId: number, planId: number, payload: PlanUpdate) =>
    apiFetch<PlanReadDetail>(
      `/api/projects/${projectId}/plans/${planId}`,
      { method: "PATCH", body: JSON.stringify(payload) },
    ),
  deletePlan: (projectId: number, planId: number) =>
    apiFetch<void>(`/api/projects/${projectId}/plans/${planId}`, {
      method: "DELETE",
    }),

  /** β.1 — Scout this app: walk the plan's target_url + write
   * recon notes to AKB. Synchronous; ~30-60s per call. */
  scoutApp: (projectId: number, planId: number) =>
    apiFetch<ScoutResult>(
      `/api/projects/${projectId}/plans/${planId}/scout`,
      { method: "POST" },
    ),

  /** Phase A.5 — load the current AppMap for the plan's target_url.
   * Returns 404 (handled as null) when no map exists yet. */
  getAppMap: (projectId: number, planId: number) =>
    apiFetch<AppMapRead>(
      `/api/projects/${projectId}/plans/${planId}/app-map`,
    ),

  /** Phase A.5 — clear the AppMap so the next agentic run rebuilds
   * it. The actual rebuild happens inside the next run after auth. */
  clearAppMap: (projectId: number, planId: number) =>
    apiFetch<void>(
      `/api/projects/${projectId}/plans/${planId}/app-map`,
      { method: "DELETE" },
    ),

  /** Phase C.3 — list all TC versions for a plan (newest first). */
  listTcVersions: (projectId: number, planId: number) =>
    apiFetch<TcVersionsListResponse>(
      `/api/projects/${projectId}/plans/${planId}/tc-versions`,
    ),

  /** Phase C.3 — fetch one TC version with its full snapshot tree. */
  getTcVersion: (projectId: number, planId: number, versionId: number) =>
    apiFetch<TcVersionDetail>(
      `/api/projects/${projectId}/plans/${planId}/tc-versions/${versionId}`,
    ),

  /** Phase C.3 — activate a version (pass versionId=0 to clear). */
  activateTcVersion: (
    projectId: number, planId: number, versionId: number,
  ) =>
    apiFetch<{ current_tc_version_id: number | null; version_number?: number }>(
      `/api/projects/${projectId}/plans/${planId}/tc-versions/${versionId}/activate`,
      { method: "PUT" },
    ),

  /** Phase C.2 — kick off TC refinement from the cached AppMap.
   * Returns the new version_id + per-submodule change counts. */
  refineFromAppMap: (projectId: number, planId: number) =>
    apiFetch<TcRefinementResponse>(
      `/api/projects/${projectId}/plans/${planId}/refine-from-app-map`,
      { method: "POST" },
    ),

  /** Phase D — dry-run validate a TC version against the live UI.
   * Opens a headless browser, logs in via auth_flow, probes each
   * step's target_hint + expected text against the running app
   * without dispatching actions. Writes per-snapshot validation
   * status + confidence; returns per-submodule rollup. */
  validateTcVersion: (
    projectId: number, planId: number, versionId: number,
  ) =>
    apiFetch<TcValidationResponse>(
      `/api/projects/${projectId}/plans/${planId}/tc-versions/${versionId}/validate`,
      { method: "POST" },
    ),

  // ── Phase E — Sub-flow modules library ────────────────────────

  listSubFlowModules: (projectId: number) =>
    apiFetch<SubFlowModuleSummary[]>(
      `/api/projects/${projectId}/sub-flow-modules`,
    ),

  getSubFlowModule: (projectId: number, moduleId: number) =>
    apiFetch<SubFlowModuleDetail>(
      `/api/projects/${projectId}/sub-flow-modules/${moduleId}`,
    ),

  promoteToSubFlowModule: (
    projectId: number,
    payload: {
      submodule_tc_node_id: number;
      name: string;
      description?: string;
      target_url_pattern?: string | null;
      tags?: string[];
      source_run_id?: number | null;
    },
  ) =>
    apiFetch<{
      module_id: number;
      name: string;
      segments: number;
      steps: number;
    }>(
      `/api/projects/${projectId}/sub-flow-modules/promote`,
      { method: "POST", body: JSON.stringify(payload) },
    ),

  importSubFlowModule: (
    projectId: number,
    moduleId: number,
    payload: { plan_id: number; parent_module_tc_node_id?: number | null },
  ) =>
    apiFetch<{
      new_submodule_id: number;
      parent_module_id: number;
      steps_created: number;
    }>(
      `/api/projects/${projectId}/sub-flow-modules/${moduleId}/import`,
      { method: "POST", body: JSON.stringify(payload) },
    ),

  updateSubFlowModule: (
    projectId: number,
    moduleId: number,
    payload: {
      name?: string;
      description?: string;
      target_url_pattern?: string | null;
      tags?: string[];
    },
  ) =>
    apiFetch<{ ok: boolean }>(
      `/api/projects/${projectId}/sub-flow-modules/${moduleId}`,
      { method: "PATCH", body: JSON.stringify(payload) },
    ),

  deleteSubFlowModule: (projectId: number, moduleId: number) =>
    apiFetch<void>(
      `/api/projects/${projectId}/sub-flow-modules/${moduleId}`,
      { method: "DELETE" },
    ),

  /** γ.1 — Resolve a runtime test-case dispute. Action accept /
   * reject / edit; optional user_note; optional apply_to_test_case
   * to annotate the TC node's description. Writes a high-confidence
   * dispute_outcome chunk to AKB so future runs benefit. */
  resolveDispute: (
    projectId: number,
    runId: number,
    stepId: number,
    payload: {
      action: "accept" | "reject" | "edit";
      user_note?: string;
      apply_to_test_case?: boolean;
    },
  ) =>
    apiFetch<{
      action: string;
      akb_chunk_id: number | null;
      target_url: string;
      applied_to_tc: boolean;
    }>(
      `/api/projects/${projectId}/agent-runs/${runId}/disputes/${stepId}/resolve`,
      { method: "POST", body: JSON.stringify(payload) },
    ),

  createCredential: (
    projectId: number,
    planId: number,
    payload: CredentialCreate,
  ) =>
    apiFetch<CredentialRead>(
      `/api/projects/${projectId}/plans/${planId}/credentials`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  updateCredential: (
    projectId: number,
    planId: number,
    credId: number,
    payload: CredentialUpdate,
  ) =>
    apiFetch<CredentialRead>(
      `/api/projects/${projectId}/plans/${planId}/credentials/${credId}`,
      { method: "PATCH", body: JSON.stringify(payload) },
    ),
  deleteCredential: (
    projectId: number,
    planId: number,
    credId: number,
  ) =>
    apiFetch<void>(
      `/api/projects/${projectId}/plans/${planId}/credentials/${credId}`,
      { method: "DELETE" },
    ),

  getHeadingSuggestions: (projectId: number, documentIds: number[]) => {
    const params = new URLSearchParams();
    documentIds.forEach((id) => params.append("document_ids", String(id)));
    return apiFetch<HeadingSuggestionsResponse>(
      `/api/projects/${projectId}/plans/heading-suggestions?${params.toString()}`,
    );
  },

  // Agent Runs
  listAgentRuns: (
    projectId: number,
    filters?: { kind?: AgentKind; status?: AgentStatus },
  ) => {
    const qs = new URLSearchParams();
    if (filters?.kind) qs.set("kind", filters.kind);
    if (filters?.status) qs.set("status", filters.status);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return apiFetch<AgentRunRead[]>(
      `/api/projects/${projectId}/agent-runs${suffix}`,
    );
  },
  getAgentRun: (projectId: number, runId: number) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/${runId}`,
    ),
  startBrdToFrd: (projectId: number, payload: BrdToFrdRunRequest) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/brd-to-frd`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  startFrdToTc: (projectId: number, payload: FrdToTcRunRequest) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/frd-to-tc`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  startExecute: (projectId: number, payload: ExecuteRunRequest) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/execute`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  listRunSteps: (projectId: number, runId: number) =>
    apiFetch<ExecutionStepRead[]>(
      `/api/projects/${projectId}/agent-runs/${runId}/steps`,
    ),
  cancelAgentRun: (projectId: number, runId: number) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/${runId}/cancel`,
      { method: "POST" },
    ),
  /**
   * Phase K.1 — strict stop. Writes status='cancelled' to the DB
   * immediately and emits the cancelled event. The runner thread
   * may keep going briefly (Python can't kill OS threads safely)
   * but the system considers the run terminal from this call.
   * Use when the cooperative ``cancelAgentRun`` doesn't take effect
   * within ~10s (stuck inside an LLM call, browser wait, or HITL
   * block).
   */
  forceCancelAgentRun: (projectId: number, runId: number) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/${runId}/cancel?force=true`,
      { method: "POST" },
    ),
  /**
   * Phase K.2 — manual orphan reaper. Cancels runs stuck in non-
   * terminal status with no completed_at (left over from a process
   * restart / crash). Server also runs this automatically at boot.
   */
  reapOrphanedRuns: (projectId: number, staleAfterSeconds = 60) =>
    apiFetch<{ reaped: number[]; count: number }>(
      `/api/projects/${projectId}/agent-runs/reap-orphans?stale_after_seconds=${staleAfterSeconds}`,
      { method: "POST" },
    ),
  deleteAgentRun: (projectId: number, runId: number) =>
    apiFetch<void>(
      `/api/projects/${projectId}/agent-runs/${runId}`,
      { method: "DELETE" },
    ),
  /** Phase W' — start a per-MODULE recording session. The browser
   * opens maximized; submodule attribution happens live on the
   * presenter via setActiveSubmodule. Returns AgentRunRead so the
   * /live/<project>/<run.id> popup plumbing carries. */
  startRecording: (
    projectId: number,
    payload: { plan_id: number; module_id: number },
  ) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/start-recording`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  /** Phase W' — set which submodule subsequent captured events
   * attribute to. Called when the operator clicks "Start chunk"
   * on the live presenter after picking a submodule. Pass
   * ``null`` to park the recording (Pause chunk) — events are
   * dropped until another submodule is selected. */
  setActiveSubmodule: (
    projectId: number,
    runId: number,
    submoduleId: number | null,
  ) =>
    apiFetch<{
      run_id: number;
      active_submodule_id: number | null;
      per_submodule_counts: Record<string, number>;
    }>(
      `/api/projects/${projectId}/agent-runs/${runId}/active-submodule`,
      { method: "POST", body: JSON.stringify({ submodule_id: submoduleId }) },
    ),
  /** Phase W' — snapshot of the recording's per-submodule counts +
   * currently-active submodule. Safe to poll every 2-3s. */
  getRecordingState: (projectId: number, runId: number) =>
    apiFetch<{
      run_id: number;
      module_id: number;
      status: string;
      exists: boolean;
      active_submodule_id: number | null;
      per_submodule_counts: Record<string, number>;
    }>(
      `/api/projects/${projectId}/agent-runs/${runId}/recording-state`,
    ),
  /** Phase W — signal end of recording. The browser closes and the
   * captured events get persisted to the submodule's frozen_path. */
  stopRecording: (projectId: number, runId: number) =>
    apiFetch<{
      run_id: number;
      stop_delivered: boolean;
      buffered_events: number;
      note: string;
    }>(
      `/api/projects/${projectId}/agent-runs/${runId}/stop-recording`,
      { method: "POST" },
    ),
  /** Phase W — discard an in-progress recording without persisting. */
  discardRecording: (projectId: number, runId: number) =>
    apiFetch<{ run_id: number; cancelled: boolean }>(
      `/api/projects/${projectId}/agent-runs/${runId}/recording`,
      { method: "DELETE" },
    ),
  pauseAgentRun: (projectId: number, runId: number) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/${runId}/pause`,
      { method: "POST" },
    ),
  resumeAgentRun: (projectId: number, runId: number) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/${runId}/resume`,
      { method: "POST" },
    ),
  provideIntervention: (
    projectId: number,
    runId: number,
    payload: InterventionPayload,
  ) =>
    apiFetch<AgentRunRead>(
      `/api/projects/${projectId}/agent-runs/${runId}/intervention`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  /** Phase 4 — fetch the open typed HITL prompt (if any) for a step. */
  getOpenPrompt: (projectId: number, runId: number, stepId: number) =>
    apiFetch<OpenPrompt>(
      `/api/projects/${projectId}/agent-runs/${runId}/intervention/open?step_id=${stepId}`,
    ),
  getRunReport: (projectId: number, runId: number) =>
    apiFetch<ReportRead>(
      `/api/projects/${projectId}/agent-runs/${runId}/report`,
    ),
  /** Absolute URL for the xlsx download — pass to an `<a download>` tag. */
  runReportXlsxUrl: (projectId: number, runId: number): string =>
    `${API_BASE}/api/projects/${projectId}/agent-runs/${runId}/report.xlsx`,
  /** Absolute URL for a screenshot at the relative path the backend stores. */
  screenshotUrl: (relativePath: string): string =>
    `${API_BASE}/static/screenshots/${relativePath.replace(/^\/+/, "")}`,
  agentRunsEventsUrl: (projectId: number, sinceSeq?: number): string => {
    const url = new URL(
      `${API_BASE}/api/projects/${projectId}/agent-runs/events`,
    );
    if (sinceSeq !== undefined) {
      url.searchParams.set("since_seq", String(sinceSeq));
    }
    return url.toString();
  },
  agentRunEventsUrl: (
    projectId: number,
    runId: number,
    sinceSeq?: number,
  ): string => {
    const url = new URL(
      `${API_BASE}/api/projects/${projectId}/agent-runs/${runId}/events`,
    );
    if (sinceSeq !== undefined) {
      url.searchParams.set("since_seq", String(sinceSeq));
    }
    return url.toString();
  },

  // Requirements
  listRequirements: (
    projectId: number,
    filters?: {
      status?: RequirementStatus;
      kind?: RequirementKind;
      source_document_id?: number;
    },
  ) => {
    const qs = new URLSearchParams();
    if (filters?.status) qs.set("status", filters.status);
    if (filters?.kind) qs.set("kind", filters.kind);
    if (filters?.source_document_id !== undefined) {
      qs.set("source_document_id", String(filters.source_document_id));
    }
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return apiFetch<RequirementRead[]>(
      `/api/projects/${projectId}/requirements${suffix}`,
    );
  },
  getRequirement: (projectId: number, reqId: number) =>
    apiFetch<RequirementDetail>(
      `/api/projects/${projectId}/requirements/${reqId}`,
    ),
  updateRequirement: (
    projectId: number,
    reqId: number,
    payload: RequirementUpdate,
  ) =>
    apiFetch<RequirementRead>(
      `/api/projects/${projectId}/requirements/${reqId}`,
      { method: "PATCH", body: JSON.stringify(payload) },
    ),
  deleteRequirement: (projectId: number, reqId: number) =>
    apiFetch<void>(
      `/api/projects/${projectId}/requirements/${reqId}`,
      { method: "DELETE" },
    ),
  bulkUpdateRequirements: (
    projectId: number,
    payload: BulkUpdateRequest,
  ) =>
    apiFetch<BulkUpdateResponse>(
      `/api/projects/${projectId}/requirements/bulk-update`,
      { method: "POST", body: JSON.stringify(payload) },
    ),

  // TC Nodes
  listTcNodes: (projectId: number, planId: number) =>
    apiFetch<TcNodeTreeRead[]>(
      `/api/projects/${projectId}/plans/${planId}/tc-nodes`,
    ),
  /** Manually create a single TcNode under a plan. Most TC trees
   * come from the BRD→FRD→TC generation pipeline; this exists for
   * Read mode — quickly add a submodule to attach a recording to.
   * When ``parent_id`` is omitted on a submodule, the backend
   * auto-creates a "Recorded flows" module to host it. */
  createTcNode: (
    projectId: number,
    planId: number,
    payload: {
      title: string;
      kind?: "module" | "submodule";
      parent_id?: number | null;
      description_md?: string;
    },
  ) =>
    apiFetch<TcNodeTreeRead>(
      `/api/projects/${projectId}/plans/${planId}/tc-nodes`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  /** Phase Y.4 — fetch the user-actions recording saved on a
   * submodule's frozen_path. Returns ``{has_recording: false}``
   * when nothing is saved on this node. */
  getNodeRecording: (
    projectId: number,
    planId: number,
    nodeId: number,
  ) =>
    apiFetch<{
      node_id: number;
      kind: string;
      title: string;
      has_recording: boolean;
      schema_version?: number;
      recorded_at?: string;
      target_url?: string;
      viewport?: { width: number; height: number };
      action_count?: number;
      actions?: Array<{
        kind: string;
        t?: number;
        x?: number;
        y?: number;
        button?: number;
        key?: string;
        value?: string;
        url?: string;
        target?: {
          tag?: string;
          role?: string;
          text?: string;
          id?: string;
          name?: string;
          type?: string;
          placeholder?: string;
          aria_label?: string;
          title?: string;
          selector?: string;
          rect?: { x: number; y: number; w: number; h: number } | null;
        };
      }>;
    }>(
      `/api/projects/${projectId}/plans/${planId}/tc-nodes/${nodeId}/recording`,
    ),
  getTcNode: (projectId: number, planId: number, nodeId: number) =>
    apiFetch<TcNodeTreeRead>(
      `/api/projects/${projectId}/plans/${planId}/tc-nodes/${nodeId}`,
    ),
  updateTcNode: (
    projectId: number,
    planId: number,
    nodeId: number,
    payload: TcNodeUpdate,
  ) =>
    apiFetch<TcNodeRead>(
      `/api/projects/${projectId}/plans/${planId}/tc-nodes/${nodeId}`,
      { method: "PATCH", body: JSON.stringify(payload) },
    ),
  deleteTcNode: (projectId: number, planId: number, nodeId: number) =>
    apiFetch<void>(
      `/api/projects/${projectId}/plans/${planId}/tc-nodes/${nodeId}`,
      { method: "DELETE" },
    ),
  bulkUpdateTcNodes: (
    projectId: number,
    planId: number,
    payload: TcNodeBulkUpdateRequest,
  ) =>
    apiFetch<TcNodeBulkUpdateResponse>(
      `/api/projects/${projectId}/plans/${planId}/tc-nodes/bulk-update`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  /** Absolute URL for the TC export download — pass to an `<a download>`. */
  exportTcNodesUrl: (
    projectId: number,
    planId: number,
    opts: {
      format: "json" | "md";
      nodeIds?: number[];
      selectedOnly?: boolean;
    },
  ): string => {
    const params = new URLSearchParams({ format: opts.format });
    if (opts.nodeIds && opts.nodeIds.length > 0) {
      params.set("node_ids", opts.nodeIds.join(","));
    } else if (opts.selectedOnly) {
      params.set("selected_only", "true");
    }
    return (
      `${API_BASE}/api/projects/${projectId}/plans/${planId}` +
      `/tc-nodes/export?${params.toString()}`
    );
  },
};
