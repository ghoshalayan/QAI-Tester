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

export type Provider = "gemini" | "openai" | "openai_compat";

export interface Settings {
  is_configured: boolean;
  provider: Provider | null;
  model: string | null;
  base_url: string | null;
  api_key_set: boolean;
  updated_at: string | null;
}

export interface SettingsWrite {
  provider?: Provider;
  model?: string;
  api_key?: string;
  base_url?: string;
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
}

export interface PlanUpdate {
  name?: string;
  target_url?: string;
  description?: string;
  scope?: string[];
  status?: PlanStatus;
  /** When present, replaces the entire set of linked docs. */
  linked_document_ids?: number[];
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
  created_at: string;
  updated_at: string;
}

export interface HeadingSuggestionsResponse {
  suggestions: string[];
  document_count: number;
  chunk_count: number;
}

// ── Agent Runs ────────────────────────────────────────────────────

export type AgentKind = "brd_to_frd" | "frd_to_tc" | "execute" | "reporter";
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
   * - "agentic":   goal-oriented QA agent loop per submodule. */
  mode?: "scripted" | "agentic";
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
  | "stop";

export interface InterventionPayload {
  step_id: number;
  choice: InterventionChoice;
  override_target_hint?: string | null;
  override_action_type?: string | null;
  apply_to_submodule?: boolean;
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
  agent_log: ReportAgentTurn[];
}

export interface ReportSubmoduleRead {
  title: string;
  total: number;
  passed: number;
  failed: number;
  blocked: number;
  skipped: number;
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
  deleteAgentRun: (projectId: number, runId: number) =>
    apiFetch<void>(
      `/api/projects/${projectId}/agent-runs/${runId}`,
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
