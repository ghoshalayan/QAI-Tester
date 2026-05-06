# Future scope

Work that's been pitched and accepted in principle but explicitly deferred.
The shape is captured here so the next session can pick it up without having
to re-derive the design.

---

## Week 6 (deferred from initial pitch) — credentials prefill + OTP HITL

The visible-cursor and narration-overlay parts of the original week-6 pitch
shipped. The HITL parts below were parked.

### 1. OTP vault primitive — `app/executor/otp_vault.py`

Process-wide thread-safe map of pending OTP requests, keyed by
`(run_id, step_id)`. Two functions:

- `request_otp(run_id, step_id, notes, timeout=180s) -> str` — orchestrator-side;
  blocks on a `threading.Event` until `provide_otp` is called or the
  timeout expires. Raises `OtpTimeout` on the latter.
- `provide_otp(run_id, step_id, value) -> bool` — HTTP-handler-side;
  stores the value, sets the Event, returns True if a waiter was found.

Vault is in-memory only — never persisted, never logged. Cleared on read.
Cancel handling: `request_cancel(run_id)` should also wake any pending
waiters under that run so the orchestrator exits cleanly instead of
holding a dead Event for 180 seconds.

### 2. Credential matcher — `app/executor/credentials.py`

Reads `plan.credentials` (already a `TestPlanCredential` list); maps
`data_needs.notes` to a credential row + field. Rules:

- If `notes` contains a credential's `role` token (case-insensitive),
  match that row. Otherwise default to the first credential.
- If `notes` contains "password" / "pwd", return the password field.
  Otherwise default to email/username.
- Single-credential plans always use that row regardless of notes.

The matcher returns the value plus a non-sensitive descriptor (role +
field name) for `details_json` and the live narration. **Never** include
the value itself in events, narration, snapshots, or screenshots.

### 3. Action dispatcher update — `app/executor/actions.py`

Drop `_check_data_block`. Each `data_needs.kind` resolves differently:

- `kind='credentials'` — call the credential matcher; type the returned
  value via `locator.fill`. Step passes when the field accepted the input.
- `kind='otp'` — flip the row to `blocked`, emit a `needs_otp` SSE event,
  call `request_otp` (blocks). On return: type the value, mark passed.
  On `OtpTimeout`: mark failed with "OTP not provided in 180s", run
  continues to siblings.
- `kind='data'` — unchanged from week 5 (extracts from `notes` or quoted
  narrative text).

### 4. Backend OTP endpoint

`POST /api/projects/{pid}/agent-runs/{run_id}/otp`
body: `{step_id: int, value: str}`

Validation:
- run must exist + belong to the project
- run.status must be `running`
- the named step row must be in `blocked` status awaiting OTP
- value: 4-12 chars, digits or alphanumerics only

On success: stores in vault, returns 204. Backend never echoes the value.

### 5. SSE event types

- `needs_otp` — orchestrator just blocked. Payload: `{step_id, ordinal,
  total, title, notes, timeout_remaining_ms}`.
- `otp_provided` — vault accepted the value. Payload: `{step_id}`.
- `otp_timeout` — 180s elapsed without input. Payload: `{step_id}`.

### 6. Frontend OTP modal

`<OtpInterventionModal>`:
- Subscribes to `needs_otp` via the existing `useAgentRunsEvents` hook
- Pops a Dialog with a 180s countdown bar, an input (autofocus, monospace,
  `inputMode="numeric"`), Submit + Cancel-run buttons
- On submit: `api.provideOtp(projectId, runId, stepId, value)`; on success
  the modal stays open until `otp_provided` confirms — then auto-closes
- On cancel-run: calls `cancelAgentRun`, closes modal
- Auto-closes on `otp_timeout`

Mounted on the run detail page; only renders for non-terminal runs.

### 7. Cancel cleanup hook

`request_cancel(run_id)` (in `agent_run_service.py`) needs to call into
the OTP vault to wake any pending waiters. Otherwise a cancel during an
OTP wait waits 180 seconds before transitioning. One-line change once
the vault is in.

### 8. Smoke test

End-to-end against a static page that:
- has a login form (credentials prefill from the plan)
- shows an OTP prompt after submit (HITL modal pops, user types code,
  page accepts and routes to a "logged in" state)
- verify step asserts the post-login text

Confirms credentials don't leak into screenshots/narration/details and
the OTP cycle finishes within timeout.

---

## Per-purpose model selection in Settings

Right now `app_settings` has a single `(provider, model)` tuple — the same
model handles BRD→FRD synthesis, FRD→TC synthesis, AI assist (text), and
AI assist (vision). That's wasteful: BRD→FRD doesn't need vision; vision
escalation doesn't need a reasoning model; FRD→TC benefits from a larger
context window than what's optimal for the recovery loop.

### What to build

- Extend `app_settings` (or a new `agent_model_overrides` table) with a
  per-purpose mapping:
  - `brd_to_frd_model` — text-only, larger context preferred
  - `frd_to_tc_model` — text-only, larger context preferred
  - `ai_assist_text_model` — text-only, fast + cheap
  - `ai_assist_vision_model` — must be vision-capable
  - Each is optional; falls back to the global `model` when unset.
- Settings page UI: a section per purpose with model picker + a hint
  about what's used where. Vision-purpose picker filters to
  vision-capable models (use the `_OPENAI_VISION_RE` /
  `_GEMINI_VISION_RE` allow-lists).
- Factory: `get_provider_from_db(db, purpose="ai_assist_vision")`
  resolves the right model. Existing call sites pass the matching
  purpose; default `purpose=None` keeps current global behavior.
- Validate at save-time that vision-purpose models actually report
  `supports_vision=True` and surface a clear error if not.

### Why deferred

- The single-model setup works today for everything we ship.
- Adds Settings-page complexity that isn't paying off until users have
  multiple keys configured AND want different cost/latency profiles per
  agent kind.
- Easy to add later once we see actual usage patterns (e.g., people
  using gpt-5-mini for the bulk LLM work and gpt-5.4-mini for vision).

---

## Other deferred items

(Add new entries here as the conversation evolves.)
