"""Auth-flow orchestrator — fills login / OTP / captcha screens
end-to-end via VL coordinates + keyboard typing.

Drives the page through auth sub-screens (email → password → OTP →
success) using the same agentic loop as the rest of the agent, but
with a dedicated set of moves:

  classify_screen → for each visible field, click-clear-type →
  click submit → wait for navigation → reclassify.

Resilience layers in order:
1. **Vault prefill** — credential matches the URL? Use it.
2. **TOTP** — vault has a TOTP seed? Generate the code with pyotp,
   no HITL needed.
3. **HITL popup** — value missing? Open a typed prompt, block on
   the user.
4. **Manual solve** — captcha / passkey can't be typed? Pop the
   "I solved it, continue" button.
5. **Error retry** — site shows "Please enter a valid email" after
   submit? Re-detect, re-clear, re-type the offending field. Cap
   at 3 retries per screen to prevent loops.
6. **Max iterations** — 8 screen transitions before we give up.

Per the user's locked policy: every field write goes through pixel
coordinates (page.mouse.click + clear_focused_field + page.keyboard.
type). DOM resolution is bypassed entirely so this works on apps
where the agent's normal resolver fails — SAP Fiori, sealed shadow
DOM, hostile rotating classes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from app.llm.base import LLMProvider
    from app.models.test_plan import TestPlan, TestPlanCredential

logger = logging.getLogger(__name__)


AuthStatus = Literal["ok", "failed", "cancelled", "blocked"]


@dataclass
class AuthResult:
    status: AuthStatus
    iterations: int = 0
    screens_seen: list[str] = field(default_factory=list)
    # True when any iteration used a value the human typed into the
    # HITL popup (vault miss, OTP-via-HITL, captcha solve). The caller
    # MUST set ``TurnRecord.manual_intervention_used`` on the
    # corresponding turn so the freeze-path gate refuses to
    # canonicalize this run as a deterministic replay candidate.
    manual_intervention_used: bool = False
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    vision_calls: int = 0


def _pick_credential(
    plan: "TestPlan",
    page_url: str,
) -> "TestPlanCredential | None":
    """Choose the credential matching the page URL.

    Order: ``url_pattern``-matching credentials first (most specific),
    then fall back to the first unscoped one.
    """
    creds = list(getattr(plan, "credentials", []) or [])
    if not creds:
        return None
    scored: list[tuple[int, "TestPlanCredential"]] = []
    for c in creds:
        pat = (c.url_pattern or "").strip()
        if pat and pat in page_url:
            scored.append((len(pat), c))
    if scored:
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]
    return creds[0]


def _type_into_field(
    page: "Page",
    x: int,
    y: int,
    value: str,
    *,
    submit: bool = False,
) -> None:
    """Click coords → clear → type → optionally press Enter.

    Clearing is mandatory: a retry after a validation error stacks
    the new value onto the old one without it (the "admin@example
    admin@example.com" bug).
    """
    from app.executor.actions import (  # noqa: PLC0415
        clear_focused_field,
    )

    page.mouse.click(x, y)
    # Let focus settle — some React inputs need a beat between
    # mousedown and keydown to wire up the controlled value.
    try:
        page.wait_for_timeout(80)
    except Exception:
        pass
    clear_focused_field(page)
    try:
        page.keyboard.type(value, delay=20)
    except Exception:
        # Fallback for ancient Playwright builds without delay kwarg.
        page.keyboard.type(value)
    if submit:
        try:
            page.wait_for_timeout(80)
        except Exception:
            pass
        page.keyboard.press("Enter")


def run_auth_loop(
    page: "Page",
    *,
    plan: "TestPlan",
    provider: "LLMProvider",
    cheap_provider: "LLMProvider | None" = None,
    max_iterations: int = 8,
    submodule_run_id: int | None = None,
    submodule_step_id: int | None = None,
    emit_event: Callable[[str, dict], None] | None = None,
    on_escalate: Any = None,
    open_typed_prompt: Callable[..., None] | None = None,
    request_intervention: Callable[..., dict | None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> AuthResult:
    """Drive the page from a login / OTP / captcha screen to logged-in.

    Returns ``AuthResult``. ``status="ok"`` when the post-login
    screen is reached; ``"failed"`` on max-iterations / detection
    error; ``"blocked"`` when HITL was needed but no channel is
    wired (programmer error); ``"cancelled"`` when the caller's
    cancel flag fired mid-flow.

    The caller writes ``manual_intervention_used=True`` onto its
    TurnRecord whenever ``out.manual_intervention_used`` is True
    so the freeze-path gate skips this run.
    """
    from app.agents.page_intel import (  # noqa: PLC0415
        capture_screenshot_for_vision,
        detect_auth_fields,
    )
    from app.security.vault import (  # noqa: PLC0415
        VaultError, generate_totp_code, read_credential,
    )

    out = AuthResult(status="failed")
    cred_row = _pick_credential(plan, page.url if page else "")
    cred_plain = None
    if cred_row is not None:
        try:
            cred_plain = read_credential(cred_row)
        except VaultError as e:
            logger.warning(
                "vault decrypt failed for credential %s: %s",
                cred_row.id, e,
            )

    # Track per-screen retry counts so we don't loop on the same
    # broken screen forever.
    retries_on_screen: dict[str, int] = {}
    MAX_RETRIES_PER_SCREEN = 3

    def _emit(t: str, d: dict) -> None:
        if emit_event is None:
            return
        try:
            emit_event(t, d)
        except Exception:
            pass

    for i in range(1, max_iterations + 1):
        if is_cancelled and is_cancelled():
            out.status = "cancelled"
            out.iterations = i - 1
            return out
        out.iterations = i

        try:
            page.wait_for_load_state(
                "domcontentloaded", timeout=8_000,
            )
        except Exception:
            pass

        try:
            shot = capture_screenshot_for_vision(page, downscale=False)
        except Exception as e:
            out.error_message = f"screenshot capture failed: {e}"
            return out

        try:
            det = detect_auth_fields(
                provider, page,
                screenshot_bytes=shot,
                cheap_provider=cheap_provider,
                on_escalate=on_escalate,
            )
            if det.input_tokens:
                out.input_tokens += det.input_tokens
            if det.output_tokens:
                out.output_tokens += det.output_tokens
            out.vision_calls += 1
        except Exception as e:
            out.error_message = (
                f"auth field detection failed: {type(e).__name__}: {e}"
            )
            return out

        out.screens_seen.append(det.kind)
        _emit("auth_screen_classified", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "iteration": i,
            "kind": det.kind,
            "confidence": det.confidence,
            "error_text": det.error_text or None,
            "error_field": det.error_field or None,
        })

        if det.kind == "success":
            out.status = "ok"
            return out

        if det.confidence < 0.6:
            # VL isn't sure what kind of screen this is. Try one
            # more iteration after a settle wait; on second low-
            # confidence in a row, give up to the agent's main loop.
            screen_key = f"unknown_{i}"
            retries_on_screen[screen_key] = (
                retries_on_screen.get(screen_key, 0) + 1
            )
            if retries_on_screen[screen_key] >= 2:
                out.error_message = (
                    f"auth field detection confidence "
                    f"{det.confidence:.2f} below threshold across "
                    f"{retries_on_screen[screen_key]} iterations"
                )
                return out
            try:
                page.wait_for_timeout(1500)
            except Exception:
                pass
            continue

        if det.kind == "captcha" or det.kind == "passkey":
            # Manual solve. Open HITL prompt; wait for user.
            if not _request_manual_solve(
                kind=det.kind,
                step_id=submodule_step_id,
                run_id=submodule_run_id,
                open_typed_prompt=open_typed_prompt,
                request_intervention=request_intervention,
                emit_event=emit_event,
            ):
                out.status = "cancelled" if (
                    is_cancelled and is_cancelled()
                ) else "blocked"
                out.error_message = (
                    f"manual solve required for {det.kind} but no "
                    "HITL response received"
                )
                return out
            out.manual_intervention_used = True
            # Loop again — the user said they solved it; re-classify.
            continue

        if det.kind == "otp":
            ok = _handle_otp_screen(
                page=page,
                det=det,
                cred=cred_plain,
                out=out,
                step_id=submodule_step_id,
                run_id=submodule_run_id,
                open_typed_prompt=open_typed_prompt,
                request_intervention=request_intervention,
                emit_event=emit_event,
                is_cancelled=is_cancelled,
            )
            if not ok:
                return out

        elif det.kind == "login":
            # Error-driven retry counter — keyed on what changed so
            # we don't bail too soon on legit multi-step login.
            screen_key = f"login_{int(det.username_visible)}_{int(det.password_visible)}"
            attempts = retries_on_screen.get(screen_key, 0)
            if attempts >= MAX_RETRIES_PER_SCREEN:
                out.status = "failed"
                out.error_message = (
                    f"login retried {attempts} times — site keeps "
                    f"rejecting (last error: {det.error_text or 'unknown'})"
                )
                return out
            retries_on_screen[screen_key] = attempts + 1

            ok = _handle_login_screen(
                page=page,
                det=det,
                cred=cred_plain,
                out=out,
                step_id=submodule_step_id,
                run_id=submodule_run_id,
                open_typed_prompt=open_typed_prompt,
                request_intervention=request_intervention,
                emit_event=emit_event,
                is_cancelled=is_cancelled,
            )
            if not ok:
                return out
        else:
            # "unknown" — out of our handled kinds. Let the main
            # agent loop decide what to do.
            out.error_message = (
                f"auth flow saw unknown screen at iteration {i}; "
                "handing back to the main agent"
            )
            out.status = "failed"
            return out

        # Settle after submit before re-classifying.
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            try:
                time.sleep(1.5)
            except Exception:
                pass

    out.status = "failed"
    out.error_message = (
        f"auth loop hit max_iterations={max_iterations} without "
        "reaching a success page"
    )
    return out


def _handle_login_screen(
    *,
    page: "Page",
    det: Any,
    cred: Any,
    out: AuthResult,
    step_id: int | None,
    run_id: int | None,
    open_typed_prompt: Callable[..., None] | None,
    request_intervention: Callable[..., dict | None] | None,
    emit_event: Callable[[str, dict], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> bool:
    """Fill username + password from vault (or HITL), click submit.

    Returns True on submit dispatched, False on unrecoverable failure.
    Error-retry logic for "wrong email format" / "wrong password"
    lives in the parent loop — this function just handles one
    fresh-fill attempt.
    """

    def _emit(t: str, d: dict) -> None:
        if emit_event is None:
            return
        try:
            emit_event(t, d)
        except Exception:
            pass

    username = cred.username if cred else ""
    password = cred.password if cred else ""

    # When the vault doesn't have creds OR the previous attempt
    # tripped a "wrong email format" error, prompt HITL for both.
    needs_hitl = (
        not username
        or not password
        or det.error_field in ("username", "password")
    )
    if needs_hitl:
        if (
            open_typed_prompt is None
            or request_intervention is None
            or step_id is None
        ):
            out.error_message = (
                "login needs HITL credentials but the popup channel "
                "isn't wired into this run"
            )
            return False
        out.manual_intervention_used = True
        question = (
            f"Authentication required. {det.error_text} "
            "Provide credentials to continue."
            if det.error_text
            else "Authentication required. Provide credentials to continue."
        )
        try:
            open_typed_prompt(
                run_id=run_id or 0,
                step_id=step_id,
                kind="request_credentials",
                question=question,
                fields=[
                    {"name": "username", "label": "Username / email"},
                    {
                        "name": "password", "label": "Password",
                        "type": "password",
                    },
                ],
            )
        except Exception as e:
            out.error_message = f"open_typed_prompt raised: {e}"
            return False
        _emit("hitl_prompt_opened", {
            "step_id": step_id,
            "kind": "request_credentials",
        })
        response = request_intervention(step_id) if step_id is not None else None
        _emit("hitl_prompt_answered", {
            "step_id": step_id,
            "status": "ok" if response else "cancelled",
        })
        if response is None:
            out.status = "cancelled"
            return False
        username = (response.get("text_value") or "").strip()
        password = (response.get("text_value_secondary") or "").strip()
        if not username or not password:
            out.error_message = "user submitted blank credentials"
            return False

    # Type into each visible field. Use coords from the VL detection.
    if det.username_visible and det.username_x > 0 and det.username_y > 0:
        _type_into_field(
            page, det.username_x, det.username_y, username,
        )
        _emit("auth_field_typed", {
            "run_id": run_id,
            "step_id": step_id,
            "field": "username",
        })
    if det.password_visible and det.password_x > 0 and det.password_y > 0:
        _type_into_field(
            page, det.password_x, det.password_y, password,
        )
        _emit("auth_field_typed", {
            "run_id": run_id,
            "step_id": step_id,
            "field": "password",
        })

    # Submit. Prefer clicking the located submit button; fall back to
    # Enter on the password field when the button wasn't detected.
    if det.submit_visible and det.submit_x > 0 and det.submit_y > 0:
        try:
            page.mouse.click(det.submit_x, det.submit_y)
            _emit("auth_submitted", {
                "run_id": run_id,
                "step_id": step_id,
                "via": "submit_button",
            })
        except Exception as e:
            logger.warning("submit click failed: %s — trying Enter", e)
            try:
                page.keyboard.press("Enter")
                _emit("auth_submitted", {
                    "run_id": run_id,
                    "step_id": step_id,
                    "via": "enter_key",
                })
            except Exception:
                out.error_message = "submit dispatch failed"
                return False
    else:
        try:
            page.keyboard.press("Enter")
            _emit("auth_submitted", {
                "run_id": run_id,
                "step_id": step_id,
                "via": "enter_key",
            })
        except Exception as e:
            out.error_message = f"submit Enter failed: {e}"
            return False
    # Cancelled mid-fill?
    if is_cancelled and is_cancelled():
        out.status = "cancelled"
        return False
    return True


def _handle_otp_screen(
    *,
    page: "Page",
    det: Any,
    cred: Any,
    out: AuthResult,
    step_id: int | None,
    run_id: int | None,
    open_typed_prompt: Callable[..., None] | None,
    request_intervention: Callable[..., dict | None] | None,
    emit_event: Callable[[str, dict], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> bool:
    """Fill the OTP field. Use TOTP from vault when available; HITL
    otherwise (SMS / email / push 2FA — can't be pre-stored)."""
    from app.security.vault import generate_totp_code  # noqa: PLC0415

    def _emit(t: str, d: dict) -> None:
        if emit_event is None:
            return
        try:
            emit_event(t, d)
        except Exception:
            pass

    seed = (
        (cred.totp_secret if cred else None)
        or None
    )
    code = generate_totp_code(seed) if seed else None
    if not code:
        # HITL — prompt for the OTP code typed by the user.
        if (
            open_typed_prompt is None
            or request_intervention is None
            or step_id is None
        ):
            out.error_message = (
                "OTP screen but no TOTP seed in vault and no HITL "
                "channel wired"
            )
            return False
        out.manual_intervention_used = True
        try:
            open_typed_prompt(
                run_id=run_id or 0,
                step_id=step_id,
                kind="request_text",
                question=(
                    det.error_text
                    or "Enter the verification code from your "
                       "authenticator / SMS / email."
                ),
                fields=[
                    {"name": "otp_code", "label": "Verification code"},
                ],
            )
        except Exception as e:
            out.error_message = f"open_typed_prompt raised: {e}"
            return False
        _emit("hitl_prompt_opened", {
            "step_id": step_id, "kind": "request_text",
        })
        response = (
            request_intervention(step_id)
            if step_id is not None else None
        )
        _emit("hitl_prompt_answered", {
            "step_id": step_id,
            "status": "ok" if response else "cancelled",
        })
        if response is None:
            out.status = "cancelled"
            return False
        code = (response.get("text_value") or "").strip()
        if not code:
            out.error_message = "user submitted blank OTP code"
            return False

    if det.otp_visible and det.otp_x > 0 and det.otp_y > 0:
        _type_into_field(page, det.otp_x, det.otp_y, code)
        _emit("auth_field_typed", {
            "run_id": run_id, "step_id": step_id, "field": "otp",
        })
    else:
        out.error_message = "OTP screen detected but OTP field not located"
        return False

    if det.submit_visible and det.submit_x > 0 and det.submit_y > 0:
        try:
            page.mouse.click(det.submit_x, det.submit_y)
        except Exception:
            page.keyboard.press("Enter")
    else:
        page.keyboard.press("Enter")
    _emit("auth_submitted", {
        "run_id": run_id, "step_id": step_id, "via": "otp",
    })

    if is_cancelled and is_cancelled():
        out.status = "cancelled"
        return False
    return True


def _request_manual_solve(
    *,
    kind: str,
    step_id: int | None,
    run_id: int | None,
    open_typed_prompt: Callable[..., None] | None,
    request_intervention: Callable[..., dict | None] | None,
    emit_event: Callable[[str, dict], None] | None,
) -> bool:
    """Open the 'I solved it, continue' HITL prompt and wait."""
    if (
        open_typed_prompt is None
        or request_intervention is None
        or step_id is None
    ):
        return False
    question = (
        "A reCAPTCHA / hCaptcha / Cloudflare challenge is blocking "
        "the form. Solve it in the browser, then click Continue."
        if kind == "captcha"
        else "A passkey / hardware key prompt is open. Approve it "
             "on your device, then click Continue."
    )
    try:
        open_typed_prompt(
            run_id=run_id or 0,
            step_id=step_id,
            kind="await_manual_solve",
            question=question,
            fields=[],
        )
    except Exception:
        return False
    if emit_event is not None:
        try:
            emit_event("hitl_prompt_opened", {
                "step_id": step_id, "kind": "await_manual_solve",
            })
        except Exception:
            pass
    response = request_intervention(step_id)
    if emit_event is not None:
        try:
            emit_event("hitl_prompt_answered", {
                "step_id": step_id,
                "status": "ok" if response else "cancelled",
            })
        except Exception:
            pass
    return response is not None
