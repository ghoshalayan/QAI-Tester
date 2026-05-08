"""LLM provider configuration endpoints.

GET   /api/settings        → public-safe read (no api_key)
PUT   /api/settings        → upsert with partial-update semantics
DELETE /api/settings       → wipe back to first-run state
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.llm.factory import build_provider, invalidate_cache
from app.models.app_settings import AppSettings
from app.schemas.app_settings import (
    AppSettingsRead,
    AppSettingsWrite,
    TestConnectionResponse,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


def _to_read(row: AppSettings | None) -> AppSettingsRead:
    if row is None:
        return AppSettingsRead()
    return AppSettingsRead(
        is_configured=bool(row.provider and row.model and row.api_key),
        provider=row.provider,  # type: ignore[arg-type]
        model=row.model,
        cheap_model=getattr(row, "cheap_model", None) or None,
        base_url=row.base_url,
        api_key_set=bool(row.api_key),
        ai_mode=bool(getattr(row, "ai_mode", False)),
        updated_at=row.updated_at,
    )


def _validate_for_create(payload: AppSettingsWrite) -> None:
    """First-time setup: provider, model, api_key all required."""
    if not payload.provider:
        raise HTTPException(400, "provider is required for initial setup")
    if not payload.model:
        raise HTTPException(400, "model is required for initial setup")
    if not payload.api_key:
        raise HTTPException(400, "api_key is required for initial setup")
    if payload.provider == "openai_compat" and not payload.base_url:
        raise HTTPException(
            400, "base_url is required when provider is 'openai_compat'",
        )


@router.get("", response_model=AppSettingsRead)
def get_settings(db: Session = Depends(get_db)):
    row = db.query(AppSettings).filter(AppSettings.id == 1).first()
    return _to_read(row)


@router.put("", response_model=AppSettingsRead)
def upsert_settings(payload: AppSettingsWrite, db: Session = Depends(get_db)):
    row = db.query(AppSettings).filter(AppSettings.id == 1).first()

    # AI-Mode-only flip: a payload that sets ONLY ``ai_mode`` shouldn't
    # be required to re-supply provider/model/api_key. This lets the
    # Settings UI flip the toggle without re-entering credentials.
    only_ai_mode_toggle = (
        payload.ai_mode is not None
        and payload.provider is None
        and payload.model is None
        and payload.cheap_model is None
        and payload.api_key is None
        and payload.base_url is None
    )

    if row is None:
        if only_ai_mode_toggle:
            raise HTTPException(
                400,
                "Configure an LLM provider before enabling AI Mode.",
            )
        _validate_for_create(payload)
        # Phase 1 — cheap_model: must NOT equal model (would be a no-op
        # tier and adds DB ambiguity). Empty string = no tiering.
        cheap = (payload.cheap_model or "").strip() or None
        if cheap and cheap == payload.model:
            raise HTTPException(
                400,
                "cheap_model must differ from model — they form the "
                "(primary, escalation) pair for tiering.",
            )
        row = AppSettings(
            id=1,
            provider=payload.provider,
            model=payload.model,
            cheap_model=cheap,
            api_key=payload.api_key,
            base_url=payload.base_url if payload.provider == "openai_compat" else None,
            ai_mode=bool(payload.ai_mode) if payload.ai_mode is not None else False,
        )
        db.add(row)
    elif only_ai_mode_toggle:
        row.ai_mode = bool(payload.ai_mode)
    else:
        new_provider = payload.provider or row.provider
        provider_changed = (
            payload.provider is not None and payload.provider != row.provider
        )

        # Switching providers requires a fresh API key — old key won't work elsewhere
        if provider_changed and not payload.api_key:
            raise HTTPException(
                400, "api_key is required when changing provider",
            )

        if new_provider == "openai_compat":
            new_base_url = (
                payload.base_url if payload.base_url is not None else row.base_url
            )
            if not new_base_url:
                raise HTTPException(
                    400, "base_url is required when provider is 'openai_compat'",
                )
        else:
            new_base_url = None  # always cleared for non-compat providers

        row.provider = new_provider
        if payload.model:
            row.model = payload.model
        # Phase 1 — cheap_model partial update. ``None`` means "leave
        # alone"; empty string means "clear tiering". Otherwise the
        # supplied value is set, with the not-equal-to-model guard.
        if payload.cheap_model is not None:
            cheap = payload.cheap_model.strip() or None
            effective_model = payload.model or row.model
            if cheap and cheap == effective_model:
                raise HTTPException(
                    400,
                    "cheap_model must differ from model — they form "
                    "the (primary, escalation) pair for tiering.",
                )
            row.cheap_model = cheap
        if payload.api_key:
            row.api_key = payload.api_key
        row.base_url = new_base_url
        if payload.ai_mode is not None:
            row.ai_mode = bool(payload.ai_mode)

    db.commit()
    db.refresh(row)
    invalidate_cache()  # provider settings changed — drop cached client
    return _to_read(row)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def reset_settings(db: Session = Depends(get_db)):
    """Wipe the settings row. App returns to first-run state."""
    db.query(AppSettings).filter(AppSettings.id == 1).delete()
    db.commit()
    invalidate_cache()


@router.post("/test", response_model=TestConnectionResponse)
def test_connection(
    payload: AppSettingsWrite | None = None,
    db: Session = Depends(get_db),
):
    """Round-trip a tiny prompt to verify the LLM is reachable.

    Two modes:

    - **Test before save** — pass ``provider``, ``model``, ``api_key``
      (and ``base_url`` for ``openai_compat``) in the body. Nothing is persisted.
    - **Test current config** — pass an empty body or omit it. Uses whatever
      is already in ``app_settings``.
    """
    use_payload = bool(
        payload and payload.provider and payload.model and payload.api_key
    )

    if use_payload:
        provider_name = payload.provider
        model = payload.model
        api_key = payload.api_key
        base_url = payload.base_url
    else:
        row = db.query(AppSettings).filter(AppSettings.id == 1).first()
        if not row or not row.api_key:
            raise HTTPException(
                400,
                "No saved settings to test. Either save settings first, "
                "or pass provider/model/api_key in the request body.",
            )
        provider_name = row.provider
        model = row.model
        api_key = row.api_key
        base_url = row.base_url

    if provider_name == "openai_compat" and not base_url:
        raise HTTPException(
            400, "base_url is required when provider is 'openai_compat'",
        )

    try:
        provider = build_provider(
            provider=provider_name,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    result = provider.test_connection()
    return TestConnectionResponse(
        ok=result.ok,
        provider=result.provider,
        model=result.model,
        base_url=result.base_url,
        echo=result.echo,
        latency_ms=result.latency_ms,
        error=result.error,
        input_tokens=result.extra.get("input_tokens"),
        output_tokens=result.extra.get("output_tokens"),
    )
