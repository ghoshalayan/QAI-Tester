"""Gemini provider — uses the new ``google-genai`` SDK (NOT the legacy
``google-generativeai`` package).

Notes for callers:
- Gemini's role for the assistant is ``"model"``, not ``"assistant"``. The
  translation happens here.
- ``system`` messages become ``GenerateContentConfig.system_instruction``.
- Reasoning-oriented models (e.g. ``gemini-3.1-pro``) accept ``temperature``
  but you can also omit it for default behavior.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from app.llm.base import ChatMessage, ChatResult, LLMProvider, TestConnectionResult

if TYPE_CHECKING:
    from google.genai import Client  # noqa: F401

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    provider_id = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ChatResult:
        from google.genai import types

        # Pull system messages out — Gemini wants them in config, not contents
        system_parts = [m.content for m in messages if m.role == "system"]
        turn_messages = [m for m in messages if m.role != "system"]

        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in turn_messages
        ]

        config_kwargs: dict = {}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        usage = getattr(response, "usage_metadata", None)
        return ChatResult(
            text=(response.text or "").strip(),
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            raw=response,
        )

    def chat_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        schema_name: str = "output",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ChatResult:
        """Gemini structured output via ``response_schema``.

        Gemini accepts a Python dict in OpenAPI-3 schema form and returns a
        JSON string in ``response.text``. We parse it here so callers see
        ``ChatResult.parsed`` populated.

        ``schema_name`` is part of the ABC for OpenAI parity; Gemini ignores it.
        """
        del schema_name
        from google.genai import types

        system_parts = [m.content for m in messages if m.role == "system"]
        turn_messages = [m for m in messages if m.role != "system"]

        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in turn_messages
        ]

        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        config = types.GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        text = (response.text or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Gemini returned invalid JSON: {e}. "
                f"Text (first 500 chars): {text[:500]}",
            ) from e

        usage = getattr(response, "usage_metadata", None)
        return ChatResult(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
            parsed=parsed,
            raw=response,
        )

    def test_connection(self) -> TestConnectionResult:
        started = time.monotonic()
        try:
            result = self.chat(
                [ChatMessage(role="user", content='Reply with the single word: ok')],
                temperature=0.0,
                max_output_tokens=8,
            )
            return TestConnectionResult(
                ok=True,
                provider=self.provider_id,
                model=self.model,
                echo=result.text[:80],
                latency_ms=int((time.monotonic() - started) * 1000),
                extra={
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            )
        except Exception as e:
            logger.warning("Gemini test_connection failed: %s", e)
            return TestConnectionResult(
                ok=False,
                provider=self.provider_id,
                model=self.model,
                error=f"{type(e).__name__}: {str(e)[:300]}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
