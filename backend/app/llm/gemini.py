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
import re
import time
from typing import TYPE_CHECKING, Any

from app.llm.base import ChatMessage, ChatResult, LLMProvider, TestConnectionResult

if TYPE_CHECKING:
    from google.genai import Client  # noqa: F401

logger = logging.getLogger(__name__)


# Gemini vision support — every Gemini 1.5/2.x model family is multimodal.
# We pattern-match conservatively rather than maintain a hard-coded list, so
# new model names ship working without a code change.
_GEMINI_VISION_RE = re.compile(r"^gemini-(?:1\.5|2\.\d|3\.\d)", re.IGNORECASE)


class GeminiProvider(LLMProvider):
    provider_id = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def supports_vision(self) -> bool:
        return bool(_GEMINI_VISION_RE.match(self.model or ""))

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _build_contents(turn_messages: list[ChatMessage]):
        """Convert ChatMessage list to Gemini ``types.Content`` parts.

        Each message becomes one Content with one or two parts: a text part
        (if ``content`` is non-empty) and an inline-image part (if
        ``image`` is set). Vision-capable Gemini models accept both in a
        single user turn; non-vision models will reject the image part.
        """
        from google.genai import types

        contents = []
        for m in turn_messages:
            parts = []
            if m.content:
                parts.append(types.Part(text=m.content))
            if m.image is not None:
                parts.append(types.Part.from_bytes(
                    data=m.image, mime_type=m.image_mime,
                ))
            if not parts:
                continue
            contents.append(types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=parts,
            ))
        return contents

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

        contents = self._build_contents(turn_messages)

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

        contents = self._build_contents(turn_messages)

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
            # Tolerant: parse the first valid JSON value and ignore any
            # trailing chain-of-thought / stray text Gemini sometimes
            # appends despite ``response_mime_type="application/json"``.
            from app.llm.openai_provider import _tolerant_json_parse
            parsed = _tolerant_json_parse(text)
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
