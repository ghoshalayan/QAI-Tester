"""OpenAI / OpenAI-compatible provider.

One class handles both ``provider="openai"`` (api.openai.com) and
``provider="openai_compat"`` (Ollama, vLLM, LM Studio, OpenRouter, Together,
Groq, Anthropic-via-proxy, etc.) — the only difference is whether
``base_url`` is set.

Token-limit parameter quirk
---------------------------
- Native OpenAI **GPT-5 series** (gpt-5, gpt-5.x, o-series) requires
  ``max_completion_tokens``. Old ``max_tokens`` will 400.
- OpenAI-compatible servers (Ollama, vLLM, LM Studio, etc.) almost universally
  speak the older ``max_tokens`` parameter.

We resolve that by:
1. ``provider="openai"`` → send ``max_completion_tokens``
2. ``provider="openai_compat"`` → send ``max_tokens``

Reasoning models (gpt-5*, o-series) also reject ``temperature`` for non-default
values; we only forward ``temperature`` when the caller passes it explicitly.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any

from app.llm.base import ChatMessage, ChatResult, LLMProvider, TestConnectionResult

logger = logging.getLogger(__name__)


def _extract_cached_tokens(usage: Any) -> int | None:
    """Pull OpenAI's automatic-cache hit count off the usage block.

    OpenAI surfaces this as ``usage.prompt_tokens_details.cached_tokens``
    (≥ 1024-token prompts hit cache automatically at ~50% the input
    rate). Returns ``None`` when the field is absent, which happens on:
      - older models that didn't cache
      - OpenAI-compatible servers (Ollama / vLLM / etc.) that don't
        forward this field
      - prompts below the cache threshold
    The cost service treats None as 0 cached.
    """
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return None
    val = getattr(details, "cached_tokens", None)
    if val is None and isinstance(details, dict):
        val = details.get("cached_tokens")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# Vision-capable model families. Native OpenAI keeps adding new ones; we
# pattern-match conservatively rather than hard-code a list. Compat
# providers usually opt-in by exposing a vision-capable model name —
# matching the same patterns is a reasonable default.
_OPENAI_VISION_RE = re.compile(
    r"^(gpt-4o|gpt-4(?:\.\d+)?-turbo|gpt-4-vision|gpt-5|o\d)",
    re.IGNORECASE,
)


def _strip_fences(text: str) -> str:
    """Remove leading/trailing ```json … ``` fences if present.

    Compat servers sometimes ignore json_object mode and wrap their output in
    markdown fences; this lets us still parse the payload.
    """
    text = text.strip()
    if text.startswith("```"):
        # Drop the first line (```json or ```)
        first_nl = text.find("\n")
        text = text[first_nl + 1 :] if first_nl != -1 else ""
    if text.endswith("```"):
        text = text[: -3]
    return text.strip()


def _tolerant_json_parse(text: str) -> Any:
    """Parse the FIRST valid JSON value in ``text`` and ignore trailing data.

    Why not ``json.loads``: with strict-schema responses, we sometimes
    get a perfectly valid JSON object followed by a trailing newline +
    a second object (the model briefly chain-of-thought-ed before
    closing) or by stray reasoning text. ``json.loads`` rejects this
    with ``Extra data: line 2 column 1`` even though the first object
    is exactly what we asked for.

    ``raw_decode`` consumes only the first value and returns the
    byte-offset where it stopped — we ignore the tail. This matches
    what every other JSON-extraction lib does (langchain, pydantic-ai,
    etc.) and is the standard fix.

    Falls back to ``_strip_fences`` for fenced output; raises
    ``json.JSONDecodeError`` only if NEITHER path yields a value.
    """
    if not text:
        return {}
    text = text.strip()
    if not text:
        return {}
    decoder = json.JSONDecoder()
    try:
        value, _idx = decoder.raw_decode(text)
        return value
    except json.JSONDecodeError:
        # Fence-wrapped output — strip and retry.
        stripped = _strip_fences(text)
        if stripped and stripped != text:
            value, _idx = decoder.raw_decode(stripped)
            return value
        raise


class OpenAIProvider(LLMProvider):
    """Backs both the ``openai`` and ``openai_compat`` settings values."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        *,
        is_compat: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.is_compat = is_compat
        self.provider_id = "openai_compat" if is_compat else "openai"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    @property
    def supports_vision(self) -> bool:
        return bool(_OPENAI_VISION_RE.match(self.model or ""))

    @staticmethod
    def _to_oai_messages(
        messages: list[ChatMessage],
    ) -> list[dict[str, Any]]:
        """Convert ChatMessage list to OpenAI's request shape.

        Plain text messages stay as ``{"role": ..., "content": "..."}``.
        Messages with an inline image switch to the multimodal content
        array shape (text part + image_url part with a base64 data URI).
        Vision-incapable models will reject the image part — callers
        gate via :prop:`supports_vision` before passing images.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.image is None:
                out.append({"role": m.role, "content": m.content})
                continue
            b64 = base64.b64encode(m.image).decode("ascii")
            data_uri = f"data:{m.image_mime};base64,{b64}"
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"type": "text", "text": m.content})
            parts.append({
                "type": "image_url",
                "image_url": {"url": data_uri},
            })
            out.append({"role": m.role, "content": parts})
        return out

    def _tokens_param(self) -> str:
        # Native OpenAI GPT-5/o series use max_completion_tokens.
        # Compat servers use the legacy max_tokens.
        return "max_tokens" if self.is_compat else "max_completion_tokens"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ChatResult:
        oai_messages = self._to_oai_messages(messages)

        kwargs: dict = {"model": self.model, "messages": oai_messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs[self._tokens_param()] = max_output_tokens

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0] if response.choices else None
        text = ""
        if choice and choice.message and choice.message.content:
            text = choice.message.content.strip()

        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            cached_input_tokens=_extract_cached_tokens(usage),
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
        """Native: ``response_format=json_schema`` strict mode.
        Compat: falls back to ``json_object`` mode + schema-in-prompt + parse-retry.
        """
        if self.is_compat:
            return self._chat_structured_compat(
                messages, schema, schema_name, temperature, max_output_tokens,
            )
        return self._chat_structured_native(
            messages, schema, schema_name, temperature, max_output_tokens,
        )

    def _chat_structured_native(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        schema_name: str,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> ChatResult:
        oai_messages = self._to_oai_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": oai_messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs[self._tokens_param()] = max_output_tokens

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0] if response.choices else None
        text = ""
        if choice and choice.message and choice.message.content:
            text = choice.message.content.strip()

        try:
            parsed = _tolerant_json_parse(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"OpenAI returned invalid JSON despite strict schema: {e}. "
                f"Text (first 500 chars): {text[:500]}",
            ) from e

        usage = getattr(response, "usage", None)
        return ChatResult(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            cached_input_tokens=_extract_cached_tokens(usage),
            parsed=parsed,
            raw=response,
        )

    def _chat_structured_compat(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        schema_name: str,
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> ChatResult:
        """Compat-mode best-effort: most servers don't support strict json_schema.

        Strategy:
        1. Inject the schema as an extra system instruction
        2. Ask for ``response_format={"type": "json_object"}`` (broadly supported)
        3. Parse the response; if invalid JSON, strip Markdown fences and retry parse
        4. If still invalid, send a follow-up "your last reply wasn't valid JSON" turn
           and try once more
        """
        del schema_name  # ignored on compat path; schema is in the prompt

        schema_str = json.dumps(schema, indent=2)
        schema_addendum = (
            "Output ONLY valid JSON matching this schema "
            "(no markdown, no commentary):\n"
            f"{schema_str}"
        )

        # Merge schema instruction into the first system message, or prepend one
        augmented: list[ChatMessage] = []
        injected = False
        for m in messages:
            if m.role == "system" and not injected:
                augmented.append(
                    ChatMessage(
                        role="system",
                        content=f"{m.content}\n\n{schema_addendum}",
                    ),
                )
                injected = True
            else:
                augmented.append(m)
        if not injected:
            augmented.insert(
                0, ChatMessage(role="system", content=schema_addendum),
            )

        oai_messages = self._to_oai_messages(augmented)

        base_kwargs: dict = {
            "model": self.model,
            "messages": oai_messages,
            "response_format": {"type": "json_object"},
        }
        if temperature is not None:
            base_kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            base_kwargs[self._tokens_param()] = max_output_tokens

        last_text = ""
        last_error: Exception | None = None
        last_response = None

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(**base_kwargs)
                last_response = response
                choice = response.choices[0] if response.choices else None
                text = ""
                if choice and choice.message and choice.message.content:
                    text = choice.message.content.strip()
                last_text = text

                # Tolerant parse: handles fences AND trailing-data cases
                # (model emits a valid object, newline, then stray text).
                parsed = _tolerant_json_parse(text)

                usage = getattr(response, "usage", None)
                return ChatResult(
                    text=text,
                    model=self.model,
                    input_tokens=(
                        getattr(usage, "prompt_tokens", None) if usage else None
                    ),
                    output_tokens=(
                        getattr(usage, "completion_tokens", None) if usage else None
                    ),
                    cached_input_tokens=_extract_cached_tokens(usage),
                    parsed=parsed,
                    raw=response,
                )
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "OpenAI-compat returned invalid JSON on attempt %s: %s",
                    attempt + 1,
                    str(e)[:200],
                )
                # Add a corrective turn for the retry
                base_kwargs = dict(base_kwargs)
                base_kwargs["messages"] = [
                    *oai_messages,
                    {"role": "assistant", "content": last_text},
                    {
                        "role": "user",
                        "content": (
                            "Your last reply was not valid JSON. Output ONLY a "
                            "single JSON object matching the schema — no markdown "
                            "fences, no prose, no extra keys."
                        ),
                    },
                ]

        # All attempts exhausted
        usage = getattr(last_response, "usage", None) if last_response else None
        raise RuntimeError(
            f"OpenAI-compat provider returned invalid JSON after 2 attempts: "
            f"{last_error}. Last text (first 500 chars): {last_text[:500]}. "
            f"Tokens used so far: in={getattr(usage, 'prompt_tokens', None) if usage else None}",
        )

    def test_connection(self) -> TestConnectionResult:
        started = time.monotonic()
        try:
            # Don't pass temperature — GPT-5 reasoning models reject non-defaults.
            # Don't pass max_output_tokens — keeps it simple and works on every
            # compat server. The test message is tiny anyway.
            result = self.chat(
                [ChatMessage(role="user", content='Reply with the single word: ok')],
            )
            return TestConnectionResult(
                ok=True,
                provider=self.provider_id,
                model=self.model,
                base_url=self.base_url,
                echo=result.text[:80],
                latency_ms=int((time.monotonic() - started) * 1000),
                extra={
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            )
        except Exception as e:
            logger.warning(
                "OpenAI%s test_connection failed: %s",
                "-compat" if self.is_compat else "",
                e,
            )
            return TestConnectionResult(
                ok=False,
                provider=self.provider_id,
                model=self.model,
                base_url=self.base_url,
                error=f"{type(e).__name__}: {str(e)[:300]}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
