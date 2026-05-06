"""Provider-agnostic LLM interface.

The factory in ``app.llm.factory`` picks an implementation based on
``app_settings.provider`` and hands it to callers as an ``LLMProvider``.
Phase 1 only uses ``test_connection``; ``chat`` is here so step 5+ have
something to call.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    # Optional inline image (PNG/JPEG bytes) for multimodal calls. Only the
    # ``user`` role typically attaches images; providers that don't support
    # vision fall back to text-only and ignore this field.
    image: bytes | None = None
    image_mime: str = "image/png"


@dataclass
class ChatResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Set by ``chat_structured()`` — the parsed JSON payload matching the schema.
    # ``None`` for free-form ``chat()`` calls.
    parsed: Any = None
    raw: Any = None  # provider-specific response object, useful for debugging


@dataclass
class TestConnectionResult:
    ok: bool
    provider: str
    model: str
    base_url: str | None = None
    echo: str | None = None
    latency_ms: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Interface every concrete provider must implement."""

    provider_id: str
    model: str

    @property
    def supports_vision(self) -> bool:
        """True if the configured model accepts inline images.

        Default False — providers override this with a model-name allow-list.
        Callers (e.g. the executor's vision-escalation branch) check this
        before attaching a screenshot via ``ChatMessage.image``; if False,
        they skip the vision call cleanly.
        """
        return False

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ChatResult: ...

    @abstractmethod
    def chat_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        schema_name: str = "output",
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> ChatResult:
        """Round-trip a chat with a JSON schema constraint.

        On success, ``ChatResult.parsed`` holds the parsed Python dict.
        Native OpenAI uses ``response_format=json_schema`` (strict); Gemini
        uses ``response_schema``; OpenAI-compatible providers fall back to
        ``json_object`` mode with the schema injected as a system message and
        a parse-retry on bad JSON.
        """

    @abstractmethod
    def test_connection(self) -> TestConnectionResult: ...
