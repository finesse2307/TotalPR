"""Production LLMClient backed by Anthropic's official SDK.


API key is read from the ``ANTHROPIC_API_KEY`` environment variable unless
provided explicitly. Using Claude Haiku by default for pricing
"""

import os
from typing import Any

from anthropic import Anthropic
from anthropic.types import TextBlock, ToolUseBlock

from sentry.llm import LLMResponse, LLMToolCall, Message, ToolDef, Usage

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicLLMClient:
    """LLMClient backed by ``anthropic.Anthropic``.

    Accepts an optional pre-configured ``client`` so tests can inject a fake
    Anthropic instance and avoid real API calls (and real spend).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: Anthropic | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
            return

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set; provide via env or api_key arg."
            )
        self._client = Anthropic(api_key=key)

    def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                arguments = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    LLMToolCall(id=block.id, name=block.name, arguments=arguments)
                )

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )