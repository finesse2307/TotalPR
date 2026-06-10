"""Tests for AnthropicLLMClient.

Fakes the Anthropic SDK with unittest.mock so tests run offline and never spend
tokens. Covers request construction, response parsing for both text and tool-use
blocks, usage propagation, and the missing-API-key error path.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic import Anthropic
from anthropic.types import TextBlock, ToolUseBlock

from sentry.anthropic_client import DEFAULT_MODEL, AnthropicLLMClient
from sentry.llm import Message, ToolDef


def _fake_anthropic(
    *,
    content: list[object],
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.stop_reason = stop_reason
    response.usage = MagicMock(
        input_tokens=input_tokens, output_tokens=output_tokens
    )

    client = MagicMock(spec=Anthropic)
    client.messages.create.return_value = response
    return client


def _text_block(text: str) -> MagicMock:
    block = MagicMock(spec=TextBlock)
    block.text = text
    return block


def _tool_use_block(
    *, id: str, name: str, input: dict[str, Any]
) -> MagicMock:
    block = MagicMock(spec=ToolUseBlock)
    block.id = id
    block.name = name
    block.input = input
    return block


def test_request_includes_model_messages_system_tools_max_tokens() -> None:
    """The SDK is called with our config translated to Anthropic's shape."""
    fake = _fake_anthropic(content=[_text_block("ok")])
    client = AnthropicLLMClient(client=fake, model="claude-test")

    tool = ToolDef(
        name="submit_plan",
        description="...",
        input_schema={"type": "object"},
    )
    client.complete(
        messages=[Message(role="user", content="hi")],
        system="be helpful",
        tools=[tool],
        max_tokens=1000,
    )

    fake.messages.create.assert_called_once()
    kwargs = fake.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-test"
    assert kwargs["max_tokens"] == 1000
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["system"] == "be helpful"
    assert kwargs["tools"][0]["name"] == "submit_plan"


def test_request_omits_system_and_tools_when_unset() -> None:
    """``system`` and ``tools`` are excluded from kwargs when not provided."""
    fake = _fake_anthropic(content=[_text_block("ok")])
    AnthropicLLMClient(client=fake).complete(
        messages=[Message(role="user", content="hi")]
    )

    kwargs = fake.messages.create.call_args.kwargs
    assert "system" not in kwargs
    assert "tools" not in kwargs


def test_text_blocks_concatenate_into_text_field() -> None:
    """Multiple TextBlocks join into a single LLMResponse.text."""
    fake = _fake_anthropic(
        content=[_text_block("hello "), _text_block("world")],
        stop_reason="end_turn",
    )
    response = AnthropicLLMClient(client=fake).complete(
        messages=[Message(role="user", content="hi")]
    )

    assert response.text == "hello world"
    assert response.tool_calls == []
    assert response.stop_reason == "end_turn"


def test_tool_use_blocks_become_llm_tool_calls() -> None:
    """ToolUseBlocks are converted to LLMToolCall with id/name/arguments."""
    fake = _fake_anthropic(
        content=[
            _tool_use_block(
                id="call_001",
                name="submit_plan",
                input={"reasoning": "go", "count": 3},
            )
        ],
        stop_reason="tool_use",
    )
    response = AnthropicLLMClient(client=fake).complete(
        messages=[Message(role="user", content="plan it")]
    )

    assert response.text == ""
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_001"
    assert call.name == "submit_plan"
    assert call.arguments == {"reasoning": "go", "count": 3}
    assert response.stop_reason == "tool_use"


def test_usage_is_propagated() -> None:
    """Token counts from the SDK pass through to LLMResponse.usage."""
    fake = _fake_anthropic(
        content=[_text_block("ok")],
        input_tokens=123,
        output_tokens=45,
    )
    response = AnthropicLLMClient(client=fake).complete(
        messages=[Message(role="user", content="hi")]
    )

    assert response.usage.input_tokens == 123
    assert response.usage.output_tokens == 45


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing without a key in env or argument raises ValueError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicLLMClient()


def test_default_model_is_haiku() -> None:
    """The default model points at a current Haiku version string."""
    assert DEFAULT_MODEL.startswith("claude-haiku-")