"""
Unit tests for the Anthropic native adapter.

The Anthropic SDK client is mocked so these run fully offline. Each test
asserts which kwargs reach client.messages.create and that the
Anthropic-shaped response is translated correctly into the uniform
(text, usage_dict, generation_id) tuple.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_router_ledger.providers.anthropic_native import AnthropicAdapter


def _fake_client(
    response_id: str = "msg_abc",
    response_text: str = "ok",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """
    Helper function used to build a MagicMock Anthropic SDK client whose
    messages.create returns a minimal response with content blocks,
    usage, and id set in Anthropic's shape.
    """
    client = MagicMock()
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    response.content = [content_block]
    response.id = response_id
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.usage = usage
    client.messages.create.return_value = response
    return client


def test_adapter_omits_system_when_none() -> None:
    """
    With system=None the SDK call omits the system kwarg entirely (rather
    than passing system=None), matching Anthropic SDK convention for
    user-only calls.
    """
    client = _fake_client()
    AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        system=None,
        user="u",
    )
    call_kwargs = client.messages.create.call_args.kwargs
    assert "system" not in call_kwargs


def test_adapter_passes_system_as_top_level_param() -> None:
    """
    When system is provided it lands as the top-level system parameter,
    not as a message in the messages list. This is the key shape
    difference from OpenAI chat completions.
    """
    client = _fake_client()
    AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        system="You are concise.",
        user="hi",
    )
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "You are concise."
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "hi"},
    ]


def test_adapter_silently_ignores_unsupported_kwargs() -> None:
    """
    user_id, extra_body, and response_format are not supported by the
    Messages API in this adapter and should not appear in the SDK call.
    Callers who set them get a quiet drop, not an error, so the same
    send_message() signature works across providers.
    """
    client = _fake_client()
    AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        system="s",
        user="u",
        user_id="run-tag-123",
        extra_body={"foo": "bar"},
        response_format={"type": "json_object"},
    )
    call_kwargs = client.messages.create.call_args.kwargs
    assert "user" not in call_kwargs
    assert "extra_body" not in call_kwargs
    assert "response_format" not in call_kwargs


def test_adapter_translates_response_shape() -> None:
    """
    AnthropicAdapter translates Anthropic's response shape (content
    blocks; input_tokens/output_tokens) into the uniform tuple shape
    (text, prompt/completion/total_tokens dict, generation_id).
    """
    client = _fake_client(
        response_id="msg_abc123",
        response_text="hello world",
        input_tokens=10,
        output_tokens=5,
    )
    text, usage, gen_id = AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        system="s",
        user="u",
    )
    assert text == "hello world"
    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert gen_id == "msg_abc123"
