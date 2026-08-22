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


def _text_block(text: str) -> MagicMock:
    """
    Helper function used to build one Anthropic TextBlock, the only
    block type that carries a text attribute.
    """
    block = MagicMock(spec=["type", "text"])
    block.type = "text"
    block.text = text
    return block


def _tool_use_block() -> MagicMock:
    """
    Helper function used to build one Anthropic ToolUseBlock. spec omits
    text deliberately: a tool_use block has no text attribute at all,
    which is what the adapter has to survive.
    """
    block = MagicMock(spec=["type", "id", "name", "input"])
    block.type = "tool_use"
    block.name = "get_weather"
    block.input = {"city": "Singapore"}
    return block


def _fake_client(
    response_id: str = "msg_abc",
    response_text: str = "ok",
    input_tokens: int = 10,
    output_tokens: int = 5,
    content: list[MagicMock] | None = None,
) -> MagicMock:
    """
    Helper function used to build a MagicMock Anthropic SDK client whose
    messages.create returns a minimal response with content blocks,
    usage, and id set in Anthropic's shape. Pass content to control the
    block list; the default is a single text block holding response_text.
    """
    client = MagicMock()
    response = MagicMock()
    response.content = (
        content
        if content is not None
        else [_text_block(response_text)]
    )
    response.id = response_id
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response.usage = usage
    client.messages.create.return_value = response
    return client


def _text_message(role: str, text: str) -> dict[str, object]:
    """
    Helper function used to build one content-parts message, the shape
    send_message's messages parameter and every adapter now share.
    """
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def test_adapter_omits_system_when_no_system_message() -> None:
    """
    With no system-role message in the list, the SDK call omits the
    system kwarg entirely (rather than passing system="" or None),
    matching Anthropic SDK convention for user-only calls.
    """
    client = _fake_client()
    AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        messages=[_text_message("user", "u")],
    )
    call_kwargs = client.messages.create.call_args.kwargs
    assert "system" not in call_kwargs


def test_adapter_lifts_system_message_to_top_level_param() -> None:
    """
    A system-role message is pulled out of messages and joined into the
    top-level system parameter, since the Messages API takes system as
    its own kwarg rather than a message in the list. This is the key
    shape difference from OpenAI chat completions.
    """
    client = _fake_client()
    AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        messages=[
            _text_message("system", "You are concise."),
            _text_message("user", "hi"),
        ],
    )
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "You are concise."
    assert call_kwargs["messages"] == [_text_message("user", "hi")]


def test_adapter_joins_multiple_system_messages() -> None:
    """
    More than one system-role message (unusual, but not forbidden by
    the shape) joins into a single system string in order, rather than
    only the first or last surviving.
    """
    client = _fake_client()
    AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        messages=[
            _text_message("system", "First."),
            _text_message("system", "Second."),
            _text_message("user", "hi"),
        ],
    )
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "First.\nSecond."


def test_adapter_forwards_multi_turn_history() -> None:
    """
    A multi-turn messages list (user, assistant, user) reaches the SDK
    with every non-system entry preserved in order, so conversation
    history round-trips unchanged.
    """
    client = _fake_client()
    messages = [
        _text_message("user", "first"),
        _text_message("assistant", "reply"),
        _text_message("user", "second"),
    ]
    AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        messages=messages,
    )
    call_kwargs = client.messages.create.call_args.kwargs
    assert call_kwargs["messages"] == messages


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
        messages=[
            _text_message("system", "s"),
            _text_message("user", "u"),
        ],
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
        messages=[
            _text_message("system", "s"),
            _text_message("user", "u"),
        ],
    )
    assert text == "hello world"
    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert gen_id == "msg_abc123"


def test_adapter_joins_every_text_block() -> None:
    """
    A response carrying more than one text block returns all of them
    joined, not just the first. Reading content[0] alone silently
    truncated the response to its opening block.
    """
    client = _fake_client(
        content=[
            _text_block("first half"),
            _text_block("second half"),
        ],
    )
    text, _, _ = AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        messages=[_text_message("user", "u")],
    )
    assert text == "first half\nsecond half"


def test_adapter_returns_empty_text_for_tool_only_turn() -> None:
    """
    A turn made up entirely of tool_use blocks has no text to report, so
    the adapter returns "" rather than raising. Recording that turn as
    something other than an empty response is a separate concern from
    this extraction.
    """
    client = _fake_client(content=[_tool_use_block()])
    text, _, _ = AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        messages=[_text_message("user", "u")],
    )
    assert text == ""


def test_adapter_skips_blocks_without_text() -> None:
    """
    A block that carries no text attribute (tool_use here, thinking
    likewise) contributes nothing and does not mask the text blocks
    around it. Reading content[0] alone lost the whole response
    whenever a non-text block happened to come first.
    """
    client = _fake_client(
        content=[
            _tool_use_block(),
            _text_block("the actual answer"),
        ],
    )
    text, _, _ = AnthropicAdapter().send(
        client=client,
        model="claude-haiku-4-5",
        messages=[_text_message("user", "u")],
    )
    assert text == "the actual answer"
