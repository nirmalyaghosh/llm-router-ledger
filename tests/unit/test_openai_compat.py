"""
Unit tests for the OpenAI-compatible adapter.

The SDK client is mocked so these run fully offline. Each test asserts
which kwargs reach client.chat.completions.create.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_router_ledger.providers.openai_compat import OpenAICompatAdapter


def _fake_client(
    response_id: str = "gen-abc",
    response_text: str = "ok",
) -> MagicMock:
    """
    Helper function used to build a MagicMock SDK client whose
    chat.completions.create returns a minimal response with usage, id,
    and choices[0].message.content set.
    """
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = response_text
    response.id = response_id
    usage = MagicMock()
    usage.prompt_tokens = 1
    usage.completion_tokens = 2
    usage.total_tokens = 3
    response.usage = usage
    client.chat.completions.create.return_value = response
    return client


def test_adapter_forwards_user_id_as_user() -> None:
    """
    The user_id kwarg lands as the SDK's "user" field on
    chat.completions.create.
    """
    client = _fake_client()
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        system="s",
        user="u",
        user_id="run-tag-123",
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["user"] == "run-tag-123"


def test_adapter_forwards_extra_body() -> None:
    """
    The extra_body kwarg is passed through to the SDK verbatim for
    vendor-specific fields like OpenRouter provider routing.
    """
    client = _fake_client()
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        system="s",
        user="u",
        extra_body={"provider": {"sort": "latency"}},
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {"provider": {"sort": "latency"}}


def test_adapter_forwards_response_format() -> None:
    """
    response_format reaches the SDK so JSON mode and json_schema
    structured outputs work.
    """
    client = _fake_client()
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        system="s",
        user="u",
        response_format={"type": "json_object"},
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_adapter_omits_system_message_when_none() -> None:
    """
    With system=None the messages list contains only the user role,
    matching the OpenAI SDK convention for user-only calls.
    """
    client = _fake_client()
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        system=None,
        user="u",
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "u"},
    ]


def test_adapter_omits_user_and_extra_body_when_none() -> None:
    """
    When user_id and extra_body are not passed the SDK call gets neither
    key, so older clients that reject unknown kwargs do not break.
    """
    client = _fake_client()
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        system="s",
        user="u",
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "user" not in call_kwargs
    assert "extra_body" not in call_kwargs
