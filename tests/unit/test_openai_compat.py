"""
Unit tests for the OpenAI-compatible adapter.

The SDK client is mocked so these run fully offline. Each test asserts
which kwargs reach client.chat.completions.create.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from llm_router_ledger.providers.openai_compat import OpenAICompatAdapter


def _client_returning(response: SimpleNamespace) -> MagicMock:
    """
    Helper function used to wrap a fake response in a MagicMock client,
    for the usage-detail tests below.
    """
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


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


def _fake_response(
    *,
    usage: SimpleNamespace | None,
    text: str = "ok",
    response_id: str = "gen-abc",
    provider: str | None = None,
) -> SimpleNamespace:
    """
    Helper function used to build a plain response object for the usage-
    detail tests below. SimpleNamespace rather than MagicMock: an unset
    attribute raises AttributeError like a real SDK object, so
    getattr(obj, key, default) actually exercises the default path
    instead of MagicMock auto-vivifying a truthy child mock that would
    silently mask a bug in the extraction logic.
    """
    kwargs: dict[str, object] = {
        "choices": [SimpleNamespace(message=SimpleNamespace(content=text))],
        "id": response_id,
        "usage": usage,
    }
    if provider is not None:
        kwargs["provider"] = provider
    return SimpleNamespace(**kwargs)


def _text_message(role: str, text: str) -> dict[str, object]:
    """
    Helper function used to build one content-parts message, the shape
    send_message's messages parameter and every adapter now share.
    """
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def test_adapter_forwards_user_id_as_user() -> None:
    """
    The user_id kwarg lands as the SDK's "user" field on
    chat.completions.create.
    """
    client = _fake_client()
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
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
        messages=[_text_message("user", "u")],
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
        messages=[_text_message("user", "u")],
        response_format={"type": "json_object"},
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_adapter_forwards_messages_unchanged() -> None:
    """
    messages reaches the SDK exactly as built by the dispatcher: no
    per-role conversion happens in this adapter, since the
    content-parts shape already matches what the SDK expects for
    system, user, and assistant roles alike.
    """
    client = _fake_client()
    messages = [
        _text_message("system", "s"),
        _text_message("user", "u"),
    ]
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=messages,
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == messages


def test_adapter_omits_user_and_extra_body_when_none() -> None:
    """
    When user_id and extra_body are not passed the SDK call gets neither
    key, so older clients that reject unknown kwargs do not break.
    """
    client = _fake_client()
    OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "user" not in call_kwargs
    assert "extra_body" not in call_kwargs


def test_adapter_captures_cost_and_reasoning_detail() -> None:
    """
    cost, is_byok, upstream_provider, and the flattened reasoning /
    cache detail keys all land in the returned usage dict, prefixed
    per block, when the provider reports them. Zero-valued and null
    keys within a detail block are omitted.
    """
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.0012,
        is_byok=False,
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=40,
            image_tokens=0,
            audio_tokens=0,
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=20,
            cache_write_tokens=0,
            audio_tokens=0,
            video_tokens=0,
        ),
    )
    client = _client_returning(
        _fake_response(usage=usage, provider="deepinfra"),
    )
    _, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert usage_out["cost"] == 0.0012
    assert usage_out["is_byok"] is False
    assert usage_out["upstream_provider"] == "deepinfra"
    assert usage_out["completion_reasoning_tokens"] == 40
    assert usage_out["prompt_cached_tokens"] == 20
    for absent_key in (
        "completion_image_tokens",
        "completion_audio_tokens",
        "completion_accepted_prediction_tokens",
        "completion_rejected_prediction_tokens",
        "prompt_cache_write_tokens",
        "prompt_audio_tokens",
    ):
        assert absent_key not in usage_out


def test_adapter_omits_optional_usage_keys_when_absent() -> None:
    """
    A provider that reports only the three base token counts (no cost,
    no detail blocks, no is_byok, no provider field) yields a usage
    dict with exactly those three keys.
    """
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )
    client = _client_returning(_fake_response(usage=usage))
    _, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert usage_out == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


def test_adapter_handles_missing_usage() -> None:
    """
    A provider that omits usage entirely (response.usage is None) still
    returns the three base keys, all zero, with no crash.
    """
    client = _client_returning(_fake_response(usage=None))
    _, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert usage_out == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
