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
    response.choices[0].finish_reason = "stop"
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
    text: str | None = "ok",
    response_id: str = "gen-abc",
    provider: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """
    Helper function used to build a plain response object for the usage-
    detail tests below. SimpleNamespace rather than MagicMock: an unset
    attribute raises AttributeError like a real SDK object, so
    getattr(obj, key, default) actually exercises the default path
    instead of MagicMock auto-vivifying a truthy child mock that would
    silently mask a bug in the extraction logic.

    text is None and tool_calls is a list on a tool-call turn, matching
    what the API returns there. The message carries no tool_calls
    attribute at all unless one is passed, since an ordinary provider
    response omits it entirely. The choice likewise carries no
    finish_reason unless passed, as some OpenAI-compatible servers
    omit it.
    """
    message = SimpleNamespace(content=text)
    if tool_calls is not None:
        message.tool_calls = tool_calls
    choice = SimpleNamespace(message=message)
    if finish_reason is not None:
        choice.finish_reason = finish_reason
    kwargs: dict[str, object] = {
        "choices": [choice],
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


def test_adapter_collects_unmapped_usage_keys() -> None:
    """
    A usage key the adapter has no mapping for is collected under
    "unmapped" rather than dropped. A top-level key keeps its own name;
    a key inside a detail block takes the same prompt_ / completion_
    prefix its mapped siblings take. The shape below is DeepSeek's
    top-level cache keys and Qwen's text_tokens, both observed live.
    """
    usage = SimpleNamespace(
        prompt_tokens=9007,
        completion_tokens=1,
        total_tokens=9008,
        prompt_cache_hit_tokens=8960,
        prompt_cache_miss_tokens=47,
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=0,
            text_tokens=352,
        ),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=8960,
            text_tokens=17,
        ),
    )
    client = _client_returning(_fake_response(usage=usage))
    _, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert usage_out["prompt_cached_tokens"] == 8960
    assert usage_out["unmapped"] == {
        "prompt_cache_hit_tokens": 8960,
        "prompt_cache_miss_tokens": 47,
        "completion_text_tokens": 352,
        "prompt_text_tokens": 17,
    }


def test_adapter_omits_unmapped_when_every_key_maps() -> None:
    """
    OpenAI's own API reports nothing this adapter cannot place, so the
    usage dict carries no "unmapped" key at all rather than an empty
    one, matching how the other optional keys are written.
    """
    usage = SimpleNamespace(
        prompt_tokens=14,
        completion_tokens=1,
        total_tokens=15,
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=0,
            audio_tokens=0,
        ),
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=0,
            audio_tokens=0,
        ),
    )
    client = _client_returning(_fake_response(usage=usage))
    _, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert "unmapped" not in usage_out


def test_adapter_counts_tool_calls_on_tool_only_turn() -> None:
    """
    A turn that returns only tool calls sets message.content to null,
    which the adapter reports as "". Without a count the ledger row is
    indistinguishable from a model that answered with nothing, so
    completion_tool_call_count records what actually happened.
    """
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    client = _client_returning(
        _fake_response(
            usage=usage,
            text=None,
            tool_calls=[
                SimpleNamespace(id="call_1"),
                SimpleNamespace(id="call_2"),
            ],
        ),
    )
    text, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert text == ""
    assert usage_out["completion_tool_call_count"] == 2


def test_adapter_omits_ordinary_finish_reason() -> None:
    """
    finish_reason "stop" is dropped rather than written on every row.
    """
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )
    client = _client_returning(
        _fake_response(usage=usage, finish_reason="stop"),
    )
    _, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert "finish_reason" not in usage_out


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


def test_adapter_records_truncated_finish_reason() -> None:
    """
    finish_reason "length" means the model was cut off at max_tokens.
    The token counts look like any other full turn, so without this key
    the ledger cannot tell a truncated response from a complete one.
    """
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=4096,
        total_tokens=4196,
    )
    client = _client_returning(
        _fake_response(usage=usage, finish_reason="length"),
    )
    _, usage_out, _ = OpenAICompatAdapter().send(
        client=client,
        model="m",
        messages=[_text_message("user", "u")],
    )
    assert usage_out["finish_reason"] == "length"


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
