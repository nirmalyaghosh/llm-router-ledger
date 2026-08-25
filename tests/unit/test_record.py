"""
Unit tests for recording calls the library did not make itself:
UsageTracker.record_request, record_response, and record_run.

The Pydantic AI message objects are stood in for by plain dataclasses
carrying the same attribute names. record_run reads them by duck typing
precisely so the core package needs no pydantic-ai dependency, and the
tests exercise it the same way. The attribute names and the RequestUsage
field names were read off pydantic-ai 2.34.0.
"""

from __future__ import annotations

import json

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from typing import Any

from llm_router_ledger.usage_tracker import UsageTracker


@dataclass
class _Usage:
    """
    Stand-in for pydantic_ai.usage.RequestUsage. total_tokens is a
    property on the real class, so it is one here too.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class _Part:
    """
    Stand-in for a message part. The real classes are distinguished by
    their part_kind discriminator, which is what record_run reads.
    """

    part_kind: str
    content: str = ""


@dataclass
class _Request:
    """
    Stand-in for pydantic_ai.messages.ModelRequest.
    """

    parts: list[_Part]
    kind: str = "request"


@dataclass
class _Response:
    """
    Stand-in for pydantic_ai.messages.ModelResponse.
    """

    parts: list[_Part]
    usage: _Usage
    model_name: str = "some-model"
    provider_name: str = "openai"
    provider_response_id: str = "chatcmpl-1"
    finish_reason: str | None = "stop"
    provider_details: dict[str, Any] = field(
        default_factory=dict,
    )
    kind: str = "response"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Helper function used to read all JSONL entries from path into a list
    of dicts.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tracker(path: Path) -> UsageTracker:
    """
    Helper function used to build a tracker whose previews are readable,
    so a test can assert on the text that reached the ledger.
    """
    return UsageTracker(
        log_path=path,
        project_id="p",
        preview_length=200,
    )


def test_record_pair_matches_the_send_message_shape(
    tmp_log_path: Path,
) -> None:
    """
    A recorded call produces the same two events, with the same keys, as
    one the library made itself. If it did not, a reconciler would need
    two readers.
    """
    tracker = _tracker(tmp_log_path)
    try:
        request_id = tracker.record_request(
            model="some-model",
            provider="openrouter",
            purpose="query-planning",
            user_prompt="hello",
            metadata={"agent": "planner"},
        )
        tracker.record_response(
            request_id=request_id,
            model="some-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
            },
            response_id="gen-abc",
            response_text="hi",
            provider="openrouter",
            purpose="query-planning",
            metadata={"agent": "planner"},
        )
    finally:
        tracker.close()

    request, response = _read_jsonl(tmp_log_path)
    assert request["event"] == "llm_request"
    assert request["provider"] == "openrouter"
    assert request["purpose"] == "query-planning"
    assert request["user_prompt_preview"] == "hello"
    assert request["metadata"] == {"agent": "planner"}
    assert response["event"] == "llm_response"
    assert response["request_id"] == request_id
    assert response["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert response["usage_details"] == {"cost": 0.001}
    assert response["response_preview"] == "hi"


def test_record_response_routes_the_id_by_prefix(
    tmp_log_path: Path,
) -> None:
    """
    A "gen-" prefixed id is an OpenRouter generation id and lands in
    generation_id; anything else is a provider response id. Recording
    must reuse that routing rather than pick its own key.
    """
    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_response(
            request_id="r-1",
            model="m",
            response_id="gen-abc",
        )
        tracker.record_response(
            request_id="r-2",
            model="m",
            response_id="chatcmpl-abc",
        )
    finally:
        tracker.close()

    routed, plain = _read_jsonl(tmp_log_path)
    assert routed["generation_id"] == "gen-abc"
    assert "provider_response_id" not in routed
    assert plain["provider_response_id"] == "chatcmpl-abc"
    assert "generation_id" not in plain


def test_record_response_without_usage_writes_zeroes(
    tmp_log_path: Path,
) -> None:
    """
    A provider that reported no usage writes zeroes, the same as an
    absent usage block already produces, and no empty usage_details key.
    """
    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_response(request_id="r-1", model="m")
    finally:
        tracker.close()

    entry = _read_jsonl(tmp_log_path)[0]
    assert entry["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert "usage_details" not in entry


def test_record_response_accepts_anthropic_token_names(
    tmp_log_path: Path,
) -> None:
    """
    Anthropic reports input_tokens and output_tokens. Splitting a
    mapping in those names by the canonical keys alone would file the
    counts under usage_details and leave the usage block reading zero
    tokens for a call that cost real ones, so the aliases are
    translated first.
    """
    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_response(
            request_id="r-1",
            model="m",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
    finally:
        tracker.close()

    entry = _read_jsonl(tmp_log_path)[0]
    assert entry["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert "usage_details" not in entry


def test_record_response_keeps_a_reported_total(
    tmp_log_path: Path,
) -> None:
    """
    total_tokens is only derived when the provider omitted it. A
    reported total is kept as given, even where it disagrees with the
    sum, since the provider's own figure is what an invoice is drawn
    from.
    """
    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_response(
            request_id="r-1",
            model="m",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 99,
            },
        )
    finally:
        tracker.close()

    assert _read_jsonl(tmp_log_path)[0]["usage"][
        "total_tokens"
    ] == 99


def test_record_run_writes_one_pair_per_model_call(
    tmp_log_path: Path,
) -> None:
    """
    A run that called a tool once made two model calls, so it must
    produce two pairs. Collapsing them into one would lose the first
    call's tokens, which are billed separately.
    """
    messages = [
        _Request(
            parts=[
                _Part("system-prompt", "be helpful"),
                _Part("user-prompt", "what is 21 plus 21?"),
            ],
        ),
        _Response(
            parts=[_Part("tool-call")],
            usage=_Usage(input_tokens=179, output_tokens=31),
            provider_response_id="chatcmpl-1",
            finish_reason="tool_call",
        ),
        _Request(parts=[_Part("tool-return", "42")]),
        _Response(
            parts=[_Part("text", "The answer is 42.")],
            usage=_Usage(input_tokens=221, output_tokens=20),
            provider_response_id="chatcmpl-2",
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        request_ids = tracker.record_run(
            messages,
            purpose="arithmetic",
            provider="lmstudio",
        )
    finally:
        tracker.close()

    entries = _read_jsonl(tmp_log_path)
    assert [entry["event"] for entry in entries] == [
        "llm_request",
        "llm_response",
        "llm_request",
        "llm_response",
    ]
    assert len(request_ids) == 2
    first_request, first, second_request, second = entries
    assert first_request["request_id"] == request_ids[0]
    assert first["request_id"] == request_ids[0]
    assert second["request_id"] == request_ids[1]
    assert first["usage"] == {
        "prompt_tokens": 179,
        "completion_tokens": 31,
        "total_tokens": 210,
    }
    assert second["usage"] == {
        "prompt_tokens": 221,
        "completion_tokens": 20,
        "total_tokens": 241,
    }
    assert first["provider_response_id"] == "chatcmpl-1"
    assert second["provider_response_id"] == "chatcmpl-2"
    assert {entry["purpose"] for entry in entries} == {
        "arithmetic",
    }


def test_record_run_stamps_the_endpoint_provider(
    tmp_log_path: Path,
) -> None:
    """
    The provider name on the message is the framework's own, and is
    "openai" for every OpenAI-compatible server. Passing the endpoint's
    provider must override it, or a local call would be filed as an
    OpenAI one.
    """
    messages = [
        _Response(
            parts=[_Part("text", "hi")],
            usage=_Usage(input_tokens=1, output_tokens=1),
            provider_name="openai",
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_run(messages, provider="lmstudio")
        tracker.record_run(messages)
    finally:
        tracker.close()

    entries = _read_jsonl(tmp_log_path)
    assert {entry["provider"] for entry in entries[:2]} == {
        "lmstudio",
    }
    assert {entry["provider"] for entry in entries[2:]} == {
        "openai",
    }


def test_record_run_records_the_turn_shape(
    tmp_log_path: Path,
) -> None:
    """
    A turn that returned only tool calls has no text, so the ledger
    would read it as an empty response. The tool call count and the
    non-stop finish reason are what keep it legible, mirroring what the
    OpenAI-compatible adapter records for the same turn.
    """
    messages = [
        _Response(
            parts=[
                _Part("thinking", "let me think"),
                _Part("tool-call"),
                _Part("tool-call"),
            ],
            usage=_Usage(input_tokens=10, output_tokens=4),
            finish_reason="tool_call",
            provider_details={"finish_reason": "tool_calls"},
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_run(messages)
    finally:
        tracker.close()

    response = _read_jsonl(tmp_log_path)[1]
    assert response["response_preview"] == ""
    assert response["response_length"] == 0
    assert response["usage_details"] == {
        "completion_tool_call_count": 2,
        "finish_reason": "tool_calls",
    }


def test_record_run_records_the_providers_finish_reason(
    tmp_log_path: Path,
) -> None:
    """
    Both adapters record the provider's own word for why a turn ended.
    The framework normalises it, so the same truncated turn would be
    "tool_calls" from send_message and "tool_call" from here. A
    reconciler filtering on finish_reason must not have to know both,
    so the raw value in provider_details wins.
    """
    messages = [
        _Response(
            parts=[_Part("text", "cut off")],
            usage=_Usage(input_tokens=1, output_tokens=1),
            finish_reason="length",
            provider_details={"finish_reason": "max_tokens"},
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_run(messages)
    finally:
        tracker.close()

    assert _read_jsonl(tmp_log_path)[1]["usage_details"] == {
        "finish_reason": "max_tokens",
    }


def test_record_run_falls_back_to_the_normalised_reason(
    tmp_log_path: Path,
) -> None:
    """
    Not every provider reports a raw finish reason. Where none is
    carried, the framework's normalised one is better than nothing.
    """
    messages = [
        _Response(
            parts=[_Part("text", "cut off")],
            usage=_Usage(input_tokens=1, output_tokens=1),
            finish_reason="length",
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_run(messages)
    finally:
        tracker.close()

    assert _read_jsonl(tmp_log_path)[1]["usage_details"] == {
        "finish_reason": "length",
    }


def test_record_run_omits_an_ordinary_finish_reason(
    tmp_log_path: Path,
) -> None:
    """
    A turn that ran to completion records no finish_reason at all, so
    the key's presence means the response is not the whole answer.
    Anthropic says end_turn where OpenAI says stop, and either can
    reach here through a framework, so both count as ordinary.
    """
    messages = [
        _Response(
            parts=[_Part("text", "done")],
            usage=_Usage(input_tokens=1, output_tokens=1),
            finish_reason="stop",
        ),
        _Response(
            parts=[_Part("text", "done")],
            usage=_Usage(input_tokens=1, output_tokens=1),
            finish_reason="stop",
            provider_details={"finish_reason": "end_turn"},
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_run(messages)
    finally:
        tracker.close()

    entries = _read_jsonl(tmp_log_path)
    assert "usage_details" not in entries[1]
    assert "usage_details" not in entries[3]


def test_record_run_maps_usage_detail_to_adapter_key_names(
    tmp_log_path: Path,
) -> None:
    """
    Pydantic AI's usage fields carry different names from the ones the
    adapters write for the same quantities. They must be translated, or
    a cost report joining both sources would have to know two
    vocabularies.

    Zero-valued detail keys are dropped, matching the adapters. cost
    becomes estimated_cost: it is computed from a local price table,
    not reported by the provider, and must never be mistaken for what
    was billed.
    """
    messages = [
        _Response(
            parts=[_Part("text", "hi")],
            usage=_Usage(
                input_tokens=100,
                output_tokens=20,
                cache_read_tokens=64,
                cache_write_tokens=8,
                cost=0.00042,
                details={
                    "reasoning_tokens": 12,
                    "rejected_prediction_tokens": 0,
                    "some_new_key": 3,
                },
            ),
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_run(messages)
    finally:
        tracker.close()

    response = _read_jsonl(tmp_log_path)[1]
    assert response["usage_details"] == {
        "completion_reasoning_tokens": 12,
        "estimated_cost": 0.00042,
        "prompt_cache_write_tokens": 8,
        "prompt_cached_tokens": 64,
        "unmapped": {"some_new_key": 3},
    }


def test_record_run_previews_the_prompt_that_led_to_the_call(
    tmp_log_path: Path,
) -> None:
    """
    The request event previews the user prompt of the request that
    preceded the call. A tool return carries no user prompt, so the turn
    it continues records an empty preview rather than repeating the
    original question and reading as though it were asked twice.
    """
    messages = [
        _Request(parts=[_Part("user-prompt", "first question")]),
        _Response(
            parts=[_Part("tool-call")],
            usage=_Usage(input_tokens=1, output_tokens=1),
        ),
        _Request(parts=[_Part("tool-return", "42")]),
        _Response(
            parts=[_Part("text", "done")],
            usage=_Usage(input_tokens=1, output_tokens=1),
        ),
    ]

    tracker = _tracker(tmp_log_path)
    try:
        tracker.record_run(messages)
    finally:
        tracker.close()

    entries = _read_jsonl(tmp_log_path)
    assert entries[0]["user_prompt_preview"] == "first question"
    assert entries[0]["user_prompt_length"] == 14
    assert entries[2]["user_prompt_preview"] == ""
    assert entries[2]["user_prompt_length"] == 0


def test_record_run_ignores_messages_it_does_not_understand(
    tmp_log_path: Path,
) -> None:
    """
    Duck typing means anything can be in the list. Something that is
    neither a request nor a response is skipped rather than recorded as
    an empty call, so a future message kind cannot put a phantom row in
    the ledger.
    """
    tracker = _tracker(tmp_log_path)
    try:
        assert tracker.record_run(["not a message", object()]) == []
    finally:
        tracker.close()

    assert not tmp_log_path.read_text(encoding="utf-8").strip()
