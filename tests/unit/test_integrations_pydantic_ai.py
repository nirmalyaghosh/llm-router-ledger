"""
Unit tests for the Pydantic AI integration.

These run against the real package rather than stand-ins: the point of
a wrapping model is what happens when the framework drives it, so the
wrapped model is Pydantic AI's own FunctionModel and the calls go
through a real Agent. No network is involved.

The other half of the integration, record_run, is covered by
test_record.py with duck-typed stand-ins, and one test here asserts the
two paths write the same rows.
"""

from __future__ import annotations

import asyncio
import json

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import (
    AgentInfo,
    FunctionModel,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.usage import RequestUsage

from llm_router_ledger import (
    UsageTracker,
    load_config,
    purpose_scope,
)
from llm_router_ledger.config import LLMConfig
from llm_router_ledger.exceptions import EndpointNotFoundError
from llm_router_ledger.integrations.pydantic_ai import (
    _LedgerModel,
    ledger_model,
)


ENDPOINTS_YAML = """\
endpoints:
  local-chat:
    provider: lmstudio
    model: qwen3-4b
    api_key_env: LMSTUDIO_API_KEY
    base_url: http://localhost:1234/v1

  reasoning-off:
    provider: openrouter
    model: openai/gpt-4.1-nano
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    extra_body:
      reasoning:
        enabled: false
"""


def _events(path: Path) -> list[dict[str, Any]]:
    """
    Helper function used to read every event out of the ledger file the
    tracker wrote.
    """
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]


def _replies(
    *responses: ModelResponse,
) -> FunctionModel:
    """
    Helper function used to build a FunctionModel returning the given
    responses in order, so a test can drive a multi-turn run.
    """
    turns = iter(responses)

    def reply(
        messages: list[Any],
        info: AgentInfo,
    ) -> ModelResponse:
        return next(turns)

    return FunctionModel(reply)


def _text_reply(
    text: str = "hello",
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 3,
) -> ModelResponse:
    """
    Helper function used to build one finished text response carrying
    usage, the shape every provider returns for an ordinary turn.
    """
    return ModelResponse(
        parts=[TextPart(text)],
        usage=RequestUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        ),
        model_name="qwen3-4b",
        provider_name="function",
        provider_response_id="resp-1",
    )


def _tracker(path: Path) -> UsageTracker:
    """
    Helper function used to build a tracker whose previews are
    readable, so a test can assert on the text that reached the ledger.
    """
    return UsageTracker(
        log_path=path,
        project_id="p",
        preview_length=200,
    )


@pytest.fixture
def config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LLMConfig:
    """
    Return a config with a local OpenAI-compatible endpoint and one
    carrying extra_body.
    """
    monkeypatch.setenv("LMSTUDIO_API_KEY", "not-needed")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    path = tmp_path / "llm_endpoints.yaml"
    path.write_text(ENDPOINTS_YAML, encoding="utf-8")
    return load_config(path)


def _wrap(
    tracker: UsageTracker,
    model: FunctionModel,
    **kwargs: Any,
) -> _LedgerModel:
    """
    Helper function used to wrap a FunctionModel the way ledger_model
    wraps the model it builds from an endpoint.
    """
    kwargs.setdefault("provider", "lmstudio")
    return _LedgerModel(model, tracker=tracker, **kwargs)


def test_a_run_writes_one_pair_per_model_call(
    tmp_log_path: Path,
) -> None:
    """
    Every call the agent makes produces its own llm_request and
    llm_response, so a run that called a tool twice is two pairs rather
    than one.
    """
    tracker = _tracker(tmp_log_path)
    model = _replies(
        ModelResponse(
            parts=[ToolCallPart("weather", {"city": "Oslo"})],
            usage=RequestUsage(input_tokens=9, output_tokens=4),
        ),
        _text_reply("cold in Oslo"),
    )
    agent = Agent(_wrap(tracker, model))

    @agent.tool_plain
    def weather(city: str) -> str:
        return "cold"

    agent.run_sync("weather in Oslo?")
    tracker.close()

    events = _events(tmp_log_path)
    kinds = [e["event"] for e in events]
    assert kinds == [
        "llm_request",
        "llm_response",
        "llm_request",
        "llm_response",
    ]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[2]["request_id"] == events[3]["request_id"]
    assert events[0]["request_id"] != events[2]["request_id"]


def test_the_endpoints_provider_is_stamped(
    tmp_log_path: Path,
) -> None:
    """
    The rows carry the endpoint's provider, not the framework's. The
    framework's is the same string for every OpenAI-compatible server.
    """
    tracker = _tracker(tmp_log_path)
    agent = Agent(
        _wrap(
            tracker,
            _replies(_text_reply()),
            provider="lmstudio",
        )
    )
    agent.run_sync("hello")
    tracker.close()

    events = _events(tmp_log_path)
    assert {e["provider"] for e in events} == {"lmstudio"}


def test_usage_reaches_the_ledgers_token_keys(
    tmp_log_path: Path,
) -> None:
    """
    The framework's renamed usage fields land in the ledger under the
    three keys the usage block holds, so a row written here sums with
    one send_message wrote.
    """
    tracker = _tracker(tmp_log_path)
    agent = Agent(
        _wrap(
            tracker,
            _replies(
                _text_reply(
                    prompt_tokens=11,
                    completion_tokens=3,
                )
            ),
        )
    )
    agent.run_sync("hello")
    tracker.close()

    response = _events(tmp_log_path)[1]
    assert response["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }
    # Routed the way log_response routes it: only an id prefixed
    # "gen-" is a generation_id, so this one is not written as both.
    assert response["provider_response_id"] == "resp-1"
    assert "generation_id" not in response


def test_a_tool_call_turn_records_its_count(
    tmp_log_path: Path,
) -> None:
    """
    A turn that returned only tool calls is not an empty response: the
    count goes to usage_details, as the adapter writes it.
    """
    tracker = _tracker(tmp_log_path)
    model = _replies(
        ModelResponse(
            parts=[
                ToolCallPart("weather", {"city": "Oslo"}),
                ToolCallPart("weather", {"city": "Bergen"}),
            ],
            usage=RequestUsage(input_tokens=9, output_tokens=8),
            provider_details={"finish_reason": "tool_calls"},
        ),
        _text_reply(),
    )
    agent = Agent(_wrap(tracker, model))

    @agent.tool_plain
    def weather(city: str) -> str:
        return "cold"

    agent.run_sync("compare Oslo and Bergen")
    tracker.close()

    first = _events(tmp_log_path)[1]
    details = first["usage_details"]
    assert details["completion_tool_call_count"] == 2
    assert details["finish_reason"] == "tool_calls"


def test_instructions_are_recorded_as_the_system_prompt(
    tmp_log_path: Path,
) -> None:
    """
    An agent given instructions puts them on every request it sends,
    including the tool returns that continue a loop, so every row
    carries them.
    """
    tracker = _tracker(tmp_log_path)
    model = _replies(
        ModelResponse(
            parts=[ToolCallPart("weather", {"city": "Oslo"})],
            usage=RequestUsage(input_tokens=9, output_tokens=4),
        ),
        _text_reply(),
    )
    agent = Agent(
        _wrap(tracker, model),
        instructions="You are terse.",
    )

    @agent.tool_plain
    def weather(city: str) -> str:
        return "cold"

    agent.run_sync("weather in Oslo?")
    tracker.close()

    requests = [
        e
        for e in _events(tmp_log_path)
        if e["event"] == "llm_request"
    ]
    assert len(requests) == 2
    assert all(
        e["system_prompt_preview"] == "You are terse."
        for e in requests
    )


def test_a_system_prompt_carries_to_later_turns(
    tmp_log_path: Path,
) -> None:
    """
    An agent given a system prompt rather than instructions sets it on
    the first request only, but that request is resent as history, so
    every turn records it.
    """
    tracker = _tracker(tmp_log_path)
    model = _replies(
        ModelResponse(
            parts=[ToolCallPart("weather", {"city": "Oslo"})],
            usage=RequestUsage(input_tokens=9, output_tokens=4),
        ),
        _text_reply(),
    )
    agent = Agent(
        _wrap(tracker, model),
        system_prompt="You are terse.",
    )

    @agent.tool_plain
    def weather(city: str) -> str:
        return "cold"

    agent.run_sync("weather in Oslo?")
    tracker.close()

    requests = [
        e
        for e in _events(tmp_log_path)
        if e["event"] == "llm_request"
    ]
    assert len(requests) == 2
    assert all(
        e["system_prompt_preview"] == "You are terse."
        for e in requests
    )


def test_the_latest_user_prompt_is_previewed(
    tmp_log_path: Path,
) -> None:
    """
    The preview is the latest user prompt, not the tool return that
    continued the loop.
    """
    tracker = _tracker(tmp_log_path)
    model = _replies(
        ModelResponse(
            parts=[ToolCallPart("weather", {"city": "Oslo"})],
            usage=RequestUsage(input_tokens=9, output_tokens=4),
        ),
        _text_reply(),
    )
    agent = Agent(_wrap(tracker, model))

    @agent.tool_plain
    def weather(city: str) -> str:
        return "cold"

    agent.run_sync("weather in Oslo?")
    tracker.close()

    requests = [
        e
        for e in _events(tmp_log_path)
        if e["event"] == "llm_request"
    ]
    assert requests[0]["user_prompt_preview"] == (
        "weather in Oslo?"
    )
    assert requests[1]["user_prompt_preview"] == ""


def test_a_failed_call_writes_an_error_event(
    tmp_log_path: Path,
) -> None:
    """
    A call that raised writes an llm_error pairing the llm_request
    written before it, so the failure is recorded rather than leaving a
    bare request.
    """
    tracker = _tracker(tmp_log_path)

    def explode(
        messages: list[Any],
        info: AgentInfo,
    ) -> ModelResponse:
        raise RuntimeError("upstream is down")

    agent = Agent(_wrap(tracker, FunctionModel(explode)))
    with pytest.raises(RuntimeError):
        agent.run_sync("hello")
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_error",
    ]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[1]["error_type"] == "RuntimeError"
    assert events[1]["error_message"] == "upstream is down"
    assert events[1]["provider"] == "lmstudio"


def test_a_bound_purpose_reaches_every_event(
    tmp_log_path: Path,
) -> None:
    """
    The purpose passed at construction is stamped on the rows without
    the agent having to carry it.
    """
    tracker = _tracker(tmp_log_path)
    agent = Agent(
        _wrap(
            tracker,
            _replies(_text_reply()),
            purpose="query-planning",
        )
    )
    agent.run_sync("hello")
    tracker.close()

    events = _events(tmp_log_path)
    assert {e["purpose"] for e in events} == {
        "query-planning",
    }


def test_a_scope_overrides_the_bound_purpose(
    tmp_log_path: Path,
) -> None:
    """
    An active purpose_scope wins over the purpose bound at
    construction, which is how one model shared by several agents keeps
    them apart in the ledger.
    """
    tracker = _tracker(tmp_log_path)
    model = _wrap(
        tracker,
        _replies(_text_reply(), _text_reply()),
        purpose="query-planning",
    )
    agent = Agent(model)
    with purpose_scope("summarising"):
        agent.run_sync("hello")
    agent.run_sync("hello again")
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["purpose"] for e in events] == [
        "summarising",
        "summarising",
        "query-planning",
        "query-planning",
    ]


def test_metadata_is_stamped_on_every_event(
    tmp_log_path: Path,
) -> None:
    """
    Metadata bound at construction reaches the rows. It never reaches
    the wire: endpoint request params belong in extra_body.
    """
    tracker = _tracker(tmp_log_path)
    agent = Agent(
        _wrap(
            tracker,
            _replies(_text_reply()),
            metadata={"experiment": "a"},
        )
    )
    agent.run_sync("hello")
    tracker.close()

    events = _events(tmp_log_path)
    assert all(
        e["metadata"] == {"experiment": "a"} for e in events
    )


def test_streaming_records_when_the_stream_is_exhausted(
    tmp_log_path: Path,
) -> None:
    """
    Usage is not final until a stream ends, so the response event is
    written after it is exhausted rather than when it opens.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        for chunk in ("hel", "lo"):
            yield chunk

    agent = Agent(_wrap(tracker, FunctionModel(stream_function=stream)))

    async def drive() -> None:
        async with agent.run_stream("hello") as result:
            async for _ in result.stream_text():
                pass

    asyncio.run(drive())
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_response",
    ]
    assert events[1]["response_preview"] == "hello"
    assert events[1]["usage"]["completion_tokens"] > 0


def test_a_stream_read_only_part_way_records_a_response(
    tmp_log_path: Path,
) -> None:
    """
    Stopping early is not an error. The tokens consumed were billed, and
    the call records a response.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        for chunk in ("hel", "lo"):
            yield chunk

    agent = Agent(_wrap(tracker, FunctionModel(stream_function=stream)))

    async def drive() -> None:
        async with agent.run_stream("hello") as result:
            async for _ in result.stream_text():
                break

    asyncio.run(drive())
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_response",
    ]


def test_a_failed_stream_writes_an_error_and_no_response(
    tmp_log_path: Path,
) -> None:
    """
    A stream that breaks partway records the failure alone.

    The response is written only on a clean exit, so one request_id
    never carries both outcomes.
    """
    tracker = _tracker(tmp_log_path)

    class Dropped(Exception):
        pass

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        yield "hel"
        raise Dropped("upstream dropped the connection")

    agent = Agent(_wrap(tracker, FunctionModel(stream_function=stream)))

    async def drive() -> None:
        async with agent.run_stream("hello") as result:
            async for _ in result.stream_text():
                pass

    with pytest.raises(Dropped):
        asyncio.run(drive())
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_error",
    ]
    assert events[1]["error_type"] == "Dropped"
    assert events[1]["request_id"] == events[0]["request_id"]
    # The tokens produced before it broke were still billed, so they
    # reach the error rather than going unaccounted.
    assert events[1]["usage"]["completion_tokens"] > 0
    assert events[1]["usage"]["prompt_tokens"] > 0


def test_a_failure_while_the_stream_closes_writes_only_an_error(
    tmp_log_path: Path,
) -> None:
    """
    A transport can fail while closing, after a clean read.

    The response is written after the context manager exits, so the
    failure is recorded as an error rather than following a response.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        for chunk in ("hel", "lo"):
            yield chunk

    class ClosesBadly(WrapperModel):
        """Reads cleanly, then fails on the way out."""

        @asynccontextmanager
        async def request_stream(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            async with self.wrapped.request_stream(
                *args,
                **kwargs,
            ) as opened:
                yield opened
            raise ConnectionResetError("close failed")

    model = ClosesBadly(FunctionModel(stream_function=stream))
    agent = Agent(_wrap(tracker, model))

    async def drive() -> None:
        async with agent.run_stream("hello") as result:
            async for _ in result.stream_text():
                pass

    with pytest.raises(BaseException):
        asyncio.run(drive())
    tracker.close()

    kinds = [e["event"] for e in _events(tmp_log_path)]
    assert "llm_response" not in kinds
    assert kinds == ["llm_request", "llm_error"]


def test_a_ledger_write_failure_does_not_lose_the_call(
    tmp_log_path: Path,
) -> None:
    """
    A tracker that cannot write must not discard a call whose tokens
    are billed.

    The write is guarded and sits outside the failure handler, so it
    neither propagates to the caller nor is misrecorded as an
    llm_error.
    """
    tracker = _tracker(tmp_log_path)

    def explode(**kwargs: Any) -> None:
        raise OSError("no space left on device")

    agent = Agent(_wrap(tracker, _replies(_text_reply("hi"))))
    tracker.record_response = explode  # type: ignore[method-assign]

    result = agent.run_sync("hello")
    tracker.close()

    assert result.output == "hi"
    kinds = [e["event"] for e in _events(tmp_log_path)]
    assert kinds == ["llm_request"]
    assert "llm_error" not in kinds


def test_a_cancelled_call_is_not_left_unpaired(
    tmp_log_path: Path,
) -> None:
    """
    CancelledError derives from BaseException. Catching Exception alone
    would leave the request without an outcome, and cancellation is
    routine in an agent framework.
    """
    tracker = _tracker(tmp_log_path)

    def cancelled(
        messages: list[Any],
        info: AgentInfo,
    ) -> ModelResponse:
        raise asyncio.CancelledError

    model = _wrap(tracker, FunctionModel(cancelled))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            model.request(
                [],
                None,
                ModelRequestParameters(),
            )
        )
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_error",
    ]
    assert events[1]["error_type"] == "CancelledError"


def test_an_ordinary_failure_carries_no_usage_block(
    tmp_log_path: Path,
) -> None:
    """
    A call that raised before producing anything writes no usage
    block rather than zeroes, so llm_error rows remain safe to sum.
    """
    tracker = _tracker(tmp_log_path)

    def explode(
        messages: list[Any],
        info: AgentInfo,
    ) -> ModelResponse:
        raise RuntimeError("no tokens were produced")

    agent = Agent(_wrap(tracker, FunctionModel(explode)))

    with pytest.raises(Exception):
        agent.run_sync("hello")
    tracker.close()

    events = _events(tmp_log_path)
    assert events[-1]["event"] == "llm_error"
    assert "usage" not in events[-1]


def test_a_scope_entered_mid_stream_does_not_split_the_pair(
    tmp_log_path: Path,
) -> None:
    """
    A streamed call writes its response row when the stream closes,
    arbitrarily later than the request row. The purpose resolved when
    the call started is the one both rows carry, so a scope entered
    while the stream is being consumed cannot split them.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        for chunk in ("hel", "lo"):
            yield chunk

    agent = Agent(_wrap(tracker, FunctionModel(stream_function=stream)))

    async def drive() -> None:
        async with agent.run_stream("hello") as result:
            with purpose_scope("entered-mid-call"):
                async for _ in result.stream_text():
                    pass

    asyncio.run(drive())
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_response",
    ]
    assert events[0]["purpose"] == events[1]["purpose"]


class _Renaming(WrapperModel):
    """
    Helper class used to rename the reply after the wrapped model has
    returned it, the way a provider answering with a dated snapshot
    or a substituted upstream does. FunctionModel stamps its own name
    on the way out, so renaming inside it would not survive.
    """

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        """
        Return the wrapped model's reply under another model name.
        """
        response = await super().request(*args, **kwargs)
        response.model_name = "provider-named-something-else"
        return response


def test_both_rows_name_the_configured_model(
    tmp_log_path: Path,
) -> None:
    """
    The response row records the endpoint's model rather than the one
    the provider named in its reply, so a dated snapshot or a
    substituted upstream cannot split the pair.
    """
    tracker = _tracker(tmp_log_path)
    model = _replies(
        ModelResponse(
            parts=[TextPart("hello")],
            usage=RequestUsage(input_tokens=5, output_tokens=2),
        ),
    )
    agent = Agent(_wrap(tracker, _Renaming(model)))
    agent.run_sync("hi")
    tracker.close()

    events = _events(tmp_log_path)
    assert events[0]["model"] == events[1]["model"]
    assert "provider-named" not in events[1]["model"]


def test_a_substituted_model_is_recorded(
    tmp_log_path: Path,
) -> None:
    """
    The configured model stays on both rows, and the one the provider
    named is kept alongside it, so a substitution is still visible.
    """
    tracker = _tracker(tmp_log_path)
    model = _replies(
        ModelResponse(
            parts=[TextPart("hello")],
            usage=RequestUsage(input_tokens=5, output_tokens=2),
        ),
    )
    agent = Agent(_wrap(tracker, _Renaming(model)))
    agent.run_sync("hi")
    tracker.close()

    response = _events(tmp_log_path)[1]
    assert response["usage_details"]["response_model"] == (
        "provider-named-something-else"
    )


def test_a_scope_left_mid_call_does_not_split_the_pair(
    tmp_log_path: Path,
) -> None:
    """
    The mirror of the case above. A scope active when the call starts
    is the one both rows carry, even when the caller leaves it before
    the stream closes and the response row is written.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        for chunk in ("hel", "lo"):
            yield chunk

    model = _wrap(tracker, FunctionModel(stream_function=stream))

    async def drive() -> None:
        scope = purpose_scope("at-request")
        scope.__enter__()
        async with model.request_stream(
            [],
            None,
            ModelRequestParameters(),
        ) as streamed:
            async for _ in streamed:
                pass
            scope.__exit__(None, None, None)

    asyncio.run(drive())
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_response",
    ]
    assert events[0]["purpose"] == "at-request"
    assert events[1]["purpose"] == "at-request"


def test_an_error_row_carries_the_request_time_purpose(
    tmp_log_path: Path,
) -> None:
    """
    A failed call records the purpose the call started under, not
    whatever the caller has moved on to by the time the stream
    breaks.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        yield "partial"
        raise RuntimeError("upstream cut the connection")

    model = _wrap(tracker, FunctionModel(stream_function=stream))

    async def drive() -> None:
        moved_on = purpose_scope("caller-moved-on")
        with purpose_scope("at-request"):
            try:
                async with model.request_stream(
                    [],
                    None,
                    ModelRequestParameters(),
                ) as streamed:
                    moved_on.__enter__()
                    async for _ in streamed:
                        pass
            finally:
                moved_on.__exit__(None, None, None)

    with pytest.raises(RuntimeError):
        asyncio.run(drive())
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_error",
    ]
    assert events[1]["purpose"] == "at-request"


def test_an_unverified_provider_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A provider with no verified adapter raises the same error
    send_message raises, rather than quietly building a client that
    cannot work.
    """
    monkeypatch.setenv("PAI_KEY", "set")
    p = tmp_path / "unverified.yaml"
    p.write_text(
        "endpoints:\n"
        "  gemini-one:\n"
        "    provider: gemini\n"
        "    model: gemini-2.0-flash\n"
        "    api_key_env: PAI_KEY\n",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="deferred"):
        ledger_model(
            endpoint_name="gemini-one",
            tracker=_tracker(tmp_path / "usage.jsonl"),
            config=load_config(p),
        )


def test_both_paths_write_the_same_rows(
    tmp_log_path: Path,
) -> None:
    """
    ledger_model and record_run produce the same rows for a successful
    run, so a consumer can move between them without the ledger
    changing shape.

    model is compared separately below rather than here: the two paths
    populate it differently by design, and a stub model that echoes
    the name it was asked for would hide that.
    """
    fields = (
        "event",
        "provider",
        "purpose",
        "system_prompt_preview",
        "user_prompt_preview",
        "response_preview",
        "usage",
        "usage_details",
        "provider_response_id",
    )

    def comparable(path: Path) -> list[dict[str, Any]]:
        return [
            {k: e.get(k) for k in fields}
            for e in _events(path)
        ]

    def responses() -> tuple[ModelResponse, ModelResponse]:
        return (
            ModelResponse(
                parts=[
                    ToolCallPart("weather", {"city": "Oslo"}),
                ],
                usage=RequestUsage(
                    input_tokens=9,
                    output_tokens=4,
                ),
                model_name="qwen3-4b",
                provider_details={
                    "finish_reason": "tool_calls",
                },
            ),
            _text_reply("cold in Oslo"),
        )

    def build(model: Any) -> Agent[None, str]:
        agent: Agent[None, str] = Agent(
            model,
            instructions="You are terse.",
        )

        @agent.tool_plain
        def weather(city: str) -> str:
            return "cold"

        return agent

    wrapped_path = tmp_log_path.parent / "wrapped.jsonl"
    wrapped_tracker = _tracker(wrapped_path)
    agent = build(
        _wrap(wrapped_tracker, _replies(*responses())),
    )
    agent.run_sync("weather in Oslo?")
    wrapped_tracker.close()

    recorded_path = tmp_log_path.parent / "recorded.jsonl"
    recorded_tracker = _tracker(recorded_path)
    plain = build(_replies(*responses()))
    result = plain.run_sync("weather in Oslo?")
    recorded_tracker.record_run(
        result.all_messages(),
        provider="lmstudio",
    )
    recorded_tracker.close()

    assert comparable(wrapped_path) == comparable(
        recorded_path,
    )


def test_the_two_paths_differ_only_on_the_model_field(
    tmp_log_path: Path,
) -> None:
    """
    Against a provider that answers with a different model than it was
    asked for, ledger_model records the configured one and keeps the
    provider's under usage_details.response_model, while record_run
    records the provider's directly and keeps nothing beside it.

    This is the difference the parity test above excludes, asserted
    here so neither side can drift unnoticed.
    """
    wrapped_path = tmp_log_path.parent / "wrapped.jsonl"
    wrapped_tracker = _tracker(wrapped_path)
    agent = Agent(
        _wrap(
            wrapped_tracker,
            _Renaming(_replies(_text_reply("hello"))),
        ),
    )
    agent.run_sync("hi")
    wrapped_tracker.close()

    recorded_path = tmp_log_path.parent / "recorded.jsonl"
    recorded_tracker = _tracker(recorded_path)
    plain = Agent(_Renaming(_replies(_text_reply("hello"))))
    result = plain.run_sync("hi")
    recorded_tracker.record_run(result.all_messages())
    recorded_tracker.close()

    wrapped = _events(wrapped_path)[1]
    recorded = _events(recorded_path)[1]

    assert "provider-named" not in wrapped["model"]
    assert wrapped["usage_details"]["response_model"] == (
        "provider-named-something-else"
    )
    assert recorded["model"] == "provider-named-something-else"
    assert "response_model" not in recorded.get(
        "usage_details",
        {},
    )


def test_ledger_model_builds_from_the_endpoint(
    tmp_log_path: Path,
    config: LLMConfig,
) -> None:
    """
    The endpoint's model name and base_url reach the built model, so
    the agent talks to the endpoint send_message would have used.
    """
    model = ledger_model(
        "local-chat",
        tracker=_tracker(tmp_log_path),
        config=config,
    )
    assert model.model_name == "qwen3-4b"
    # The SDK client normalises the trailing slash.
    assert model.base_url == "http://localhost:1234/v1/"


def test_ledger_model_applies_endpoint_extra_body(
    tmp_log_path: Path,
    config: LLMConfig,
) -> None:
    """
    The endpoint's extra_body reaches the model's settings. It is deep
    copied, and the config's own dict is not shared.
    """
    model = ledger_model(
        "reasoning-off",
        tracker=_tracker(tmp_log_path),
        config=config,
    )
    settings = model.settings
    assert settings is not None
    extra_body = settings["extra_body"]
    assert extra_body == {"reasoning": {"enabled": False}}
    assert extra_body is not (
        config.endpoints["reasoning-off"].extra_body
    )


def test_ledger_model_rejects_an_unknown_endpoint(
    tmp_log_path: Path,
    config: LLMConfig,
) -> None:
    """
    A name that is not in the config raises rather than falling back to
    a default endpoint.
    """
    with pytest.raises(EndpointNotFoundError):
        ledger_model(
            "no-such-endpoint",
            tracker=_tracker(tmp_log_path),
            config=config,
        )


def test_a_suppressed_failure_leaves_an_unpaired_request(
    tmp_log_path: Path,
) -> None:
    """
    A wrapped model may suppress the caller's exception, leaving the
    stream incomplete. No shipped model does, but a user's wrapper may.

    The request is left unpaired rather than raising from inside this
    package.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        for chunk in ("hel", "lo"):
            yield chunk

    class Suppresses(WrapperModel):
        """Swallows whatever the caller raised."""

        @asynccontextmanager
        async def request_stream(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            async with self.wrapped.request_stream(
                *args,
                **kwargs,
            ) as opened:
                try:
                    yield opened
                except Exception:
                    pass

    class Boom(Exception):
        pass

    model = _wrap(
        tracker,
        Suppresses(FunctionModel(stream_function=stream)),
    )

    # Driven directly rather than through an Agent: an Agent handles
    # the caller's exception itself, so it never reaches the wrapping
    # model's yield and the suppression never comes into play.
    async def drive() -> None:
        async with model.request_stream(
            [],
            None,
            ModelRequestParameters(),
        ) as opened:
            async for _ in opened:
                pass
            raise Boom("caller blew up")

    asyncio.run(drive())
    tracker.close()

    kinds = [e["event"] for e in _events(tmp_log_path)]
    assert kinds == ["llm_request"]


def test_an_abandoned_stream_records_a_response_not_an_error(
    tmp_log_path: Path,
) -> None:
    """
    Abandoning a stream arrives as GeneratorExit. The caller did not
    fail, so it records a response, as breaking out of the loop does.
    """
    tracker = _tracker(tmp_log_path)

    async def stream(
        messages: list[Any],
        info: AgentInfo,
    ) -> Any:
        for chunk in ("hel", "lo"):
            yield chunk

    model = _wrap(tracker, FunctionModel(stream_function=stream))

    async def drive() -> None:
        opened = model.request_stream(
            [],
            None,
            ModelRequestParameters(),
        )
        await opened.__aenter__()
        # Close the generator without exiting the context manager,
        # which is what abandoning the stream does.
        await opened.gen.aclose()

    asyncio.run(drive())
    tracker.close()

    events = _events(tmp_log_path)
    assert [e["event"] for e in events] == [
        "llm_request",
        "llm_response",
    ]
