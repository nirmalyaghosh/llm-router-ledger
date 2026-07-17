"""
Dry-run unit tests for endpoint-level extra_body resolution.

The OpenAI SDK call is mocked so these run fully offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_router_ledger import (
    load_config,
    send_message,
)
from llm_router_ledger.config import LLMConfig


EXTRA_BODY_YAML = """\
endpoints:
  ep-with-extra:
    provider: openrouter
    model: openai/gpt-4.1-nano
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    extra_body:
      reasoning:
        effort: low
        enabled: false

  ep-without-extra:
    provider: openrouter
    model: openai/gpt-4.1-nano
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
"""


def _patch_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """
    Helper function used to mock both get_client and _select_adapter so
    send_message runs entirely offline. Returns the fake adapter so tests
    can inspect adapter.send.call_args.
    """
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher.get_client",
        lambda **kw: MagicMock(),
    )
    fake = MagicMock()
    fake.send.return_value = (
        "hello world",
        {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
        "gen-abc123",
    )
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher._select_adapter",
        lambda provider: fake,
    )
    return fake


@pytest.fixture
def extra_body_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LLMConfig:
    """
    Return a config with one endpoint carrying extra_body and one
    without, so tests can cover both layers.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    path = tmp_path / "llm_endpoints.yaml"
    path.write_text(EXTRA_BODY_YAML, encoding="utf-8")
    return load_config(path)


def test_call_extra_body_replaces_endpoint_value(
    monkeypatch: pytest.MonkeyPatch,
    extra_body_config: LLMConfig,
) -> None:
    """
    A call-level extra_body replaces the endpoint's outright. The layers
    do not merge, so the endpoint's sibling keys are absent rather than
    blended in.
    """
    fake = _patch_adapter(monkeypatch)
    send_message(
        endpoint_name="ep-with-extra",
        user="u",
        config=extra_body_config,
        extra_body={"provider": {"sort": "latency"}},
    )
    sent = fake.send.call_args.kwargs["extra_body"]
    assert sent == {"provider": {"sort": "latency"}}
    assert "reasoning" not in sent


def test_call_extra_body_used_when_endpoint_omits_it(
    monkeypatch: pytest.MonkeyPatch,
    extra_body_config: LLMConfig,
) -> None:
    """
    An endpoint without extra_body still honours a call-level one.
    """
    fake = _patch_adapter(monkeypatch)
    send_message(
        endpoint_name="ep-without-extra",
        user="u",
        config=extra_body_config,
        extra_body={"provider": {"sort": "latency"}},
    )
    sent = fake.send.call_args.kwargs["extra_body"]
    assert sent == {"provider": {"sort": "latency"}}


def test_endpoint_extra_body_is_copied_not_shared(
    monkeypatch: pytest.MonkeyPatch,
    extra_body_config: LLMConfig,
) -> None:
    """
    The dict handed to the adapter is a copy. EndpointConfig is not
    frozen, so a mutation downstream must not leak into the config and
    contaminate every later call on that endpoint.
    """
    fake = _patch_adapter(monkeypatch)
    send_message(
        endpoint_name="ep-with-extra",
        user="u",
        config=extra_body_config,
    )
    first = fake.send.call_args.kwargs["extra_body"]
    first["reasoning"]["enabled"] = True

    send_message(
        endpoint_name="ep-with-extra",
        user="u",
        config=extra_body_config,
    )
    second = fake.send.call_args.kwargs["extra_body"]
    assert second["reasoning"]["enabled"] is False

    endpoint = extra_body_config.endpoints["ep-with-extra"]
    assert endpoint.extra_body is not None
    assert endpoint.extra_body["reasoning"]["enabled"] is False


def test_endpoint_extra_body_used_when_call_omits_it(
    monkeypatch: pytest.MonkeyPatch,
    extra_body_config: LLMConfig,
) -> None:
    """
    With no call-level extra_body, the endpoint's value reaches the
    adapter verbatim.
    """
    fake = _patch_adapter(monkeypatch)
    send_message(
        endpoint_name="ep-with-extra",
        user="u",
        config=extra_body_config,
    )
    sent = fake.send.call_args.kwargs["extra_body"]
    assert sent == {
        "reasoning": {"effort": "low", "enabled": False},
    }


def test_no_extra_body_anywhere_passes_none(
    monkeypatch: pytest.MonkeyPatch,
    extra_body_config: LLMConfig,
) -> None:
    """
    An endpoint without extra_body, called without one, sends None rather
    than an empty dict, preserving the pre-0.1.3 behaviour.
    """
    fake = _patch_adapter(monkeypatch)
    send_message(
        endpoint_name="ep-without-extra",
        user="u",
        config=extra_body_config,
    )
    assert fake.send.call_args.kwargs["extra_body"] is None
