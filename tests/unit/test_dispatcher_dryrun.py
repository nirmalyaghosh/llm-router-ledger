"""
Dry-run unit tests for llm_router_ledger.dispatcher.send_message.

The OpenAI SDK call is mocked so these run fully offline.
"""

from __future__ import annotations

import json

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_router_ledger import (
    UsageTracker,
    load_config,
    send_message,
)
from llm_router_ledger.exceptions import EndpointNotFoundError


def _patch_adapter(
    monkeypatch: pytest.MonkeyPatch,
    response_text: str = "hello world",
    generation_id: str = "gen-abc123",
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
        response_text,
        {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
        generation_id,
    )
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher._select_adapter",
        lambda provider: fake,
    )
    return fake


def test_send_message_anthropic_raises_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Calling send_message against an anthropic endpoint raises
    NotImplementedError until the native adapter ships in a later minor
    release.
    """
    yaml_text = (
        "endpoints:\n"
        "  anthropic-claude:\n"
        "    provider: anthropic\n"
        "    model: claude-sonnet\n"
        "    api_key_env: ANTHROPIC_API_KEY\n"
    )
    p = tmp_path / "with_anthropic.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    config = load_config(p)
    with pytest.raises(NotImplementedError):
        send_message(
            endpoint_name="anthropic-claude",
            system="s",
            user="u",
            config=config,
        )


def test_send_message_azure_dispatches_to_openai_compat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Calling send_message against an azure endpoint dispatches through
    the gate and reaches the OpenAI-compatible adapter.
    """
    yaml_text = (
        "endpoints:\n"
        "  azure-test:\n"
        "    provider: azure\n"
        "    model: gpt-4.1-nano\n"
        "    api_key_env: AZURE_OPENAI_API_KEY\n"
        "    base_url: https://example.openai.azure.com/openai/v1/\n"
    )
    p = tmp_path / "with_azure.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x")
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher.get_client",
        lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "llm_router_ledger.providers.openai_compat."
        "OpenAICompatAdapter.send",
        lambda self, **kw: (
            "ok",
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "gen-test",
        ),
    )
    config = load_config(p)
    text, usage, gen = send_message(
        endpoint_name="azure-test",
        system="s",
        user="u",
        config=config,
    )
    assert text == "ok"
    assert gen == "gen-test"


def test_send_message_invokes_adapter_with_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    The adapter receives the model + prompt + optional kwargs from
    send_message.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    fake = _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    send_message(
        endpoint_name="ollama-local",
        system="sys",
        user="usr",
        config=config,
        max_tokens=100,
        temperature=0.2,
    )
    kwargs = fake.send.call_args.kwargs
    assert kwargs["model"] == "llama3.1"
    assert kwargs["system"] == "sys"
    assert kwargs["user"] == "usr"
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.2


def test_send_message_forwards_response_format(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    response_format reaches the adapter so JSON mode and structured
    outputs work.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    fake = _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    send_message(
        endpoint_name="ollama-local",
        system="s",
        user="u",
        config=config,
        response_format={"type": "json_object"},
    )
    kwargs = fake.send.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_send_message_system_optional(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    system=None reaches the adapter unchanged so user-only calls (e.g.
    JSON-mode prompts) work.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    fake = _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    send_message(endpoint_name="ollama-local", user="u", config=config)
    kwargs = fake.send.call_args.kwargs
    assert kwargs["system"] is None


def test_send_message_forwards_user_id_and_extra_body(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    user_id and extra_body kwargs reach the adapter so OpenRouter
    run-tagging and provider routing work.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    fake = _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    send_message(
        endpoint_name="ollama-local",
        system="sys",
        user="usr",
        config=config,
        user_id="run-2026-05-22",
        extra_body={"provider": {"sort": "latency"}},
    )
    kwargs = fake.send.call_args.kwargs
    assert kwargs["user_id"] == "run-2026-05-22"
    assert kwargs["extra_body"] == {"provider": {"sort": "latency"}}


def test_send_message_omits_user_id_and_extra_body_by_default(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    When user_id and extra_body are not passed, they reach the adapter
    as None so the SDK receives no spurious kwargs.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    fake = _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    send_message(
        endpoint_name="ollama-local",
        system="sys",
        user="usr",
        config=config,
    )
    kwargs = fake.send.call_args.kwargs
    assert kwargs["user_id"] is None
    assert kwargs["extra_body"] is None


def test_send_message_logs_paired_events(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
    tmp_log_path: Path,
) -> None:
    """
    With a tracker, both llm_request and llm_response events land in
    the log file in that order.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    send_message(
        endpoint_name="ollama-local",
        system="sys",
        user="usr",
        config=config,
        tracker=tracker,
    )
    tracker.close()
    entries = [
        json.loads(line)
        for line in tmp_log_path.read_text(encoding="utf-8").splitlines()
    ]
    events = [e["event"] for e in entries]
    assert events == ["llm_request", "llm_response"]


def test_send_message_logs_provider_from_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
    tmp_log_path: Path,
) -> None:
    """
    With a tracker, the EndpointConfig.provider value reaches both the
    llm_request and llm_response entries in the JSONL log.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    send_message(
        endpoint_name="ollama-local",
        system="sys",
        user="usr",
        config=config,
        tracker=tracker,
    )
    tracker.close()
    entries = [
        json.loads(line)
        for line in tmp_log_path.read_text(encoding="utf-8").splitlines()
    ]
    providers = [e["provider"] for e in entries]
    assert providers == [
        "local-openai-compat",
        "local-openai-compat",
    ]


def test_send_message_no_tracker_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
    tmp_log_path: Path,
) -> None:
    """
    Without a tracker, no log file is created.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    text, _, _ = send_message(
        endpoint_name="ollama-local",
        system="sys",
        user="usr",
        config=config,
    )
    assert text == "hello world"
    assert not tmp_log_path.exists()


def test_send_message_returns_adapter_tuple(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    Return value is the adapter's tuple unchanged.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    _patch_adapter(monkeypatch)
    config = load_config(sample_yaml_file)
    text, usage, gen = send_message(
        endpoint_name="ollama-local",
        system="s",
        user="u",
        config=config,
    )
    assert text == "hello world"
    assert usage["total_tokens"] == 12
    assert gen == "gen-abc123"


def test_send_message_unknown_endpoint_raises(
    sample_yaml_file: Path,
) -> None:
    """
    Unknown endpoint name raises EndpointNotFoundError.
    """
    config = load_config(sample_yaml_file)
    with pytest.raises(EndpointNotFoundError):
        send_message(
            endpoint_name="nope",
            system="s",
            user="u",
            config=config,
        )
