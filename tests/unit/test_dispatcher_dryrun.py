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


def test_send_message_anthropic_dispatches_to_native_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Calling send_message against an anthropic endpoint dispatches
    through the gate and reaches the native AnthropicAdapter (not the
    OpenAI-compatible adapter), since Anthropic's Messages API has a
    different request/response shape.
    """
    yaml_text = (
        "endpoints:\n"
        "  anthropic-test:\n"
        "    provider: anthropic\n"
        "    model: claude-haiku-4-5\n"
        "    api_key_env: ANTHROPIC_API_KEY\n"
    )
    p = tmp_path / "with_anthropic.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher.get_client",
        lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "llm_router_ledger.providers.anthropic_native."
        "AnthropicAdapter.send",
        lambda self, **kw: (
            "ok",
            {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            "msg_test",
        ),
    )
    config = load_config(p)
    text, usage, gen = send_message(
        endpoint_name="anthropic-test",
        system="s",
        user="u",
        config=config,
    )
    assert text == "ok"
    assert gen == "msg_test"


@pytest.mark.parametrize(
    "provider,model,api_key_env,base_url,mock_gen_id",
    [
        # Azure OpenAI v1 endpoint is OpenAI-compatible; deployment
        # name carried in model field.
        (
            "azure",
            "gpt-4.1-nano",
            "AZURE_OPENAI_API_KEY",
            "https://example.openai.azure.com/openai/v1/",
            "chatcmpl-test",
        ),
        # DeepSeek API is OpenAI-compatible.
        (
            "deepseek",
            "deepseek-chat",
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com/v1",
            "chatcmpl-test",
        ),
        # MiniMax text API is OpenAI-compatible.
        (
            "minimax",
            "MiniMax-M2.5",
            "MINIMAX_API_KEY",
            "https://api.minimax.io/v1",
            "chatcmpl-test",
        ),
        # Ollama speaks the OpenAI chat API locally; uses gen-* style id.
        (
            "ollama",
            "llama3.1",
            "OLLAMA_API_KEY",
            "http://localhost:11434/v1",
            "gen-test",
        ),
        # OpenAI direct: no base_url override needed (SDK default).
        (
            "openai",
            "gpt-4.1-nano",
            "OPENAI_API_KEY",
            None,
            "chatcmpl-test",
        ),
        # Qwen via Alibaba DashScope is OpenAI-compatible.
        (
            "qwen",
            "qwen-plus",
            "QWEN_API_KEY",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "chatcmpl-test",
        ),
        # Z.AI / Zhipu GLM family is OpenAI-compatible.
        (
            "zhipu",
            "glm-4.7-flash",
            "ZHIPU_API_KEY",
            "https://api.z.ai/api/paas/v4/",
            "chatcmpl-test",
        ),
    ],
    ids=["azure", "deepseek", "minimax", "ollama", "openai", "qwen", "zhipu"],
)
def test_send_message_dispatches_to_openai_compat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    model: str,
    api_key_env: str,
    base_url: str | None,
    mock_gen_id: str,
) -> None:
    """
    Every verified OpenAI-compatible provider dispatches through the
    gate and reaches the shared OpenAICompatAdapter, returning the
    adapter's response unchanged. The gen_id varies per provider to
    cover both the gen-* (OpenRouter convention) and provider-native
    (chatcmpl-*) routing paths exercised by UsageTracker.
    """
    base_url_line = (
        f"    base_url: {base_url}\n"
        if base_url
        else ""
    )
    yaml_text = (
        "endpoints:\n"
        f"  {provider}-test:\n"
        f"    provider: {provider}\n"
        f"    model: {model}\n"
        f"    api_key_env: {api_key_env}\n"
        f"{base_url_line}"
    )
    p = tmp_path / f"with_{provider}.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv(api_key_env, "x")
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
            mock_gen_id,
        ),
    )
    config = load_config(p)
    text, usage, gen = send_message(
        endpoint_name=f"{provider}-test",
        system="s",
        user="u",
        config=config,
    )
    assert text == "ok"
    assert gen == mock_gen_id


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
