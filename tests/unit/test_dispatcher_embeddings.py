"""
Dry-run unit tests for llm_router_ledger.dispatcher.create_embeddings.

The embedding adapter is mocked so these run fully offline.
"""

from __future__ import annotations

import json

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_router_ledger import (
    UsageTracker,
    create_embeddings,
    load_config,
)
from llm_router_ledger.config import LLMConfig
from llm_router_ledger.exceptions import EndpointNotFoundError


EMBEDDING_YAML = """\
endpoints:
  embed-with-dims:
    provider: openrouter
    model: baai/bge-m3
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    embedding_dimensions: 1024
    extra_body:
      provider:
        order:
          - DeepInfra

  embed-without-dims:
    provider: openrouter
    model: baai/bge-m3
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1

  ollama-local:
    provider: ollama
    model: llama3.1
    api_key_env: OLLAMA_API_KEY
    base_url: http://localhost:11434/v1
"""


def _patch_embedding_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """
    Helper function used to mock both get_client and
    _select_embedding_adapter so create_embeddings runs entirely
    offline. Returns the fake adapter so tests can inspect
    adapter.embed.call_args.
    """
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher.get_client",
        lambda **kw: MagicMock(),
    )
    fake = MagicMock()
    fake.embed.return_value = (
        [[0.1, 0.2], [0.3, 0.4]],
        {
            "prompt_tokens": 9,
            "completion_tokens": 0,
            "total_tokens": 9,
            "cost": 4.5e-08,
            "dimensions": 2,
            "embedding_count": 2,
            "is_byok": False,
            "upstream_provider": "DeepInfra",
        },
        "gen-emb-abc123",
    )
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher"
        "._select_embedding_adapter",
        lambda provider: fake,
    )
    return fake


@pytest.fixture
def embedding_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LLMConfig:
    """
    Return a config with an embedding endpoint that declares
    embedding_dimensions, one that does not, and a chat endpoint on a
    provider with no verified embedding adapter.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    path = tmp_path / "llm_endpoints.yaml"
    path.write_text(EMBEDDING_YAML, encoding="utf-8")
    return load_config(path)


def test_create_embeddings_forwards_expected_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
) -> None:
    """
    The endpoint's embedding_dimensions reaches the adapter, which is
    what arms the width guard.
    """
    fake = _patch_embedding_adapter(monkeypatch)
    create_embeddings(
        endpoint_name="embed-with-dims",
        texts=["a", "b"],
        config=embedding_config,
    )
    kwargs = fake.embed.call_args.kwargs
    assert kwargs["expected_dimensions"] == 1024
    assert kwargs["model"] == "baai/bge-m3"
    assert kwargs["texts"] == ["a", "b"]


def test_create_embeddings_layers_extra_body(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
) -> None:
    """
    extra_body follows the same one-layer rule as send_message: a
    call-level value replaces the endpoint's outright rather than
    merging into it.
    """
    fake = _patch_embedding_adapter(monkeypatch)
    create_embeddings(
        endpoint_name="embed-with-dims",
        texts=["a"],
        config=embedding_config,
    )
    assert fake.embed.call_args.kwargs["extra_body"] == {
        "provider": {"order": ["DeepInfra"]},
    }

    create_embeddings(
        endpoint_name="embed-with-dims",
        texts=["a"],
        config=embedding_config,
        extra_body={"provider": {"sort": "latency"}},
    )
    sent = fake.embed.call_args.kwargs["extra_body"]
    assert sent == {"provider": {"sort": "latency"}}
    assert "order" not in sent["provider"]


def test_create_embeddings_logs_paired_embedding_events(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
    tmp_log_path: Path,
) -> None:
    """
    Both ledger entries are stamped modality "embedding" so a
    reconciler can separate embedding spend from text spend, and the
    response entry records no response text, since an embedding
    response carries none.
    """
    _patch_embedding_adapter(monkeypatch)
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
    )
    create_embeddings(
        endpoint_name="embed-with-dims",
        texts=["first", "second"],
        config=embedding_config,
        tracker=tracker,
    )
    tracker.close()
    entries = [
        json.loads(line)
        for line in tmp_log_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert [e["event"] for e in entries] == [
        "llm_request",
        "llm_response",
    ]
    assert [e["modality"] for e in entries] == [
        "embedding",
        "embedding",
    ]
    assert entries[0]["user_prompt_length"] == len(
        "first\nsecond",
    )
    assert entries[1]["response_length"] == 0
    assert (
        entries[1]["generation_id"] == "gen-emb-abc123"
    )


def test_create_embeddings_no_tracker_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
    tmp_log_path: Path,
) -> None:
    """
    Without a tracker, no log file is created.
    """
    _patch_embedding_adapter(monkeypatch)
    create_embeddings(
        endpoint_name="embed-with-dims",
        texts=["a"],
        config=embedding_config,
    )
    assert not tmp_log_path.exists()


def test_create_embeddings_omits_expected_dimensions_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
) -> None:
    """
    An endpoint that declares no embedding_dimensions passes None, so
    the adapter accepts whatever width comes back.
    """
    fake = _patch_embedding_adapter(monkeypatch)
    create_embeddings(
        endpoint_name="embed-without-dims",
        texts=["a"],
        config=embedding_config,
    )
    kwargs = fake.embed.call_args.kwargs
    assert kwargs["expected_dimensions"] is None


def test_create_embeddings_returns_adapter_tuple(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
) -> None:
    """
    The caller gets the adapter's tuple unchanged, including the
    embedding extras in usage. Only the ledger splits them out.
    """
    _patch_embedding_adapter(monkeypatch)
    vectors, usage, generation_id = create_embeddings(
        endpoint_name="embed-with-dims",
        texts=["a", "b"],
        config=embedding_config,
    )
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert usage["total_tokens"] == 9
    assert usage["dimensions"] == 2
    assert usage["cost"] == 4.5e-08
    assert generation_id == "gen-emb-abc123"


def test_create_embeddings_splits_usage_into_details(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
    tmp_log_path: Path,
) -> None:
    """
    The ledger's usage block holds only the three normalised token
    keys; everything else the provider reported lands in usage_details,
    so the fixed token shape stays the same across modalities.
    """
    _patch_embedding_adapter(monkeypatch)
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
    )
    create_embeddings(
        endpoint_name="embed-with-dims",
        texts=["a", "b"],
        config=embedding_config,
        tracker=tracker,
    )
    tracker.close()
    response = json.loads(
        tmp_log_path.read_text(
            encoding="utf-8",
        ).splitlines()[1]
    )
    assert response["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 0,
        "total_tokens": 9,
    }
    assert response["usage_details"] == {
        "cost": 4.5e-08,
        "dimensions": 2,
        "embedding_count": 2,
        "is_byok": False,
        "upstream_provider": "DeepInfra",
    }


def test_create_embeddings_unknown_endpoint_raises(
    embedding_config: LLMConfig,
) -> None:
    """
    Unknown endpoint name raises EndpointNotFoundError, matching
    send_message.
    """
    with pytest.raises(EndpointNotFoundError):
        create_embeddings(
            endpoint_name="nope",
            texts=["a"],
            config=embedding_config,
        )


def test_create_embeddings_unverified_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
    embedding_config: LLMConfig,
) -> None:
    """
    A provider whose chat adapter is verified is still refused for
    embeddings until its embeddings endpoint has been exercised
    end-to-end. The two capabilities are gated independently.
    """
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher.get_client",
        lambda **kw: MagicMock(),
    )
    with pytest.raises(NotImplementedError) as excinfo:
        create_embeddings(
            endpoint_name="ollama-local",
            texts=["a"],
            config=embedding_config,
        )
    assert "ollama" in str(excinfo.value)
