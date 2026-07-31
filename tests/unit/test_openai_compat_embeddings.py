"""
Unit tests for the OpenAI-compatible embedding adapter.

The SDK client is mocked so these run fully offline. Response objects are
built from SimpleNamespace rather than MagicMock so that a missing
attribute is genuinely missing; MagicMock would auto-create cost and
is_byok and defeat the tests that assert they are omitted.

The shapes asserted here were captured from live OpenRouter embedding
calls on 2026-07-29.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_router_ledger.exceptions import ProviderError
from llm_router_ledger.providers.openai_compat import (
    OpenAICompatEmbeddingAdapter,
)


def _fake_client(
    vectors: list[list[float]] | None = None,
    usage: SimpleNamespace | None = None,
    response_id: str = "gen-emb-abc",
    provider: str | None = "DeepInfra",
    indices: list[int] | None = None,
) -> MagicMock:
    """
    Helper function used to build a MagicMock SDK client whose
    embeddings.create returns a minimal OpenRouter-shaped response.
    """
    if vectors is None:
        vectors = [[0.1, 0.2, 0.3]]
    if indices is None:
        indices = list(range(len(vectors)))
    if usage is None:
        usage = SimpleNamespace(
            prompt_tokens=9,
            total_tokens=9,
        )
    client = MagicMock()
    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                embedding=vector,
                index=index,
                object="embedding",
            )
            for vector, index in zip(vectors, indices)
        ],
        id=response_id,
        model="BAAI/bge-m3",
        object="list",
        provider=provider,
        usage=usage,
    )
    client.embeddings.create.return_value = response
    return client


def test_adapter_accepts_matching_dimensions() -> None:
    """
    The guard is silent when the declared and returned widths agree.
    """
    client = _fake_client(vectors=[[0.1, 0.2, 0.3]])
    vectors, usage, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a"],
        expected_dimensions=3,
    )
    assert vectors == [[0.1, 0.2, 0.3]]
    assert usage["dimensions"] == 3


def test_adapter_always_sends_float_encoding_format() -> None:
    """
    encoding_format is pinned to "float" on every call. The SDK default
    is base64, which OpenRouter's nvidia/nemotron-3-embed-1b rejects with
    a 400, so leaving it unset breaks that endpoint.
    """
    client = _fake_client()
    OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["hello"],
    )
    call_kwargs = client.embeddings.create.call_args.kwargs
    assert call_kwargs["encoding_format"] == "float"


def test_adapter_falls_back_to_prompt_tokens_for_total() -> None:
    """
    A provider that reports only prompt_tokens still yields a
    well-formed total, so the ledger never records a zero total against
    a non-zero prompt count.
    """
    client = _fake_client(
        usage=SimpleNamespace(prompt_tokens=7),
    )
    _, usage, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a"],
    )
    assert usage["total_tokens"] == 7


def test_adapter_forwards_extra_body() -> None:
    """
    extra_body passes through verbatim so OpenRouter provider routing
    hints reach the embeddings endpoint.
    """
    client = _fake_client()
    OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["hello"],
        extra_body={"provider": {"order": ["DeepInfra"]}},
    )
    call_kwargs = client.embeddings.create.call_args.kwargs
    assert call_kwargs["extra_body"] == {
        "provider": {"order": ["DeepInfra"]},
    }


def test_adapter_handles_missing_usage() -> None:
    """
    usage=None yields zeroed token counts rather than raising, matching
    how the chat adapter treats a provider that omits usage.
    """
    client = _fake_client(usage=None)
    client.embeddings.create.return_value.usage = None
    _, usage, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a"],
    )
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert usage["embedding_count"] == 1


def test_adapter_normalises_usage() -> None:
    """
    completion_tokens is fixed at 0 (embedding endpoints bill input
    only), and dimensions plus embedding_count are derived from the
    returned vectors.
    """
    client = _fake_client(
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        usage=SimpleNamespace(
            prompt_tokens=10,
            total_tokens=10,
        ),
    )
    _, usage, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a", "b"],
    )
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 10
    assert usage["embedding_count"] == 2
    assert usage["dimensions"] == 2


def test_adapter_omits_cost_fields_when_absent() -> None:
    """
    A provider that reports no cost or upstream leaves those keys out
    entirely, so the ledger never records a fabricated zero cost.
    """
    client = _fake_client(
        usage=SimpleNamespace(
            prompt_tokens=9,
            total_tokens=9,
        ),
        provider=None,
    )
    _, usage, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a"],
    )
    assert "cost" not in usage
    assert "is_byok" not in usage
    assert "upstream_provider" not in usage


def test_adapter_omits_timeout_and_extra_body_when_none() -> None:
    """
    Unset optionals are left off the SDK call entirely rather than sent
    as None.
    """
    client = _fake_client()
    OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["hello"],
    )
    call_kwargs = client.embeddings.create.call_args.kwargs
    assert "timeout" not in call_kwargs
    assert "extra_body" not in call_kwargs


def test_adapter_orders_vectors_by_index() -> None:
    """
    Vectors come back ordered to match the input texts even when the
    provider returns data out of order. The API's index field exists
    because arrival order is not promised.
    """
    client = _fake_client(
        vectors=[[3.0], [1.0], [2.0]],
        indices=[2, 0, 1],
    )
    vectors, _, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a", "b", "c"],
    )
    assert vectors == [[1.0], [2.0], [3.0]]


def test_adapter_raises_on_dimension_mismatch() -> None:
    """
    A returned width other than the endpoint's declared
    embedding_dimensions raises rather than handing back vectors that
    would corrupt a fixed-width index. OpenRouter can move an endpoint
    between upstreams, so this is a live failure mode, not a
    hypothetical one.
    """
    client = _fake_client(vectors=[[0.1, 0.2, 0.3]])
    with pytest.raises(ProviderError) as excinfo:
        OpenAICompatEmbeddingAdapter().embed(
            client=client,
            model="baai/bge-m3",
            texts=["a"],
            expected_dimensions=1024,
        )
    message = str(excinfo.value)
    assert "1024" in message
    assert "3" in message
    assert "baai/bge-m3" in message


def test_adapter_reports_cost_and_upstream_provider() -> None:
    """
    OpenRouter returns the actual USD charge and the upstream that
    served the call. Both ride in usage_dict for the ledger, since
    routing can move between calls and change what an endpoint costs.
    """
    client = _fake_client(
        usage=SimpleNamespace(
            prompt_tokens=9,
            total_tokens=9,
            cost=4.5e-08,
            is_byok=False,
        ),
        provider="DeepInfra",
    )
    _, usage, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a"],
    )
    assert usage["cost"] == 4.5e-08
    assert usage["is_byok"] is False
    assert usage["upstream_provider"] == "DeepInfra"


def test_adapter_returns_generation_id() -> None:
    """
    The response id is returned as-is. OpenRouter embedding ids carry a
    "gen-emb-" prefix, which the tracker's existing "gen-" test routes
    to generation_id for CSV reconciliation.
    """
    client = _fake_client(
        response_id="gen-emb-1785319849-F3HAmwoR",
    )
    _, _, generation_id = (
        OpenAICompatEmbeddingAdapter().embed(
            client=client,
            model="m",
            texts=["a"],
        )
    )
    assert generation_id == "gen-emb-1785319849-F3HAmwoR"
    assert generation_id.startswith("gen-")


def test_adapter_skips_guard_when_dimensions_unset() -> None:
    """
    An endpoint that declares no embedding_dimensions accepts whatever
    width the provider returns, so chat-era configs keep working.
    """
    client = _fake_client(vectors=[[0.1, 0.2, 0.3]])
    vectors, _, _ = OpenAICompatEmbeddingAdapter().embed(
        client=client,
        model="m",
        texts=["a"],
    )
    assert vectors == [[0.1, 0.2, 0.3]]
