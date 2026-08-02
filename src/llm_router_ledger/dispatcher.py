"""
Public send_message and create_embeddings entry points.

Each resolves an endpoint, gets the SDK client, picks the adapter for
that capability, calls it, optionally appends paired llm_request and
llm_response events via a UsageTracker, and returns a result object:
ChatResult for send_message, EmbeddingResult for create_embeddings.
"""

from __future__ import annotations

import copy

from typing import Any

from llm_router_ledger.client_factory import (
    get_client,
    get_model_name,
)
from llm_router_ledger.config import (
    EndpointConfig,
    LLMConfig,
    load_config,
)
from llm_router_ledger.exceptions import (
    EndpointNotFoundError,
)
from llm_router_ledger.providers._base import (
    EmbeddingAdapter,
    ProviderAdapter,
)
from llm_router_ledger.providers.anthropic_native import (
    AnthropicAdapter,
)
from llm_router_ledger.providers.openai_compat import (
    OpenAICompatAdapter,
    OpenAICompatEmbeddingAdapter,
)
from llm_router_ledger.results import (
    ChatResult,
    EmbeddingResult,
)
from llm_router_ledger.usage_tracker import UsageTracker


_TOKEN_USAGE_KEYS = frozenset({
    "completion_tokens",
    "prompt_tokens",
    "total_tokens",
})

_VERIFIED_EMBEDDING_PROVIDERS = frozenset({
    "ollama",
    "openrouter",
})

_VERIFIED_PROVIDERS = frozenset({
    "anthropic",
    "azure",
    "deepseek",
    "local-openai-compat",
    "minimax",
    "ollama",
    "openai",
    "openrouter",
    "qwen",
    "zhipu",
})


def _resolve_endpoint(
    *,
    config: LLMConfig | None,
    endpoint_name: str,
) -> tuple[LLMConfig, EndpointConfig]:
    """
    Helper function used to load the default config when the caller
    passed none, and look up a named endpoint in it.

    Returns the config as well as the endpoint, because callers need
    the config itself for get_client and get_model_name, not just the
    endpoint it resolved to.

    Raises EndpointNotFoundError when the name is not in the config.
    """
    if config is None:
        config = load_config()
    if endpoint_name not in config.endpoints:
        raise EndpointNotFoundError(
            f"Endpoint '{endpoint_name}' not"
            f" found in config"
        )
    return config, config.endpoints[endpoint_name]


def _resolve_extra_body(
    *,
    call_value: dict[str, Any] | None,
    endpoint_value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Helper function used to pick the extra_body for a single call. The
    endpoint value acts as a default: a call-level extra_body replaces it
    outright rather than merging into it, so the effective value is
    always exactly one layer and never a blend of the two.

    The result is deep-copied because EndpointConfig is not frozen; the
    config's own dict must never be handed to an adapter, or a mutation
    downstream would persist onto every later call on that endpoint.
    """
    chosen = (
        call_value
        if call_value is not None
        else endpoint_value
    )
    if chosen is None:
        return None
    return copy.deepcopy(chosen)


def _select_adapter(provider: str) -> ProviderAdapter:
    """
    Helper function used to pick the provider adapter for a given provider
    name. Raises NotImplementedError for providers whose adapter has not
    been verified end-to-end in this release (anything outside the
    _VERIFIED_PROVIDERS set above).
    """
    if provider not in _VERIFIED_PROVIDERS:
        verified = ", ".join(sorted(_VERIFIED_PROVIDERS))
        raise NotImplementedError(
            f"The '{provider}' adapter is deferred to a later minor"
            f" release. Verified providers in this release: {verified}."
            f" Use OpenRouter as a workaround to reach most model"
            f" families."
        )
    if provider == "anthropic":
        return AnthropicAdapter()
    return OpenAICompatAdapter()


def _select_embedding_adapter(provider: str) -> EmbeddingAdapter:
    """
    Helper function used to pick the embedding adapter for a given
    provider name.

    Embedding verification is tracked separately from text, in
    _VERIFIED_EMBEDDING_PROVIDERS, because a working chat adapter says
    nothing about embeddings: most providers in _VERIFIED_PROVIDERS
    serve no embedding models at all, and the rest were not exercised
    end-to-end in this release.

    Note that "local-openai-compat" is deliberately absent even though
    "ollama" is present. Ollama's embeddings endpoint was verified
    directly; LM Studio, llama.cpp and vLLM share that provider name
    and were not.
    """
    if provider not in _VERIFIED_EMBEDDING_PROVIDERS:
        verified = ", ".join(
            sorted(_VERIFIED_EMBEDDING_PROVIDERS),
        )
        raise NotImplementedError(
            f"Embeddings are not available for the '{provider}'"
            f" provider. Verified embedding providers in this"
            f" release: {verified}. Point the endpoint at OpenRouter"
            f" for hosted embedding models, or at Ollama to run one"
            f" locally."
        )
    return OpenAICompatEmbeddingAdapter()


def _split_embedding_usage(
    usage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Helper function used to divide an embedding adapter's usage dict into
    the three normalised token keys the ledger's usage block holds and
    the remainder, which belongs in usage_details (dimensions,
    embedding_count, cost, is_byok, upstream_provider).

    The split is by token key rather than by a list of known extras, so a
    value a provider starts returning later lands in usage_details on its
    own instead of being silently dropped.
    """
    tokens = {
        key: value
        for key, value in usage.items()
        if key in _TOKEN_USAGE_KEYS
    }
    details = {
        key: value
        for key, value in usage.items()
        if key not in _TOKEN_USAGE_KEYS
    }
    return tokens, details


def create_embeddings(
    *,
    endpoint_name: str,
    texts: list[str],
    config: LLMConfig | None = None,
    tracker: UsageTracker | None = None,
    purpose: str = "",
    metadata: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
    extra_body: dict[str, Any] | None = None,
) -> EmbeddingResult:
    """
    Embed texts on the named endpoint and return an EmbeddingResult.

    result.vectors is ordered to match texts, one vector per input.

    result.usage carries the normalised prompt_tokens, completion_tokens,
    and total_tokens keys plus whatever the provider reported about the
    call: dimensions and embedding_count always, and cost, is_byok, and
    upstream_provider when available. The caller gets all of it; the
    ledger splits it, keeping the token keys in the usage block and the
    rest under usage_details.

    When tracker is provided, paired llm_request and llm_response events
    are appended to its JSONL log, both stamped modality "embedding" so
    a reconciler can separate embedding spend from text spend. When
    tracker is None, no logging happens.

    The endpoint's embedding_dimensions, when set, is enforced: the
    adapter raises ProviderError if the provider returns vectors of a
    different width. extra_body works exactly as it does for
    send_message, including the endpoint-level default that a call-level
    value replaces outright.

    Raises EndpointNotFoundError if the endpoint name is missing, and
    NotImplementedError if the endpoint's provider has no verified
    embedding adapter.
    """
    config, ep = _resolve_endpoint(
        config=config,
        endpoint_name=endpoint_name,
    )
    adapter = _select_embedding_adapter(ep.provider)
    effective_extra_body = _resolve_extra_body(
        call_value=extra_body,
        endpoint_value=ep.extra_body,
    )
    model = get_model_name(
        endpoint_name=endpoint_name,
        config=config,
    )
    client = get_client(
        endpoint_name=endpoint_name,
        config=config,
    )

    request_id = ""
    if tracker is not None:
        # The inputs are joined for the preview and the character count
        # only; the ledger stores neither the full text nor the vectors.
        request_id = tracker.log_request(
            model=model,
            system_prompt="",
            user_prompt="\n".join(texts),
            purpose=purpose,
            provider=ep.provider,
            modality="embedding",
            metadata=metadata,
        )

    vectors, usage, generation_id = adapter.embed(
        client=client,
        model=model,
        texts=texts,
        expected_dimensions=ep.embedding_dimensions,
        timeout_seconds=timeout_seconds,
        extra_body=effective_extra_body,
    )

    if tracker is not None:
        token_usage, usage_details = (
            _split_embedding_usage(usage)
        )
        # response_text is empty because an embedding response carries no
        # text: recording a stand-in would put a fabricated
        # response_length in the ledger. The shape of what came back is
        # in usage_details as embedding_count and dimensions.
        tracker.log_response(
            request_id=request_id,
            model=model,
            response_text="",
            usage=token_usage,
            generation_id=generation_id,
            purpose=purpose,
            provider=ep.provider,
            modality="embedding",
            usage_details=usage_details,
            metadata=metadata,
        )

    return EmbeddingResult(
        vectors=vectors,
        usage=usage,
        generation_id=generation_id,
    )


def send_message(
    *,
    endpoint_name: str,
    user: str,
    system: str | None = None,
    config: LLMConfig | None = None,
    tracker: UsageTracker | None = None,
    purpose: str = "",
    metadata: dict[str, Any] | None = None,
    max_tokens: int = 4096,
    temperature: float | None = None,
    timeout_seconds: float | None = None,
    user_id: str | None = None,
    extra_body: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
) -> ChatResult:
    """
    Send a system + user message to the named endpoint and return a
    ChatResult.

    When tracker is provided, paired llm_request and llm_response events
    are appended to its JSONL log. When tracker is None, no logging
    happens.

    system is optional; pass None for user-only calls (common with
    JSON-mode prompts that embed all instructions in the user message).

    user_id is forwarded as the SDK's "user" field (OpenRouter run
    tagging, OpenAI end-user identifier). extra_body is a vendor-specific
    passthrough dict, e.g. {"provider": {"sort": "latency"}} for
    OpenRouter provider routing. response_format requests structured
    output, e.g. {"type": "json_object"} for OpenAI JSON mode.

    extra_body may also be set per endpoint in llm_endpoints.yaml. The
    two layers do not merge: an extra_body passed here replaces the
    endpoint's value outright, so a caller wanting both must combine them
    itself. Note that the Anthropic adapter ignores extra_body entirely,
    so the endpoint field has no effect on provider "anthropic".

    Raises EndpointNotFoundError if the endpoint name is missing, and
    NotImplementedError if the endpoint points at the Anthropic provider
    (the Anthropic adapter lands in a later minor release).
    """
    config, ep = _resolve_endpoint(
        config=config,
        endpoint_name=endpoint_name,
    )
    adapter = _select_adapter(ep.provider)
    effective_extra_body = _resolve_extra_body(
        call_value=extra_body,
        endpoint_value=ep.extra_body,
    )
    model = get_model_name(
        endpoint_name=endpoint_name,
        config=config,
    )
    client = get_client(
        endpoint_name=endpoint_name,
        config=config,
    )

    request_id = ""
    if tracker is not None:
        request_id = tracker.log_request(
            model=model,
            system_prompt=system or "",
            user_prompt=user,
            purpose=purpose,
            provider=ep.provider,
            metadata=metadata,
        )

    text, usage, generation_id = adapter.send(
        client=client,
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        user_id=user_id,
        extra_body=effective_extra_body,
        response_format=response_format,
    )

    if tracker is not None:
        tracker.log_response(
            request_id=request_id,
            model=model,
            response_text=text,
            usage=usage,
            generation_id=generation_id,
            purpose=purpose,
            provider=ep.provider,
            metadata=metadata,
        )

    return ChatResult(
        text=text,
        usage=usage,
        generation_id=generation_id,
    )
