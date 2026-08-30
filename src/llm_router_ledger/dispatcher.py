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

from llm_router_ledger._errors import wrap_provider_exception
from llm_router_ledger._messages import (
    build_messages,
    extract_system_text,
    extract_text,
)
from llm_router_ledger._usage import split_usage
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


_VERIFIED_EMBEDDING_PROVIDERS = frozenset({
    "lmstudio",
    "ollama",
    "openrouter",
})

_VERIFIED_PROVIDERS = frozenset({
    "anthropic",
    "azure",
    "deepseek",
    "lmstudio",
    "minimax",
    "nvidia",
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


def _resolve_messages(
    *,
    messages: list[dict[str, Any]] | None,
    system: str | None,
    user: str | None,
) -> list[dict[str, Any]]:
    """
    Helper function used to pick the messages list for a single call. A
    call-level messages list replaces system/user outright rather than
    merging with them, mirroring the rule _resolve_extra_body already
    follows for the endpoint/call layering.

    Raises ValueError when neither messages nor user is supplied, since
    a call needs some content to send.
    """
    if messages is not None:
        return messages
    if user is None:
        raise ValueError(
            "send_message requires either 'user' or 'messages'"
        )
    return build_messages(system=system, user=user)


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

    Both local-server names are verified. Neither returns a response
    id, so their rows carry an empty provider_response_id. LM Studio
    additionally leaves prompt_tokens and total_tokens at zero for
    embeddings, so those rows record the vectors and their width but
    no token count. Nothing is billed either way, so there is no
    invoice to reconcile against.
    """
    if provider not in _VERIFIED_EMBEDDING_PROVIDERS:
        verified = ", ".join(
            sorted(_VERIFIED_EMBEDDING_PROVIDERS),
        )
        raise NotImplementedError(
            f"Embeddings are not available for the '{provider}'"
            f" provider. Verified embedding providers in this"
            f" release: {verified}. Point the endpoint at OpenRouter"
            f" for hosted embedding models, or at Ollama or LM Studio"
            f" to run one locally."
        )
    return OpenAICompatEmbeddingAdapter()


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

    try:
        vectors, usage, generation_id = adapter.embed(
            client=client,
            model=model,
            texts=texts,
            expected_dimensions=ep.embedding_dimensions,
            timeout_seconds=timeout_seconds,
            extra_body=effective_extra_body,
        )
    except Exception as exc:
        wrapped = wrap_provider_exception(exc, endpoint_name)
        if tracker is not None:
            tracker.log_error(
                request_id=request_id,
                model=model,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=getattr(
                    wrapped, "status_code", None
                ),
                purpose=purpose,
                provider=ep.provider,
                modality="embedding",
                metadata=metadata,
            )
        if wrapped is exc:
            raise
        raise wrapped from exc

    if tracker is not None:
        token_usage, usage_details = (
            split_usage(usage)
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
    user: str | None = None,
    system: str | None = None,
    messages: list[dict[str, Any]] | None = None,
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
    Send a message to the named endpoint and return a ChatResult.

    result.usage carries the normalised prompt_tokens, completion_tokens,
    and total_tokens keys plus whatever else the provider reported: cost,
    is_byok, and upstream_provider when available, and the flattened
    contents of completion_tokens_details / prompt_tokens_details under a
    completion_ / prompt_ prefix (e.g. completion_reasoning_tokens,
    prompt_cached_tokens). The caller gets all of it; the ledger splits
    it, keeping the token keys in the usage block and the rest under
    usage_details, the same rule create_embeddings follows.

    When tracker is provided, paired llm_request and llm_response events
    are appended to its JSONL log. When tracker is None, no logging
    happens.

    system + user is the single-turn convenience form; system is
    optional, pass None for user-only calls (common with JSON-mode
    prompts that embed all instructions in the user message).

    messages is the multi-turn form, for conversation history, tool
    loops, or any call system/user cannot express. Each entry is
    {"role": "system" | "user" | "assistant", "content":
    [{"type": "text", "text": ...}]}, the OpenAI content-parts shape,
    kept even though image parts are not supported yet so that adding
    them later is additive. When messages is supplied it replaces
    system and user outright rather than merging with them, the same
    rule extra_body follows against its endpoint-level default. Exactly
    one of messages or user must be supplied.

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

    Raises EndpointNotFoundError if the endpoint name is missing,
    NotImplementedError if the endpoint's provider has no verified chat
    adapter, and ValueError if neither messages nor user is supplied.
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
    effective_messages = _resolve_messages(
        messages=messages,
        system=system,
        user=user,
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
        latest_user_messages = [
            message
            for message in effective_messages
            if message.get("role") == "user"
        ]
        latest_user_prompt = (
            extract_text(latest_user_messages[-1].get("content"))
            if latest_user_messages
            else ""
        )
        request_id = tracker.log_request(
            model=model,
            system_prompt=extract_system_text(effective_messages),
            user_prompt=latest_user_prompt,
            purpose=purpose,
            provider=ep.provider,
            metadata=metadata,
        )

    try:
        text, usage, generation_id = adapter.send(
            client=client,
            model=model,
            messages=effective_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            user_id=user_id,
            extra_body=effective_extra_body,
            response_format=response_format,
        )
    except Exception as exc:
        wrapped = wrap_provider_exception(exc, endpoint_name)
        if tracker is not None:
            tracker.log_error(
                request_id=request_id,
                model=model,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=getattr(
                    wrapped, "status_code", None
                ),
                purpose=purpose,
                provider=ep.provider,
                metadata=metadata,
            )
        if wrapped is exc:
            raise
        raise wrapped from exc

    if tracker is not None:
        token_usage, usage_details = (
            split_usage(usage)
        )
        tracker.log_response(
            request_id=request_id,
            model=model,
            response_text=text,
            usage=token_usage,
            generation_id=generation_id,
            purpose=purpose,
            provider=ep.provider,
            usage_details=usage_details,
            metadata=metadata,
        )

    return ChatResult(
        text=text,
        usage=usage,
        generation_id=generation_id,
    )
