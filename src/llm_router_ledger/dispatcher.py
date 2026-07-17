"""
Public send_message entry point.

Resolves an endpoint, gets the SDK client, picks the provider adapter,
sends, optionally appends paired llm_request and llm_response events via
a UsageTracker, and returns the standard tuple (response_text, usage_dict,
generation_id).
"""

from __future__ import annotations

import copy

from typing import Any

from llm_router_ledger.client_factory import (
    get_client,
    get_model_name,
)
from llm_router_ledger.config import (
    LLMConfig,
    load_config,
)
from llm_router_ledger.exceptions import (
    EndpointNotFoundError,
)
from llm_router_ledger.providers._base import (
    ProviderAdapter,
)
from llm_router_ledger.providers.anthropic_native import (
    AnthropicAdapter,
)
from llm_router_ledger.providers.openai_compat import (
    OpenAICompatAdapter,
)
from llm_router_ledger.usage_tracker import UsageTracker


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
) -> tuple[str, dict[str, int], str]:
    """
    Send a system + user message to the named endpoint and return
    (response_text, usage_dict, generation_id).

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
    if config is None:
        config = load_config()
    if endpoint_name not in config.endpoints:
        raise EndpointNotFoundError(
            f"Endpoint '{endpoint_name}' not"
            f" found in config"
        )

    ep = config.endpoints[endpoint_name]
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

    return text, usage, generation_id
