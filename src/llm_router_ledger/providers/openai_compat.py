"""
OpenAI-compatible provider adapters.

OpenAICompatAdapter wraps client.chat.completions.create(...) and
OpenAICompatEmbeddingAdapter wraps client.embeddings.create(...) for
every OpenAI-compatible endpoint: OpenAI, Azure (via AzureOpenAI
client), OpenRouter, DeepSeek, MiniMax, Qwen, Zhipu, Xiaomi, ByteDance,
Gemini's compatibility endpoint, and local servers (Ollama, LM Studio,
vLLM).

Neither adapter catches SDK exceptions; openai.APIError and friends
propagate so the caller can distinguish rate limits, timeouts, and auth
failures by subtype.
"""

from __future__ import annotations

from typing import Any

from llm_router_ledger.exceptions import ProviderError
from llm_router_ledger.providers._base import (
    EmbeddingAdapter,
    ProviderAdapter,
    collect_unmapped,
)


ENCODING_FORMAT = "float"

# The finish_reason of a turn that ran to completion. Other values
# (length, tool_calls, content_filter) mean the response is not the
# whole answer, and are recorded as finish_reason.
ORDINARY_FINISH_REASON = "stop"

# completion_tokens_details and prompt_tokens_details field names, per
# OpenRouter's chat completion response. audio_tokens appears in both
# blocks with a different meaning in each, which is why every key is
# flattened with a prompt_ or completion_ prefix rather than merged.
_COMPLETION_DETAIL_KEYS = (
    "accepted_prediction_tokens",
    "audio_tokens",
    "image_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
)
_PROMPT_DETAIL_KEYS = (
    "audio_tokens",
    "cache_write_tokens",
    "cached_tokens",
    "video_tokens",
)

# Top-level usage keys this adapter maps itself. Anything else the
# provider reports is collected under usage_details["unmapped"],
# e.g. DeepSeek's prompt_cache_hit_tokens or Azure's
# latency_checkpoint.
_MAPPED_USAGE_KEYS = (
    "completion_tokens",
    "completion_tokens_details",
    "cost",
    "is_byok",
    "prompt_tokens",
    "prompt_tokens_details",
    "total_tokens",
)


def _flatten_detail(
    detail: Any,
    keys: tuple[str, ...],
    prefix: str,
) -> dict[str, Any]:
    """
    Helper function used to flatten one nested usage detail block
    (completion_tokens_details or prompt_tokens_details) into prefixed
    top-level keys. Keys whose value is zero, null, or absent are
    omitted, matching how the embedding adapter only writes cost,
    is_byok, and upstream_provider when the provider actually reports
    them.
    """
    if detail is None:
        return {}
    flattened: dict[str, Any] = {}
    for key in keys:
        value = getattr(detail, key, None)
        if value:
            flattened[f"{prefix}{key}"] = value
    return flattened


class OpenAICompatAdapter(ProviderAdapter):
    """
    Single adapter for every OpenAI-compatible endpoint.
    """

    def send(
        self,
        *,
        client: Any,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
        user_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        """
        Send messages to an OpenAI-compat endpoint and return
        (response_text, usage_dict, generation_id).

        messages is forwarded to the SDK unchanged: the content-parts
        shape ({"content": [{"type": "text", "text": ...}]}) the
        dispatcher builds is exactly what ChatCompletionMessageParam
        accepts for system, user, and assistant roles alike, so no
        per-role conversion is needed here.

        response_text is "" on a turn that returns only tool calls,
        since the API sets message.content to null there. That turn cost
        real tokens, so usage_dict carries completion_tool_call_count to
        keep the ledger row from reading as an empty response; the key
        is omitted when the turn made no tool calls.

        usage_dict also carries finish_reason, in the provider's own
        vocabulary, when the turn ended as anything other than "stop",
        e.g. "length" for an answer truncated at max_tokens.

        usage_dict always has prompt_tokens, completion_tokens, and
        total_tokens, all zero if the provider omits usage. When the
        provider reports more, usage_dict also carries cost, is_byok,
        and upstream_provider (mirroring the embedding adapter), plus
        the flattened contents of completion_tokens_details and
        prompt_tokens_details under a completion_ / prompt_ prefix
        (e.g. completion_reasoning_tokens, prompt_cached_tokens).
        Any usage key this adapter has no mapping for is collected
        under an "unmapped" sub-dict rather than dropped.
        generation_id is response.id; the downstream tracker routes
        "gen-" prefixed IDs to generation_id and everything else to
        provider_response_id.

        user_id is forwarded as the SDK's "user" field (e.g. OpenRouter
        run tag). extra_body is passed through verbatim for
        vendor-specific fields like OpenRouter's {"provider": {...}}
        routing hints.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        if user_id is not None:
            kwargs["user"] = user_id
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = (
            client.chat.completions.create(
                **kwargs,
            )
        )

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )
        finish_reason = getattr(
            choice,
            "finish_reason",
            None,
        )
        raw = response.usage
        usage: dict[str, Any] = {
            k: (
                getattr(raw, k, 0)
                if raw is not None
                else 0
            )
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        }
        cost = (
            getattr(raw, "cost", None)
            if raw is not None
            else None
        )
        if cost is not None:
            usage["cost"] = cost
        is_byok = (
            getattr(raw, "is_byok", None)
            if raw is not None
            else None
        )
        if is_byok is not None:
            usage["is_byok"] = is_byok
        if tool_calls:
            usage["completion_tool_call_count"] = len(
                tool_calls
            )
        if finish_reason and finish_reason != ORDINARY_FINISH_REASON:
            usage["finish_reason"] = finish_reason
        completion_details = (
            getattr(
                raw,
                "completion_tokens_details",
                None,
            )
            if raw is not None
            else None
        )
        usage.update(
            _flatten_detail(
                completion_details,
                _COMPLETION_DETAIL_KEYS,
                "completion_",
            )
        )
        prompt_details = (
            getattr(
                raw,
                "prompt_tokens_details",
                None,
            )
            if raw is not None
            else None
        )
        usage.update(
            _flatten_detail(
                prompt_details,
                _PROMPT_DETAIL_KEYS,
                "prompt_",
            )
        )
        unmapped = collect_unmapped(
            raw,
            _MAPPED_USAGE_KEYS,
            (
                (
                    "completion_tokens_details",
                    _COMPLETION_DETAIL_KEYS,
                    "completion_",
                ),
                (
                    "prompt_tokens_details",
                    _PROMPT_DETAIL_KEYS,
                    "prompt_",
                ),
            ),
        )
        if unmapped:
            usage["unmapped"] = unmapped
        upstream = getattr(
            response,
            "provider",
            None,
        )
        if upstream:
            usage["upstream_provider"] = upstream

        return text, usage, response.id or ""


class OpenAICompatEmbeddingAdapter(EmbeddingAdapter):
    """
    Single embedding adapter for every OpenAI-compatible endpoint.
    """

    def embed(
        self,
        *,
        client: Any,
        model: str,
        texts: list[str],
        expected_dimensions: int | None = None,
        timeout_seconds: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[list[list[float]], dict[str, Any], str]:
        """
        Embed texts against an OpenAI-compat endpoint and return
        (vectors, usage_dict, generation_id).

        encoding_format is always sent as "float" rather than left to the
        SDK. The SDK otherwise defaults to base64 and decodes client
        side, which some upstreams reject outright (OpenRouter's
        nvidia/nemotron-3-embed-1b returns a 400 naming base64). Pinning
        it keeps one wire format across every provider.

        usage_dict always carries prompt_tokens, completion_tokens, and
        total_tokens, with completion_tokens fixed at 0 because embedding
        endpoints bill input only. dimensions and embedding_count are
        always present; cost, is_byok, and upstream_provider are included
        only when the provider reports them. OpenRouter returns cost as
        the actual USD charge for the call, and upstream_provider names
        the service that served it (routing can move between calls and
        change what a given endpoint costs).

        Raises ProviderError when expected_dimensions is set and the
        provider returns vectors of a different width.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "input": texts,
            "encoding_format": ENCODING_FORMAT,
        }
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        response = client.embeddings.create(
            **kwargs,
        )

        # Ordered by index rather than trusting arrival order: the index
        # field exists precisely because the API does not promise one.
        items = sorted(
            response.data,
            key=lambda item: getattr(
                item,
                "index",
                0,
            ),
        )
        vectors = [
            list(item.embedding) for item in items
        ]

        # Guard the declared width. OpenRouter routes across upstreams
        # and the one serving a given endpoint can change between
        # calls; a width change would otherwise be written into a
        # fixed-width index and only surface as degraded retrieval.
        if expected_dimensions is not None and vectors:
            actual = len(vectors[0])
            if actual != expected_dimensions:
                raise ProviderError(
                    f"Endpoint declares"
                    f" embedding_dimensions"
                    f" {expected_dimensions} but model"
                    f" '{model}' returned {actual}."
                    f" Refusing to return vectors of an"
                    f" unexpected width; correct the"
                    f" config or pin the upstream"
                    f" provider."
                )

        raw = response.usage
        prompt_tokens = (
            getattr(raw, "prompt_tokens", 0)
            if raw is not None
            else 0
        )
        total_tokens = (
            getattr(raw, "total_tokens", 0)
            if raw is not None
            else 0
        )
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": (
                total_tokens or prompt_tokens
            ),
            "embedding_count": len(vectors),
            "dimensions": (
                len(vectors[0]) if vectors else 0
            ),
        }
        cost = (
            getattr(raw, "cost", None)
            if raw is not None
            else None
        )
        if cost is not None:
            usage["cost"] = cost
        is_byok = (
            getattr(raw, "is_byok", None)
            if raw is not None
            else None
        )
        if is_byok is not None:
            usage["is_byok"] = is_byok
        upstream = getattr(
            response,
            "provider",
            None,
        )
        if upstream:
            usage["upstream_provider"] = upstream

        return (
            vectors,
            usage,
            getattr(response, "id", "") or "",
        )
