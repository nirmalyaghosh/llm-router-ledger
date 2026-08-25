"""
Provider adapter abstract base classes.

Each provider adapter wraps a single SDK call into a uniform tuple
return so the dispatcher does not branch on provider. One base class per
capability: ProviderAdapter for chat completions, EmbeddingAdapter for
vector embeddings.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import Any


def as_usage_dict(obj: Any) -> dict[str, Any]:
    """
    Best-effort dict view of an SDK usage object, which may be a
    pydantic model or a plain dict depending on the provider's SDK.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump())
        except Exception:  # noqa: BLE001
            pass
    return {
        key: value
        for key, value in vars(obj).items()
        if not key.startswith("_")
    }


def collect_unmapped(
    raw: Any,
    mapped: tuple[str, ...],
    blocks: tuple[tuple[str, tuple[str, ...], str], ...] = (),
) -> dict[str, Any]:
    """
    Gather the usage keys an adapter has no mapping for, so a
    provider-specific field reaches the ledger under
    usage_details["unmapped"] rather than being dropped.

    mapped names the top-level keys the adapter handles itself. blocks
    names each nested detail block as (block name, keys the adapter
    maps, prefix), so an unmapped key inside one takes the same prefix
    its mapped siblings take. Falsy values are omitted throughout,
    matching how the adapters write cost and the detail keys only when
    the provider actually reports them.

    Keys collected here are unstable by design: one that later gains an
    explicit mapping moves out of "unmapped" and up a level.
    """
    usage = as_usage_dict(raw)
    unmapped: dict[str, Any] = {}
    for key, value in usage.items():
        if key not in mapped and value:
            unmapped[key] = value
    for block, known, prefix in blocks:
        block_items = as_usage_dict(usage.get(block)).items()
        for key, value in block_items:
            if key not in known and value:
                unmapped[f"{prefix}{key}"] = value
    return unmapped


class EmbeddingAdapter(ABC):
    """
    Uniform embed interface for a single provider family.
    """

    @abstractmethod
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
        Embed one or more texts and return (vectors, usage_dict,
        generation_id).

        vectors is ordered to match texts, one vector per input. usage_dict
        carries the normalised prompt_tokens, completion_tokens, and
        total_tokens keys; completion_tokens is always 0, since embedding
        endpoints bill input only. Embedding-specific values ride alongside
        in the same dict when the provider returns them (dimensions,
        embedding_count, cost, upstream_provider); the dispatcher moves
        those into the ledger's usage_details rather than forcing them into
        the fixed token block.

        expected_dimensions is the endpoint's declared vector width. When
        set, an adapter raises ProviderError if the provider returns a
        different width rather than handing back vectors that would
        corrupt a fixed-width index.

        generation_id is the provider's response identifier, or "" if the
        provider does not return one. When timeout_seconds is None the
        client-level default applies. extra_body is a vendor-specific
        passthrough dict (e.g. OpenRouter provider routing hints);
        adapters that do not support it can ignore.
        """


class ProviderAdapter(ABC):
    """
    Uniform send-message interface for a single provider family.
    """

    @abstractmethod
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
        Send messages to the provider and return (response_text,
        usage_dict, generation_id).

        messages is already normalised by the dispatcher into the
        OpenAI content-parts shape: each entry is
        {"role": "system" | "user" | "assistant", "content":
        [{"type": "text", "text": ...}]}. A "system" entry, if present,
        may be anywhere in the list but is conventionally first;
        adapters whose provider takes system as a top-level parameter
        rather than a message (Anthropic) pull it out themselves via
        llm_router_ledger._messages.extract_system_text.

        usage_dict always carries prompt_tokens, completion_tokens, and
        total_tokens. Adapters that can report more (reasoning tokens,
        cache pricing, cost) add it to the same dict under its own key;
        the dispatcher splits the three normalised keys from the rest
        before logging, keeping the rest under usage_details. generation_id
        is the provider's response identifier, or "" if the provider does
        not return one. When timeout_seconds is None the client-level
        default applies.

        user_id is forwarded as the SDK's "user" field (end-user identifier;
        OpenRouter also uses this for request tagging). extra_body is a
        vendor-specific passthrough dict (e.g. OpenRouter provider routing
        hints). response_format requests structured output (e.g.
        {"type": "json_object"} for OpenAI JSON mode). All optional;
        adapters that do not support them can ignore.
        """
