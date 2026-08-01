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
        system: str | None,
        user: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        timeout_seconds: float | None = None,
        user_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, int], str]:
        """
        Send system + user to the provider and return (response_text,
        usage_dict, generation_id).

        usage_dict is normalised to keys prompt_tokens, completion_tokens,
        total_tokens. generation_id is the provider's response identifier,
        or "" if the provider does not return one. When timeout_seconds
        is None the client-level default applies.

        system may be None for user-only calls. user_id is forwarded as
        the SDK's "user" field (end-user identifier; OpenRouter also uses
        this for request tagging). extra_body is a vendor-specific
        passthrough dict (e.g. OpenRouter provider routing hints).
        response_format requests structured output (e.g.
        {"type": "json_object"} for OpenAI JSON mode). All optional;
        adapters that do not support them can ignore.
        """
