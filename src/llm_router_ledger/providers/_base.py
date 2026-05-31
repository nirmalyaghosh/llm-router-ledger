"""
Provider adapter abstract base class.

Each provider adapter wraps a single SDK call into a uniform tuple
return so the dispatcher does not branch on provider.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import Any


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
