"""
Anthropic native provider adapter.

Wraps Anthropic's Messages API into the uniform ProviderAdapter
interface. Anthropic's API differs from OpenAI chat completions in
several ways the adapter normalises away:

- `system` is a top-level parameter, not a message in the list.
- `usage.input_tokens` / `usage.output_tokens` are translated to
  prompt_tokens / completion_tokens for ledger consistency;
  total_tokens is computed locally since Anthropic does not return it.
- `max_tokens` is required by the Messages API; the adapter passes
  the caller-supplied value (default 4096 from the base interface).

The adapter does not catch SDK exceptions; anthropic.APIError and
friends propagate so the caller can distinguish rate limits, timeouts,
and auth failures by subtype.
"""

from __future__ import annotations

from typing import Any

from llm_router_ledger.providers._base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    """
    Adapter for Anthropic's native Messages API (api.anthropic.com).
    """

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
        Send system + user to Anthropic Messages API and return
        (response_text, usage_dict, generation_id).

        generation_id is response.id (a `msg_*`-prefixed string); the
        downstream tracker routes it to provider_response_id since it
        does not start with `gen-`.

        Unsupported by the Messages API in this adapter (silently
        ignored): user_id (no end-user identifier field), extra_body
        (no passthrough mechanism), response_format (Anthropic uses
        tool-use for structured output, not a response_format kwarg).
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": user},
            ],
        }
        if system is not None:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds

        response = client.messages.create(**kwargs)

        text = ""
        if response.content:
            first = response.content[0]
            text = getattr(first, "text", "") or ""

        raw = response.usage
        input_tokens = (
            getattr(raw, "input_tokens", 0)
            if raw is not None
            else 0
        )
        output_tokens = (
            getattr(raw, "output_tokens", 0)
            if raw is not None
            else 0
        )
        usage: dict[str, int] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        return text, usage, response.id or ""
