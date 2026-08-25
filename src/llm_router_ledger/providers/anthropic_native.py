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

from llm_router_ledger._messages import extract_system_text
from llm_router_ledger.providers._base import (
    ProviderAdapter,
    collect_unmapped,
)

# Top-level usage keys this adapter maps itself. The Messages API
# reports cache and tier fields besides these, which are collected
# under usage_details["unmapped"].
_MAPPED_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
)


# The Messages API name for an ordinary completion, its spelling of the
# OpenAI-compatible "stop". Other values are recorded as finish_reason.
ORDINARY_STOP_REASON = "end_turn"


class AnthropicAdapter(ProviderAdapter):
    """
    Adapter for Anthropic's native Messages API (api.anthropic.com).
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
        Send messages to Anthropic Messages API and return
        (response_text, usage_dict, generation_id).

        Any system-role entries in messages are pulled out and joined
        into the top-level system parameter, since the Messages API
        takes system as its own kwarg rather than a message in the
        list; the remaining entries are forwarded as-is, since their
        {"role", "content": [{"type": "text", "text": ...}]} shape
        already matches Anthropic's TextBlockParam.

        response_text is every text block in the response joined with a
        newline, in order. A Messages API response is a list of content
        blocks, and only a text block carries a text attribute: thinking
        and tool_use blocks are skipped rather than read as empty. A
        turn made up entirely of non-text blocks therefore returns "".

        usage_dict carries the three normalised token keys. The
        Messages API reports no cost, and no reasoning detail on an
        ordinary call, but it does report cache fields
        (cache_read_input_tokens, cache_creation_input_tokens and a
        cache_creation breakdown) plus service_tier and
        inference_geo. This adapter maps none of those, so they are
        collected under an "unmapped" sub-dict rather than dropped.
        Also present is completion_tool_call_count, the
        number of tool_use blocks in the response, added so a tool-only
        turn does not read as an empty response. It is unreachable
        today: this library exposes no tools parameter and this adapter
        ignores extra_body, so nothing can ask the Messages API for
        tools. It is here so the ledger stays honest the moment tools
        are plumbed through.

        stop_reason is recorded as finish_reason, the key the
        OpenAI-compatible adapter uses, unless it is "end_turn". The
        value stays in Anthropic's vocabulary: the two APIs' stop
        reasons do not map onto each other cleanly enough to normalise.

        generation_id is response.id (a `msg_*`-prefixed string); the
        downstream tracker routes it to provider_response_id since it
        does not start with `gen-`.

        Unsupported by the Messages API in this adapter (silently
        ignored): user_id (no end-user identifier field), extra_body
        (no passthrough mechanism), response_format (Anthropic uses
        tool-use for structured output, not a response_format kwarg).
        """
        system = extract_system_text(messages)
        conversation = [
            message
            for message in messages
            if message.get("role") != "system"
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": conversation,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds

        response = client.messages.create(**kwargs)

        text_blocks: list[str] = []
        tool_call_count = 0
        for block in response.content or []:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str) and block_text:
                text_blocks.append(block_text)
            if getattr(block, "type", None) == "tool_use":
                tool_call_count += 1
        text = "\n".join(text_blocks)

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
        usage: dict[str, Any] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if tool_call_count:
            usage["completion_tool_call_count"] = tool_call_count
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason and stop_reason != ORDINARY_STOP_REASON:
            usage["finish_reason"] = stop_reason
        unmapped = collect_unmapped(raw, _MAPPED_USAGE_KEYS)
        if unmapped:
            usage["unmapped"] = unmapped

        return text, usage, response.id or ""
