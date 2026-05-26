"""
OpenAI-compatible provider adapter.

Wraps client.chat.completions.create(...) for every OpenAI-compatible
endpoint: OpenAI, Azure (via AzureOpenAI client), OpenRouter, DeepSeek,
MiniMax, Qwen, Zhipu, Xiaomi, ByteDance, Gemini's compatibility endpoint,
and local servers (Ollama, LM Studio, vLLM).

The adapter does not catch SDK exceptions; openai.APIError and friends
propagate so the caller can distinguish rate limits, timeouts, and auth
failures by subtype.
"""

from __future__ import annotations

from typing import Any

from llm_router_ledger.providers._base import ProviderAdapter


class OpenAICompatAdapter(ProviderAdapter):
    """
    Single adapter for every OpenAI-compatible endpoint.
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
        Send system + user to an OpenAI-compat endpoint and return
        (response_text, usage_dict, generation_id).

        usage_dict has prompt_tokens, completion_tokens, total_tokens,
        all zero if the provider omits usage. generation_id is
        response.id; the downstream tracker routes "gen-" prefixed IDs
        to generation_id and everything else to provider_response_id.

        user_id is forwarded as the SDK's "user" field (e.g. OpenRouter
        run tag). extra_body is passed through verbatim for
        vendor-specific fields like OpenRouter's {"provider": {...}}
        routing hints.
        """
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
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

        text = (
            response.choices[0].message.content
            or ""
        )
        raw = response.usage
        usage: dict[str, int] = {
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
        return text, usage, response.id or ""
