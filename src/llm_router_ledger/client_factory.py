"""
Provider-native client factory.

get_client(endpoint_name) returns the SDK client object for a given
endpoint:

- anthropic -> anthropic.Anthropic
- all others (azure, bytedance, deepseek, gemini, local-openai-compat,
  minimax, ollama, openai, openrouter, qwen, xiaomi, zhipu) ->
  openai.OpenAI with the endpoint's base_url. Azure's modern v1
  endpoint (base_url ending in /openai/v1/) is fully OpenAI-compatible,
  so it uses the standard OpenAI client just like every other
  OpenAI-compat provider.

SDK imports are deferred so that consumers who only use OpenAI-compatible
providers do not pay the anthropic import cost.
"""

from __future__ import annotations

from typing import Any

from llm_router_ledger.config import (
    LLMConfig,
    load_config,
)
from llm_router_ledger.exceptions import (
    ConfigError,
    EndpointNotFoundError,
)


def get_client(endpoint_name: str, config: LLMConfig | None = None) -> Any:
    """
    Return a configured provider-native client for the named endpoint.

    Raises EndpointNotFoundError if the name is not present in the loaded
    config.
    """
    if config is None:
        config = load_config()

    if endpoint_name not in config.endpoints:
        raise EndpointNotFoundError(
            f"Endpoint '{endpoint_name}' not"
            f" found in config"
        )

    ep = config.endpoints[endpoint_name]

    if ep.provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ConfigError(
                "The 'anthropic' SDK is not installed. Install"
                " the optional extra: uv pip install"
                " llm-router-ledger[anthropic]"
            ) from e
        return Anthropic(
            api_key=ep.api_key,
            timeout=ep.timeout_seconds,
            max_retries=ep.max_retries,
        )

    if ep.provider == "azure" and not ep.base_url:
        raise ConfigError(
            f"Azure endpoint"
            f" '{endpoint_name}' requires"
            f" base_url (e.g."
            f" https://<resource>.openai.azure.com/openai/v1/)"
        )

    from openai import OpenAI
    kwargs: dict[str, Any] = {
        "api_key": ep.api_key,
        "timeout": ep.timeout_seconds,
        "max_retries": ep.max_retries,
    }
    if ep.base_url:
        kwargs["base_url"] = ep.base_url
    return OpenAI(**kwargs)


def get_model_name(endpoint_name: str, config: LLMConfig | None = None) -> str:
    """
    Return the model string to pass in API calls for the named endpoint.

    Raises EndpointNotFoundError if the name is not present in the loaded
    config.
    """
    if config is None:
        config = load_config()
    if endpoint_name not in config.endpoints:
        raise EndpointNotFoundError(
            f"Endpoint '{endpoint_name}' not"
            f" found in config"
        )
    return config.endpoints[endpoint_name].model
