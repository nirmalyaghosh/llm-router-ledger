"""
llm-router-ledger: route any LLM call and keep a JSONL ledger of every
request and response for offline cost reconciliation.

Public surface:
- send_message: primary entry point.
- UsageTracker: append-only JSONL logger.
- load_config, LLMConfig: YAML config + types.
- Exceptions rooted at LLMCallError.
"""

from importlib.metadata import (
    PackageNotFoundError,
    version,
)

from llm_router_ledger.client_factory import (
    get_client,
    get_model_name,
)
from llm_router_ledger.config import (
    CostConfig,
    DefaultsConfig,
    EndpointConfig,
    LLMConfig,
    ProviderName,
    get_context_window,
    load_config,
)
from llm_router_ledger.dispatcher import (
    send_message,
)
from llm_router_ledger.exceptions import (
    ConfigError,
    EndpointNotFoundError,
    LLMCallError,
    MissingApiKeyError,
    ProviderError,
    UsageTrackerError,
)
from llm_router_ledger.usage_tracker import (
    UsageTracker,
)


try:
    __version__ = version("llm-router-ledger")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


__all__ = [
    "ConfigError",
    "CostConfig",
    "DefaultsConfig",
    "EndpointConfig",
    "EndpointNotFoundError",
    "LLMCallError",
    "LLMConfig",
    "MissingApiKeyError",
    "ProviderError",
    "ProviderName",
    "UsageTracker",
    "UsageTrackerError",
    "__version__",
    "get_client",
    "get_context_window",
    "get_model_name",
    "load_config",
    "send_message",
]
