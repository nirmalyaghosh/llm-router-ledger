"""
llm-router-ledger: route any LLM call and keep a JSONL ledger of every
request and response for offline cost reconciliation.

Public surface:
- send_message: primary entry point, returns a ChatResult.
- create_embeddings: vector embeddings entry point, returns an
  EmbeddingResult.
- UsageTracker: append-only JSONL logger, with record_request,
  record_response, and record_run for calls the library did not make.
- purpose_scope: set the purpose an agent framework cannot pass.
- integrations.pydantic_ai.ledger_model: a Pydantic AI model that
  records every call it makes; needs the [pydantic-ai] extra.
- load_config, LLMConfig: YAML config + types.
- route: choose an endpoint from a route group without calling it.
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
    RouteGroupConfig,
    RouteStrategy,
    get_context_window,
    load_config,
)
from llm_router_ledger.dispatcher import (
    create_embeddings,
    send_message,
)
from llm_router_ledger.exceptions import (
    AuthenticationError,
    ConfigError,
    EndpointNotFoundError,
    InsufficientBalanceError,
    LLMCallError,
    MissingApiKeyError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitedError,
    RoutingError,
    UsageTrackerError,
)
from llm_router_ledger.purpose import (
    current_purpose,
    purpose_scope,
)
from llm_router_ledger.results import (
    ChatResult,
    EmbeddingResult,
)
from llm_router_ledger.routing import (
    RouteDecision,
    route,
)
from llm_router_ledger.usage_tracker import (
    UsageTracker,
)


try:
    __version__ = version("llm-router-ledger")
except PackageNotFoundError:
    __version__ = "0.0.0+local"


__all__ = [
    "ChatResult",
    "AuthenticationError",
    "ConfigError",
    "CostConfig",
    "DefaultsConfig",
    "EmbeddingResult",
    "EndpointConfig",
    "EndpointNotFoundError",
    "LLMCallError",
    "LLMConfig",
    "InsufficientBalanceError",
    "MissingApiKeyError",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitedError",
    "ProviderName",
    "RouteDecision",
    "RouteGroupConfig",
    "RouteStrategy",
    "RoutingError",
    "UsageTracker",
    "UsageTrackerError",
    "__version__",
    "create_embeddings",
    "current_purpose",
    "get_client",
    "get_context_window",
    "get_model_name",
    "load_config",
    "purpose_scope",
    "route",
    "send_message",
]
