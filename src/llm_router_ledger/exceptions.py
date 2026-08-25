"""
Exception hierarchy for llm-router-ledger.

All exceptions raised by this library inherit from LLMCallError so
consumers can catch every library failure with a single except clause.
"""

from __future__ import annotations


class LLMCallError(Exception):
    """
    Root exception for all llm-router-ledger failures.
    """


class ConfigError(LLMCallError):
    """
    Raised when YAML config is missing, invalid, or fails Pydantic
    validation.
    """


class EndpointNotFoundError(ConfigError):
    """
    Raised when an endpoint name is requested but is not defined in the
    loaded config.
    """


class MissingApiKeyError(ConfigError):
    """
    Raised when an endpoint's api_key_env environment variable is not set.
    """


class ProviderError(LLMCallError):
    """
    Raised when a provider API call fails (HTTP error, malformed response,
    timeout).

    status_code carries the HTTP status the provider returned, and is
    None for transport failures and malformed responses. The subclasses
    below name the statuses worth branching on; anything else a provider
    returns raises this class directly.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(ProviderError):
    """
    Raised when the provider rejects the credential (HTTP 401 or 403).

    Distinct from MissingApiKeyError, which fires before any call is made
    because the environment variable is unset. This one means a key was
    sent and refused.
    """


class InsufficientBalanceError(ProviderError):
    """
    Raised when the account cannot pay for the call (HTTP 402).
    """


class ProviderUnavailableError(ProviderError):
    """
    Raised when the provider is unreachable or failing: HTTP 5xx, or a
    connection or timeout failure that carries no status at all.
    """


class RateLimitedError(ProviderError):
    """
    Raised when the provider rate-limits the call (HTTP 429).

    The SDK retries internally first, so this means its retries were
    exhausted, not that a single attempt was throttled.
    """


class UsageTrackerError(LLMCallError):
    """
    Raised when the usage tracker cannot write to its JSONL log (e.g.
    disk full, permission denied).
    """
