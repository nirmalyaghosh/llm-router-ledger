"""
Exception hierarchy for llm-router-ledger.

All exceptions raised by this library inherit from LLMCallError so
consumers can catch every library failure with a single except clause.
"""


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
    """


class UsageTrackerError(LLMCallError):
    """
    Raised when the usage tracker cannot write to its JSONL log (e.g.
    disk full, permission denied).
    """
