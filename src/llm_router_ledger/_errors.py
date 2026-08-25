"""
Internal helper for turning a provider SDK exception into a library one.

Callers should not have to import a provider SDK to catch a failure, and
with the anthropic SDK behind an optional extra they may not be able to.
So the dispatcher wraps whatever the adapter raises into the
ProviderError family before it reaches the caller.

The status is read with getattr rather than by importing SDK exception
classes. Both the openai and anthropic SDKs expose status_code on their
APIStatusError, so duck typing covers both and keeps anthropic optional.
A transport failure carries no status and maps to
ProviderUnavailableError, which is the useful reading of it.
"""

from __future__ import annotations

from llm_router_ledger.exceptions import (
    AuthenticationError,
    InsufficientBalanceError,
    LLMCallError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitedError,
)

_STATUS_MAP: dict[int, type[ProviderError]] = {
    401: AuthenticationError,
    402: InsufficientBalanceError,
    403: AuthenticationError,
    429: RateLimitedError,
}


def wrap_provider_exception(
    exc: Exception,
    endpoint_name: str,
) -> LLMCallError:
    """
    Map an exception raised by an adapter onto the ProviderError family.

    A library exception passes through unchanged, so a ProviderError an
    adapter raised itself (e.g. an embedding width mismatch) is not
    rewrapped or reclassified. The returned exception is meant to be
    raised `from` the original, keeping the SDK's own traceback and
    message reachable through __cause__.
    """
    if isinstance(exc, LLMCallError):
        return exc
    status = getattr(exc, "status_code", None)
    # bool is a subclass of int, so exclude it explicitly rather than
    # letting True through as an HTTP status.
    if isinstance(status, bool) or not isinstance(status, int):
        status = None
    if status is None:
        error_class: type[ProviderError] = ProviderUnavailableError
    elif status in _STATUS_MAP:
        error_class = _STATUS_MAP[status]
    elif status >= 500:
        error_class = ProviderUnavailableError
    else:
        error_class = ProviderError
    return error_class(
        f"Endpoint '{endpoint_name}' failed:"
        f" {type(exc).__name__}: {exc}",
        status_code=status,
    )
