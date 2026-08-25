"""
Unit tests for provider-error wrapping and the llm_error ledger event.

The SDK is mocked so these run fully offline. The wrapping tests use
plain exception classes carrying a status_code attribute rather than real
SDK types: the mapper reads the status by duck typing precisely so it
does not import a provider SDK, and the tests exercise it the same way.
"""

from __future__ import annotations

import json

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_router_ledger import (
    AuthenticationError,
    InsufficientBalanceError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitedError,
    UsageTracker,
    load_config,
    send_message,
)
from llm_router_ledger._errors import wrap_provider_exception


class _FakeStatusError(Exception):
    """
    Stand-in for an SDK APIStatusError: an exception carrying an HTTP
    status_code, which is all the mapper looks at.
    """

    def __init__(self, status_code: int) -> None:
        super().__init__(f"boom {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, AuthenticationError),
        (402, InsufficientBalanceError),
        (403, AuthenticationError),
        (429, RateLimitedError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
        (400, ProviderError),
        (404, ProviderError),
    ],
)
def test_wrap_maps_status_to_exception_class(
    status: int,
    expected: type[ProviderError],
) -> None:
    """
    Each HTTP status the classifier cares about maps to its own class,
    and the status is preserved on the wrapped exception. A status with
    no dedicated class falls back to ProviderError itself rather than
    being forced into a wrong one.
    """
    wrapped = wrap_provider_exception(
        _FakeStatusError(status),
        "some-endpoint",
    )
    assert type(wrapped) is expected
    assert wrapped.status_code == status
    assert "some-endpoint" in str(wrapped)


def test_wrap_treats_missing_status_as_unavailable() -> None:
    """
    A transport failure carries no status_code at all. It maps to
    ProviderUnavailableError with a None status, which is the useful
    reading: the provider could not be reached.
    """
    wrapped = wrap_provider_exception(
        ConnectionError("connection reset"),
        "some-endpoint",
    )
    assert type(wrapped) is ProviderUnavailableError
    assert wrapped.status_code is None


def test_wrap_passes_library_exceptions_through() -> None:
    """
    A ProviderError an adapter raised itself, e.g. an embedding width
    mismatch, is returned unchanged rather than rewrapped and
    reclassified as a transport failure.
    """
    original = ProviderError("width mismatch")
    assert (
        wrap_provider_exception(original, "some-endpoint")
        is original
    )


def _patch_failing_adapter(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
) -> None:
    """
    Helper function used to mock get_client and _select_adapter so the
    adapter's send raises exc, letting the dispatcher's failure path run
    offline.
    """
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher.get_client",
        lambda **kw: MagicMock(),
    )
    fake = MagicMock()
    fake.send.side_effect = exc
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher._select_adapter",
        lambda provider: fake,
    )


def test_send_message_raises_wrapped_error(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    The SDK exception does not reach the caller. A wrapped library
    exception does, with the original still reachable through __cause__
    so nothing is lost.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    original = _FakeStatusError(429)
    _patch_failing_adapter(monkeypatch, original)
    config = load_config(sample_yaml_file)
    with pytest.raises(RateLimitedError) as caught:
        send_message(
            endpoint_name="ollama-local",
            user="hi",
            config=config,
        )
    assert caught.value.status_code == 429
    assert caught.value.__cause__ is original


def test_send_message_writes_error_event(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
    tmp_log_path: Path,
) -> None:
    """
    A failed call writes an llm_error event sharing the request_id of
    the llm_request that preceded it, so the request is not left
    orphaned. The original exception's class name is recorded, since the
    wrapped class the caller sees is coarser than what the SDK raised.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    _patch_failing_adapter(monkeypatch, _FakeStatusError(402))
    config = load_config(sample_yaml_file)
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
    )
    try:
        with pytest.raises(InsufficientBalanceError):
            send_message(
                endpoint_name="ollama-local",
                user="hi",
                config=config,
                tracker=tracker,
            )
    finally:
        tracker.close()

    entries = [
        json.loads(line)
        for line in tmp_log_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert [e["event"] for e in entries] == [
        "llm_request",
        "llm_error",
    ]
    request, error = entries
    assert error["request_id"] == request["request_id"]
    assert error["error_type"] == "_FakeStatusError"
    assert error["status_code"] == 402
    assert error["provider"] == "ollama"


def test_error_event_omits_status_when_absent(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
    tmp_log_path: Path,
) -> None:
    """
    A transport failure has no HTTP status, so status_code is omitted
    from the event rather than written as null.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    _patch_failing_adapter(
        monkeypatch,
        ConnectionError("connection reset"),
    )
    config = load_config(sample_yaml_file)
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
    )
    try:
        with pytest.raises(ProviderUnavailableError):
            send_message(
                endpoint_name="ollama-local",
                user="hi",
                config=config,
                tracker=tracker,
            )
    finally:
        tracker.close()

    error = json.loads(
        tmp_log_path.read_text(
            encoding="utf-8",
        ).splitlines()[-1],
    )
    assert error["event"] == "llm_error"
    assert "status_code" not in error
