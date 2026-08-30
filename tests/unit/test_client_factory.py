"""
Unit tests for llm_router_ledger.client_factory.

The point of these is that unusable_reason and get_client agree:
routing skips a candidate on the strength of the first, so anything
the second refuses must be something the first names.
"""

from __future__ import annotations

import sys

from pathlib import Path

import pytest

from llm_router_ledger.client_factory import (
    get_client,
    unusable_reason,
)
from llm_router_ledger.config import load_config
from llm_router_ledger.exceptions import LLMCallError

_ENDPOINTS = {
    "anthropic-no-sdk": (
        "  a:\n"
        "    provider: anthropic\n"
        "    model: claude\n"
        "    api_key_env: CF_KEY\n"
    ),
    "azure-no-base-url": (
        "  a:\n"
        "    provider: azure\n"
        "    model: gpt-4o\n"
        "    api_key_env: CF_KEY\n"
    ),
    "no-api-key": (
        "  a:\n"
        "    provider: openrouter\n"
        "    model: m\n"
        "    api_key_env: CF_MISSING\n"
    ),
}


@pytest.mark.parametrize(
    "block",
    list(_ENDPOINTS.values()),
    ids=list(_ENDPOINTS),
)
def test_get_client_refuses_what_unusable_reason_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    block: str,
) -> None:
    """
    Every condition unusable_reason reports is one get_client also
    refuses, so routing cannot skip a candidate the dispatcher would
    have accepted, or pick one it would not.
    """
    monkeypatch.setenv("CF_KEY", "set")
    monkeypatch.delenv("CF_MISSING", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    p = tmp_path / "cf.yaml"
    p.write_text("endpoints:\n" + block, encoding="utf-8")
    config = load_config(p)
    endpoint = config.endpoints["a"]

    assert unusable_reason(endpoint=endpoint) is not None
    with pytest.raises(LLMCallError):
        get_client(endpoint_name="a", config=config)


def test_unusable_reason_passes_a_working_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An endpoint get_client can build a client for is not skipped.
    """
    monkeypatch.setenv("CF_KEY", "set")
    p = tmp_path / "cf.yaml"
    p.write_text(
        "endpoints:\n"
        "  a:\n"
        "    provider: openrouter\n"
        "    model: m\n"
        "    api_key_env: CF_KEY\n",
        encoding="utf-8",
    )
    config = load_config(p)
    assert (
        unusable_reason(endpoint=config.endpoints["a"]) is None
    )
    assert get_client(endpoint_name="a", config=config)
