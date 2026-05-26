"""
Integration-tier fixtures and skip logic.

Any test under tests/integration/ marked @pytest.mark.integration is
auto-skipped when OPENROUTER_API_KEY is not set in the environment, so a
default `pytest` run does no real API calls.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """
    Add a skip marker to every integration test in this dir tree when
    the required API key is absent. Tests already marked with a different
    skip reason are left alone.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    skip = pytest.mark.skip(
        reason="OPENROUTER_API_KEY not set; skipping integration tests",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
