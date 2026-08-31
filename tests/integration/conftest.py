"""
Integration-tier fixtures and skip logic.

Any test under tests/integration/ marked @pytest.mark.integration is
skipped unless LRL_RUN_INTEGRATION is set, so a default `pytest` run
makes no real API calls.

The skip hook read OPENROUTER_API_KEY until 0.3.0. That stopped
working in 0.2.2, when .env resolution moved to the working
directory: importing the library loads .env, so by the time this
hook runs the key is always set and the tests always ran. An
explicit opt-in cannot be turned on as a side effect of an import.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """
    Add a skip marker to every integration test in this dir tree
    unless the opt-in is set. Tests already marked with a different
    skip reason are left alone.
    """
    if os.environ.get("LRL_RUN_INTEGRATION"):
        return
    skip = pytest.mark.skip(
        reason=(
            "LRL_RUN_INTEGRATION not set; skipping integration"
            " tests, which make billed API calls"
        ),
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
