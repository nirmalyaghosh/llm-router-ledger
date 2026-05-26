"""
Integration tests against a real OpenRouter endpoint.

These tests issue paid API calls and require OPENROUTER_API_KEY in the
environment. The conftest in this directory auto-skips them when the env
var is unset, so a default `pytest` run does nothing real.

Run explicitly:

    pytest -m integration tests/integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_router_ledger import (
    UsageTracker,
    load_config,
    send_message,
)


OPENROUTER_YAML = """\
endpoints:
  openrouter-test:
    provider: openrouter
    model: openai/gpt-4.1-nano
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    timeout_seconds: 30
"""


@pytest.fixture
def openrouter_config_path(tmp_path: Path) -> Path:
    """
    Write a minimal OpenRouter-only YAML to a per-test path and return
    it.
    """
    p = tmp_path / "llm_endpoints.yaml"
    p.write_text(OPENROUTER_YAML, encoding="utf-8")
    return p


@pytest.mark.integration
def test_openrouter_send_message_roundtrip(
    openrouter_config_path: Path,
    tmp_path: Path,
) -> None:
    """
    Hit OpenRouter with a short prompt and verify the response shape:
    non-empty text, positive token counts, and a "gen-" prefixed
    generation_id (the OpenRouter convention that the tracker routes to
    the generation_id field).
    """
    config = load_config(openrouter_config_path)
    tracker = UsageTracker(
        log_path=tmp_path / "usage.jsonl",
        project_id="integration-test",
    )
    text, usage, gen_id = send_message(
        endpoint_name="openrouter-test",
        system="You are concise.",
        user="Reply with exactly: hello.",
        config=config,
        tracker=tracker,
        max_tokens=32,
    )
    tracker.close()
    assert text.strip()
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] > 0
    assert gen_id.startswith("gen-")
