"""
Shared pytest fixtures for the llm-router-ledger test suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest


SAMPLE_YAML = """\
defaults:
  timeout_seconds: 30
  max_retries: 2

endpoints:
  ollama-local:
    provider: local-openai-compat
    model: llama3.1
    api_key_env: OLLAMA_API_KEY
    base_url: http://localhost:11434/v1
    context_window: 8192

  openrouter-test:
    provider: openrouter
    model: openai/gpt-4.1-nano
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    cost:
      input_per_1m: 0.15
      output_per_1m: 0.60
      pricing_url: https://example.com/pricing
      pricing_checked: 2026-05-01

roles:
  default:
    quick: ollama-local
    capable: openrouter-test
"""


@pytest.fixture
def sample_yaml_file(tmp_path: Path, sample_yaml_text: str) -> Path:
    """
    Write the sample YAML to a per-test tmp file and return its Path.
    """
    path = tmp_path / "llm_endpoints.yaml"
    path.write_text(sample_yaml_text, encoding="utf-8")
    return path


@pytest.fixture
def sample_yaml_text() -> str:
    """
    Return the canonical sample YAML string used across config tests.
    """
    return SAMPLE_YAML


@pytest.fixture
def tmp_log_path(tmp_path: Path) -> Path:
    """
    Return a per-test JSONL log path. The parent directory is left
    non-existent so callers can verify the auto-mkdir behaviour.
    """
    return tmp_path / "logs" / "usage.jsonl"
