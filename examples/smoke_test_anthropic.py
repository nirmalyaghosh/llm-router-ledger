"""
End-to-end smoke test against Anthropic direct (Messages API).

Runs every Workload defined in _smoke_test_common.py through
send_message() and writes paired llm_request / llm_response events
to the JSONL ledger for each call.

Uses the native AnthropicAdapter (not the OpenAI-compat path), so this
exercises the Messages API shape end-to-end.

Prerequisites:
- ANTHROPIC_API_KEY set in .env or shell environment.
- The [anthropic] extra installed: uv pip install llm-router-ledger[anthropic]
- llm_endpoints.yaml in the working directory with an endpoint
  matching --endpoint (default endpoint anthropic-claude-haiku in
  examples/llm_endpoints.example.yaml uses claude-haiku-4-5, the
  cheapest model in the current Claude line; switch to a different
  model id by editing the YAML).

Run from the project root:
    python examples/smoke_test_anthropic.py
"""
import argparse
import datetime
import os
import sys

from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

try:
    _LIBRARY_VERSION = version("llm-router-ledger")
except PackageNotFoundError:
    _LIBRARY_VERSION = "0.0.0+local"

_today = datetime.date.today().isoformat()

# Set env defaults before importing the library: load_dotenv() runs at
# library import time, after which any value already in .env wins over a
# later os.environ.setdefault() call.
os.environ.setdefault("LRL_RUN_TAG", "smoke")
os.environ.setdefault(
    "LRL_RUN_LABEL",
    f"{_LIBRARY_VERSION}-anthropic-verify-{_today}",
)

from _smoke_test_common import run_smoke_test  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test an Anthropic direct endpoint via "
            "llm-router-ledger using a set of representative LLM "
            "workloads."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="anthropic-claude-haiku",
        help=(
            "Endpoint name from llm_endpoints.yaml "
            "(default: anthropic-claude-haiku)."
        ),
    )
    args = parser.parse_args(argv)

    return run_smoke_test(
        endpoint_name=args.endpoint,
        log_path=Path("logs/anthropic_smoke_test_token_usage.jsonl"),
        project_id="anthropic-smoke-test",
    )


if __name__ == "__main__":
    sys.exit(main())
