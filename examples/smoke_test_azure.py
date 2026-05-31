"""
End-to-end smoke test against Azure OpenAI.

Runs every Workload defined in _smoke_test_common.py through
send_message() and writes paired llm_request / llm_response events
to the JSONL ledger for each call.

Azure note: the `model` field in llm_endpoints.yaml carries the
Azure deployment name, not a model name.

Prerequisites:
- AZURE_OPENAI_API_KEY set in .env or shell environment.
- llm_endpoints.yaml in the working directory with an endpoint
  matching --endpoint (default endpoint azure-gpt-4.1-nano in
  examples/llm_endpoints.example.yaml uses gpt-4.1-nano as the
  deployment name; override base_url with your actual Azure
  resource).

Run from the project root:
    python examples/smoke_test_azure.py
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
    f"{_LIBRARY_VERSION}-azure-verify-{_today}",
)

from _smoke_test_common import run_smoke_test  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test an Azure OpenAI endpoint via llm-router-ledger "
            "using a set of representative LLM workloads."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="azure-gpt-4.1-nano",
        help=(
            "Endpoint name from llm_endpoints.yaml "
            "(default: azure-gpt-4.1-nano)."
        ),
    )
    args = parser.parse_args(argv)

    return run_smoke_test(
        endpoint_name=args.endpoint,
        log_path=Path("logs/azure_smoke_test_token_usage.jsonl"),
        project_id="azure-smoke-test",
    )


if __name__ == "__main__":
    sys.exit(main())
