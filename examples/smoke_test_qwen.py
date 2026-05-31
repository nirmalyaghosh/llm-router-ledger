"""
End-to-end smoke test against Qwen direct (Alibaba Cloud DashScope).

Runs every Workload defined in _smoke_test_common.py through
send_message() and writes paired llm_request / llm_response events
to the JSONL ledger for each call.

Prerequisites:
- QWEN_API_KEY set in .env or shell environment.
- The model named in the endpoint must be enabled in your DashScope
  workspace (console: enable per-model before any key can call it;
  a 403 Model.AccessDenied means the model is not enabled, not that
  the key is invalid).
- llm_endpoints.yaml in the working directory with an endpoint
  matching --endpoint. The default endpoint qwen-direct in
  examples/llm_endpoints.example.yaml uses qwen3.5-plus against the
  DashScope international endpoint
  (https://dashscope-intl.aliyuncs.com/compatible-mode/v1), which is
  the exact pairing verified in 0.1.2.

DashScope exposes an OpenAI-compatible API per
https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope.

Run from the project root:
    python examples/smoke_test_qwen.py
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
    f"{_LIBRARY_VERSION}-qwen-verify-{_today}",
)

from _smoke_test_common import run_smoke_test  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test a Qwen direct (DashScope) endpoint via "
            "llm-router-ledger using a set of representative LLM "
            "workloads."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="qwen-direct",
        help=(
            "Endpoint name from llm_endpoints.yaml "
            "(default: qwen-direct)."
        ),
    )
    args = parser.parse_args(argv)

    return run_smoke_test(
        endpoint_name=args.endpoint,
        log_path=Path("logs/qwen_smoke_test_token_usage.jsonl"),
        project_id="qwen-smoke-test",
    )


if __name__ == "__main__":
    sys.exit(main())
