"""
End-to-end smoke test against a local LM Studio server.

Runs every Workload defined in _smoke_test_common.py through
send_message() and writes paired llm_request / llm_response events
to the JSONL ledger for each call.

Prerequisites:
- LM Studio running with its local server started (Developer tab,
  Status: Running), listening on http://localhost:1234
- A model loaded in LM Studio whose id matches the model field of
  your endpoint in llm_endpoints.yaml. The id is the one reported by
  GET http://localhost:1234/v1/models, which is the repository-style
  path rather than the display name shown in the UI.
- llm_endpoints.yaml in the working directory.

Run from the project root:
    python examples/smoke_test_lmstudio.py
    python examples/smoke_test_lmstudio.py --endpoint local-lmstudio
"""
import argparse
import datetime
import os
import sys

from importlib.metadata import (
    PackageNotFoundError,
    version,
)

try:
    _LIBRARY_VERSION = version("llm-router-ledger")
except PackageNotFoundError:
    _LIBRARY_VERSION = "0.0.0+local"

_today = datetime.date.today().isoformat()

# Set env defaults before importing the library: load_dotenv() runs at
# library import time, after which any value already in .env wins over a
# later os.environ.setdefault() call.
os.environ.setdefault("LMSTUDIO_API_KEY", "lmstudio")
os.environ.setdefault("LRL_RUN_TAG", "smoke")
os.environ.setdefault(
    "LRL_RUN_LABEL",
    f"{_LIBRARY_VERSION}-lmstudio-verify-{_today}",
)

from _smoke_test_common import run_smoke_test  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test an LM Studio endpoint via llm-router-ledger "
            "using a set of representative LLM workloads."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="local-lmstudio",
        help=(
            "Endpoint name from llm_endpoints.yaml "
            "(default: local-lmstudio)."
        ),
    )
    args = parser.parse_args(argv)

    return run_smoke_test(
        endpoint_name=args.endpoint,
        project_id="lmstudio-smoke-test",
    )


if __name__ == "__main__":
    sys.exit(main())
