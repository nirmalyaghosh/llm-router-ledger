"""
End-to-end smoke test of create_embeddings() against OpenRouter.

Embeds a small corpus through create_embeddings() and appends paired
llm_request / llm_response events to the shared OpenRouter ledger,
logs/openrouter_smoke_test_token_usage.jsonl, both stamped modality
"embedding" so they are distinguishable from the send_message()
entries smoke_test_openrouter.py writes to the same file. The
response event carries the provider's own reported cost under
usage_details, alongside the vector count and width, so a run leaves
a durable record of what the call cost.

Prerequisites:
- OPENROUTER_API_KEY set in .env or shell environment.
- llm_endpoints.yaml in the working directory with an endpoint
  matching --endpoint. The default endpoint
  openrouter-embed-nemotron-3-1b-free in
  examples/llm_endpoints.example.yaml uses
  nvidia/nemotron-3-embed-1b:free, so a default run is free; override
  with --endpoint to exercise a paid model.

The default corpus is the three passages in _smoke_test_common.py.
Pass --input-file to embed your own, one text per line.

Run from the project root:
    python examples/smoke_test_openrouter_embeddings.py
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
    f"{_LIBRARY_VERSION}-openrouter-embeddings-verify-{_today}",
)

from _smoke_test_common import (  # noqa: E402
    load_embedding_texts,
    run_embedding_smoke_test,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test an OpenRouter embedding endpoint via "
            "llm-router-ledger, recording tokens and cost to the "
            "JSONL ledger."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="openrouter-embed-nemotron-3-1b-free",
        help=(
            "Endpoint name from llm_endpoints.yaml (default: "
            "openrouter-embed-nemotron-3-1b-free, which is free)."
        ),
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Path to a text file, one text to embed per line. "
            "Blank lines are skipped. Defaults to the built-in "
            "sample corpus. Note that cost scales with input, so a "
            "large file against a paid endpoint bills accordingly."
        ),
    )
    args = parser.parse_args(argv)

    texts = (
        load_embedding_texts(Path(args.input_file))
        if args.input_file
        else None
    )

    return run_embedding_smoke_test(
        endpoint_name=args.endpoint,
        # Same ledger and project_id as smoke_test_openrouter.py: one
        # file per provider, not per capability. The tracker opens in
        # append mode, so text and embedding runs interleave safely and
        # a reader separates them on the modality field.
        log_path=Path(
            "logs/openrouter_smoke_test_token_usage.jsonl",
        ),
        project_id="openrouter-smoke-test",
        texts=texts,
    )


if __name__ == "__main__":
    sys.exit(main())
