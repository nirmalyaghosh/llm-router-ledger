"""
End-to-end smoke test of create_embeddings() against local Ollama.

Embeds a small corpus through create_embeddings() and appends paired
llm_request / llm_response events to the shared ledger at
_smoke_test_common.LEDGER_PATH, both stamped modality "embedding" so
they are distinguishable from the send_message() entries every other
smoke test writes to the same file.

Nothing is billed, so unlike the OpenRouter counterpart the ledger
rows carry no cost: Ollama reports no cost field, and the endpoint's
configured rate is 0.00. Tokens are still recorded, which keeps a
local run comparable with a hosted one on the same corpus.

Ollama also returns no response id, so these rows carry an empty
provider_response_id. There is no provider invoice to reconcile them
against, so nothing is lost.

Prerequisites:
- Ollama running on http://localhost:11434
- The embedding model pulled: ollama pull qwen3-embedding:0.6b
- llm_endpoints.yaml in the working directory with an endpoint
  matching --endpoint (default ollama-embed-qwen3-0.6b).

The default corpus is the five passages in _smoke_test_common.py.
Pass --input-file to embed your own, one text per line.

Run from the project root:
    python examples/smoke_test_ollama_embeddings.py
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
os.environ.setdefault("OLLAMA_API_KEY", "ollama")
os.environ.setdefault("LRL_RUN_TAG", "smoke")
os.environ.setdefault(
    "LRL_RUN_LABEL",
    f"{_LIBRARY_VERSION}-ollama-embeddings-verify-{_today}",
)

from _smoke_test_common import (  # noqa: E402
    load_embedding_texts,
    run_embedding_smoke_test,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test a local Ollama embedding endpoint via "
            "llm-router-ledger, recording tokens to the JSONL ledger."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="ollama-embed-qwen3-0.6b",
        help=(
            "Endpoint name from llm_endpoints.yaml "
            "(default: ollama-embed-qwen3-0.6b)."
        ),
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Path to a text file, one text to embed per line. "
            "Blank lines are skipped. Defaults to the built-in "
            "sample corpus."
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
        project_id="ollama-smoke-test",
        texts=texts,
    )


if __name__ == "__main__":
    sys.exit(main())
