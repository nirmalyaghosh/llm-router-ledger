"""
Agentic RAG over a synthetic annual report, recorded automatically.

This is option 2. Each agent gets its model from ledger_model(), which
builds the endpoint's model and wraps it, so every call records itself.
See posthoc.py for option 1, which records after the fact. The two
files differ only in the two integration functions below.

Option 2 has no granularity limit. purpose binds per model, and a
purpose_scope can vary it per call within one run, so a single agent
making several calls in different roles still attributes cost
correctly. It also records a call that raised, which option 1 cannot
see at all.

Two questions worth running:

- "What was Marine Systems revenue in FY2025?" flows through all five
  agents unclarified.
- "What was revenue in 2024?" trips the Disambiguator, which asks which
  segment and whether restated. The clarified run writes two
  disambiguation rows, told apart by metadata.

Prerequisites:
1. Install the extra: `uv pip install llm-router-ledger[pydantic-ai]`.
2. Copy `examples/llm_endpoints.example.yaml` to `llm_endpoints.yaml`
   in the working directory.
3. Set OPENROUTER_API_KEY.
4. Python 3.12 or later, which this library requires.
5. Run LM Studio's server on port 1234 with
   text-embedding-qwen3-embedding-0.6b loaded, and set
   LMSTUDIO_API_KEY to any value. Query embedding runs locally.
6. Build the corpus: `python prepare_corpus.py`. The vectors are
   not in the repository.

Embeddings are gated to providers verified end to end, so
provider: openai raises. The corpus vectors and every query are
embedded with lmstudio-embed-qwen3-0.6b; a different embedding
endpoint degrades similarity silently.

Every endpoint defaults to a free one. Small models are less
reliable at tool calling; --answerer moves the Answerer to a paid
one.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from pathlib import Path
from typing import Any

os.environ.setdefault("LRL_RUN_TAG", "agentic-rag-example")

from llm_router_ledger import (  # noqa: E402
    ProviderUnavailableError,
    UsageTracker,
)
from llm_router_ledger.integrations.pydantic_ai import (  # noqa: E402
    ledger_model,
)

from _common import (  # noqa: E402
    ANSWERER_ENDPOINT,
    CAPABLE_ENDPOINT,
    EMBED_ENDPOINT,
    configure_stdout,
    count_lines,
    print_ledger_rows,
    run_pipeline,
)

LOG_PATH = Path("logs/usage.jsonl")

DEFAULT_QUESTION = "What was Marine Systems revenue in FY2025?"


def main() -> int:
    """
    Run the pipeline and print the rows it wrote.
    """
    configure_stdout()

    parser = argparse.ArgumentParser(
        description=(
            "Agentic RAG recorded by the wrapping ledger model."
        ),
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--endpoint", default=EMBED_ENDPOINT)
    parser.add_argument(
        "--answerer",
        default=ANSWERER_ENDPOINT,
        help=(
            "Endpoint for the Answerer. Free by default;"
            f" {CAPABLE_ENDPOINT} is a paid endpoint that is more"
            " reliable at tool calling."
        ),
    )
    args = parser.parse_args()

    tracker = UsageTracker(
        log_path=LOG_PATH,
        project_id="agentic-rag-example",
        preview_length=120,
    )
    since = count_lines(LOG_PATH)

    def build_model(
        endpoint_name: str,
        purpose: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        """
        Build the model wrapped. It records as it goes.
        """
        return ledger_model(
            endpoint_name,
            tracker=tracker,
            purpose=purpose,
            metadata=metadata,
        )

    def after_run(
        result: Any,
        endpoint_name: str,
        purpose: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """
        Nothing to do. Each call recorded itself.
        """

    try:
        answer = asyncio.run(
            run_pipeline(
                args.question,
                build_model=build_model,
                after_run=after_run,
                tracker=tracker,
                embed_endpoint=args.endpoint,
                answerer_endpoint=args.answerer,
            )
        )
        print()
        print(answer)
        print_ledger_rows(LOG_PATH, since)
    except ProviderUnavailableError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print(
            "Check the endpoint is reachable. The default embedding"
            " endpoint needs LM Studio serving on port 1234."
        )
        return 1
    except Exception as exc:
        # A free endpoint shares its pool, so a 429 here is contention.
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        tracker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
