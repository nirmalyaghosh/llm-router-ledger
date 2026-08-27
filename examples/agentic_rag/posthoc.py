"""
Agentic RAG over a synthetic annual report, recorded after each run.

This is option 1. Agents are built normally, and the finished message
list is handed to UsageTracker.record_run() once each agent returns.
See ledger_model.py for option 2, which records automatically. The two
files differ only in the two integration functions below.

Option 1 has a granularity limit. A finished message list carries no
record of why each call was made, so record_run() stamps one purpose
across a whole run and any retry inside that run inherits it. Here each
agent is its own run, so each gets its own purpose. Reach for option 2
when one run needs several.

A run that raises produces no rows at all, unlike send_message(), which
logs the request before the call.

Two questions worth running:

- "What was Marine Systems revenue in FY2025?" flows through all five
  agents unclarified.
- "What was revenue in 2024?" trips the Disambiguator, which asks which
  segment and whether restated. The clarified run writes two
  disambiguation rows, told apart by metadata.

Prerequisites:
1. Install the extra: `uv pip install llm-router-ledger[pydantic-ai]`.
2. Copy `examples/llm_endpoints.example.yaml` to `llm_endpoints.yaml`
   in the working directory. That file and logs/usage.jsonl are both
   resolved from there, so run every command from the same directory.
3. Set OPENROUTER_API_KEY.
4. Python 3.12 or later, which this library requires.
5. Run LM Studio's server on port 1234 with
   text-embedding-qwen3-embedding-0.6b loaded, and set
   LMSTUDIO_API_KEY to any value. Query embedding runs locally.
6. Build the corpus: `python prepare_corpus.py`. The vectors are
   not in the repository.

Embeddings are gated to providers verified end to end, so
provider: openai raises. The corpus vectors and every query are
embedded with lmstudio-embed-qwen3-0.6b. A corpus built with a
different endpoint is rejected at load time.

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

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import (
    OpenAIProvider,
)

from llm_router_ledger import (
    ProviderUnavailableError,
    UsageTracker,
)

from _common import (
    ANSWERER_ENDPOINT,
    CAPABLE_ENDPOINT,
    EMBED_ENDPOINT,
    configure_stdout,
    count_lines,
    endpoint_config,
    get_provider_name,
    print_ledger_rows,
    run_pipeline,
)

DEFAULT_QUESTION = "What was Marine Systems revenue in FY2025?"

LOG_PATH = Path("logs/usage.jsonl")


def main() -> int:
    """
    Run the pipeline and print the rows it wrote.
    """
    configure_stdout()

    parser = argparse.ArgumentParser(
        description=(
            "Agentic RAG recorded with record_run() after each agent."
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

    # After _common's load_dotenv, so shell wins, then .env, then
    # this. UsageTracker reads LRL_RUN_TAG when constructed.
    os.environ.setdefault("LRL_RUN_TAG", "agentic-rag-example")
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
        Build the model directly. Nothing records yet.
        """
        endpoint = endpoint_config(endpoint_name)
        return OpenAIChatModel(
            endpoint.model,
            provider=OpenAIProvider(
                base_url=endpoint.base_url,
                api_key=endpoint.api_key,
            ),
        )

    def after_run(
        result: Any,
        endpoint_name: str,
        purpose: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """
        Hand the finished message list to the tracker.
        """
        tracker.record_run(
            result.all_messages(),
            purpose=purpose,
            provider=get_provider_name(endpoint_name),
            metadata=metadata,
        )

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
