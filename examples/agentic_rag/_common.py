"""
Shared pipeline for the two agentic RAG examples.

Both run this pipeline and differ only in how the ledger is written.
The diff between posthoc.py and ledger_model.py is the teaching
device.

Five agents run in order over a synthetic annual report: Intent
Classifier, Disambiguator, Query Rewriter, Query Planner, and an
Answerer holding a search tool.

The query is embedded after clarification, through this library's own
create_embeddings(). That call is the only place where this library's
path runs alongside Pydantic AI's, both writing to the same JSONL.

Embeddings are gated to verified providers, so provider: openai
raises. prepare_corpus.py builds the corpus vectors and the
examples embed each query at run time, both through
lmstudio-embed-qwen3-0.6b, which runs locally, so LM Studio must
be serving on port 1234 with the embedding model loaded. Both
must use the same endpoint, which load_corpus checks.
"""

from __future__ import annotations

import json
import math
import os
import sys

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import (
    find_dotenv,
    load_dotenv,
)
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from llm_router_ledger import (
    EndpointNotFoundError,
    UsageTracker,
    create_embeddings,
    load_config,
)

load_dotenv(find_dotenv(usecwd=True))

# Agents 1 to 4 do structured output with no tools.
# (Intent Classifier, Disambiguator, Query Rewriter, Query Planner)
# Only the Answerer calls a tool.
#
# CHEAP_ENDPOINT and ANSWERER_ENDPOINT both default to free ones.
# Small models are less reliable at tool calling, so CAPABLE_ENDPOINT
# names a paid one, for a caller that wants the Answerer on it via
# run_pipeline's answerer_endpoint.
#
# Each is overridable by the matching LRL_RAG_EXAMPLE_ variable. The
# prefix keeps them clear of the library's own LRL_ variables.
# The values are endpoint names from llm_endpoints.yaml, not
# model ids.
ANSWERER_ENDPOINT = os.environ.get(
    "LRL_RAG_EXAMPLE_ANSWERER_ENDPOINT",
    "openrouter-nemotron-3.5-lightning-free",
)
CAPABLE_ENDPOINT = os.environ.get(
    "LRL_RAG_EXAMPLE_CAPABLE_ENDPOINT",
    "openrouter-mimo-v2.5",
)
CHEAP_ENDPOINT = os.environ.get(
    "LRL_RAG_EXAMPLE_CHEAP_ENDPOINT",
    "openrouter-nemotron-3.5-lightning-free",
)
EMBED_ENDPOINT = os.environ.get(
    "LRL_RAG_EXAMPLE_EMBED_ENDPOINT",
    "lmstudio-embed-qwen3-0.6b",
)

CORPUS_PATH = Path(__file__).parent / "corpus" / "prepared.json"

TOP_K = 4

# Smaller / general-purpose models occasionally produce malformed
# responses on the first attempt. Therefore retries are required.
# A retry keeps the purpose and carries the attempt count in metadata
# (not appended to purpose) to preserve role-based cost aggregation.
MAX_ATTEMPTS = 3


class Ambiguity(BaseModel):
    """
    Whether the question can be answered as asked.
    """

    ambiguous: bool
    clarifying_question: str = Field(
        default="",
        description="Asked only when ambiguous.",
    )


@dataclass(frozen=True)
class Chunk:
    """
    One retrievable passage of the corpus.
    """

    heading: str
    text: str
    vector: tuple[float, ...]


class Intent(BaseModel):
    """
    What kind of question was asked.
    """

    kind: Literal["factual", "comparative", "other"]
    reason: str = Field(description="One short sentence.")


class Plan(BaseModel):
    """
    The search terms to retrieve_chunks on.
    """

    terms: list[str]


class Rewrite(BaseModel):
    """
    The question restated for retrieval.
    """

    query: str


# (result, endpoint_name, purpose, metadata) -> None.
AfterRun = Callable[
    [Any, str, str, dict[str, Any] | None],
    None,
]

# (endpoint_name, purpose, metadata) -> a Pydantic AI model.
BuildModel = Callable[
    [str, str, dict[str, Any] | None],
    Any,
]


@lru_cache(maxsize=1)
def _config() -> Any:
    """
    Helper function used to read llm_endpoints.yaml once.

    Cached because the pipeline resolves an endpoint per agent.
    """
    return load_config()


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """
    Helper function used to score one chunk against the query.

    Written out rather than taken from numpy: an example should not add
    a dependency for a dot product.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(
        sum(y * y for y in b)
    )
    return dot / norm if norm else 0.0


def _format_passages(chunks: list[Chunk]) -> str:
    """
    Helper function used to format chunks for the model.
    """
    return "\n\n".join(
        f"## {chunk.heading}\n{chunk.text}"
        for chunk in chunks
    )


async def _run_agent(
    *,
    build_model: BuildModel,
    after_run: AfterRun,
    endpoint: str,
    purpose: str,
    instructions: str,
    question: str,
    output_type: type[BaseModel],
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Helper function used to run one structured-output agent.

    The result goes to whichever recording strategy the caller supplied.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        stamp = _stamp_attempt(
            attempt=attempt,
            metadata=metadata,
        )
        agent: Agent[None, Any] = Agent(
            build_model(endpoint, purpose, stamp),
            instructions=instructions,
            output_type=output_type,
        )
        try:
            result = await agent.run(question)
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            continue
        after_run(result, endpoint, purpose, stamp)
        return result.output
    raise AssertionError("unreachable")


def _stamp_attempt(
    *,
    attempt: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Helper function used to add the attempt number to a call's
    metadata, from the second attempt onwards.
    """
    meta = dict(metadata or {})
    if attempt > 1:
        meta["attempt"] = attempt
    return meta or None


def configure_stdout() -> None:
    """
    Helper function used to keep model output printable when stdout is
    not UTF-8.

    Redirecting or piping on Windows gives stdout the cp1252 locale
    encoding, and a model answer carrying a narrow no-break space then
    raises UnicodeEncodeError at print time, after every call in the
    run has been paid for. A UTF-8 stdout, the normal case on Linux
    and macOS, is left alone.
    """
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower().replace("-", "") == "utf8":
        return
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def count_lines(log_path: Path) -> int:
    """
    Helper function used to mark where this run's rows begin.
    """
    if not log_path.exists():
        return 0
    return len(
        log_path.read_text(encoding="utf-8").splitlines()
    )


def embed_query(
    question: str,
    *,
    tracker: UsageTracker,
    endpoint: str = EMBED_ENDPOINT,
) -> list[float]:
    """
    Helper function used to embed a query through this library.

    Writes a query-embed row to the same ledger the agents write to.
    """
    result = create_embeddings(
        endpoint_name=endpoint,
        texts=[question],
        tracker=tracker,
        purpose="query-embed",
    )
    return list(result.vectors[0])


def endpoint_config(endpoint_name: str) -> Any:
    """
    Helper function used to look up one endpoint.

    Raises EndpointNotFoundError rather than KeyError, matching what
    the library raises for an unknown name.
    """
    config = _config()
    if endpoint_name not in config.endpoints:
        raise EndpointNotFoundError(
            f"Endpoint '{endpoint_name}' not found in config"
        )
    return config.endpoints[endpoint_name]


def get_provider_name(endpoint_name: str) -> str:
    """
    Helper function used to read an endpoint's provider.

    record_run needs it explicitly. The name on the messages is the
    framework's provider id, which is openai for every OpenAI-compatible
    server.
    """
    return endpoint_config(endpoint_name).provider


def load_corpus(*, endpoint: str) -> list[Chunk]:
    """
    Helper function used to read the prepared corpus.

    prepare_corpus.py writes it. Run that first: the vectors
    are not in the repository.

    The query endpoint must match the one the corpus was built
    with. Otherwise _cosine_similarity zips vectors from two
    different spaces and returns a confident wrong answer.
    """
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"No prepared corpus at {CORPUS_PATH}."
            " Run prepare_corpus.py first."
        )
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    built_with = raw.get("endpoint")
    if built_with != endpoint:
        named = built_with or "an unrecorded endpoint"
        raise SystemExit(
            f"The corpus was built with {named}, and the query"
            f" would be embedded with {endpoint}. Rebuild it"
            f" with prepare_corpus.py --endpoint {endpoint}."
        )
    return [
        Chunk(
            heading=chunk["heading"],
            text=chunk["text"],
            vector=tuple(chunk["vector"]),
        )
        for chunk in raw["chunks"]
    ]


def print_ledger_rows(log_path: Path, since: int) -> None:
    """
    Print the rows this run wrote, grouped by purpose.

    Without it both examples look like ordinary Pydantic AI usage.
    """
    lines = log_path.read_text(
        encoding="utf-8",
    ).splitlines()[since:]
    rows: dict[str, list[dict[str, Any]]] = {}
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "llm_response":
            continue
        rows.setdefault(event.get("purpose", ""), []).append(event)

    print()
    print("ledger rows by purpose:")
    for purpose, events in rows.items():
        print(f"  {purpose}")
        for event in events:
            usage = event.get("usage", {})
            note = event.get("metadata") or ""
            print(
                f"    {event.get('provider', ''):11}"
                f" {event.get('model', ''):34}"
                f" {usage.get('total_tokens', 0):>6}"
                f"  {event.get('generation_id') or ''}"
                f" {note}".rstrip()
            )


def retrieve_chunks(
    *,
    query_vector: list[float],
    chunks: list[Chunk],
    top_k: int = TOP_K,
) -> list[Chunk]:
    """
    Helper function used to return the closest chunks to the query.
    """
    query = tuple(query_vector)
    scored = sorted(
        chunks,
        key=lambda c: _cosine_similarity(query, c.vector),
        reverse=True,
    )
    return scored[:top_k]


async def run_pipeline(
    question: str,
    *,
    build_model: BuildModel,
    after_run: AfterRun,
    tracker: UsageTracker,
    embed_endpoint: str = EMBED_ENDPOINT,
    answerer_endpoint: str = ANSWERER_ENDPOINT,
    ask_user: Callable[[str], str] = input,
) -> str:
    """
    Run all five agents and return the answer.

    ask_user defaults to the builtin input. A disambiguator that cannot
    ask is not a disambiguator, so there is no scripted answer and no
    early exit.
    """
    chunks = load_corpus(endpoint=embed_endpoint)

    intent = await _run_agent(
        build_model=build_model,
        after_run=after_run,
        endpoint=CHEAP_ENDPOINT,
        purpose="intent-classification",
        instructions=(
            "Classify the reader's question about a company annual"
            " report. Answer only with the structured output."
        ),
        question=question,
        output_type=Intent,
    )
    print(f"  intent: {intent.kind} ({intent.reason})")

    verdict = await _run_agent(
        build_model=build_model,
        after_run=after_run,
        endpoint=CHEAP_ENDPOINT,
        purpose="disambiguation",
        instructions=(
            "The report covers two segments, Marine Systems and"
            " Industrial Coatings, and two years, FY2024 and"
            " FY2025. A question is answerable when it names a"
            " segment or the group, and a year. A question about"
            " FY2024 must also say whether it wants the restated"
            " or the originally reported figure, because FY2024"
            " was restated. Set ambiguous only when one of those"
            " is missing, and then ask for the missing one. Never"
            " ask the reader for the answer."
        ),
        question=question,
        output_type=Ambiguity,
        metadata={"pass": 1},
    )

    if verdict.ambiguous:
        # A clarified run produces two disambiguation rows, told apart
        # by metadata rather than by a suffixed purpose. Suffixing
        # would fragment cost by role and force consumers to prefix
        # match.
        answer = ask_user(f"  {verdict.clarifying_question} ")
        question = f"{question} ({answer.strip()})"
        verdict = await _run_agent(
            build_model=build_model,
            after_run=after_run,
            endpoint=CHEAP_ENDPOINT,
            purpose="disambiguation",
            instructions=(
                "Confirm the clarified question now has one answer."
            ),
            question=question,
            output_type=Ambiguity,
            metadata={"pass": 2},
        )

    rewritten = await _run_agent(
        build_model=build_model,
        after_run=after_run,
        endpoint=CHEAP_ENDPOINT,
        purpose="query-rewrite",
        instructions=(
            "Restate the question as a retrieval query over an annual"
            " report. Keep every segment, year and restatement"
            " qualifier."
        ),
        question=question,
        output_type=Rewrite,
    )
    print(f"  rewritten: {rewritten.query}")

    plan = await _run_agent(
        build_model=build_model,
        after_run=after_run,
        endpoint=CHEAP_ENDPOINT,
        purpose="query-planning",
        instructions=(
            "List the search terms that would find the answer in an"
            " annual report. Three at most."
        ),
        question=rewritten.query,
        output_type=Plan,
    )
    print(f"  plan: {', '.join(plan.terms)}")

    # Embedded after clarification, so the vector reflects what the
    # reader actually meant.
    vector = embed_query(
        rewritten.query,
        tracker=tracker,
        endpoint=embed_endpoint,
    )
    retrieved = retrieve_chunks(
        query_vector=vector,
        chunks=chunks,
    )

    prompt = f"{rewritten.query}\n\n{_format_passages(retrieved)}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        stamp = _stamp_attempt(attempt=attempt)
        seen: set[Chunk] = set(retrieved)

        def search(query: str) -> str:
            """
            Search the report for passages not already shown.
            """
            fresh = [c for c in chunks if c not in seen]
            if not fresh:
                return "No further passages in the report."
            found = retrieve_chunks(
                query_vector=embed_query(
                    query,
                    tracker=tracker,
                    endpoint=embed_endpoint,
                ),
                chunks=fresh,
            )
            seen.update(found)
            return _format_passages(found)

        # Rebuilt each attempt so the stamp reaches build_model. A
        # strategy that binds metadata when the model is built
        # cannot see it otherwise. _run_agent does the same.
        answerer: Agent[None, str] = Agent(
            build_model(
                answerer_endpoint,
                "answer-synthesis",
                stamp,
            ),
            instructions=(
                "Answer from the passages alone. Quote the"
                " figure and name the heading it came from."
                " Call search again if they do not contain the"
                " answer; each call returns passages you have not"
                " yet seen. If they still do not, say so."
            ),
            tools=[search],
        )
        try:
            result = await answerer.run(prompt)
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            continue
        after_run(
            result,
            answerer_endpoint,
            "answer-synthesis",
            stamp,
        )
        return result.output
    raise AssertionError("unreachable")
