"""
Chunk and embed a Markdown document into the format the examples read.

Run this before either example. They read corpus/prepared.json,
which is not in the repository; with no arguments this builds it
from corpus/synthetic_annual_report.md. Point it at your own
document with --source.

Embedding a corpus is a one off. Query embedding is live on every run,
inside the examples themselves.

Usage:

    python prepare_corpus.py
    python prepare_corpus.py --source my_report.md --out my.json

Prerequisites:
1. Copy `examples/llm_endpoints.example.yaml` to `llm_endpoints.yaml`
   in the working directory.
2. Run LM Studio's server on port 1234 with
   text-embedding-qwen3-embedding-0.6b loaded, and set
   LMSTUDIO_API_KEY to any value.

Embeddings are gated to providers verified end to end, so
provider: openai raises. The endpoint is recorded in prepared.json.
The examples compare it against their own embedding endpoint and
reject a corpus built with a different one. prepared.json is not
kept in version control.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from contextlib import suppress
from pathlib import Path

from dotenv import (
    find_dotenv,
    load_dotenv,
)

from llm_router_ledger import (
    ProviderUnavailableError,
    UsageTracker,
    create_embeddings,
)

load_dotenv(find_dotenv(usecwd=True))

HERE = Path(__file__).parent

DEFAULT_SOURCE = HERE / "corpus" / "synthetic_annual_report.md"

DEFAULT_OUT = HERE / "corpus" / "prepared.json"

# A second copy of the default in _common.py, so that this script
# does not import the examples and acquire the pydantic-ai
# dependency. load_corpus rejects a corpus whose endpoint differs
# from the query endpoint.
EMBED_ENDPOINT = os.environ.get(
    "LRL_RAG_EXAMPLE_EMBED_ENDPOINT",
    "lmstudio-embed-qwen3-0.6b",
)

# Target chunk size in words. Small enough that a chunk holds one
# figure under one heading, large enough to carry its qualifiers.
MIN_WORDS = 140

# A chunk below this is merged into the next one. A heading with a
# single short paragraph under it retrieves noise on its own.
MERGE_BELOW = 70


def _join_chunks(
    *,
    first: dict[str, str],
    second: dict[str, str],
) -> dict[str, str]:
    """
    Helper function used to merge two chunks, keeping both headings.
    """
    heading = first["heading"]
    if second["heading"] != heading:
        heading = f"{heading} / {second['heading']}"
    return {
        "heading": heading,
        "text": first["text"] + "\n\n" + second["text"],
    }


def _merge_small_chunks(
    chunks: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Helper function used to merge a chunk that is too short into the
    chunk after it, or into the chunk before it if it is the last one.

    Both headings are kept, joined with a slash.
    """
    merged: list[dict[str, str]] = []
    for chunk in chunks:
        if (
            merged
            and len(merged[-1]["text"].split()) < MERGE_BELOW
        ):
            chunk = _join_chunks(
                first=merged.pop(),
                second=chunk,
            )
        merged.append(chunk)
    if (
        len(merged) > 1
        and len(merged[-1]["text"].split()) < MERGE_BELOW
    ):
        last = merged.pop()
        previous = merged.pop()
        merged.append(
            _join_chunks(first=previous, second=last)
        )
    return merged


def _write_atomically(*, path: Path, text: str) -> None:
    """
    Helper function used to replace a file in one step.

    A direct write truncates the existing file before the new
    content arrives, so an interrupted run leaves neither the old
    corpus nor a complete new one.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            tmp.unlink()
        raise


def chunk_markdown(text: str) -> list[dict[str, str]]:
    """
    Split a Markdown document into passages, each under one heading.

    Paragraphs accumulate to MIN_WORDS. A heading starts a new chunk.
    A long paragraph is never split: a figure separated from its
    qualifier retrieves worse than a long chunk.
    """
    chunks: list[dict[str, str]] = []
    heading = "Introduction"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        chunks.append(
            {
                "heading": heading,
                "text": "\n\n".join(buffer).strip(),
            }
        )
        buffer.clear()

    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            flush()
            heading = block.lstrip("#").strip()
            continue
        buffer.append(block)
        if len(" ".join(buffer).split()) >= MIN_WORDS:
            flush()
    flush()
    return _merge_small_chunks(chunks)


def main() -> int:
    """
    Chunk the source document, embed every chunk, and write the result.
    """
    parser = argparse.ArgumentParser(
        description="Prepare a Markdown corpus for the RAG examples.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", default=EMBED_ENDPOINT)
    args = parser.parse_args()

    if not args.source.exists():
        print(f"No document at {args.source}")
        return 1

    chunks = chunk_markdown(
        args.source.read_text(encoding="utf-8"),
    )
    print(f"{len(chunks)} chunks from {args.source.name}")

    # After load_dotenv, so shell wins, then .env, then this.
    os.environ.setdefault("LRL_RUN_TAG", "agentic-rag-example")
    tracker = UsageTracker(
        log_path=Path("logs/usage.jsonl"),
        project_id="agentic-rag-example",
    )
    try:
        result = create_embeddings(
            endpoint_name=args.endpoint,
            texts=[c["text"] for c in chunks],
            tracker=tracker,
            purpose="corpus-index",
        )
    except ProviderUnavailableError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print(
            "Check the endpoint is reachable. The default embedding"
            " endpoint needs LM Studio serving on port 1234."
        )
        return 1
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        tracker.close()

    vectors = list(result.vectors)
    if len(vectors) != len(chunks):
        print(
            f"Embedded {len(vectors)} of {len(chunks)} chunks"
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(
        path=args.out,
        text=json.dumps(
            {
                "source": args.source.name,
                "endpoint": args.endpoint,
                "dimensions": len(vectors[0]),
                "chunks": [
                    {**chunk, "vector": list(vector)}
                    for chunk, vector in zip(
                        chunks, vectors, strict=True
                    )
                ],
            },
            indent=1,
        ),
    )
    print(
        f"wrote {args.out} "
        f"({len(vectors)} vectors, {len(vectors[0])} dims)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
