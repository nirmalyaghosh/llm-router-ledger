"""
Result objects returned by the public entry points.

send_message and create_embeddings returned a bare (payload, usage,
generation_id) tuple through 0.1.x. Reasoning traces, tool calls, and
finish reasons had nowhere to go in a fixed-width tuple, so 0.2.0 moves
to these result objects instead.

Both are NamedTuple subclasses rather than plain classes so that
existing positional-unpacking callers (text, usage, gen_id =
send_message(...)) keep working through 0.2.x instead of breaking
immediately; unpacking logs a deprecation warning that points at the
attribute-access form. This is a soft-deprecation window, not a
permanent shim: 0.3.0 adds the fields this migration exists to make
room for (a reasoning trace, tool calls), at which point a 3-element
unpack raises ValueError regardless, since the tuple is then the wrong
width. Prefer attribute access (result.text, result.usage,
result.generation_id) in new code.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
    NamedTuple,
)

from llm_router_ledger._logger import get_logger


logger = get_logger(__name__)


class ChatResult(NamedTuple):
    """
    Return value of send_message.

    text is the response content, "" for a turn that returned only tool
    calls (see usage_details["completion_tool_call_count"]). usage
    carries the normalised prompt_tokens, completion_tokens, and
    total_tokens keys; generation_id is the provider's response
    identifier, or "" if the provider does not return one.
    """

    text: str
    usage: dict[str, Any]
    generation_id: str

    def __iter__(self) -> Iterator[Any]:
        logger.warning(
            "ChatResult unpacked as a tuple (e.g. `text, usage,"
            " gen_id = send_message(...)`); this is deprecated and"
            " will stop working in 0.3.0, when new fields are added"
            " to ChatResult. Use attribute access instead:"
            " result.text, result.usage, result.generation_id.",
            stacklevel=2,
        )
        return tuple.__iter__(self)


class EmbeddingResult(NamedTuple):
    """
    Return value of create_embeddings.

    vectors is ordered to match the input texts, one vector per input.
    usage carries the normalised prompt_tokens, completion_tokens, and
    total_tokens keys; generation_id is the provider's response
    identifier, or "" if the provider does not return one.
    """

    vectors: list[list[float]]
    usage: dict[str, Any]
    generation_id: str

    def __iter__(self) -> Iterator[Any]:
        logger.warning(
            "EmbeddingResult unpacked as a tuple (e.g. `vectors,"
            " usage, gen_id = create_embeddings(...)`); this is"
            " deprecated and will stop working in 0.3.0, when new"
            " fields are added to EmbeddingResult. Use attribute"
            " access instead: result.vectors, result.usage,"
            " result.generation_id.",
            stacklevel=2,
        )
        return tuple.__iter__(self)
