"""
Result objects returned by the public entry points.

send_message and create_embeddings returned a bare (payload, usage,
generation_id) tuple through 0.1.x. Reasoning traces, tool calls, and
finish reasons had nowhere to go in a fixed-width tuple, so 0.2.0
moved to these result objects instead.

Both were NamedTuple subclasses through 0.2.x so that existing
positional-unpacking callers (text, usage, gen_id =
send_message(...)) kept working during that soft-deprecation window;
unpacking logged a warning that pointed at the attribute-access form.
That window is now closed: both are frozen dataclasses, unpacking
raises TypeError, and attribute access (result.text, result.usage,
result.generation_id) is the only supported form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatResult:
    """
    Return value of send_message.

    text is the response content, "" for a turn that returned only tool
    calls (see usage_details["completion_tool_call_count"]). usage is
    the provider's usage mapping as reported, unsplit: the normalised
    prompt_tokens, completion_tokens, and total_tokens keys plus
    whatever else the provider returned (e.g. cost, is_byok,
    upstream_provider, completion_reasoning_tokens, unmapped). The
    split into those three keys plus usage_details happens only when
    UsageTracker writes the ledger. generation_id is the provider's
    response identifier, or "" if the provider does not return one.
    """

    text: str
    usage: dict[str, Any]
    generation_id: str


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Return value of create_embeddings.

    vectors is ordered to match the input texts, one vector per input.
    usage is the provider's usage mapping as reported, unsplit: the
    normalised prompt_tokens, completion_tokens, and total_tokens keys
    plus whatever else the provider returned (e.g. cost, is_byok,
    upstream_provider, dimensions, embedding_count). The split into
    those three keys plus usage_details happens only when
    UsageTracker writes the ledger. generation_id is the provider's
    response identifier, or "" if the provider does not return one.
    """

    vectors: list[list[float]]
    usage: dict[str, Any]
    generation_id: str
