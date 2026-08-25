"""
Internal helpers for reshaping usage dicts.

split_usage divides a provider's usage mapping into the three
normalised token keys the ledger's usage block holds and the remainder,
which belongs in usage_details. Both entry points in the dispatcher and
UsageTracker.record_response use it, so a call the library made and a
call it only recorded produce the same row shape.

normalise_token_keys puts a usage mapping into the vocabulary the two
of them expect, for the one case where it may not already be in it: a
mapping handed to record_response by a caller, rather than produced by
an adapter.

map_request_usage translates a Pydantic AI RequestUsage into that same
mapping. It lives here, next to the OpenAI-compatible key names it
mirrors, and reads the object by duck typing so the core package gains
no dependency on pydantic-ai.
"""

from __future__ import annotations

from typing import Any


# The finish reasons meaning a turn ran to completion, so nothing is
# recorded for it. The union of what the two adapters each treat as
# ordinary: openai_compat.ORDINARY_FINISH_REASON ("stop") and
# anthropic_native.ORDINARY_STOP_REASON ("end_turn"). record_run sees
# whichever of the two a framework passed through, so it has to know
# both.
ORDINARY_FINISH_REASONS = frozenset({
    "end_turn",
    "stop",
})

TOKEN_USAGE_KEYS = frozenset({
    "completion_tokens",
    "prompt_tokens",
    "total_tokens",
})

# The other name each token key goes by. Anthropic reports input_tokens
# and output_tokens, and so does anything modelled on it.
_TOKEN_ALIASES = (
    ("input_tokens", "prompt_tokens"),
    ("output_tokens", "completion_tokens"),
)

# RequestUsage attribute -> the key openai_compat already writes for the
# same quantity, so a row recorded through Pydantic AI reconciles
# against one send_message wrote. cost is deliberately absent: see
# map_request_usage.
_REQUEST_USAGE_MAP = (
    ("cache_audio_read_tokens", "prompt_cache_audio_read_tokens"),
    ("cache_read_tokens", "prompt_cached_tokens"),
    ("cache_write_tokens", "prompt_cache_write_tokens"),
    ("input_audio_tokens", "prompt_audio_tokens"),
    ("output_audio_tokens", "completion_audio_tokens"),
)

# Keys Pydantic AI collects into RequestUsage.details unprefixed, having
# flattened them out of completion_tokens_details. openai_compat writes
# the same quantities with a completion_ prefix, so they are prefixed
# here too. Anything else in details is provider-specific and goes to
# unmapped, the same contract the adapters follow.
_DETAIL_KEYS = frozenset({
    "accepted_prediction_tokens",
    "audio_tokens",
    "image_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
})


def map_request_usage(usage: Any) -> dict[str, Any]:
    """
    Turn a Pydantic AI RequestUsage into the flat usage mapping the
    tracker expects, using the same key names openai_compat produces.

    input_tokens and output_tokens carry the names they had before
    Pydantic AI renamed them, so the mapping is written against the
    current names and pinned by the optional extra.

    Zero and absent values are omitted, matching how the adapters only
    write a detail key the provider actually reported.

    RequestUsage.cost is recorded as estimated_cost, never as cost. The
    adapters' cost is what the provider said it billed; this one is
    computed locally from a price table. Writing both under one key
    would put an estimate into rows this library exists to reconcile
    against an invoice.

    Returns a mapping with prompt_tokens, completion_tokens, and
    total_tokens always present, so a caller can hand it straight to
    record_response.
    """
    mapped: dict[str, Any] = {
        "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
        "completion_tokens": (
            getattr(usage, "output_tokens", 0) or 0
        ),
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    for attribute, key in _REQUEST_USAGE_MAP:
        value = getattr(usage, attribute, None)
        if value:
            mapped[key] = value
    cost = getattr(usage, "cost", None)
    if cost is not None:
        mapped["estimated_cost"] = cost
    details = getattr(usage, "details", None) or {}
    unmapped: dict[str, Any] = {}
    for key, value in details.items():
        if not value:
            continue
        if key in _DETAIL_KEYS:
            mapped[f"completion_{key}"] = value
        else:
            unmapped[key] = value
    if unmapped:
        mapped["unmapped"] = unmapped
    return mapped


def normalise_token_keys(
    usage: dict[str, Any],
) -> dict[str, Any]:
    """
    Return a copy of a usage mapping with its token counts under the
    three keys the ledger's usage block holds.

    An adapter always reports prompt_tokens and completion_tokens, but
    a caller recording someone else's call passes whatever the provider
    gave them, and Anthropic gives input_tokens and output_tokens.
    Without this they would be split into usage_details as unrecognised
    keys and the usage block would record zero tokens for a call that
    cost real ones.

    A canonical key that is already present and non-zero wins; the
    alias is then left alone and lands in usage_details, since two
    disagreeing counts are not something to silently pick between.

    total_tokens is filled from the other two when it is missing or
    zero, because a mapping carrying only the aliases carries no total
    either.
    """
    normalised = dict(usage)
    for alias, canonical in _TOKEN_ALIASES:
        if normalised.get(canonical):
            continue
        value = normalised.pop(alias, None)
        if value is not None:
            normalised[canonical] = value
    if not normalised.get("total_tokens"):
        total = (normalised.get("prompt_tokens") or 0) + (
            normalised.get("completion_tokens") or 0
        )
        if total:
            normalised["total_tokens"] = total
    return normalised


def split_usage(
    usage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Divide a usage dict into the three normalised token keys the
    ledger's usage block holds and the remainder, which belongs in
    usage_details (e.g. dimensions and embedding_count for
    create_embeddings; cost, is_byok, and the flattened cache/reasoning
    detail keys for send_message).

    The split is by token key rather than by a list of known extras, so
    a value a provider starts returning later lands in usage_details on
    its own instead of being silently dropped. The adapters uphold the
    same rule upstream: a usage key they have no mapping for is
    collected under usage_details["unmapped"] rather than discarded.
    Modality-agnostic: the same function serves both entry points.
    """
    tokens = {
        key: value
        for key, value in usage.items()
        if key in TOKEN_USAGE_KEYS
    }
    details = {
        key: value
        for key, value in usage.items()
        if key not in TOKEN_USAGE_KEYS
    }
    return tokens, details
