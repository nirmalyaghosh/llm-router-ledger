"""
Readers for an agent framework's message objects.

record_run and the Pydantic AI integration both need the prompts for
the previews, and the finish reason and tool call count for the usage
block.

Duck typed. Nothing here imports pydantic-ai. Attribute names come
from pydantic-ai 2.34.0, the version the optional extra pins.
"""

from __future__ import annotations

from typing import Any

from llm_router_ledger._usage import (
    ORDINARY_FINISH_REASONS,
    map_request_usage,
)


def count_tool_calls(message: Any) -> int:
    """
    Helper function used to count the tool calls in one model response.

    A turn returning only tool calls would otherwise read as empty.
    Mirrors completion_tool_call_count in the adapter.

    Counts the ordinary tool call part only. Provider-side call parts
    carry their own discriminators and are excluded. The result is a
    lower bound.
    """
    return sum(
        1
        for part in getattr(message, "parts", ())
        if getattr(part, "part_kind", None) == "tool-call"
    )


def finish_reason(message: Any) -> str:
    """
    Helper function used to read the finish reason from one model
    response, in the provider's vocabulary. Returns "" for a completed
    turn.

    The raw value is preferred over the framework's normalised one. A
    reconciler filtering on finish_reason then sees "tool_calls" from
    either path. Pydantic AI keeps the raw value in provider_details.
    """
    details = getattr(message, "provider_details", None)
    raw = (
        details.get("finish_reason")
        if isinstance(details, dict)
        else None
    )
    reason = raw or getattr(message, "finish_reason", None)
    if not reason or reason in ORDINARY_FINISH_REASONS:
        return ""
    return str(reason)


def latest_user_prompt(message: Any) -> str:
    """
    Helper function used to read the last user prompt from one model
    request, for the request event's preview and character count.

    Tool returns and system prompts are skipped, matching the
    dispatcher. Returns "" for a request with no user prompt, and for a
    multimodal prompt whose content is not a string.
    """
    latest = ""
    for part in getattr(message, "parts", ()):
        if getattr(part, "part_kind", None) != "user-prompt":
            continue
        content = getattr(part, "content", "")
        if isinstance(content, str):
            latest = content
    return latest


def response_text(message: Any) -> str:
    """
    Helper function used to join the text parts of one model response,
    for the response event's preview and character count.

    Thinking parts are excluded. Their tokens are accounted for under
    completion_reasoning_tokens. A turn returning only tool calls yields
    "", matching the adapter.
    """
    return "".join(
        part.content
        for part in getattr(message, "parts", ())
        if getattr(part, "part_kind", None) == "text"
        and isinstance(getattr(part, "content", None), str)
    )


def response_usage(message: Any) -> dict[str, Any]:
    """
    Helper function used to build the usage mapping for one model
    response, ready to pass to record_response.

    Token counts come from map_request_usage. The finish reason and tool
    call count are added here, each omitted when absent.

    Both entry points use this. record_run and the wrapping model
    produce identical rows.
    """
    usage = map_request_usage(getattr(message, "usage", None))
    reason = finish_reason(message)
    if reason:
        usage["finish_reason"] = reason
    tool_calls = count_tool_calls(message)
    if tool_calls:
        usage["completion_tool_call_count"] = tool_calls
    return usage


def system_prompt(message: Any) -> str:
    """
    Helper function used to read the system prompt from one model
    request, for the request event's preview.

    Both shapes are read. instructions= appears on every request.
    system_prompt= appears on the first only, and that request is resent
    as history thereafter.

    Returns "" for a request carrying neither. The caller treats the
    result as sticky.
    """
    instructions = getattr(message, "instructions", None)
    if isinstance(instructions, str) and instructions:
        return instructions
    return "".join(
        part.content
        for part in getattr(message, "parts", ())
        if getattr(part, "part_kind", None) == "system-prompt"
        and isinstance(getattr(part, "content", None), str)
    )
