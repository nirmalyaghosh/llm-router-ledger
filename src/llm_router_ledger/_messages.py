"""
Shared OpenAI content-parts message helpers.

Every chat entry point works with the same message shape:
{"role": ..., "content": [{"type": "text", "text": ...}]}. content is a
list of parts rather than a plain string so that adding image input
later is a new part type, not another break; only "text" parts exist
today. Both dispatcher.py (building messages from system/user, and
deriving ledger preview text) and the Anthropic adapter (splitting
system out of the message list) need to build and read this shape,
hence the shared module.
"""

from __future__ import annotations

from typing import Any


def build_messages(
    *,
    system: str | None,
    user: str,
) -> list[dict[str, Any]]:
    """
    Build a content-parts messages list from the system/user convenience
    args. system is omitted entirely when None, matching how the
    single-turn call has always treated an unset system prompt.
    """
    messages: list[dict[str, Any]] = []
    if system is not None:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": user}],
        }
    )
    return messages


def extract_system_text(messages: list[dict[str, Any]]) -> str:
    """
    Join the text of every system-role message into one string, in
    order. Anthropic's Messages API takes system as a top-level
    parameter rather than a message in the list, and the ledger's
    system_prompt preview needs a single string regardless of provider.
    """
    parts = [
        extract_text(message.get("content"))
        for message in messages
        if message.get("role") == "system"
    ]
    return "\n".join(part for part in parts if part)


def extract_text(content: Any) -> str:
    """
    Join the text parts of a content-parts list into one string. A
    plain string is returned unchanged, so content built by a caller
    who skipped the content-parts form still works.
    """
    if isinstance(content, str):
        return content
    if not content:
        return ""
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )
