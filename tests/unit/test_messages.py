"""
Unit tests for llm_router_ledger._messages.
"""

from __future__ import annotations

from llm_router_ledger._messages import (
    build_messages,
    extract_system_text,
    extract_text,
)


def test_build_messages_omits_system_when_none() -> None:
    """
    system=None produces a messages list with only the user entry, so
    user-only calls carry no system-role message at all.
    """
    messages = build_messages(system=None, user="hi")
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]


def test_build_messages_includes_system_first() -> None:
    """
    A non-None system produces a system-role message ahead of the
    user-role message, both in content-parts form.
    """
    messages = build_messages(system="be concise", user="hi")
    assert messages == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "be concise"}],
        },
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]


def test_extract_text_from_content_parts() -> None:
    """
    A content-parts list joins the text of every "text" part.
    """
    content = [{"type": "text", "text": "hello"}]
    assert extract_text(content) == "hello"


def test_extract_text_joins_multiple_text_parts() -> None:
    """
    Multiple text parts concatenate in order, with no separator.
    """
    content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    assert extract_text(content) == "hello world"


def test_extract_text_skips_non_text_parts() -> None:
    """
    A non-text part (e.g. a future image_url part) is skipped rather
    than raising, so callers do not need to filter first.
    """
    content = [
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "hello"},
    ]
    assert extract_text(content) == "hello"


def test_extract_text_passes_through_plain_string() -> None:
    """
    A plain string content value is returned unchanged, so callers who
    built messages without the content-parts wrapper still work.
    """
    assert extract_text("hello") == "hello"


def test_extract_text_empty_for_none_or_empty_content() -> None:
    """
    None or an empty list content returns "" rather than raising.
    """
    assert extract_text(None) == ""
    assert extract_text([]) == ""


def test_extract_system_text_joins_all_system_messages() -> None:
    """
    Every system-role message's text joins into one string, in order,
    separated by newlines.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "First."}],
        },
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "system",
            "content": [{"type": "text", "text": "Second."}],
        },
    ]
    assert extract_system_text(messages) == "First.\nSecond."


def test_extract_system_text_empty_when_no_system_messages() -> None:
    """
    A messages list with no system-role entry returns "".
    """
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]
    assert extract_system_text(messages) == ""
