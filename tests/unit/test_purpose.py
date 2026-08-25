"""
Unit tests for the ambient purpose contextvar and the precedence rule
UsageTracker applies when resolving a purpose for an entry.
"""

from __future__ import annotations

import asyncio
import json

from pathlib import Path
from typing import Any

import pytest

from llm_router_ledger import (
    current_purpose,
    purpose_scope,
)
from llm_router_ledger.usage_tracker import UsageTracker


def _logged_purpose(
    path: Path,
    tracker: UsageTracker,
    **kwargs: Any,
) -> str:
    """
    Helper function used to write one request event and return the
    purpose that reached the ledger.
    """
    tracker.log_request(
        model="m",
        system_prompt="",
        user_prompt="u",
        **kwargs,
    )
    return str(_read_jsonl(path)[-1]["purpose"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Helper function used to read all JSONL entries from path into a list
    of dicts.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_current_purpose_is_empty_outside_a_scope() -> None:
    """
    With no scope entered, there is no ambient purpose. This is the
    state every existing caller is in, so it must stay "" rather than
    become a sentinel.
    """
    assert current_purpose() == ""


def test_scope_sets_and_restores() -> None:
    """
    The scope is visible inside the block and gone after it, so a
    purpose cannot leak into unrelated work that follows.
    """
    with purpose_scope("planning"):
        assert current_purpose() == "planning"
    assert current_purpose() == ""


def test_scopes_nest_innermost_first() -> None:
    """
    Nested scopes stack: the innermost wins, and leaving it restores the
    one around it rather than clearing everything.
    """
    with purpose_scope("outer"):
        with purpose_scope("inner"):
            assert current_purpose() == "inner"
        assert current_purpose() == "outer"


def test_empty_scope_clears_an_outer_one() -> None:
    """
    Entering a scope with "" is how a nested call records with no
    purpose instead of inheriting the surrounding one.
    """
    with purpose_scope("outer"):
        with purpose_scope(""):
            assert current_purpose() == ""
        assert current_purpose() == "outer"


def test_scope_restores_when_the_block_raises() -> None:
    """
    An exception inside the block must not leave the purpose set, or
    every later call in the process would be stamped with it.
    """
    with pytest.raises(ValueError):
        with purpose_scope("planning"):
            raise ValueError("boom")
    assert current_purpose() == ""


def test_concurrent_tasks_do_not_see_each_others_purpose() -> None:
    """
    Being a contextvar rather than a module global is the point: two
    agents running concurrently each keep their own purpose. The awaits
    force the tasks to interleave, so a shared global would fail here.
    """

    async def scoped(name: str) -> str:
        with purpose_scope(name):
            await asyncio.sleep(0)
            observed = current_purpose()
            await asyncio.sleep(0)
            assert current_purpose() == observed
            return observed

    async def main() -> list[str]:
        return list(
            await asyncio.gather(
                scoped("first"),
                scoped("second"),
            )
        )

    assert asyncio.run(main()) == ["first", "second"]


def test_explicit_purpose_beats_the_scope(
    tmp_log_path: Path,
) -> None:
    """
    A purpose passed to the call is the narrowest statement of intent,
    so it wins over an ambient scope.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    try:
        with purpose_scope("scoped"):
            assert (
                _logged_purpose(
                    tmp_log_path,
                    tracker,
                    purpose="explicit",
                )
                == "explicit"
            )
    finally:
        tracker.close()


def test_scope_beats_the_tracker_default(
    tmp_log_path: Path,
) -> None:
    """
    The scope is set around specific work while default_purpose covers
    the whole tracker, so the scope is the narrower of the two and wins.
    """
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
        default_purpose="fallback",
    )
    try:
        with purpose_scope("scoped"):
            assert (
                _logged_purpose(tmp_log_path, tracker)
                == "scoped"
            )
    finally:
        tracker.close()


def test_default_purpose_applies_with_no_scope(
    tmp_log_path: Path,
) -> None:
    """
    With nothing narrower set, default_purpose reaches the ledger. It
    previously did not for anything routed through send_message, which
    passes purpose="" on every call.
    """
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
        default_purpose="fallback",
    )
    try:
        assert (
            _logged_purpose(tmp_log_path, tracker, purpose="")
            == "fallback"
        )
    finally:
        tracker.close()


def test_response_and_error_events_resolve_the_same_way(
    tmp_log_path: Path,
) -> None:
    """
    All three event types share one resolution rule. A response or an
    error stamped differently from its own request would break grouping
    a run's rows by purpose.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    try:
        with purpose_scope("scoped"):
            request_id = tracker.log_request(
                model="m",
                system_prompt="",
                user_prompt="u",
            )
            tracker.log_response(
                request_id=request_id,
                model="m",
                response_text="r",
                usage={"prompt_tokens": 1},
            )
            tracker.log_error(
                request_id=request_id,
                model="m",
                error_type="E",
                error_message="boom",
            )
    finally:
        tracker.close()

    entries = _read_jsonl(tmp_log_path)
    assert [entry["event"] for entry in entries] == [
        "llm_request",
        "llm_response",
        "llm_error",
    ]
    assert {entry["purpose"] for entry in entries} == {"scoped"}
