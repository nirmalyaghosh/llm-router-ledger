"""
Unit tests for llm_router_ledger.usage_tracker.UsageTracker.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pytest

from llm_router_ledger.exceptions import UsageTrackerError
from llm_router_ledger.usage_tracker import UsageTracker


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


def test_run_id_property_matches_request_id_prefix(
    tmp_log_path: Path,
) -> None:
    """
    The run_id property returns the same id that prefixes every
    request_id, so sibling log files can stamp the same value to group
    records cross-file.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    run_id = tracker.run_id
    assert run_id
    request_id = tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    tracker.close()
    assert request_id.startswith(f"{run_id}-")


def test_start_run_updates_run_id_property(tmp_log_path: Path) -> None:
    """
    Calling start_run() mints a fresh run_id that the run_id property
    then returns.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    first = tracker.run_id
    second = tracker.start_run()
    assert second == tracker.run_id
    assert second != first
    tracker.close()


def test_close_then_write_raises(tmp_log_path: Path) -> None:
    """
    Calling log_request after close() raises UsageTrackerError.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    tracker.close()
    with pytest.raises(UsageTrackerError):
        tracker.log_request(
            model="m",
            system_prompt="s",
            user_prompt="u",
        )


def test_constructor_run_tag_overrides_env(
    tmp_log_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A constructor-provided run_tag wins over the LRL_RUN_TAG env var.
    """
    monkeypatch.setenv("LRL_RUN_TAG", "from-env")
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
        run_tag="from-arg",
    )
    tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    assert entries[0]["run_tag"] == "from-arg"


def test_context_manager_closes_on_exit(tmp_log_path: Path) -> None:
    """
    Using the tracker as a context manager closes it on exit, so
    subsequent writes raise.
    """
    with UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
    ) as tracker:
        tracker.log_request(
            model="m",
            system_prompt="s",
            user_prompt="u",
        )
    with pytest.raises(UsageTrackerError):
        tracker.log_request(
            model="m",
            system_prompt="s",
            user_prompt="u",
        )


def test_counter_width_configurable(tmp_log_path: Path) -> None:
    """
    request_id pads the counter to the configured counter_width digits.
    """
    tracker = UsageTracker(
        log_path=tmp_log_path,
        project_id="p",
        counter_width=6,
    )
    rid = tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    tracker.close()
    run_id, counter = rid.rsplit("-", 1)
    assert len(run_id) == 8
    assert counter == "000001"


def test_log_path_parent_auto_mkdir(tmp_path: Path) -> None:
    """
    Parent directory of log_path is created automatically when it does
    not exist.
    """
    nested = tmp_path / "deep" / "deeper" / "log.jsonl"
    assert not nested.parent.exists()
    tracker = UsageTracker(log_path=nested, project_id="p")
    tracker.close()
    assert nested.parent.exists()


def test_log_request_writes_expected_fields(tmp_log_path: Path) -> None:
    """
    A single log_request call appends one llm_request event with the
    documented fields.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="my-proj")
    request_id = tracker.log_request(
        model="gpt-4.1-nano",
        system_prompt="sys",
        user_prompt="hi",
        purpose="test",
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["event"] == "llm_request"
    assert e["project_id"] == "my-proj"
    assert e["model"] == "gpt-4.1-nano"
    assert e["purpose"] == "test"
    assert e["request_id"] == request_id
    assert e["system_prompt_preview"] == "sys"
    assert e["user_prompt_preview"] == "hi"
    assert e["user_prompt_length"] == 2


def test_log_response_routes_gen_prefix(tmp_log_path: Path) -> None:
    """
    generation_id starting with "gen-" lands in the "generation_id"
    field.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    tracker.log_response(
        request_id="r-0001",
        model="m",
        response_text="hi",
        usage={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
        generation_id="gen-abc",
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    assert entries[0]["generation_id"] == "gen-abc"
    assert "provider_response_id" not in entries[0]


def test_log_response_routes_non_gen_to_provider_id(
    tmp_log_path: Path,
) -> None:
    """
    Non-"gen-" generation_id values land in provider_response_id.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    tracker.log_response(
        request_id="r-0001",
        model="m",
        response_text="hi",
        usage={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
        generation_id="chatcmpl-xyz",
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    e = entries[0]
    assert e["provider_response_id"] == "chatcmpl-xyz"
    assert "generation_id" not in e


def test_log_response_token_normalisation_input_output(
    tmp_log_path: Path,
) -> None:
    """
    Usage dicts using input_tokens / output_tokens (Anthropic style) get
    normalised to the standard keys.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    tracker.log_response(
        request_id="r-0001",
        model="m",
        response_text="hi",
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    u = entries[0]["usage"]
    assert u["prompt_tokens"] == 10
    assert u["completion_tokens"] == 20
    assert u["total_tokens"] == 30


def test_provider_field_omitted_when_empty(tmp_log_path: Path) -> None:
    """
    When provider is left as the default empty string, neither
    llm_request nor llm_response writes a "provider" key (matches the
    metadata / usage_details pattern for optional fields).
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    request_id = tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    tracker.log_response(
        request_id=request_id,
        model="m",
        response_text="hi",
        usage={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    assert len(entries) == 2
    assert "provider" not in entries[0]
    assert "provider" not in entries[1]


def test_provider_field_round_trips(tmp_log_path: Path) -> None:
    """
    log_request and log_response both write a "provider" field carrying
    the value passed by the caller, so ledger entries can be grouped by
    which server produced the tokens.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    request_id = tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
        provider="ollama",
    )
    tracker.log_response(
        request_id=request_id,
        model="m",
        response_text="hi",
        usage={
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
        provider="ollama",
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    assert len(entries) == 2
    assert entries[0]["event"] == "llm_request"
    assert entries[0]["provider"] == "ollama"
    assert entries[1]["event"] == "llm_response"
    assert entries[1]["provider"] == "ollama"


def test_request_id_increments(tmp_log_path: Path) -> None:
    """
    Successive log_request calls bump the counter and reuse the same
    run_id.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    r1 = tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    r2 = tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    tracker.close()
    run1, c1 = r1.rsplit("-", 1)
    run2, c2 = r2.rsplit("-", 1)
    assert run1 == run2
    assert int(c1) + 1 == int(c2)


def test_run_tag_falls_back_to_env(
    tmp_log_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    run_tag defaults to LRL_RUN_TAG when the constructor arg is None.
    """
    monkeypatch.setenv("LRL_RUN_TAG", "envtag")
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    tracker.close()
    entries = _read_jsonl(tmp_log_path)
    assert entries[0]["run_tag"] == "envtag"


def test_start_run_resets_counter(tmp_log_path: Path) -> None:
    """
    Calling start_run mints a fresh run_id and resets the counter to
    zero.
    """
    tracker = UsageTracker(log_path=tmp_log_path, project_id="p")
    tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    new_run = tracker.start_run()
    rid_b = tracker.log_request(
        model="m",
        system_prompt="s",
        user_prompt="u",
    )
    tracker.close()
    assert rid_b.startswith(f"{new_run}-")
    assert rid_b.endswith("-0001")
