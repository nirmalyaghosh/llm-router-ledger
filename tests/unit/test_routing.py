"""
Unit tests for llm_router_ledger.routing and for send_message when it
is given a route group.

Everything here runs offline: route() makes no request, and the
send_message tests mock the adapter the way test_dispatcher_dryrun
does.
"""

from __future__ import annotations

import dataclasses
import json
import sys

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_router_ledger import (
    UsageTracker,
    load_config,
    route,
    send_message,
)
from llm_router_ledger.config import LLMConfig
from llm_router_ledger.exceptions import (
    ConfigError,
    LLMCallError,
    ProviderUnavailableError,
    RoutingError,
)

_KEYS = (
    "ROUTING_KEY_A",
    "ROUTING_KEY_B",
    "ROUTING_KEY_C",
    "ROUTING_KEY_D",
    "ROUTING_KEY_E",
)

_ENDPOINTS = (
    "endpoints:\n"
    "  azure-no-url:\n"
    "    provider: azure\n"
    "    model: gpt-4o\n"
    "    api_key_env: ROUTING_KEY_D\n"
    "  claude-one:\n"
    "    provider: anthropic\n"
    "    model: claude\n"
    "    api_key_env: ROUTING_KEY_E\n"
    "  cheap-one:\n"
    "    provider: openrouter\n"
    "    model: cheap\n"
    "    api_key_env: ROUTING_KEY_A\n"
    "    cost:\n"
    "      input_per_1m: 0.10\n"
    "      output_per_1m: 0.20\n"
    "  dear-one:\n"
    "    provider: openrouter\n"
    "    model: dear\n"
    "    api_key_env: ROUTING_KEY_B\n"
    "    cost:\n"
    "      input_per_1m: 1.00\n"
    "      output_per_1m: 2.00\n"
    "  twin-one:\n"
    "    provider: openrouter\n"
    "    model: twin\n"
    "    api_key_env: ROUTING_KEY_B\n"
    "    cost:\n"
    "      input_per_1m: 0.10\n"
    "      output_per_1m: 0.20\n"
    "  unverified-one:\n"
    "    provider: gemini\n"
    "    model: unverified\n"
    "    api_key_env: ROUTING_KEY_C\n"
    "    cost:\n"
    "      input_per_1m: 0.00\n"
    "      output_per_1m: 0.00\n"
)


def _write(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    block: str,
    keys: tuple[str, ...] = _KEYS,
) -> Path:
    """
    Helper function used to write a config carrying the test
    endpoints plus the given route_groups block, and to set only the
    API keys named in keys.
    """
    for name in _KEYS:
        monkeypatch.delenv(name, raising=False)
    for name in keys:
        monkeypatch.setenv(name, "set")
    path = tmp_path / "routing.yaml"
    path.write_text(_ENDPOINTS + block, encoding="utf-8")
    return path


def _config(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    block: str,
    keys: tuple[str, ...] = _KEYS,
) -> LLMConfig:
    """
    Helper function used to load the config _write produces.
    """
    return load_config(
        _write(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            block=block,
            keys=keys,
        ),
    )


def _patch_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raises: Exception | None = None,
) -> MagicMock:
    """
    Helper function used to mock the client and the adapter so
    send_message runs offline, optionally making the call raise.
    """
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher.get_client",
        lambda **kw: MagicMock(),
    )
    fake = MagicMock()
    if raises is None:
        fake.send.return_value = (
            "hello",
            {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
            },
            "gen-1",
        )
    else:
        fake.send.side_effect = raises
    monkeypatch.setattr(
        "llm_router_ledger.dispatcher._select_adapter",
        lambda provider: fake,
    )
    return fake


def _rows(log: Path) -> list[dict[str, Any]]:
    """
    Helper function used to read the ledger rows a test wrote.
    """
    return [
        json.loads(line)
        for line in log.read_text(
            encoding="utf-8",
        ).splitlines()
        if line
    ]


def test_cheapest_breaks_a_tie_by_list_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Two candidates on the same combined rate resolve to the one
    listed first, whichever order they are written in, which is what
    the README promises.
    """
    for first, second in (
        ("cheap-one", "twin-one"),
        ("twin-one", "cheap-one"),
    ):
        config = _config(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            block=(
                "route_groups:\n"
                "  default:\n"
                "    g:\n"
                "      strategy: cheapest\n"
                f"      candidates: [{first}, {second}]\n"
            ),
        )
        assert route(
            config=config,
            name="g",
        ).endpoint_name == first


def test_cheapest_picks_the_lowest_combined_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    cheapest compares the declared input rate plus output rate, so the
    order the candidates are listed in does not decide the choice.
    """
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      strategy: cheapest\n"
            "      candidates: [dear-one, cheap-one]\n"
        ),
    )
    decision = route(config=config, name="g")
    assert decision.endpoint_name == "cheap-one"
    assert decision.strategy == "cheapest"
    assert decision.chosen_by == "cheapest of 2 usable candidates"


def test_no_usable_candidate_raises_and_names_each_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A group whose candidates are all skipped raises RoutingError, and
    the message says why each one was skipped.
    """
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one, unverified-one]\n"
        ),
        keys=(),
    )
    with pytest.raises(RoutingError) as excinfo:
        route(config=config, name="g")
    message = str(excinfo.value)
    assert "ROUTING_KEY_A is not set" in message
    assert "no verified adapter" in message


def test_priority_takes_the_first_usable_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    priority walks the list in order, skipping a candidate whose key
    is unset and recording why, so a dry run explains the choice.
    """
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one, dear-one]\n"
        ),
        keys=("ROUTING_KEY_B",),
    )
    decision = route(config=config, name="g")
    assert decision.endpoint_name == "dear-one"
    assert decision.chosen_by == "first of 1 usable candidate"
    assert dict(decision.skipped) == {
        "cheap-one": "ROUTING_KEY_A is not set",
    }


def test_route_falls_back_to_the_config_on_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    route() with no config loads the one LRL_CONFIG_PATH names, the
    same way the rest of the library resolves a config.
    """
    path = _write(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
        ),
    )
    monkeypatch.setenv("LRL_CONFIG_PATH", str(path))
    assert route(name="g").endpoint_name == "cheap-one"


def test_route_records_the_project_it_resolved_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A group inherited from the default project reports "default", and
    a project's own group of the same name shadows it and reports
    that project.
    """
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    shared:\n"
            "      candidates: [cheap-one]\n"
            "    both:\n"
            "      candidates: [cheap-one]\n"
            "  reporting:\n"
            "    both:\n"
            "      candidates: [dear-one]\n"
        ),
    )
    inherited = route(
        config=config,
        name="shared",
        project="reporting",
    )
    shadowed = route(
        config=config,
        name="both",
        project="reporting",
    )
    from_default = route(config=config, name="both")
    assert inherited.project == "default"
    assert shadowed.project == "reporting"
    assert shadowed.endpoint_name == "dear-one"
    assert from_default.project == "default"
    assert from_default.endpoint_name == "cheap-one"


def test_route_skips_what_get_client_would_refuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An azure endpoint with no base_url, and an anthropic endpoint
    whose SDK is absent, are both skipped rather than chosen and then
    refused by get_client after the ledger row would have been lost.
    """
    monkeypatch.setitem(sys.modules, "anthropic", None)
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates:"
            " [azure-no-url, claude-one, cheap-one]\n"
        ),
    )
    decision = route(config=config, name="g")
    assert decision.endpoint_name == "cheap-one"
    assert dict(decision.skipped) == {
        "azure-no-url": "no base_url is declared",
        "claude-one": "the anthropic SDK is not installed",
    }


def test_route_unknown_group_raises_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An unknown group is a config problem rather than a routing one,
    so it raises the same ConfigError get_route_group raises.
    """
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
        ),
    )
    with pytest.raises(ConfigError, match="not found"):
        route(config=config, name="nope")


def test_route_decision_is_frozen_and_not_shared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A decision's fields cannot be reassigned. skipped is an ordinary
    dict, so it can be edited, but each call builds its own, and
    editing one decision cannot change the next.
    """
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one, dear-one]\n"
        ),
        keys=("ROUTING_KEY_B",),
    )
    first = route(config=config, name="g")
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.endpoint_name = "cheap-one"  # type: ignore[misc]
    first.skipped["invented"] = "not a real reason"  # type: ignore[index]
    second = route(config=config, name="g")
    assert dict(second.skipped) == {
        "cheap-one": "ROUTING_KEY_A is not set",
    }


def test_send_message_honours_an_explicit_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An explicit project wins over the tracker's project_id, and the
    row records which project the group came from.
    """
    _patch_adapter(monkeypatch)
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
            "  reporting:\n"
            "    g:\n"
            "      candidates: [dear-one]\n"
        ),
    )
    log = tmp_path / "usage.jsonl"
    with UsageTracker(
        project_id="default",
        log_path=log,
    ) as tracker:
        send_message(
            route_group="g",
            project="reporting",
            user="hi",
            config=config,
            tracker=tracker,
        )
    rows = _rows(log)
    assert rows[0]["model"] == "dear"
    assert rows[0]["route_project"] == "reporting"


def test_send_message_ignores_the_tracker_project_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The project is asked for, never inferred, so a project_id that
    happens to name a route_groups section does not silently change
    which endpoint a call reaches.
    """
    _patch_adapter(monkeypatch)
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
            "  reporting:\n"
            "    g:\n"
            "      candidates: [dear-one]\n"
        ),
    )
    log = tmp_path / "usage.jsonl"
    with UsageTracker(
        project_id="reporting",
        log_path=log,
    ) as tracker:
        send_message(
            route_group="g",
            user="hi",
            config=config,
            tracker=tracker,
        )
    assert _rows(log)[0]["model"] == "cheap"


def test_send_message_writes_no_row_when_nothing_is_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A group with nothing callable raises before the call is attempted,
    and RoutingError is an LLMCallError, so one except clause catches
    it. No row is written, since no request was made.
    """
    _patch_adapter(monkeypatch)
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
        ),
        keys=(),
    )
    log = tmp_path / "usage.jsonl"
    with UsageTracker(
        project_id="proj",
        log_path=log,
    ) as tracker:
        with pytest.raises(LLMCallError) as excinfo:
            send_message(
                route_group="g",
                user="hi",
                config=config,
                tracker=tracker,
            )
    assert isinstance(excinfo.value, RoutingError)
    assert not log.exists() or log.read_text(
        encoding="utf-8",
    ) == ""


def test_send_message_records_the_group_on_an_error_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A routed call that raises still records which group chose the
    endpoint, so a failure is as traceable as a success.
    """
    _patch_adapter(
        monkeypatch,
        raises=RuntimeError("upstream exploded"),
    )
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
        ),
    )
    log = tmp_path / "usage.jsonl"
    with UsageTracker(
        project_id="proj",
        log_path=log,
    ) as tracker:
        with pytest.raises(ProviderUnavailableError):
            send_message(
                route_group="g",
                user="hi",
                config=config,
                tracker=tracker,
            )
    error_row = _rows(log)[-1]
    assert error_row["event"] == "llm_error"
    assert error_row["route_group"] == "g"
    assert error_row["route_project"] == "default"
    assert error_row["route_strategy"] == "priority"
    assert error_row["route_endpoint"] == "cheap-one"


def test_send_message_records_the_group_and_the_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A routed call reaches the chosen endpoint and stamps the group,
    the project, the strategy and the reason on both ledger rows.
    """
    _patch_adapter(monkeypatch)
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      strategy: cheapest\n"
            "      candidates: [dear-one, cheap-one]\n"
        ),
    )
    log = tmp_path / "usage.jsonl"
    with UsageTracker(
        project_id="proj",
        log_path=log,
    ) as tracker:
        result = send_message(
            route_group="g",
            user="hi",
            config=config,
            tracker=tracker,
        )
    assert result.text == "hello"
    rows = _rows(log)
    assert len(rows) == 2
    for row in rows:
        assert row["route_group"] == "g"
        assert row["route_project"] == "default"
        assert row["route_strategy"] == "cheapest"
        assert row["route_endpoint"] == "cheap-one"
        assert row["model"] == "cheap"


def test_send_message_rejects_both_and_neither(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    endpoint_name and route_group are alternatives, so passing both or
    neither is a ValueError, and project only means something
    alongside a group.
    """
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
        ),
    )
    with pytest.raises(ValueError, match="not both"):
        send_message(
            endpoint_name="cheap-one",
            route_group="g",
            user="hi",
            config=config,
        )
    with pytest.raises(ValueError, match="requires either"):
        send_message(user="hi", config=config)
    with pytest.raises(ValueError, match="only uses 'project'"):
        send_message(
            endpoint_name="cheap-one",
            project="reporting",
            user="hi",
            config=config,
        )


def test_send_message_writes_no_routing_fields_when_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A call that names its endpoint writes the rows it always did, so
    an existing consumer reading the ledger sees no new keys.
    """
    _patch_adapter(monkeypatch)
    config = _config(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        block=(
            "route_groups:\n"
            "  default:\n"
            "    g:\n"
            "      candidates: [cheap-one]\n"
        ),
    )
    log = tmp_path / "usage.jsonl"
    with UsageTracker(
        project_id="proj",
        log_path=log,
    ) as tracker:
        send_message(
            endpoint_name="cheap-one",
            user="hi",
            config=config,
            tracker=tracker,
        )
    text = log.read_text(encoding="utf-8")
    for key in (
        "route_group",
        "route_project",
        "route_strategy",
        "route_endpoint",
    ):
        assert key not in text
