"""
Choosing one endpoint from a route group.

A route group names candidate endpoints and a strategy. route()
applies the strategy and returns a RouteDecision without calling
anything, so it costs no tokens; send_message(route_group=...) uses
the same function and then calls the endpoint it chose.

A candidate is skipped when its provider has no verified adapter, or
when client_factory.unusable_reason says get_client would refuse it.
Skipping happens before the choice rather than at call time because
the dispatcher resolves the client before it writes any ledger row,
so an unusable candidate would otherwise fail leaving no trace of the
call at all, and under the priority strategy it would take down a
group that has a working candidate behind it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from llm_router_ledger._verified import VERIFIED_PROVIDERS
from llm_router_ledger.client_factory import unusable_reason
from llm_router_ledger.config import (
    EndpointConfig,
    LLMConfig,
    RouteStrategy,
    load_config,
)
from llm_router_ledger.exceptions import RoutingError


@dataclass(frozen=True)
class RouteDecision:
    """
    What route() decided, and the record of how send_message chose an
    endpoint when it was given a route group.

    endpoint_name is the chosen endpoint. group is the route group,
    and project is the project the group resolved through, which is
    "default" when the named project does not carry a group of that
    name. chosen_by is a short phrase for a person reading a ledger
    row and is not written to the ledger; the group, the project, the
    strategy and the chosen endpoint are recorded as fields of their
    own, so nothing needs to be parsed. skipped maps each candidate
    that could not be chosen to why.
    """

    endpoint_name: str
    group: str
    project: str
    strategy: RouteStrategy
    chosen_by: str
    skipped: Mapping[str, str]


def _combined_rate(endpoint: EndpointConfig) -> float:
    """
    Helper function used to score a candidate under the cheapest
    strategy: the declared input rate plus the declared output rate,
    both per 1M tokens.
    """
    cost = endpoint.cost
    return 0.0 if cost is None else (
        cost.input_per_1m + cost.output_per_1m
    )


def _skip_reason(
    *,
    endpoint: EndpointConfig | None,
    strategy: RouteStrategy,
) -> str | None:
    """
    Helper function used to say why a candidate cannot be chosen, or
    to return None when it can.
    """
    if endpoint is None:
        return "not declared in this config"
    if endpoint.provider not in VERIFIED_PROVIDERS:
        return (
            f"provider '{endpoint.provider}' has no verified"
            f" adapter"
        )
    reason = unusable_reason(endpoint=endpoint)
    if reason is not None:
        return reason
    if strategy == "cheapest" and endpoint.cost is None:
        return "declares no cost to compare"
    return None


def route(
    *,
    config: LLMConfig | None = None,
    name: str,
    project: str = "default",
) -> RouteDecision:
    """
    Choose an endpoint from a route group without calling it.

    name is the group. project selects which set of groups to read,
    falling back per group to the 'default' project. Raises
    ConfigError when neither carries the group, and RoutingError when
    the group carries no candidate that can be called.

    The decision records which candidates were skipped and why, so an
    unexpected choice can be explained without making a request.
    """
    if config is None:
        config = load_config()
    group = config.get_route_group(project=project, name=name)

    skipped: dict[str, str] = {}
    eligible: list[str] = []
    for candidate in group.candidates:
        reason = _skip_reason(
            endpoint=config.endpoints.get(candidate),
            strategy=group.strategy,
        )
        if reason is None:
            eligible.append(candidate)
        else:
            skipped[candidate] = reason

    if not eligible:
        detail = (
            "; ".join(
                f"{candidate} ({reason})"
                for candidate, reason in skipped.items()
            )
            or "it names none"
        )
        raise RoutingError(
            f"Route group '{name}' in project"
            f" '{group.project}' has no"
            f" candidate that can be called: {detail}"
        )

    noun = (
        "candidate" if len(eligible) == 1 else "candidates"
    )
    if group.strategy == "cheapest":
        chosen = min(
            eligible,
            key=lambda candidate: _combined_rate(
                config.endpoints[candidate],
            ),
        )
        chosen_by = (
            f"cheapest of {len(eligible)} usable {noun}"
        )
    else:
        chosen = eligible[0]
        chosen_by = f"first of {len(eligible)} usable {noun}"

    return RouteDecision(
        endpoint_name=chosen,
        group=name,
        project=group.project,
        strategy=group.strategy,
        chosen_by=chosen_by,
        skipped=skipped,
    )
