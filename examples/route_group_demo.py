"""
Ask a route group for an endpoint instead of naming one.

Prints the dry run for every group in the config, showing which
candidate each strategy picks and why the others were skipped, then
sends one prompt through the low-cost group. Both steps read
`llm_endpoints.yaml` in the working directory.

The dry run costs nothing: route() makes no request. The call at the
end goes through whichever candidate the group chooses, which in the
example config is a free OpenRouter model, so running this example
should not be billed. Check the endpoint it names before running if
you have edited the groups.

Prerequisites:
1. Copy `examples/llm_endpoints.example.yaml` to `llm_endpoints.yaml`
   in the working directory.
2. Set `OPENROUTER_API_KEY`, or edit the low-cost group to name
   endpoints you can reach.
"""

from llm_router_ledger import (
    RoutingError,
    UsageTracker,
    load_config,
    route,
    send_message,
)


def main() -> None:
    """
    Show what each group would choose, then call one of them.
    """
    config = load_config()
    groups = config.route_groups.get("default", {})

    for name in sorted(groups):
        try:
            decision = route(config=config, name=name)
        except RoutingError as exc:
            print(f"{name}: no usable candidate")
            print(f"  {exc}")
            print()
            continue
        print(f"{name}: {decision.endpoint_name}")
        print(f"  {decision.chosen_by}")
        for candidate, reason in decision.skipped.items():
            print(f"  skipped {candidate}: {reason}")
        print()

    try:
        chosen = route(config=config, name="low-cost")
    except RoutingError as exc:
        print(f"Not calling anything: {exc}")
        return

    tracker = UsageTracker(
        log_path="logs/usage.jsonl",
        project_id="route-group-demo",
    )
    try:
        result = send_message(
            route_group="low-cost",
            config=config,
            system="You are concise.",
            user="Name one benefit of routing by cost.",
            tracker=tracker,
        )
        print(f"{chosen.endpoint_name} answered:")
        print(result.text)
        print()
        print(f"tokens used: {result.usage['total_tokens']}")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
