"""
Command-line interface for llm-router-ledger.

Subcommands:
- chat: ad-hoc single-shot call.
- list: print the endpoint table with key
  status, cost, and staleness warnings.
- stale: list endpoints whose pricing has
  not been verified in the last N days.
- validate: load a YAML config and report any errors.

Exit codes: 0 on success, 1 on library errors
raised as subclasses of LLMCallError.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from llm_router_ledger.config import load_config
from llm_router_ledger.dispatcher import send_message
from llm_router_ledger.exceptions import LLMCallError
from llm_router_ledger.usage_tracker import UsageTracker


STALE_THRESHOLD_DAYS = 30


def _cmd_chat(args: argparse.Namespace) -> int:
    """
    Helper function used to run the chat subcommand: send a system + user
    message to the named endpoint and print the response text. When
    --log-path is given, paired llm_request / llm_response events are
    written to that JSONL file.
    """
    tracker: UsageTracker | None = None
    if args.log_path:
        tracker = UsageTracker(
            log_path=Path(args.log_path),
            project_id=args.project_id,
        )
    try:
        text, usage, gen_id = send_message(
            endpoint_name=args.endpoint,
            system=args.system,
            user=args.user,
            tracker=tracker,
            purpose="cli-chat",
        )
        print(text)
        if tracker is not None:
            print(
                f"\n[logged] tokens:"
                f" {usage['prompt_tokens']}p"
                f" + {usage['completion_tokens']}c"
                f" = {usage['total_tokens']}t"
                f" | id: {gen_id}",
                file=sys.stderr,
            )
    finally:
        if tracker is not None:
            tracker.close()
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """
    Helper function used to run the list subcommand: print all configured
    endpoints with key status, cost, and staleness warnings.
    """
    cfg = load_config()
    print(
        f"Loaded {len(cfg.endpoints)}"
        f" endpoint(s):"
    )
    for name, ep in cfg.endpoints.items():
        key_status = (
            "OK"
            if ep.api_key_available
            else "MISSING KEY"
        )
        cost_info = ""
        stale_warning = ""
        if ep.cost:
            cost_info = (
                f"  ("
                f"${ep.cost.input_per_1m:.2f}"
                f" / "
                f"${ep.cost.output_per_1m:.2f}"
                f" per 1M)"
            )
            days = ep.cost.days_since_checked
            if days is None:
                stale_warning = (
                    "  NEVER CHECKED"
                )
            elif days > STALE_THRESHOLD_DAYS:
                stale_warning = (
                    f"  STALE ({days}d ago)"
                )
        print(
            f"  {name:<28}"
            f" {ep.provider:<20}"
            f" {ep.model:<30}"
            f" [{key_status}]"
            f"{cost_info}"
            f"{stale_warning}"
        )
    print(
        f"\nAvailable (key set):"
        f" {len(cfg.available())}"
        f"/{len(cfg.endpoints)}"
    )
    return 0


def _cmd_stale(args: argparse.Namespace) -> int:
    """
    Helper function used to run the stale subcommand: list endpoints whose
    pricing has not been verified in the last args.days days.
    """
    cfg = load_config()
    threshold = args.days
    stale = []
    for name, ep in cfg.endpoints.items():
        days = (
            ep.cost.days_since_checked
            if ep.cost
            else None
        )
        if days is None or days <= threshold:
            continue
        url = (
            ep.cost.pricing_url or ""
            if ep.cost
            else ""
        )
        stale.append((name, days, url))
    if not stale:
        print(
            f"No endpoints with pricing"
            f" older than {threshold} days."
        )
        return 0
    print(
        f"{len(stale)} endpoint(s) with"
        f" pricing older than {threshold}"
        f" days:"
    )
    for name, days, url in stale:
        print(f"  {name}: {days}d -> {url}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """
    Helper function used to run the validate subcommand: load the YAML
    config at args.path and print a one-line summary.
    """
    cfg = load_config(args.path)
    print(
        f"OK: {len(cfg.endpoints)}"
        f" endpoint(s),"
        f" {len(cfg.roles)} role mapping(s)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and dispatch to the right subcommand. Returns a
    Unix-style exit code (0 on success, 1 on library errors).
    """
    parser = argparse.ArgumentParser(
        prog="llm-router-ledger",
        description=(
            "Route LLM calls and keep a JSONL"
            " ledger of every request and"
            " response."
        ),
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p_chat = sub.add_parser(
        "chat",
        help=(
            "Send a single system+user message."
        ),
    )
    p_chat.add_argument(
        "--endpoint",
        required=True,
    )
    p_chat.add_argument(
        "--system",
        required=True,
    )
    p_chat.add_argument(
        "--user",
        required=True,
    )
    p_chat.add_argument(
        "--log-path",
        default=None,
        help=(
            "JSONL file to append llm_request"
            " / llm_response events to."
        ),
    )
    p_chat.add_argument(
        "--project-id",
        default="cli",
        help=(
            "project_id stamped on log events"
            " (default: cli)."
        ),
    )
    p_chat.set_defaults(func=_cmd_chat)

    p_list = sub.add_parser(
        "list",
        help="Show all endpoints.",
    )
    p_list.set_defaults(func=_cmd_list)

    p_stale = sub.add_parser(
        "stale",
        help=(
            "List endpoints with stale pricing."
        ),
    )
    p_stale.add_argument(
        "--days",
        type=int,
        default=STALE_THRESHOLD_DAYS,
    )
    p_stale.set_defaults(func=_cmd_stale)

    p_val = sub.add_parser(
        "validate",
        help=(
            "Load and validate a YAML config."
        ),
    )
    p_val.add_argument(
        "path",
        help="Path to llm_endpoints.yaml",
    )
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    try:
        exit_code: int = args.func(args)
        return exit_code
    except LLMCallError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
