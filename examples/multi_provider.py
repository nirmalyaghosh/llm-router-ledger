"""
Iterate over every configured endpoint with an available API key and send the
same prompt to each. Endpoints missing their `api_key_env` are silently
skipped so this script doubles as a smoke test even when only a subset of
providers is set up.

Prerequisites:
- llm_endpoints.yaml in the working directory.
- At least one provider configured with a valid `api_key_env`
  (Ollama works with no real key).
"""

from llm_router_ledger import (
    UsageTracker,
    load_config,
    send_message,
)


def main() -> None:
    """
    Send the same prompt to every available endpoint and print a one-line
    summary per provider. Endpoints whose `api_key_env` is unset never appear
    in `available()`, so a partially configured setup just reports on the
    providers it can reach.
    """
    config = load_config()
    available = config.available()
    if not available:
        print("No endpoints have api_key_env set; nothing to send.")
        return

    tracker = UsageTracker(
        log_path="logs/usage.jsonl",
        project_id="multi-provider-demo",
    )
    for ep in available:
        try:
            result = send_message(
                endpoint_name=ep.name,
                system="You are concise.",
                user="Say hello in one short sentence.",
                config=config,
                tracker=tracker,
                purpose="multi-provider-smoke",
            )
        except Exception as exc:
            print(f"[{ep.name}] FAILED: {exc}")
            continue
        preview = result.text[:80].replace("\n", " ")
        tokens = result.usage["total_tokens"]
        print(f"[{ep.name}] tokens={tokens} text={preview!r}")
    tracker.close()


if __name__ == "__main__":
    main()
