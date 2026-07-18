"""
Frictionless first call: hit a local Ollama server
and append a single request + response pair to `logs/usage.jsonl`.

Ollama is the default endpoint so this example runs without signing up for any
paid provider.

Prerequisites:
1. Install and run Ollama (https://ollama.com).
2. Pull the model that the example config's
   local-llama endpoint points at: `ollama pull qwen2.5vl:3b`.
3. Copy `llm_endpoints.example.yaml` to `llm_endpoints.yaml` in the working
   directory.

To swap providers, change endpoint_name below to any other name from your 
`llm_endpoints.yaml` and ensure the matching `api_key_env` is set.
"""

from llm_router_ledger import (
    UsageTracker,
    send_message,
)


def main() -> None:
    """
    Send one prompt to Ollama and print the response plus the usage summary.
    """
    tracker = UsageTracker(
        log_path="logs/usage.jsonl",
        project_id="ollama-quickstart",
    )
    text, usage, gen_id = send_message(
        endpoint_name="local-llama",
        system="You are concise.",
        user="Explain prompt caching in two sentences.",
        tracker=tracker,
    )
    print(text)
    print()
    print(f"tokens used: {usage['total_tokens']}")
    print(f"response id: {gen_id}")
    tracker.close()


if __name__ == "__main__":
    main()
