# llm-router-ledger

Route any LLM call through one `send_message()` and keep a JSONL ledger
of every request and response for offline cost reconciliation.

## Provider support

| Status | Adapter | Providers |
|---|---|---|
| Supported | direct | Anthropic |
| Supported | OpenAI-compat | Azure OpenAI, DeepSeek, Local Ollama, MiniMax, OpenAI, OpenRouter, Qwen, Zhipu / GLM |
| Supported | via OpenRouter | ByteDance Seed, Xiaomi MiMo |
| Planned | direct | Gemini |

- All "Supported" rows in 0.1.2 are live-smoke-verified end-to-end.
- Anthropic requires the optional `[anthropic]` extra: `uv pip install llm-router-ledger[anthropic]`.
- For ByteDance Seed and Xiaomi MiMo, use `provider: openrouter` with the appropriate model id.

## Install

```bash
uv pip install llm-router-ledger
```

## Quickstart

Set `OPENROUTER_API_KEY` in `.env` and create `llm_endpoints.yaml` in the working directory. The fastest path is to copy `examples/llm_endpoints.example.yaml` to `llm_endpoints.yaml` in your working directory and edit it.

```python
from llm_router_ledger import UsageTracker, send_message

tracker = UsageTracker(
    log_path="logs/usage.jsonl",
    project_id="my-blog",
)
text, usage, gen_id = send_message(
    endpoint_name="openrouter-mimo-v2-flash",
    system="You are concise.",
    user="Explain prompt caching in two sentences.",
    tracker=tracker,
)
```

Or against a local Ollama server, with no API costs:

```python
text, usage, gen_id = send_message(
    endpoint_name="local-llama",
    system="You are concise.",
    user="Explain prompt caching in two sentences.",
    tracker=tracker,
)
```

- `send_message()` returns `(response_text, usage_dict, generation_id)`.
- `UsageTracker` appends paired `llm_request` / `llm_response` events to the JSONL log, stamped with `project_id`, `run_tag`, `run_label`, and `purpose` for later grouping.

## JSONL ledger schema

- `UsageTracker` writes two events per `send_message()` call: an `llm_request` before the call, and an `llm_response` after.
- Both share a `request_id` so they can be paired. Top-level fields on each event include `project_id`, `provider`, `model`, `purpose`, `run_tag`, `run_label`, and `timestamp`.
- The `llm_response` event additionally carries `usage` (with `prompt_tokens`, `completion_tokens`, `total_tokens`) and a response preview.

**Identifying a response for billing reconciliation:** the response id is routed to one of two fields based on prefix:

- `generation_id`: set when the id starts with `"gen-"` (OpenRouter convention). Use this when joining against OpenRouter's CSV export, which calls the column `generation_id`.
- `provider_response_id`: set for everything else. OpenAI, Azure OpenAI, Ollama, and most direct-provider endpoints return ids like `"chatcmpl-..."` that land here. Use this when joining against OpenAI-family billing exports or any provider-native log that exposes a
  chat completion id.

Exactly one of the two fields is populated per `llm_response` event; queries that join the ledger to billing data should `COALESCE` over both or branch on `provider`.

## CLI

```bash
llm-router-ledger list                          # show configured endpoints
llm-router-ledger validate llm_endpoints.yaml   # validate the YAML
llm-router-ledger stale --days 30               # endpoints with stale pricing
llm-router-ledger chat --endpoint openrouter-mimo-v2-flash --system "You are concise." --user "Hello." --log-path logs/usage.jsonl --project-id my-project
```

## Env vars

| Variable | Purpose |
|---|---|
| `LRL_RUN_TAG` | Stamped on every JSONL event. |
| `LRL_RUN_LABEL` | Stamped on every JSONL event. |
| `LRL_CONFIG_PATH` | Default YAML path when `load_config()` is called with no argument. |

## Development

```bash
git clone https://github.com/nirmalyaghosh/llm-router-ledger
cd llm-router-ledger
uv sync --extra dev
pytest tests/unit
```

Verify a local Ollama setup end-to-end with
`python examples/smoke_test_ollama.py` (see prerequisites at the top
of the script).

## License

MIT. See `LICENSE`.
