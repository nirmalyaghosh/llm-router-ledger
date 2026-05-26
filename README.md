# llm-router-ledger

Route any LLM call through one `send_message()` and keep a JSONL ledger
of every request and response for offline cost reconciliation.

## Provider support

| Provider | Status | Adapter |
|---|---|---|
| OpenRouter | Supported | OpenAI-compat |
| Anthropic | Planned | direct |
| Azure OpenAI | Planned | direct |
| DeepSeek | Planned | direct |
| Gemini | Planned | direct |
| MiniMax | Planned | direct |
| OpenAI | Planned | OpenAI-compat |
| Qwen | Planned | direct |
| Zhipu / GLM | Planned | direct |
| ByteDance Seed | Planned | via OpenRouter |
| Xiaomi MiMo | Planned | via OpenRouter |
| Local Ollama | Supported | OpenAI-compat |

In 0.1.0 OpenRouter and Local Ollama are verified end-to-end. All other
rows describe planned providers and land in their own minor releases.

## Install

```bash
uv pip install llm-router-ledger
```

## Quickstart

Set `OPENROUTER_API_KEY` in `.env` and create `llm_endpoints.yaml` in
the working directory. The fastest path is to copy the example file
and edit it:

```bash
cp examples/llm_endpoints.example.yaml llm_endpoints.yaml
```

```python
from llm_router_ledger import UsageTracker, send_message

tracker = UsageTracker(
    log_path="logs/usage.jsonl",
    project_id="my-blog",
)
text, usage, gen_id = send_message(
    endpoint_name="openrouter-gpt-4.1-nano",
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

`send_message()` returns `(response_text, usage_dict, generation_id)`.
`UsageTracker` appends paired `llm_request` / `llm_response` events to
the JSONL log, stamped with `project_id`, `run_tag`, `run_label`, and
`purpose` for later grouping.

## CLI

```bash
llm-router-ledger list                          # show configured endpoints
llm-router-ledger validate llm_endpoints.yaml   # validate the YAML
llm-router-ledger stale --days 30               # endpoints with stale pricing
llm-router-ledger chat --endpoint openrouter-gpt-4.1-nano \
    --system "You are concise." --user "Hello." \
    --log-path logs/usage.jsonl --project-id my-project
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
