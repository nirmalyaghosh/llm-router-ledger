# llm-router-ledger

Route LLM calls through one `send_message()` for text and one
`create_embeddings()` for vectors, and keep a JSONL ledger of every
request and response for offline cost reconciliation.

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
- The table above is about text. Embeddings are gated separately and verified on OpenRouter and Ollama only; see [Embeddings](#embeddings).

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
result = send_message(
    endpoint_name="openrouter-mimo-v2.5",
    system="You are concise.",
    user="Explain prompt caching in two sentences.",
    tracker=tracker,
)
```

Or against a local Ollama server, with no API costs:

```python
result = send_message(
    endpoint_name="local-llama",
    system="You are concise.",
    user="Explain prompt caching in two sentences.",
    tracker=tracker,
)
```

- `send_message()` returns a `ChatResult` with `.text`, `.usage`, and `.generation_id`.
- `.usage` adds `cost`, `is_byok`, and `upstream_provider` to the token keys when the provider reports them, plus flattened reasoning / cache detail keys (e.g. `completion_reasoning_tokens`, `prompt_cached_tokens`); see [JSONL ledger schema](#jsonl-ledger-schema).
- `UsageTracker` appends paired `llm_request` / `llm_response` events to the JSONL log, stamped with `project_id`, `run_tag`, `run_label`, and `purpose` for later grouping.
- Prompt and response previews are redacted by default; pass `preview_length` to opt in to storing truncated text, see [JSONL ledger schema](#jsonl-ledger-schema).
- For multi-turn conversations, tool loops, or anything `system` + `user` can't express, pass `messages` instead; it replaces `system` and `user` outright rather than merging with them:

```python
result = send_message(
    endpoint_name="openrouter-mimo-v2.5",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "You are concise."}]},
        {"role": "user", "content": [{"type": "text", "text": "Explain prompt caching."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [{"type": "text", "text": "Now in one sentence."}]},
    ],
    tracker=tracker,
)
```

Each entry is `{"role": ..., "content": [{"type": "text", "text": ...}]}`, the OpenAI content-parts shape. It's kept even though only `"text"` parts are supported today, so adding image input later is additive rather than another break.

## Embeddings

`create_embeddings()` embeds a list of texts and writes the same paired ledger events as `send_message()`.

```python
from llm_router_ledger import UsageTracker, create_embeddings

tracker = UsageTracker(
    log_path="logs/usage.jsonl",
    project_id="my-blog",
)
result = create_embeddings(
    endpoint_name="openrouter-embed-bge-m3",
    texts=["first passage", "second passage"],
    tracker=tracker,
)
```

- `create_embeddings()` returns an `EmbeddingResult` with `.vectors` (one per input, in input order), `.usage`, and `.generation_id`.
- `.usage` adds `dimensions` and `embedding_count` to the token keys, plus `cost`, `is_byok`, and `upstream_provider` when the provider reports them.
- `completion_tokens` is always 0. Embeddings bill input only.

### Verified models

Via OpenRouter:

| Model | Dims | Context |
|---|---|---|
| `baai/bge-base-en-v1.5` | 768 | 512 |
| `baai/bge-m3` | 1024 | 8194 |
| `mistralai/mistral-embed-2312` | 1024 | 8192 |
| `nvidia/nemotron-3-embed-1b:free` | 2048 | 32768 |
| `openai/text-embedding-3-large` | 3072 | 8192 |
| `openai/text-embedding-3-small` | 1536 | 8192 |
| `perplexity/pplx-embed-v1-0.6b` | 1024 | 32000 |
| `qwen/qwen3-embedding-4b` | 2560 | 32768 |
| `qwen/qwen3-embedding-8b` | 4096 | 32768 |

Locally via Ollama: `qwen3-embedding:0.6b`, 1024 dims, 32768 context (`ollama pull qwen3-embedding:0.6b`).

`baai/bge-base-en-v1.5` is English only. Non-English input still returns vectors, with no error.

Prices are per endpoint in `llm_endpoints.yaml`, each with a `pricing_url` and `pricing_checked` date. See `examples/llm_endpoints.example.yaml` for the verified values, and `llm-router-ledger stale` for ones that need rechecking.

### `embedding_dimensions`

An optional endpoint field declaring the vector width.

- Never sent on the wire, so a vector column or collection can be sized without first making a call. It is not OpenAI's `dimensions` request parameter and truncates nothing.
- Enforced on the response: a different width raises `ProviderError` instead of returning vectors that would corrupt a fixed-width index.
- OpenRouter re-routes between calls. `baai/bge-m3` has been served by DeepInfra on one call and Parasail on the next.
- Leave it unset to accept any width.

### Provider gate

Embeddings are refused for providers not verified end-to-end, even where the chat adapter works: `provider: openai` raises `NotImplementedError`.

`ollama` is verified. `local-openai-compat` is not, so vLLM and LM Studio are refused despite serving the same OpenAI-compatible API.

### Smoke tests

```bash
python examples/smoke_test_openrouter_embeddings.py                                       # free endpoint
python examples/smoke_test_openrouter_embeddings.py --endpoint openrouter-embed-qwen3-8b
python examples/smoke_test_ollama_embeddings.py                                           # local, no cost
```

Both take `--input-file`, one text to embed per line, in place of the sample corpus.

## Per-endpoint request params

Model-specific knobs belong in config, not in every caller. Give an endpoint an `extra_body` and it is sent on every call to that endpoint:

```yaml
endpoints:
  openrouter-deepseek:
    provider: openrouter
    model: deepseek/deepseek-chat
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    extra_body:
      reasoning:
        enabled: false
```

- An `extra_body` passed to `send_message()` replaces the endpoint's value outright. The two layers are not merged, so a caller that wants both must combine them itself. An opaque vendor passthrough carries no merge rules to memorise as a result.
- **Known limitation:** `provider: anthropic` ignores `extra_body`, so the field has no effect there. `provider: openrouter` reaches Claude with `extra_body` intact.

## Mirroring usage elsewhere

`UsageTracker.subscribe()` registers a callback that receives every ledger entry, so usage can be mirrored to another store without this library depending on it:

```python
tracker.subscribe(lambda entry: my_container.upsert_item(entry))
```

- Each entry is written to the JSONL ledger before any subscriber runs.
- A callback that raises is logged and skipped. The entry is already in the ledger, the call that produced it is unaffected, and the remaining subscribers still run.
- Each subscriber receives its own copy of the entry.
- Callbacks are synchronous and run on the calling thread, so a slow one delays every call. Queue the work inside the callback if the destination is remote.

## JSONL ledger schema

- `UsageTracker` writes two events per `send_message()` or `create_embeddings()` call: an `llm_request` before the call, and an `llm_response` after.
- Both share a `request_id` so they can be paired. Top-level fields on each event include `project_id`, `provider`, `model`, `purpose`, `run_tag`, `run_label`, and `timestamp`.
- The `llm_response` event additionally carries `usage` (with `prompt_tokens`, `completion_tokens`, `total_tokens`) and a response preview.
- Previews are redacted by default: `system_prompt_preview`, `user_prompt_preview`, and `response_preview` are written as `"[REDACTED]"` when the underlying text is non-empty, `""` when it genuinely is empty. Pass `preview_length` (a positive character count) to `UsageTracker()` to opt in to storing a truncated preview instead; the length and token counts are always recorded either way.
- A failed call leaves an `llm_request` with no matching `llm_response`, because the request is logged before the call is made. Readers should expect unpaired requests.
- `usage_details` on the response holds everything the provider reported beyond the three token keys, written only when non-empty. `usage` keeps the same fixed three-key shape regardless of what lands in `usage_details`, across both modalities.
  - **Chat calls** (OpenAI-compatible providers): `cost`, `is_byok`, and `upstream_provider` where available, plus the flattened contents of `completion_tokens_details` / `prompt_tokens_details` under a `completion_` / `prompt_` prefix (e.g. `completion_reasoning_tokens`, `prompt_cached_tokens`), plus `completion_tool_call_count` on a turn that returned tool calls. A tool-call turn has no text, so it records `response_length` 0; the count is what distinguishes it from a model that answered with nothing. The Anthropic adapter reports none of the cost or detail keys, so Anthropic rows carry no `usage_details` on an ordinary call.
  - **Embedding calls**: `dimensions` and `embedding_count` always, plus `cost`, `is_byok` and `upstream_provider` where available.

**Embedding calls** additionally set `modality: "embedding"` on both events. The key is omitted entirely on text calls, so existing rows are unchanged and an absent `modality` means text. The response preview is empty and `response_length` is 0 for embeddings, since an embedding response carries no text. Neither the input text in full nor the vectors are ever written to the ledger.

**Identifying a response for billing reconciliation:** the response id is routed to one of two fields based on prefix:

- `generation_id`: set when the id starts with `"gen-"` (OpenRouter convention). Use this when joining against OpenRouter's CSV export, which calls the column `generation_id`.
- `provider_response_id`: set for everything else. OpenAI, Azure OpenAI, Ollama, and most direct-provider endpoints return ids like `"chatcmpl-..."` that land here. Use this when joining against OpenAI-family billing exports or any provider-native log that exposes a
  chat completion id.

OpenRouter embedding ids are prefixed `gen-emb-`, so they route to `generation_id` and reconcile like any other OpenRouter call. Ollama returns no id at all, leaving `provider_response_id` empty; nothing is billed there, so there is nothing to reconcile against.

Exactly one of the two fields is populated per `llm_response` event; queries that join the ledger to billing data should `COALESCE` over both or branch on `provider`.

## CLI

```bash
llm-router-ledger list                          # show configured endpoints
llm-router-ledger validate llm_endpoints.yaml   # validate the YAML
llm-router-ledger stale --days 30               # endpoints with stale pricing
llm-router-ledger chat --endpoint openrouter-mimo-v2.5 --system "You are concise." --user "Hello." --log-path logs/usage.jsonl --project-id my-project
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
