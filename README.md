# llm-router-ledger

Route LLM calls through one `send_message()` for text and one
`create_embeddings()` for vectors, and keep a JSONL ledger of every
request and response for offline cost reconciliation.

## Provider support

| Status | Adapter | Providers |
|---|---|---|
| Supported | direct | Anthropic |
| Supported | OpenAI-compat | Azure OpenAI, DeepSeek, Local LM Studio, Local Ollama, MiniMax, NVIDIA NIM, OpenAI, OpenRouter, Qwen, Zhipu / GLM |
| Supported | via OpenRouter | ByteDance Seed, InclusionAI Ling, Nvidia Nemotron, Xiaomi MiMo |
| Planned | direct | Gemini |

- Every "Supported" row is live-smoke-verified end-to-end.
- Anthropic requires the optional `[anthropic]` extra: `uv pip install llm-router-ledger[anthropic]`.
- For the "via OpenRouter" families, use `provider: openrouter` with the appropriate model id.
- Nemotron is reachable both ways. `provider: nvidia` goes direct to NIM, which reports token counts only; `provider: openrouter` reports cost and upstream and is the route to reconcile against.
- The table above is about text. Embeddings are gated separately and verified on OpenRouter, Ollama and LM Studio only; see [Embeddings](#embeddings).

### Free chat models

Verified end-to-end via OpenRouter and configured in `examples/llm_endpoints.example.yaml`. Rates are USD per 1M tokens. Both endpoints declare an explicit `0.00` rather than omitting `cost`, so the ledger records their tokens the same way it does a paid endpoint.

| Model | In | Out | Context |
|---|---|---|---|
| `nvidia/nemotron-3.5-content-safety:free` | 0.00 | 0.00 | 128000 |
| `nvidia/nemotron-3.5-lightning:free` | 0.00 | 0.00 | 1000000 |

Free models share their capacity with everyone else using them, so a call fails with HTTP 429 when they are busy. One of the two above failed nine times in a row during verification.

`nvidia/nemotron-3.5-content-safety:free` is a safety classifier, not a general chat model. It answers every prompt with a verdict, so `What is 17 * 23?` returns `User Safety: safe`.

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

Locally via Ollama: `qwen3-embedding:0.6b`, 1024 dims, 32768 context (`ollama pull qwen3-embedding:0.6b`). The same model at Q8_0 runs under LM Studio as `text-embedding-qwen3-embedding-0.6b`, downloaded from the Discover tab, so local runs on either server are directly comparable.

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

`ollama` and `lmstudio` are verified for embeddings. Other local servers are not, so they are refused despite serving the same OpenAI-compatible API.

Neither local server returns a response id, leaving `provider_response_id` empty. LM Studio additionally reports `prompt_tokens` and `total_tokens` as zero for embeddings, at any input size, so its rows record the vectors and their width but a token count of 0 rather than the true figure. Ollama reports real counts. Nothing is billed on either, so there is no invoice to reconcile against.

### Smoke tests

```bash
python examples/smoke_test_openrouter_embeddings.py                                       # free endpoint
python examples/smoke_test_openrouter_embeddings.py --endpoint openrouter-embed-qwen3-8b
python examples/smoke_test_ollama_embeddings.py                                           # local, no cost
python examples/smoke_test_lmstudio_embeddings.py                                         # local, no cost
```

Each takes `--input-file`, one text to embed per line, in place of the sample corpus.

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

## Recording calls this library did not make

An agent framework owns its own call path, so `send_message()` never
runs and the ledger never sees the tokens. `UsageTracker` records those
calls too, producing the same two events, with the same keys, that a
call through this library produces.

Record a call directly:

```python
request_id = tracker.record_request(
    model="xiaomi/mimo-v2.5",
    provider="openrouter",
    purpose="query-planning",
)
# ... something else makes the call ...
tracker.record_response(
    request_id=request_id,
    model="xiaomi/mimo-v2.5",
    usage=raw_usage,
    response_id=response.id,
    provider="openrouter",
)
```

`usage` is the provider's usage mapping as reported. The three token
keys are lifted into the `usage` block and everything else is written
under `usage_details`, the same split `send_message()` performs.
`response_id` is routed the same way too: an id prefixed `gen-` lands in
`generation_id`, anything else in `provider_response_id`.

Or record a whole [Pydantic AI](https://ai.pydantic.dev) run at once:

```python
result = await agent.run("...")
tracker.record_run(
    result.all_messages(),
    purpose="query-planning",
    provider="openrouter",
)
```

One request and response pair is written per model call, so a run that
called a tool three times produces three pairs. The messages are read by
duck typing, so this costs no dependency on pydantic-ai.

- Pass `provider`. The provider name on the message is the framework's
  own, and is `openai` for every OpenAI-compatible server, so without it
  a local call is filed as an OpenAI one.
- Pydantic AI's usage fields are translated to the names the adapters
  already write, so rows from both sources join on one vocabulary. That
  includes `finish_reason`, recorded in the provider's own words rather
  than the framework's normalised ones.
- `RequestUsage.cost` is recorded as `usage_details.estimated_cost`,
  never as `cost`: it is computed from a local price table rather than
  reported by the provider, and `cost` is reserved for what the provider
  said it billed. A provider's own reported cost does not survive the
  trip at all, since the framework keeps only integer usage fields, so
  reconcile these rows against the provider's export by response id.
- A finished message list carries no record of why each call was made,
  so one `purpose` is stamped across the whole run and retries within it
  inherit it. Use a purpose scope where per-call purpose matters.
- A run that raises produces no rows at all, unlike `send_message()`,
  which logs the request before making the call. Both are correct, but
  it changes what an unpaired `llm_request` means. `ledger_model()`
  below does log the request first, so it records a failed call.

## Recording a Pydantic AI agent automatically

`record_run()` needs a second call and only sees a finished run.
`ledger_model()` builds the model for an endpoint in
`llm_endpoints.yaml` and wraps it, so the agent records itself:

```python
from pydantic_ai import Agent

from llm_router_ledger.integrations.pydantic_ai import ledger_model

model = ledger_model("openrouter-mimo-v2.5", tracker=tracker)
agent = Agent(model, instructions="Answer briefly.")
result = await agent.run("...")
```

Needs the extra: `uv pip install llm-router-ledger[pydantic-ai]`.

The endpoint's `provider`, `model`, `base_url`, `api_key_env`,
`extra_body`, `timeout_seconds` and `max_retries` all apply, so the
agent talks to the endpoint on the same terms `send_message()` would,
and the rows carry the endpoint's own provider rather than the
framework's.

For a successful run the two paths write the same rows. Prefer this one
unless the model is not yours to build, because it also:

- records a call that raised, as an `llm_error` pairing the
  `llm_request` written before the call, exactly as `send_message()`
  does. A run that raises never reaches `record_run()`.
- resolves `purpose` per call rather than once per run, so a
  `purpose_scope` entered inside a run is honoured.
- covers streaming. Usage is not final until a stream ends, so the
  response event is written once it is exhausted; a stream cut short
  still records the tokens it spent.

`purpose` and `metadata` can also bind for the life of the model:

```python
model = ledger_model(
    "openrouter-mimo-v2.5",
    tracker=tracker,
    purpose="query-planning",
    metadata={"experiment": "a"},
)
```

An active purpose scope overrides the bound `purpose`, which is how one
model shared by several agents keeps them apart in the ledger.

The cost limitation above applies here too: the framework keeps only
integer usage fields, so a provider's reported cost never reaches these
rows either. Reconcile by response id.

See `examples/agentic_rag/posthoc.py` and `ledger_model.py`, which run
the same five-agent pipeline through each option.

## Setting a purpose an agent cannot pass

`send_message()` takes `purpose` per call, but by the time a framework's
request reaches the ledger there is no argument left to carry it. Set it
around the work instead:

```python
from llm_router_ledger import purpose_scope

with purpose_scope("query-planning"):
    result = await agent.run("...")
    tracker.record_run(result.all_messages())
```

- The scope is a context variable, so it is per-task and per-thread: two
  agents running concurrently under asyncio each keep their own purpose.
- Scopes nest and the innermost wins. Entering a scope with `""` is how
  a nested call records with no purpose rather than inheriting the one
  around it.
- A `purpose` passed to the call wins over the scope, and the scope wins
  over `UsageTracker(default_purpose=...)`. The narrowest thing that was
  actually set is what reaches the ledger.

## JSONL ledger schema

- `UsageTracker` writes two events per `send_message()` or `create_embeddings()` call: an `llm_request` before the call, and an `llm_response` after.
- Both share a `request_id` so they can be paired. Top-level fields on each event include `project_id`, `provider`, `model`, `purpose`, `run_tag`, `run_label`, and `timestamp`.
- The `llm_response` event additionally carries `usage` (with `prompt_tokens`, `completion_tokens`, `total_tokens`) and a response preview.
- Previews are redacted by default: `system_prompt_preview`, `user_prompt_preview`, and `response_preview` are written as `"[REDACTED]"` when the underlying text is non-empty, `""` when it genuinely is empty. Pass `preview_length` (a positive character count) to `UsageTracker()` to opt in to storing a truncated preview instead; the length and token counts are always recorded either way.
- A failed call writes an `llm_error` event sharing the `request_id` of its `llm_request`, carrying `error_type` (the original SDK exception's class name), `error_message`, and `status_code` where the provider returned one. A third event type rather than an `llm_response` with an error field, because a failed call has no tokens and writing zeroes would corrupt anyone summing them. The SDK retries internally before raising, so one `llm_error` stands for however many attempts it made.
- `usage_details` on the response holds everything the provider reported beyond the three token keys, written only when non-empty. `usage` keeps the same fixed three-key shape regardless of what lands in `usage_details`, across both modalities.
  - **Embedding calls**: `dimensions` and `embedding_count` always, plus `cost`, `is_byok` and `upstream_provider` where available.

Chat calls map provider fields onto ledger keys as follows. A key is written only when the provider reports a non-zero value for it.

| Provider reports | Ledger key | Observed on |
|---|---|---|
| `usage.prompt_tokens` / `completion_tokens` / `total_tokens` | `usage.*`, unchanged | all |
| Anthropic `usage.input_tokens` / `output_tokens` | `usage.prompt_tokens` / `completion_tokens` | Anthropic |
| `usage.cost`, `usage.is_byok` | `usage_details.cost`, `.is_byok` | OpenRouter |
| response `provider` | `usage_details.upstream_provider` | OpenRouter |
| `completion_tokens_details.reasoning_tokens` | `usage_details.completion_reasoning_tokens` | OpenRouter, Qwen, Zhipu |
| `prompt_tokens_details.cached_tokens` | `usage_details.prompt_cached_tokens` | OpenRouter, DeepSeek, Zhipu |
| other keys in either `*_tokens_details` block | same name, `completion_` / `prompt_` prefixed | varies |
| anything else the provider reports | `usage_details.unmapped.<key>` | see below |

`usage_details.completion_reasoning_tokens` and `usage_details.prompt_cached_tokens` are subsets of `usage.completion_tokens` and `usage.prompt_tokens`, not additions to them, so adding either to its parent double-counts. Verified against three paid OpenRouter endpoints: the reported `cost` matched the inclusive reading to the cent, while the additive reading overstated it by 21 to 58 percent.

Two keys are derived rather than reported: `completion_tool_call_count`, the number of tool calls on a turn that made any, and `finish_reason`, written only when the turn ended abnormally (e.g. `length`, truncated at `max_tokens`) in the provider's own vocabulary. A tool-call turn has no text, so it records `response_length` 0; the count is what distinguishes it from a model that answered with nothing.

`usage_details.unmapped` holds provider fields the library has no mapping for, so nothing a provider reports is silently discarded. Observed examples: DeepSeek's `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`, which duplicate `prompt_cached_tokens`; Qwen's `completion_text_tokens` and `prompt_text_tokens`; Azure's `latency_checkpoint` timing block; Anthropic's `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation`, `service_tier` and `inference_geo`. OpenAI reports nothing unmapped.

Treat `unmapped` as unstable. A key that later gains an explicit mapping moves out of `unmapped` and up a level, so read from it defensively.

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
