# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Added: `usage_details` on chat responses, holding `cost`, `is_byok`, `upstream_provider`, flattened `completion_*` / `prompt_*` token details, `completion_tool_call_count`, and `finish_reason`; `CostConfig.cache_write_input_per_1m`; `send_message(messages=...)` for multi-turn calls.
- Changed: `send_message()` and `create_embeddings()` return a `ChatResult` / `EmbeddingResult` namedtuple; access as `.text` / `.vectors`, `.usage`, `.generation_id`. `CostConfig.estimate_cost()` takes `cached_tokens` / `cache_write_tokens` counts instead of a `cache_hit` bool, so a single call can mix cached, cache-write, and uncached input; raises `ValueError` if `cached_tokens + cache_write_tokens > input_tokens`. An explicit `0.0` rate is now honored as free rather than treated as unset. `UsageTracker` prompt/response previews are redacted by default (`preview_length` now defaults to 0); pass a positive `preview_length` to opt back in to stored previews. `ProviderAdapter.send()` takes `messages` in place of `system` / `user`, a break for custom adapters; `send_message()`'s `user` is now optional and raises `ValueError` unless exactly one of `user` or `messages` is supplied.
- Deprecated: positional unpacking of the result tuple (logs a warning on each use); removed in 0.3.0 when new fields land on either result type.

## [0.1.4] - 2026-08-01

- Added: `create_embeddings()` returning `(vectors, usage, generation_id)`; `EmbeddingAdapter` with an OpenAI-compatible implementation; `EndpointConfig.embedding_dimensions`, declarative metadata never sent on the wire but enforced on the response (`ProviderError` on a width mismatch); `modality` in JSONL, omitted on text calls so existing rows are unchanged; `usage_details` on embedding responses (`dimensions`, `embedding_count`, and `cost` / `is_byok` / `upstream_provider` where reported); 9 OpenRouter embedding models and local Ollama `qwen3-embedding:0.6b`; embedding smoke tests for both.
- Known limitation: embeddings are verified for `openrouter` and `ollama` only, and every other provider raises `NotImplementedError`, including `local-openai-compat`; Ollama returns no response id, leaving `provider_response_id` empty.

## [0.1.3] - 2026-07-20

- Added: per-endpoint `extra_body` in `llm_endpoints.yaml` (a call-level `extra_body` replaces it, no merge); `UsageTracker.subscribe()` for mirroring ledger entries to another store.
- Changed: pin `openai<3` to guard against future major-version breaks; sdist ships an explicit allowlist of files; example config and README quickstart use `xiaomi/mimo-v2.5` (mimo-v2-flash retired at OpenRouter).
- Fixed: `load_config()` now honors the documented `LRL_CONFIG_PATH` env var.
- Known limitation: `provider: anthropic` ignores `extra_body`.

## [0.1.2] - 2026-05-31

- Added: `provider` field in JSONL; verified Anthropic (native Messages API) / Azure / DeepSeek / MiniMax / OpenAI / Qwen / Zhipu; `[anthropic]` extra.
- Changed: Azure uses `OpenAI(base_url=.../openai/v1/)`.
- Deprecated: `provider: local-openai-compat` (use `ollama` etc.).
- Removed: `azure_deployment`, `azure_api_version`.

## [0.1.1] - 2026-05-26

Initial release. See [README](README.md) for features and usage.
