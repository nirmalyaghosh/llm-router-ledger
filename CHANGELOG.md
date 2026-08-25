# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Added: `usage_details` on chat responses, holding `cost`, `is_byok`, `upstream_provider`, flattened `completion_*` / `prompt_*` token details, `completion_tool_call_count`, and `finish_reason`; `CostConfig.cache_write_input_per_1m`; `send_message(messages=...)` for multi-turn calls; `provider: lmstudio`, verified end-to-end against a local LM Studio server for both chat and embeddings. LM Studio reports `prompt_tokens` and `total_tokens` as zero on embeddings, so those rows record a token count of 0. `usage_details.unmapped`, holding any usage key an adapter has no mapping for, so nothing a provider reports is dropped; it surfaces Anthropic's cache fields, DeepSeek's `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`, Qwen's `text_tokens` and Azure's `latency_checkpoint`. Keys under `unmapped` are unstable: one that later gains an explicit mapping moves up a level. `AuthenticationError` (401/403), `InsufficientBalanceError` (402), `ProviderUnavailableError` (5xx and transport failures) and `RateLimitedError` (429), all subclasses of `ProviderError`, each carrying `status_code`. An `llm_error` ledger event, written when a call raises, sharing the failed request's `request_id`.
- Changed: `send_message()` / `create_embeddings()` return `ChatResult` / `EmbeddingResult` namedtuples; use `.text` / `.vectors`, `.usage`, `.generation_id`. `CostConfig.estimate_cost()` takes `cached_tokens` / `cache_write_tokens` counts instead of a `cache_hit` bool, and honours an explicit `0.0` rate as free. `UsageTracker` previews are redacted by default; pass `preview_length` to opt back in. `ProviderAdapter.send()` takes `messages` instead of `system` / `user`, breaking custom adapters. Provider SDK exceptions no longer reach the caller: they are wrapped in the `ProviderError` family, with the original on `__cause__`, so catching a failure no longer means importing `openai` or `anthropic`.
- Removed: `provider: local-openai-compat`, deprecated since 0.1.2; `load_config()` raises `ConfigError` naming `ollama` / `lmstudio` as replacements.
- Deprecated: positional unpacking of the result tuple (logs a warning on each use); removed in 0.3.0 when new fields land on either result type. The `roles` config block and `LLMConfig.get_role_endpoints()` (logs a warning on each use); removed in 0.3.0 in favour of route groups.
- Fixed: `get_context_window()` returned the default for an OpenRouter `:free` model string, reducing it to `free` while stripping a provider prefix. The string is now matched as given first.

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
