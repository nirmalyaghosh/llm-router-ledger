# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
