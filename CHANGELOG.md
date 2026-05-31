# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-31

- Added: `provider` field in JSONL; verified Anthropic (native Messages API) / Azure / DeepSeek / MiniMax / OpenAI / Qwen / Zhipu; `[anthropic]` extra.
- Changed: Azure uses `OpenAI(base_url=.../openai/v1/)`.
- Deprecated: `provider: local-openai-compat` (use `ollama` etc.).
- Removed: `azure_deployment`, `azure_api_version`.

## [0.1.1] - 2026-05-26

Initial release. See [README](README.md) for features and usage.
