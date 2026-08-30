"""
The providers whose adapters have been exercised against a live API.

Kept in a module of its own so that routing can filter candidates on
the chat list without importing the dispatcher, which imports
routing. Routing covers send_message only; create_embeddings takes no
route group.
"""

from __future__ import annotations


VERIFIED_EMBEDDING_PROVIDERS = frozenset({
    "lmstudio",
    "ollama",
    "openrouter",
})

VERIFIED_PROVIDERS = frozenset({
    "anthropic",
    "azure",
    "deepseek",
    "lmstudio",
    "minimax",
    "nvidia",
    "ollama",
    "openai",
    "openrouter",
    "qwen",
    "zhipu",
})
