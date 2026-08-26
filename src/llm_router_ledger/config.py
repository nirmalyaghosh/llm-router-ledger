"""
LLM endpoints configuration loader.

Loads llm_endpoints.yaml, validates with Pydantic, resolves API keys from
environment variables, and exposes typed accessors.

Usage:
    from llm_router_ledger.config import load_config

    config = load_config()
    ep = config.endpoints["deepseek-v3.2"]
    print(ep.cost.input_per_1m)

A .env file in the current working directory is loaded automatically at
import time so API keys referenced via api_key_env resolve without extra
setup.
"""

from __future__ import annotations

import os

from datetime import date
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import yaml

from dotenv import (
    find_dotenv,
    load_dotenv,
)
from pydantic import (
    BaseModel,
    Field,
)

from llm_router_ledger._logger import get_logger
from llm_router_ledger.exceptions import (
    ConfigError,
    MissingApiKeyError,
)

logger = get_logger(__name__)

# find_dotenv searches upward from this module's own directory by
# default, which resolves to site-packages for an installed
# distribution and therefore never reaches the caller's project. A
# .env in the working directory was ignored unless the package had
# been installed from source within that project. usecwd=True searches
# from the working directory instead, matching where
# llm_endpoints.yaml is resolved.
#
# load_dotenv does not override variables already present in the
# environment, so existing configuration takes precedence.
load_dotenv(find_dotenv(usecwd=True))


ProviderName = Literal[
    "anthropic",
    "azure",
    "bytedance",
    "deepseek",
    "gemini",
    "lmstudio",
    "minimax",
    "ollama",
    "openai",
    "openrouter",
    "qwen",
    "xiaomi",
    "zhipu",
]


class CostConfig(BaseModel):
    """
    Token pricing. All rates USD per 1M tokens.

    pricing_url must point to a first-party official page only. No
    aggregators, no third-party calculators.
    """

    input_per_1m: float
    output_per_1m: float
    cache_hit_input_per_1m: float | None = None
    cache_write_input_per_1m: float | None = None
    pricing_url: str | None = None
    pricing_checked: date | None = None
    pricing_notes: str | None = None

    @property
    def days_since_checked(self) -> int | None:
        """
        Days since pricing was last verified. None if never checked.
        """
        if self.pricing_checked is None:
            return None
        return (
            date.today() - self.pricing_checked
        ).days

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """
        Estimate cost in USD for a single request.

        input_tokens is the full prompt token count, matching how a
        provider reports it; cached_tokens and cache_write_tokens are
        the subsets of it billed at cache_hit_input_per_1m and
        cache_write_input_per_1m respectively, since a real call can mix
        cached, cache-write, and uncached input in one prompt rather
        than being purely one or the other. Whichever rate is unset
        falls back to input_per_1m, including when it is explicitly 0.0
        (a genuinely free tier, not "unset").

        Raises ValueError if cached_tokens plus cache_write_tokens
        exceeds input_tokens, which would otherwise silently produce a
        negative charge for the remaining uncached tokens.
        """
        if cached_tokens + cache_write_tokens > input_tokens:
            raise ValueError(
                f"cached_tokens ({cached_tokens}) plus"
                f" cache_write_tokens ({cache_write_tokens})"
                f" exceeds input_tokens ({input_tokens})"
            )
        uncached_tokens = (
            input_tokens
            - cached_tokens
            - cache_write_tokens
        )
        cache_hit_rate = (
            self.cache_hit_input_per_1m
            if self.cache_hit_input_per_1m is not None
            else self.input_per_1m
        )
        cache_write_rate = (
            self.cache_write_input_per_1m
            if self.cache_write_input_per_1m is not None
            else self.input_per_1m
        )
        return (
            uncached_tokens * self.input_per_1m
            + cached_tokens * cache_hit_rate
            + cache_write_tokens * cache_write_rate
            + output_tokens * self.output_per_1m
        ) / 1_000_000


class DefaultsConfig(BaseModel):
    """
    Default values inherited by all endpoints.
    """

    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_factor: float = 2.0


class EndpointConfig(BaseModel):
    """
    Single LLM endpoint definition.

    embedding_dimensions is the width of the vectors an embedding
    endpoint returns. It is declarative metadata, never sent on the
    wire, so consumers can size a vector column or collection without
    first making a call; it is not OpenAI's `dimensions` request
    parameter and setting it does not truncate anything. When set,
    create_embeddings raises ProviderError if the provider returns a
    different width, which catches an upstream re-route before malformed
    vectors reach an index. Leave unset on chat endpoints.
    """

    name: str = ""
    provider: ProviderName
    model: str
    api_key_env: str
    base_url: str | None = None
    region: str | None = None
    context_window: int | None = None
    embedding_dimensions: int | None = None
    cost: CostConfig | None = None
    extra_body: dict[str, Any] | None = None
    notes: str | None = None

    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_factor: float = 2.0

    @property
    def api_key(self) -> str:
        """
        Resolve API key from environment. Raises MissingApiKeyError if
        the variable named by api_key_env is unset or empty.
        """
        value = os.environ.get(self.api_key_env)
        if not value:
            raise MissingApiKeyError(
                f"Missing env var"
                f" '{self.api_key_env}'"
                f" required by endpoint"
                f" '{self.name}'"
            )
        return value

    @property
    def api_key_available(self) -> bool:
        """
        Check if API key is set without raising.
        """
        return bool(
            os.environ.get(self.api_key_env)
        )


class LLMConfig(BaseModel):
    """
    Top-level config: all endpoints plus role mappings.
    """

    defaults: DefaultsConfig = Field(
        default_factory=DefaultsConfig,
    )
    endpoints: dict[str, EndpointConfig] = (
        Field(default_factory=dict)
    )
    roles: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
    )

    def available(self) -> list[EndpointConfig]:
        """
        Return endpoints whose API keys are actually set.
        """
        return [
            ep
            for ep in self.endpoints.values()
            if ep.api_key_available
        ]

    def by_provider(self, provider: ProviderName) -> list[EndpointConfig]:
        """
        Return all endpoints for a given provider.
        """
        return [
            ep
            for ep in self.endpoints.values()
            if ep.provider == provider
        ]

    def by_region(self, region: str) -> list[EndpointConfig]:
        """
        Return all endpoints in a given region (e.g. 'cn-beijing',
        'cn-hangzhou').
        """
        return [
            ep
            for ep in self.endpoints.values()
            if ep.region == region
        ]

    def get_role_endpoints(
        self,
        project: str,
        role: str,
    ) -> list[EndpointConfig]:
        """
        Resolve role assignment to endpoint configs. Returns list (role
        may map to a list or a single string in YAML).

        Deprecated: removed in 0.3.0, replaced by route groups.
        """
        logger.warning(
            "get_role_endpoints() and the `roles` config block are"
            " deprecated and will be removed in 0.3.0, when route"
            " groups replace them. Route groups carry a selection"
            " strategy and stamp the chosen endpoint in the ledger.",
            stacklevel=2,
        )
        mapping = self.roles.get(
            project,
            self.roles.get("default", {}),
        )
        value = mapping.get(role)
        if value is None:
            raise ConfigError(
                f"Role '{role}' not found"
                f" in project '{project}'"
                f" or defaults"
            )
        names = (
            value
            if isinstance(value, list)
            else [value]
        )
        return [
            self.endpoints[n] for n in names
        ]


def get_context_window(
    model: str,
    config: LLMConfig | None = None,
    default: int = 8192,
) -> int:
    """
    Look up context window for a model string.

    The string is matched as given first, so an OpenRouter ":free"
    variant such as "nvidia/nemotron-3.5-lightning:free" resolves to its
    own endpoint. Only when that misses is a single leading provider
    prefix stripped, so "openrouter:qwen/qwen3.5-9b" still matches an
    endpoint whose model field is "qwen/qwen3.5-9b". Returns default if
    no match is found.
    """
    if config is None:
        config = load_config()
    candidates = [model]
    if ":" in model:
        candidates.append(model.split(":", 1)[-1])
    for candidate in candidates:
        for ep in config.endpoints.values():
            if (
                ep.model == candidate
                and ep.context_window
            ):
                return ep.context_window
    return default


def load_config(path: str | Path | None = None) -> LLMConfig:
    """
    Load and validate config from YAML.

    When path is None, falls back to the LRL_CONFIG_PATH environment
    variable, then to llm_endpoints.yaml in the current working
    directory. An explicit path argument always wins over the env var.
    Raises ConfigError if the file does not exist.
    """
    if path is None:
        env_path = os.environ.get("LRL_CONFIG_PATH")
        path = (
            Path(env_path)
            if env_path
            else Path.cwd() / "llm_endpoints.yaml"
        )
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"Config not found: {path}"
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = DefaultsConfig(
        **(raw.get("defaults") or {}),
    )

    endpoints: dict[str, EndpointConfig] = {}
    for name, ep_data in (
        raw.get("endpoints") or {}
    ).items():
        merged = {
            "name": name,
            "timeout_seconds": (
                defaults.timeout_seconds
            ),
            "max_retries": (
                defaults.max_retries
            ),
            "retry_backoff_factor": (
                defaults.retry_backoff_factor
            ),
            **ep_data,
        }
        if merged.get("provider") == "local-openai-compat":
            raise ConfigError(
                f"Endpoint '{name}': provider"
                " 'local-openai-compat' was removed in 0.2.0,"
                " having been deprecated in 0.1.2. Specify the"
                " local server explicitly: 'ollama' or"
                " 'lmstudio'. The ledger records the configured"
                " provider name, so the two are not"
                " interchangeable."
            )
        endpoints[name] = EndpointConfig(
            **merged,
        )

    return LLMConfig(
        defaults=defaults,
        endpoints=endpoints,
        roles=raw.get("roles") or {},
    )
