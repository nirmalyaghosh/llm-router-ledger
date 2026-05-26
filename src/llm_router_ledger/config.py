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

import yaml

from datetime import date
from pathlib import Path
from typing import (
    Any,
    Literal,
)

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
)

from llm_router_ledger.exceptions import (
    ConfigError,
    MissingApiKeyError,
)

load_dotenv()


ProviderName = Literal[
    "anthropic",
    "azure",
    "bytedance",
    "deepseek",
    "gemini",
    "local-openai-compat",
    "minimax",
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
    pricing_url: str | None = None
    pricing_checked: date | None = None
    pricing_notes: str | None = None

    @property
    def days_since_checked(self) -> int | None:
        """Days since pricing was last verified. None if never checked."""
        if self.pricing_checked is None:
            return None
        return (
            date.today() - self.pricing_checked
        ).days

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool = False,
    ) -> float:
        """Estimate cost in USD for a single request."""
        if (
            cache_hit
            and self.cache_hit_input_per_1m
        ):
            input_rate = (
                self.cache_hit_input_per_1m
            )
        else:
            input_rate = self.input_per_1m
        return (
            input_tokens * input_rate
            + output_tokens * self.output_per_1m
        ) / 1_000_000


class DefaultsConfig(BaseModel):
    """Default values inherited by all endpoints."""

    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_factor: float = 2.0


class EndpointConfig(BaseModel):
    """Single LLM endpoint definition."""

    name: str = ""
    provider: ProviderName
    model: str
    api_key_env: str
    base_url: str | None = None
    region: str | None = None
    context_window: int | None = None
    cost: CostConfig | None = None
    notes: str | None = None

    azure_deployment: str | None = None
    azure_api_version: str | None = None

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
        """Check if API key is set without raising."""
        return bool(
            os.environ.get(self.api_key_env)
        )


class LLMConfig(BaseModel):
    """Top-level config: all endpoints plus role mappings."""

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
        """Return endpoints whose API keys are actually set."""
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
        """
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

    Strips a single leading provider prefix so e.g.
    "openrouter:qwen/qwen3.5-9b" matches an endpoint whose model field is
    "qwen/qwen3.5-9b". Returns default if no match is found.
    """
    if config is None:
        config = load_config()
    bare = (
        model.split(":", 1)[-1]
        if ":" in model
        else model
    )
    for ep in config.endpoints.values():
        if (
            ep.model == bare
            and ep.context_window
        ):
            return ep.context_window
    return default


def load_config(path: str | Path | None = None) -> LLMConfig:
    """
    Load and validate config from YAML.

    Defaults to llm_endpoints.yaml in the current working directory.
    Raises ConfigError if the file does not exist.
    """
    if path is None:
        path = (
            Path.cwd() / "llm_endpoints.yaml"
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
        endpoints[name] = EndpointConfig(
            **merged,
        )

    return LLMConfig(
        defaults=defaults,
        endpoints=endpoints,
        roles=raw.get("roles") or {},
    )
