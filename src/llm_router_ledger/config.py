"""
LLM endpoints configuration loader.

Reads llm_endpoints.yaml, resolves API keys from environment
variables, and exposes typed accessors. A file the loader cannot read,
cannot parse, or cannot use raises ConfigError, so a consumer catching
LLMCallError sees a readable message rather than a YAML or Pydantic
error. A key the loader does not recognise is rejected, so a typo in a
setting name is reported rather than quietly doing nothing.

The file declares endpoints, optional defaults inherited by every
endpoint, and optional route groups, which name candidate endpoints
and the strategy for choosing between them.

Usage:
    from llm_router_ledger.config import load_config

    config = load_config()
    ep = config.endpoints["deepseek-direct"]
    print(ep.cost.input_per_1m)

    group = config.get_route_group(project="reporting", name="quick")

The path comes from the argument, then LRL_CONFIG_PATH, then
llm_endpoints.yaml in the working directory. A .env file in the
working directory is loaded at import time, so API keys referenced by
api_key_env resolve without extra setup.
"""

from __future__ import annotations

import math
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
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from llm_router_ledger.exceptions import (
    ConfigError,
    MissingApiKeyError,
)

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
    "nvidia",
    "ollama",
    "openai",
    "openrouter",
    "qwen",
    "xiaomi",
    "zhipu",
]

RouteStrategy = Literal[
    "cheapest",
    "priority",
]


class _UniqueKeyLoader(yaml.SafeLoader):
    """
    Helper class used to reject a mapping that names the same key
    twice. PyYAML keeps the last one, so a duplicated endpoint block
    would silently replace the one above it and the file would still
    load.
    """

    def construct_mapping(
        self,
        node: Any,
        deep: bool = False,
    ) -> dict[Any, Any]:
        """
        Build a mapping, raising ConfigError on a repeated key.
        """
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                # The parent resolves a << merge key, and a merged
                # key that the block also sets explicitly is an
                # override rather than a duplicate.
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen:
                    raise ConfigError(
                        f"Duplicate key {key!r} on line"
                        f" {key_node.start_mark.line + 1}"
                    )
                seen.add(key)
            except TypeError:
                continue
        return super().construct_mapping(node, deep=deep)


class CostConfig(BaseModel):
    """
    Token pricing. All rates USD per 1M tokens.

    pricing_url must point to a first-party official page only. No
    aggregators, no third-party calculators.
    """

    model_config = ConfigDict(extra="forbid")

    input_per_1m: float
    output_per_1m: float
    cache_hit_input_per_1m: float | None = None
    cache_write_input_per_1m: float | None = None
    pricing_url: str | None = None
    pricing_checked: date | None = None
    pricing_notes: str | None = None

    @field_validator(
        "input_per_1m",
        "output_per_1m",
        "cache_hit_input_per_1m",
        "cache_write_input_per_1m",
    )
    @classmethod
    def _check_rate(cls, value: float | None) -> float | None:
        """
        Reject a rate that is negative or not a finite number, and
        normalise negative zero. A negative rate writes a negative
        charge to the ledger, and NaN compares false against every
        other rate, so any comparison of one endpoint against another
        would depend on the order they were read in.
        """
        if value is None:
            return value
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "must be a finite number and not negative"
            )
        if value == 0:
            return 0.0
        return value

    @field_validator(
        "input_per_1m",
        "output_per_1m",
        "cache_hit_input_per_1m",
        "cache_write_input_per_1m",
        mode="before",
    )
    @classmethod
    def _reject_boolean_rate(cls, value: Any) -> Any:
        """
        Reject a rate written as a boolean. YAML reads yes, no, true
        and false as booleans, which would otherwise be read as 1.0
        and 0.0, so an endpoint priced no would look free.
        """
        return _reject_boolean(value)

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

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_factor: float = Field(
        default=2.0,
        gt=0,
        allow_inf_nan=False,
    )

    @field_validator(
        "max_retries",
        "retry_backoff_factor",
        "timeout_seconds",
        mode="before",
    )
    @classmethod
    def _reject_boolean_number(cls, value: Any) -> Any:
        """
        Reject a value written as a boolean, which Pydantic would
        otherwise read as 1 or 0.
        """
        return _reject_boolean(value)


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

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    provider: ProviderName
    model: str
    api_key_env: str
    base_url: str | None = None
    region: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    embedding_dimensions: int | None = Field(
        default=None,
        gt=0,
    )
    cost: CostConfig | None = None
    extra_body: dict[str, Any] | None = None
    notes: str | None = None

    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_factor: float = Field(
        default=2.0,
        gt=0,
        allow_inf_nan=False,
    )

    @field_validator(
        "context_window",
        "embedding_dimensions",
        "max_retries",
        "retry_backoff_factor",
        "timeout_seconds",
        mode="before",
    )
    @classmethod
    def _reject_boolean_number(cls, value: Any) -> Any:
        """
        Reject a value written as a boolean. An endpoint given
        timeout_seconds: no would otherwise be read as 0 and time
        every call out immediately.
        """
        return _reject_boolean(value)

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
    Top-level config: the inherited defaults, all endpoints, and the
    route groups.
    """

    defaults: DefaultsConfig = Field(
        default_factory=DefaultsConfig,
    )
    endpoints: dict[str, EndpointConfig] = (
        Field(default_factory=dict)
    )
    route_groups: dict[
        str, dict[str, RouteGroupConfig]
    ] = Field(default_factory=dict)

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

    def get_route_group(
        self,
        *,
        project: str,
        name: str,
    ) -> RouteGroupConfig:
        """
        Resolve a route group, preferring the named project and
        falling back per group to the 'default' project, so a project
        that declares groups of its own still inherits the shared
        ones. Raises ConfigError when neither carries the group.
        """
        group = self.route_groups.get(project, {}).get(name)
        if group is None:
            group = self.route_groups.get(
                "default", {}
            ).get(name)
        if group is None:
            raise ConfigError(
                f"Route group '{name}' not found"
                f" in project '{project}'"
                f" or in the 'default' project"
            )
        return group


class RouteGroupConfig(BaseModel):
    """
    A named set of candidate endpoints and the strategy for choosing
    between them.

    candidates are endpoint names. The priority strategy records that
    they are in preference order; the cheapest strategy records that
    they should be compared on their declared rate, and so requires
    every one of them to declare a cost. notes is free text for
    whoever edits the config.

    name and project are filled in from the two YAML keys the group
    sits under, so a group always knows where it was declared and
    routing does not have to work it out again. Neither may be set in
    the group body.

    The model is frozen and candidates is a tuple, so assigning to a
    field raises. The mappings holding the groups are ordinary dicts,
    so this guards a slip inside one group rather than the config as
    a whole.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = ""
    project: str = ""
    candidates: tuple[str, ...] = ()
    strategy: RouteStrategy = "priority"
    notes: str | None = None


def _build_route_groups(
    *,
    raw_groups: Any,
    endpoints: dict[str, EndpointConfig],
) -> dict[str, dict[str, RouteGroupConfig]]:
    """
    Helper function used to build and validate the route_groups block.

    Every level must be a mapping with string keys, a project or group
    name must not be blank or padded, a group name must not collide
    with an endpoint name, a group must not set its own name or
    project, it must
    name at least one candidate and must not name the same one twice,
    every candidate must name an endpoint the config declares, and
    every candidate of a cheapest group must declare a cost, since a
    candidate without one cannot be compared.
    """
    groups: dict[str, dict[str, RouteGroupConfig]] = {}
    if raw_groups is None:
        raw_groups = {}
    if not isinstance(raw_groups, dict):
        raise ConfigError(
            "'route_groups' must be a mapping of project name to"
            " groups"
        )
    for project, mapping in raw_groups.items():
        if not isinstance(project, str):
            raise ConfigError(
                f"Project name in 'route_groups' must be a"
                f" string, not {type(project).__name__}. Put"
                f" quotes around it in the YAML."
            )
        _check_name(
            value=project,
            where="Project name in 'route_groups'",
        )
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, dict):
            raise ConfigError(
                f"Project '{project}' must be a mapping of group"
                f" name to group"
            )
        groups[project] = {}
        for name, data in mapping.items():
            if not isinstance(name, str):
                raise ConfigError(
                    f"Route group name in project '{project}'"
                    f" must be a string, not"
                    f" {type(name).__name__}. Put quotes around"
                    f" it in the YAML."
                )
            _check_name(
                value=name,
                where=(
                    f"Route group name in project '{project}'"
                ),
            )
            if data is None:
                raise ConfigError(
                    f"Route group '{name}' in project"
                    f" '{project}' has an empty body"
                )
            if not isinstance(data, dict):
                raise ConfigError(
                    f"Route group '{name}' in project"
                    f" '{project}' must be a mapping, not"
                    f" {type(data).__name__}"
                )
            _check_string_keys(
                data=data,
                where=(
                    f"Route group '{name}' in project"
                    f" '{project}'"
                ),
            )
            if data.get("candidates", ()) is None:
                data = {**data, "candidates": ()}
            for key in ("name", "project"):
                if key in data:
                    raise ConfigError(
                        f"Route group '{name}' in project"
                        f" '{project}' must not set '{key}'. It"
                        f" comes from the YAML key it sits under."
                    )
            if name in endpoints:
                raise ConfigError(
                    f"Route group '{name}' in project"
                    f" '{project}' collides with an endpoint of"
                    f" the same name"
                )
            try:
                group = RouteGroupConfig(
                    name=name,
                    project=project,
                    **data,
                )
            except ValidationError as exc:
                raise ConfigError(
                    f"Route group '{name}' in project"
                    f" '{project}' is invalid."
                    f" {_format_errors(exc)}"
                ) from exc
            if not group.candidates:
                raise ConfigError(
                    f"Route group '{name}' in project"
                    f" '{project}' declares no candidates"
                )
            if len(set(group.candidates)) != len(
                group.candidates
            ):
                raise ConfigError(
                    f"Route group '{name}' in project"
                    f" '{project}' names the same candidate"
                    f" twice"
                )
            if not endpoints:
                raise ConfigError(
                    f"Route group '{name}' in project"
                    f" '{project}' names candidates but the"
                    f" config declares no endpoints"
                )
            for candidate in group.candidates:
                _check_name(
                    value=candidate,
                    where=(
                        f"Candidate of route group '{name}' in"
                        f" project '{project}'"
                    ),
                )
                if candidate not in endpoints:
                    raise ConfigError(
                        f"Route group '{name}' in project"
                        f" '{project}' names unknown endpoint"
                        f" '{candidate}'"
                    )
            if group.strategy == "cheapest":
                for candidate in group.candidates:
                    if endpoints[candidate].cost is None:
                        raise ConfigError(
                            f"Route group '{name}' in project"
                            f" '{project}' uses the cheapest"
                            f" strategy, but candidate"
                            f" '{candidate}' declares no cost,"
                            f" so it cannot be compared"
                        )
            groups[project][name] = group
    return groups


def _check_name(*, value: str, where: str) -> None:
    """
    Helper function used to reject a name that is blank or padded with
    spaces, since nobody typing it can then match it.
    """
    if not value.strip() or value != value.strip():
        raise ConfigError(
            f"{where} must not be blank or padded with spaces:"
            f" {value!r}"
        )


def _check_string_keys(
    *,
    data: dict[Any, Any],
    where: str,
) -> None:
    """
    Helper function used to reject a mapping whose keys are not all
    strings, which would otherwise fail when the mapping is passed as
    keyword arguments.
    """
    for key in data:
        if not isinstance(key, str):
            raise ConfigError(
                f"{where} has a setting name that is not a"
                f" string: {key!r}. Put quotes around it in the"
                f" YAML."
            )


def _format_errors(exc: ValidationError) -> str:
    """
    Helper function used to render a Pydantic error as one plain line
    naming each field and what is wrong with it.
    """
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
        for err in exc.errors()
    )


def _reject_boolean(value: Any) -> Any:
    """
    Helper function used to reject a number written as a YAML boolean.
    YAML reads yes, no, true, false, on and off as booleans, and
    Pydantic reads a boolean as 1 or 0, so the mistake is silent.
    """
    if isinstance(value, bool):
        raise ValueError(
            "must be a number, not true, false, yes or no"
        )
    return value


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

    Raises ConfigError for a file that is missing, unreadable, not
    UTF-8, not valid YAML, empty, or whose contents the loader cannot
    use, so a caller catching LLMCallError catches every failure.
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
            f"Config at {path} does not exist"
        )

    if not path.is_file():
        raise ConfigError(
            f"Config at {path} is not a file"
        )

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.load(f, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Config at {path} is not valid YAML. {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"Config at {path} is not UTF-8 text. {exc}"
        ) from exc
    except RecursionError as exc:
        raise ConfigError(
            f"Config at {path} is nested too deeply to read"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Config at {path} could not be read. {exc}"
        ) from exc

    if raw is None or raw == {}:
        raise ConfigError(
            f"Config at {path} is empty"
        )
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config at {path} must be a mapping at the top"
            f" level, not {type(raw).__name__}"
        )
    if "roles" in raw:
        raise ConfigError(
            "The 'roles' block was removed in 0.3.0. Use"
            " 'route_groups' instead: same project and group"
            " nesting, plus a strategy."
        )
    unknown = sorted(
        set(raw) - {"defaults", "endpoints", "route_groups"}
    )
    if unknown:
        raise ConfigError(
            f"Config at {path} has a block the loader does not"
            f" recognise:"
            f" {', '.join(repr(key) for key in unknown)}"
        )

    raw_defaults = raw.get("defaults")
    if raw_defaults is None:
        raw_defaults = {}
    if not isinstance(raw_defaults, dict):
        raise ConfigError(
            "'defaults' must be a mapping"
        )
    _check_string_keys(
        data=raw_defaults,
        where="'defaults'",
    )
    try:
        defaults = DefaultsConfig(**raw_defaults)
    except ValidationError as exc:
        raise ConfigError(
            f"'defaults' is invalid. {_format_errors(exc)}"
        ) from exc

    raw_endpoints = raw.get("endpoints")
    if raw_endpoints is None:
        raw_endpoints = {}
    if not isinstance(raw_endpoints, dict):
        raise ConfigError(
            "'endpoints' must be a mapping of endpoint name to"
            " endpoint definition"
        )

    endpoints: dict[str, EndpointConfig] = {}
    for name, ep_data in raw_endpoints.items():
        if not isinstance(name, str):
            raise ConfigError(
                f"Endpoint name must be a string, not"
                f" {type(name).__name__}. Put quotes around it"
                f" in the YAML."
            )
        _check_name(
            value=name,
            where="Endpoint name",
        )
        if ep_data is None:
            raise ConfigError(
                f"Endpoint '{name}' has an empty body"
            )
        if not isinstance(ep_data, dict):
            raise ConfigError(
                f"Endpoint '{name}' must be a mapping, not"
                f" {type(ep_data).__name__}"
            )
        _check_string_keys(
            data=ep_data,
            where=f"Endpoint '{name}'",
        )
        if "name" in ep_data:
            raise ConfigError(
                f"Endpoint '{name}' must not set 'name'. The"
                f" endpoint name comes from the YAML key."
            )
        merged = {
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
            "name": name,
        }
        if merged.get("provider") == "local-openai-compat":
            raise ConfigError(
                f"Endpoint '{name}' names provider"
                " 'local-openai-compat', which was removed in"
                " 0.2.0,"
                " having been deprecated in 0.1.2. Specify the"
                " local server explicitly: 'ollama' or"
                " 'lmstudio'. The ledger records the configured"
                " provider name, so the two are not"
                " interchangeable."
            )
        try:
            endpoints[name] = EndpointConfig(**merged)
        except ValidationError as exc:
            raise ConfigError(
                f"Endpoint '{name}' is invalid."
                f" {_format_errors(exc)}"
            ) from exc

    return LLMConfig(
        defaults=defaults,
        endpoints=endpoints,
        route_groups=_build_route_groups(
            raw_groups=raw.get("route_groups"),
            endpoints=endpoints,
        ),
    )
