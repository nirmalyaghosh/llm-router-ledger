"""
Unit tests for llm_router_ledger.config.
"""

from __future__ import annotations

import re
import subprocess
import sys

from datetime import date
from pathlib import Path

import pytest

from pydantic import ValidationError

from llm_router_ledger.config import (
    CostConfig,
    LLMConfig,
    get_context_window,
    load_config,
)
from llm_router_ledger.exceptions import (
    ConfigError,
    MissingApiKeyError,
)

# An OpenRouter ":free" variant, whose colon sits at the end of the
# model string rather than in front of a provider prefix.
_FREE_VARIANT_YAML = """endpoints:
  free-model:
    provider: openrouter
    model: nvidia/nemotron-3.5-lightning:free
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1
    context_window: 1000000
"""


_COST_ENDPOINT = (
    "endpoints:\n"
    "  a:\n"
    "    provider: openai\n"
    "    model: m\n"
    "    api_key_env: K\n"
    "    cost:\n"
    "      input_per_1m: RATE\n"
    "      output_per_1m: 1.0\n"
)

_GROUP = "route_groups:\n  default:\n    q:\n"

_NUMERIC_ENDPOINT = (
    "endpoints:\n"
    "  a:\n"
    "    provider: openai\n"
    "    model: m\n"
    "    api_key_env: K\n"
    "    SETTING\n"
)

_LOAD_CONFIG_CASES = {
    "not valid yaml": (
        "endpoints:\n  a: [1, 2\n",
        "is not valid YAML",
    ),
    "empty file": ("", "is empty"),
    "top level is a list": (
        "- a\n- b\n",
        "must be a mapping at the top level, not list",
    ),
    "empty mapping": ("{}\n", "is empty"),
    "stale roles block": (
        "roles:\n  default:\n    quick: a\n",
        "The 'roles' block was removed in 0.3.0",
    ),
    "unknown top-level block": (
        "route_group:\n  default: {}\n",
        "has a block the loader does not recognise: 'route_group'",
    ),
    "endpoint has an empty body": (
        "endpoints:\n  a:\n",
        "Endpoint 'a' has an empty body",
    ),
    "endpoints is a list": (
        "endpoints:\n  - a\n",
        "'endpoints' must be a mapping",
    ),
    "endpoint is a scalar": (
        "endpoints:\n  a: b\n",
        "Endpoint 'a' must be a mapping",
    ),
    "endpoint name not a string": (
        "endpoints:\n  1:\n    provider: openai\n",
        "Endpoint name must be a string",
    ),
    "endpoint setting not a string": (
        "endpoints:\n  a:\n    1: 2\n",
        "Endpoint 'a' has a setting name",
    ),
    "endpoint sets its own name": (
        "endpoints:\n  a:\n    name: b\n"
        "    provider: ollama\n    model: m\n"
        "    api_key_env: K\n",
        "Endpoint 'a' must not set 'name'",
    ),
    "duplicate key": (
        "endpoints:\n  a:\n    provider: openai\n"
        "    model: one\n    api_key_env: K\n"
        "  a:\n    provider: deepseek\n"
        "    model: two\n    api_key_env: K\n",
        "Duplicate key 'a'",
    ),
    "endpoint name blank": (
        "endpoints:\n  '':\n    provider: openai\n"
        "    model: m\n    api_key_env: K\n",
        "Endpoint name must not be blank",
    ),
    "number written as a boolean": (
        _NUMERIC_ENDPOINT.replace("SETTING", "max_retries: no"),
        "max_retries: Value error, must be a number, not true,"
        " false, yes or no",
    ),
    "timeout is negative": (
        _NUMERIC_ENDPOINT.replace("SETTING", "timeout_seconds: -30"),
        "Endpoint 'a' is invalid. timeout_seconds",
    ),
    "backoff is not a number": (
        _NUMERIC_ENDPOINT.replace(
            "SETTING",
            "retry_backoff_factor: .nan",
        ),
        "Endpoint 'a' is invalid. retry_backoff_factor",
    ),
    "unknown endpoint setting": (
        _NUMERIC_ENDPOINT.replace("SETTING", "timeout_second: 900"),
        "Endpoint 'a' is invalid. timeout_second",
    ),
    "unknown cost setting": (
        _COST_ENDPOINT.replace("RATE", "1.0")
        + "      outpt_per_1m: 2.0\n",
        "Endpoint 'a' is invalid. cost.outpt_per_1m",
    ),
    "unknown defaults setting": (
        "defaults:\n  timout_seconds: 5\n",
        "'defaults' is invalid. timout_seconds",
    ),
    "unknown provider": (
        "endpoints:\n  a:\n    provider: nosuch\n"
        "    model: m\n    api_key_env: K\n",
        "Endpoint 'a' is invalid. provider: Input should be",
    ),
    "defaults is a scalar": (
        "defaults: hello\n",
        "'defaults' must be a mapping",
    ),
    "defaults bad type": (
        "defaults:\n  timeout_seconds: notanint\n",
        "'defaults' is invalid. timeout_seconds",
    ),
    "negative rate": (
        _COST_ENDPOINT.replace("RATE", "-5.0"),
        "cost.input_per_1m: Value error, must be a finite number",
    ),
    "rate not a number": (
        _COST_ENDPOINT.replace("RATE", ".nan"),
        "cost.input_per_1m: Value error, must be a finite number",
    ),
    "rate written as yes": (
        _COST_ENDPOINT.replace("RATE", "yes"),
        "cost.input_per_1m: Value error, must be a number, not"
        " true, false, yes or no",
    ),
}

_ROUTE_GROUP_ENDPOINTS = (
    "endpoints:\n"
    "  free-one:\n"
    "    provider: ollama\n"
    "    model: llama3.1\n"
    "    api_key_env: OLLAMA_API_KEY\n"
    "  paid-one:\n"
    "    provider: openrouter\n"
    "    model: openai/gpt-4.1-nano\n"
    "    api_key_env: OPENROUTER_API_KEY\n"
    "    cost:\n"
    "      input_per_1m: 0.15\n"
    "      output_per_1m: 0.60\n"
)

_ROUTE_GROUP_CASES = {
    "block is a list": (
        "route_groups: []\n",
        "mapping of project name",
    ),
    "project name not a string": (
        "route_groups:\n  1:\n    q:\n"
        "      candidates: [paid-one]\n",
        "Project name in 'route_groups'",
    ),
    "project name padded": (
        "route_groups:\n  ' proj ':\n    q:\n"
        "      candidates: [paid-one]\n",
        "Project name in 'route_groups' must not be blank",
    ),
    "project is a scalar": (
        "route_groups:\n  default: 3\n",
        "mapping of group name",
    ),
    "group name not a string": (
        "route_groups:\n  default:\n    1:\n"
        "      candidates: [paid-one]\n",
        "Route group name in project 'default' must be a string",
    ),
    "group name padded": (
        "route_groups:\n  default:\n    '  q  ':\n"
        "      candidates: [paid-one]\n",
        "Route group name in project 'default' must not be blank"
        " or padded",
    ),
    "group name blank": (
        "route_groups:\n  default:\n    '':\n"
        "      candidates: [paid-one]\n",
        "Route group name in project 'default' must not be blank",
    ),
    "group has an empty body": (
        "route_groups:\n  default:\n    q:\n",
        "Route group 'q' in project 'default' has an empty body",
    ),
    "candidate padded": (
        _GROUP + "      candidates: ['  paid-one  ']\n",
        "Candidate of route group 'q' in project 'default' must"
        " not be blank or padded",
    ),
    "group is a scalar": (
        "route_groups:\n  default:\n    q: paid-one\n",
        "Route group 'q' in project 'default' must be a mapping,"
        " not str",
    ),
    "setting name not a string": (
        _GROUP + "      1: 2\n      candidates: [paid-one]\n",
        "setting name that is not a string",
    ),
    "group sets its own name": (
        _GROUP + "      name: other\n"
        "      candidates: [paid-one]\n",
        "must not set 'name'",
    ),
    "name collides with endpoint": (
        "route_groups:\n  default:\n    paid-one:\n"
        "      candidates: [paid-one]\n",
        "collides with an endpoint",
    ),
    "unknown setting": (
        _GROUP + "      bogus: 1\n      candidates: [paid-one]\n",
        "is invalid. bogus",
    ),
    "unknown strategy": (
        _GROUP + "      strategy: fastest\n"
        "      candidates: [paid-one]\n",
        "is invalid. strategy",
    ),
    "no candidates": (
        _GROUP + "      strategy: priority\n",
        "declares no candidates",
    ),
    "candidates is null": (
        _GROUP + "      candidates: ~\n",
        "declares no candidates",
    ),
    "candidate named twice": (
        _GROUP + "      candidates: [paid-one, paid-one]\n",
        "names the same candidate twice",
    ),
    "candidate not an endpoint": (
        _GROUP + "      candidates: [nope]\n",
        "names unknown endpoint",
    ),
    "cheapest candidate unpriced": (
        _GROUP + "      strategy: cheapest\n"
        "      candidates: [free-one]\n",
        "declares no cost",
    ),
}


def test_cost_days_since_checked_none() -> None:
    """
    days_since_checked is None when pricing_checked is unset.
    """
    cost = CostConfig(input_per_1m=1.0, output_per_1m=2.0)
    assert cost.days_since_checked is None


def test_cost_days_since_checked_today() -> None:
    """
    days_since_checked returns 0 when pricing was checked today.
    """
    cost = CostConfig(
        input_per_1m=1.0,
        output_per_1m=2.0,
        pricing_checked=date.today(),
    )
    assert cost.days_since_checked == 0


def test_cost_estimate_basic() -> None:
    """
    estimate_cost sums input * rate plus output * rate divided by 1M.
    """
    cost = CostConfig(input_per_1m=1.0, output_per_1m=2.0)
    actual = cost.estimate_cost(input_tokens=1000, output_tokens=2000)
    assert actual == pytest.approx(0.005)


def test_cost_estimate_fully_cached_uses_cache_rate() -> None:
    """
    When every input token is cached, the cache rate replaces the
    regular input rate entirely.
    """
    cost = CostConfig(
        input_per_1m=1.0,
        output_per_1m=2.0,
        cache_hit_input_per_1m=0.1,
    )
    actual = cost.estimate_cost(
        input_tokens=1000,
        output_tokens=1000,
        cached_tokens=1000,
    )
    assert actual == pytest.approx(0.0021)


def test_cost_estimate_partial_cache_hit_mixes_rates() -> None:
    """
    A real call can mix cached and uncached input in one prompt; only
    the cached subset bills at the cache rate.
    """
    cost = CostConfig(
        input_per_1m=1.0,
        output_per_1m=2.0,
        cache_hit_input_per_1m=0.1,
    )
    actual = cost.estimate_cost(
        input_tokens=1000,
        output_tokens=500,
        cached_tokens=400,
    )
    assert actual == pytest.approx(0.00164)


def test_cost_estimate_cache_write_uses_cache_write_rate() -> None:
    """
    cache_write_tokens bills at cache_write_input_per_1m, which is
    typically higher than the base input rate.
    """
    cost = CostConfig(
        input_per_1m=1.0,
        output_per_1m=2.0,
        cache_write_input_per_1m=3.0,
    )
    actual = cost.estimate_cost(
        input_tokens=1000,
        output_tokens=0,
        cache_write_tokens=300,
    )
    assert actual == pytest.approx(0.0016)


def test_cost_estimate_missing_cache_write_rate_uses_input_rate() -> None:
    """
    With no cache_write_input_per_1m configured, cache-write tokens
    bill at the standard input rate instead.
    """
    cost = CostConfig(input_per_1m=2.0, output_per_1m=0.0)
    actual = cost.estimate_cost(
        input_tokens=1000,
        output_tokens=0,
        cache_write_tokens=300,
    )
    assert actual == pytest.approx(0.002)


def test_cost_estimate_zero_cache_rate_is_free_not_unset() -> None:
    """
    An explicit cache_hit_input_per_1m of 0.0 is a genuinely free cache
    tier and must not be treated as "unset" and fall back to the
    regular input rate.
    """
    cost = CostConfig(
        input_per_1m=5.0,
        output_per_1m=0.0,
        cache_hit_input_per_1m=0.0,
    )
    actual = cost.estimate_cost(
        input_tokens=1000,
        output_tokens=0,
        cached_tokens=1000,
    )
    assert actual == 0.0


def test_cost_estimate_raises_when_cached_tokens_exceed_input() -> None:
    """
    cached_tokens alone exceeding input_tokens raises rather than
    silently producing a negative charge for the remainder.
    """
    cost = CostConfig(input_per_1m=1.0, output_per_1m=2.0)
    with pytest.raises(ValueError):
        cost.estimate_cost(
            input_tokens=100,
            output_tokens=0,
            cached_tokens=150,
        )


def test_cost_estimate_raises_when_cache_tokens_sum_exceeds_input() -> None:
    """
    cached_tokens and cache_write_tokens together exceeding
    input_tokens also raises, even when neither alone does.
    """
    cost = CostConfig(input_per_1m=1.0, output_per_1m=2.0)
    with pytest.raises(ValueError):
        cost.estimate_cost(
            input_tokens=100,
            output_tokens=0,
            cached_tokens=60,
            cache_write_tokens=60,
        )


def test_dotenv_is_read_from_the_working_directory(
    tmp_path: Path,
) -> None:
    """
    A .env in the working directory is loaded even when the package
    itself lives outside the caller's project.

    find_dotenv otherwise walks up from config.py's own directory,
    which reaches this repo's .env only because the dev environment
    installs from source in-tree. Installed from a wheel it walks up
    from site-packages and finds nothing.

    Run as a script file in a subprocess rather than with -c, because
    the loading happens once at import and with -c the calling frame
    has no file, so find_dotenv falls back to the working directory
    regardless and the test would pass against the unfixed code.
    """
    (tmp_path / ".env").write_text(
        "LRL_TEST_DOTENV_KEY=from-working-directory\n",
        encoding="utf-8",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os\n"
        "import llm_router_ledger.config  # noqa: F401\n"
        "print(os.environ.get('LRL_TEST_DOTENV_KEY', ''))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "from-working-directory"


def test_embedding_dimensions_loads_and_defaults_to_none(
    tmp_path: Path,
) -> None:
    """
    embedding_dimensions is read from YAML when present and stays None
    otherwise, so chat endpoints written before the field existed keep
    loading unchanged.
    """
    yaml_text = (
        "endpoints:\n"
        "  embed-endpoint:\n"
        "    provider: openrouter\n"
        "    model: baai/bge-m3\n"
        "    api_key_env: OPENROUTER_API_KEY\n"
        "    embedding_dimensions: 1024\n"
        "  chat-endpoint:\n"
        "    provider: openrouter\n"
        "    model: xiaomi/mimo-v2.5\n"
        "    api_key_env: OPENROUTER_API_KEY\n"
    )
    path = tmp_path / "llm_endpoints.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    config = load_config(path)
    assert (
        config.endpoints[
            "embed-endpoint"
        ].embedding_dimensions
        == 1024
    )
    assert (
        config.endpoints[
            "chat-endpoint"
        ].embedding_dimensions
        is None
    )


def test_endpoint_api_key_available_false_without_env(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    api_key_available returns False when the env var is not set.
    """
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    config = load_config(sample_yaml_file)
    ep = config.endpoints["ollama-local"]
    assert ep.api_key_available is False


def test_endpoint_api_key_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    Accessing .api_key when the env var is unset raises
    MissingApiKeyError.
    """
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    config = load_config(sample_yaml_file)
    ep = config.endpoints["ollama-local"]
    with pytest.raises(MissingApiKeyError):
        _ = ep.api_key


def test_endpoint_api_key_returns_env_value(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    api_key returns the env-var value when set.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    config = load_config(sample_yaml_file)
    ep = config.endpoints["ollama-local"]
    assert ep.api_key == "secret"


def test_get_context_window_default_when_no_match(
    sample_yaml_file: Path,
) -> None:
    """
    Returns the default when no endpoint's model field matches.
    """
    config = load_config(sample_yaml_file)
    actual = get_context_window(
        model="unknown-model",
        config=config,
        default=999,
    )
    assert actual == 999


def test_get_context_window_finds_match(sample_yaml_file: Path) -> None:
    """
    get_context_window returns the endpoint's value when the model string
    matches.
    """
    config = load_config(sample_yaml_file)
    actual = get_context_window(model="llama3.1", config=config)
    assert actual == 8192


def test_get_context_window_matches_free_variant(
    tmp_path: Path,
) -> None:
    """
    An OpenRouter ":free" model string matches its own endpoint. Reading
    the colon as a provider prefix would leave "free" as the bare name,
    which matches nothing and silently returns the default.
    """
    path = tmp_path / "llm_endpoints.yaml"
    path.write_text(_FREE_VARIANT_YAML, encoding="utf-8")
    config = load_config(path)
    actual = get_context_window(
        model="nvidia/nemotron-3.5-lightning:free",
        config=config,
    )
    assert actual == 1000000


def test_get_context_window_strips_prefix_from_free_variant(
    tmp_path: Path,
) -> None:
    """
    A provider prefix is still stripped when the remainder is itself a
    ":free" variant, so both spellings resolve to the same endpoint.

    This spelling was never broken: splitting on the first colon
    already left the ":free" tail intact. The test guards the prefix
    path now that it runs through a candidate loop.
    """
    path = tmp_path / "llm_endpoints.yaml"
    path.write_text(_FREE_VARIANT_YAML, encoding="utf-8")
    config = load_config(path)
    actual = get_context_window(
        model="openrouter:nvidia/nemotron-3.5-lightning:free",
        config=config,
    )
    assert actual == 1000000


def test_get_context_window_strips_provider_prefix(
    sample_yaml_file: Path,
) -> None:
    """
    A leading 'provider:' prefix on the model string is stripped before
    matching.
    """
    config = load_config(sample_yaml_file)
    actual = get_context_window(model="ollama:llama3.1", config=config)
    assert actual == 8192


def test_get_route_group_resolves_default(
    sample_yaml_file: Path,
) -> None:
    """
    get_route_group falls back to the default project.
    """
    config = load_config(sample_yaml_file)
    group = config.get_route_group(
        project="any-project",
        name="quick",
    )
    assert group.strategy == "priority"
    assert group.candidates == (
        "ollama-local",
        "openrouter-test",
    )


def test_get_route_group_unknown_name_raises(
    sample_yaml_file: Path,
) -> None:
    """
    Unknown group raises ConfigError, not KeyError, for a single
    library exception hierarchy.
    """
    config = load_config(sample_yaml_file)
    with pytest.raises(ConfigError):
        config.get_route_group(
            project="any",
            name="nonexistent",
        )


def test_llm_config_available_filters_to_set_keys(
    monkeypatch: pytest.MonkeyPatch,
    sample_yaml_file: Path,
) -> None:
    """
    available() returns only endpoints whose api_key_env is set.
    """
    monkeypatch.setenv("OLLAMA_API_KEY", "x")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = load_config(sample_yaml_file)
    available = config.available()
    names = [ep.name for ep in available]
    assert names == ["ollama-local"]


def test_llm_config_by_provider(sample_yaml_file: Path) -> None:
    """
    by_provider returns endpoints matching the given provider literal.
    """
    config = load_config(sample_yaml_file)
    eps = config.by_provider("openrouter")
    assert len(eps) == 1
    assert eps[0].name == "openrouter-test"


def test_load_config_explicit_path_overrides_env(
    tmp_path: Path,
    sample_yaml_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test for precedence: an explicit path argument wins over
    LRL_CONFIG_PATH, so the env var never overrides a caller that names
    a file directly.
    """
    monkeypatch.setenv(
        "LRL_CONFIG_PATH",
        str(tmp_path / "never-read.yaml"),
    )
    config = load_config(sample_yaml_file)
    assert "ollama-local" in config.endpoints


def test_load_config_honors_lrl_config_path_env(
    sample_yaml_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Test for LRL_CONFIG_PATH: with no path argument, load_config reads
    the file named by the env var rather than the cwd default.
    """
    monkeypatch.setenv(
        "LRL_CONFIG_PATH",
        str(sample_yaml_file),
    )
    config = load_config()
    assert "ollama-local" in config.endpoints


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    """
    load_config raises ConfigError (not FileNotFoundError) when the path
    does not exist.
    """
    bad = tmp_path / "does-not-exist.yaml"
    with pytest.raises(ConfigError):
        load_config(bad)


def test_load_config_normalises_negative_zero_rate(
    tmp_path: Path,
) -> None:
    """
    A rate written as -0.0 is stored as 0.0, so a price never renders
    with a minus sign.
    """
    p = tmp_path / "zero.yaml"
    p.write_text(
        _COST_ENDPOINT.replace("RATE", "-0.0"),
        encoding="utf-8",
    )
    cost = load_config(p).endpoints["a"].cost
    assert cost is not None
    assert str(cost.input_per_1m) == "0.0"


@pytest.mark.parametrize(
    ("text", "expected"),
    list(_LOAD_CONFIG_CASES.values()),
    ids=list(_LOAD_CONFIG_CASES),
)
def test_load_config_rejects(
    tmp_path: Path,
    text: str,
    expected: str,
) -> None:
    """
    A malformed file, a bad defaults or endpoints block, and a value
    the loader cannot use all raise ConfigError rather than the
    underlying YAML or Pydantic error.
    """
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=re.escape(expected)):
        load_config(p)


def test_load_config_reads_a_merge_key(tmp_path: Path) -> None:
    """
    A << merge key is resolved, and a key the block also sets
    explicitly overrides the merged one rather than counting as a
    duplicate.
    """
    yaml_text = (
        "endpoints:\n"
        "  base: &base\n"
        "    provider: ollama\n"
        "    model: llama3.1\n"
        "    api_key_env: OLLAMA_API_KEY\n"
        "  derived:\n"
        "    <<: *base\n"
        "    model: other\n"
    )
    p = tmp_path / "merge.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    derived = load_config(p).endpoints["derived"]
    assert derived.provider == "ollama"
    assert derived.model == "other"


def test_load_config_rejects_a_directory(tmp_path: Path) -> None:
    """
    A path that exists but is not a file raises ConfigError rather
    than the operating system error from opening it.
    """
    with pytest.raises(ConfigError, match="is not a file"):
        load_config(tmp_path)


def test_load_config_rejects_a_non_utf8_file(
    tmp_path: Path,
) -> None:
    """
    A config saved in another encoding raises ConfigError rather than
    UnicodeDecodeError.
    """
    p = tmp_path / "latin.yaml"
    p.write_bytes(
        "endpoints:\n  a:\n    notes: caf\xe9\n".encode(
            "cp1252",
        )
    )
    with pytest.raises(ConfigError, match="is not UTF-8"):
        load_config(p)


def test_load_config_returns_typed_objects(
    sample_yaml_file: Path,
) -> None:
    """
    load_config returns an LLMConfig with EndpointConfig values populated
    from the YAML.
    """
    config = load_config(sample_yaml_file)
    assert isinstance(config, LLMConfig)
    assert len(config.endpoints) == 2
    ep = config.endpoints["ollama-local"]
    assert ep.provider == "ollama"
    assert ep.context_window == 8192


def test_local_openai_compat_is_rejected_with_migration_hint(
    tmp_path: Path,
) -> None:
    """
    provider: local-openai-compat was deprecated in 0.1.2 and removed
    in 0.2.0, so load_config now rejects it. The error identifies the
    endpoint and the replacement provider names rather than relying on
    Pydantic's enumeration of every valid value.
    """
    yaml_text = (
        "endpoints:\n"
        "  legacy-local:\n"
        "    provider: local-openai-compat\n"
        "    model: llama3.1\n"
        "    api_key_env: OLLAMA_API_KEY\n"
        "    base_url: http://localhost:11434/v1\n"
    )
    p = tmp_path / "with_legacy.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(p)
    message = str(excinfo.value)
    assert "was removed in 0.2.0" in message
    assert "legacy-local" in message
    assert "ollama" in message
    assert "lmstudio" in message


def test_nvidia_provider_loads_with_its_base_url(
    tmp_path: Path,
) -> None:
    """
    provider: nvidia is accepted by the ProviderName literal, which
    Pydantic enforces, so this fails before the name is added. NIM
    serves an OpenAI-compatible API, so no adapter of its own is
    needed; only the name had to be added.
    """
    yaml_text = (
        "endpoints:\n"
        "  nvidia-test:\n"
        "    provider: nvidia\n"
        "    model: nvidia/nemotron-3.5-lightning-30b-a3b\n"
        "    api_key_env: NVIDIA_API_KEY\n"
        "    base_url: https://integrate.api.nvidia.com/v1\n"
    )
    p = tmp_path / "with_nvidia.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    ep = load_config(p).endpoints["nvidia-test"]
    assert ep.provider == "nvidia"
    assert ep.base_url == "https://integrate.api.nvidia.com/v1"


def test_route_group_falls_back_per_group(tmp_path: Path) -> None:
    """
    A project that declares groups of its own still inherits the
    groups it does not declare from the default project, and its own
    group of the same name wins.
    """
    yaml_text = _ROUTE_GROUP_ENDPOINTS + (
        "route_groups:\n"
        "  default:\n"
        "    shared:\n"
        "      candidates: [paid-one]\n"
        "    both:\n"
        "      candidates: [paid-one]\n"
        "  reporting:\n"
        "    both:\n"
        "      candidates: [free-one]\n"
    )
    p = tmp_path / "groups.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    config = load_config(p)
    inherited = config.get_route_group(
        project="reporting",
        name="shared",
    )
    overridden = config.get_route_group(
        project="reporting",
        name="both",
    )
    assert inherited.candidates == ("paid-one",)
    assert overridden.candidates == ("free-one",)


def test_route_group_is_read_only(tmp_path: Path) -> None:
    """
    A loaded group cannot be reassigned, and its candidates are a
    tuple, so walking a group cannot change the config by accident.
    """
    yaml_text = _ROUTE_GROUP_ENDPOINTS + (
        _GROUP + "      candidates: [paid-one]\n"
    )
    p = tmp_path / "groups.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    group = load_config(p).get_route_group(
        project="any",
        name="q",
    )
    assert group.candidates == ("paid-one",)
    with pytest.raises(ValidationError):
        group.strategy = "cheapest"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("block", "expected"),
    list(_ROUTE_GROUP_CASES.values()),
    ids=list(_ROUTE_GROUP_CASES),
)
def test_route_group_rejects(
    tmp_path: Path,
    block: str,
    expected: str,
) -> None:
    """
    Every route-group rule reports its own ConfigError, so a consumer
    catching LLMCallError sees a readable message rather than a
    Pydantic or YAML error.
    """
    p = tmp_path / "groups.yaml"
    p.write_text(
        _ROUTE_GROUP_ENDPOINTS + block,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=re.escape(expected)):
        load_config(p)


def test_route_group_reports_an_unknown_candidate_first(
    tmp_path: Path,
) -> None:
    """
    A cheapest group naming both an uncosted endpoint and one the
    config does not declare reports the unknown endpoint whichever
    order they are listed in.
    """
    yaml_text = _ROUTE_GROUP_ENDPOINTS + (
        _GROUP + "      strategy: cheapest\n"
        "      candidates: [free-one, nope]\n"
    )
    p = tmp_path / "groups.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match="names unknown endpoint 'nope'",
    ):
        load_config(p)


def test_route_group_round_trips_notes_and_absence(
    tmp_path: Path,
) -> None:
    """
    A config with no route_groups block exposes an empty mapping, and
    a group's notes are kept as written.
    """
    p = tmp_path / "none.yaml"
    p.write_text(_ROUTE_GROUP_ENDPOINTS, encoding="utf-8")
    assert load_config(p).route_groups == {}

    p = tmp_path / "notes.yaml"
    p.write_text(
        _ROUTE_GROUP_ENDPOINTS
        + _GROUP
        + "      candidates: [paid-one]\n"
        + "      notes: why this group exists\n",
        encoding="utf-8",
    )
    group = load_config(p).get_route_group(
        project="any",
        name="q",
    )
    assert group.notes == "why this group exists"

def test_route_group_strategy_defaults_to_priority(
    tmp_path: Path,
) -> None:
    """
    A group that names no strategy takes the first candidate, which
    needs no declared rates.
    """
    yaml_text = _ROUTE_GROUP_ENDPOINTS + (
        _GROUP + "      candidates: [free-one, paid-one]\n"
    )
    p = tmp_path / "groups.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    group = load_config(p).get_route_group(
        project="any",
        name="q",
    )
    assert group.strategy == "priority"
    assert group.candidates == ("free-one", "paid-one")


def test_route_group_without_endpoints_says_so(
    tmp_path: Path,
) -> None:
    """
    A config whose groups name candidates but which declares no
    endpoints reports the missing endpoints rather than blaming the
    group for naming one that does not exist.
    """
    yaml_text = _GROUP + "      candidates: [paid-one]\n"
    p = tmp_path / "groups.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(
        ConfigError,
        match="declares no endpoints",
    ):
        load_config(p)
