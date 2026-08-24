"""
Unit tests for llm_router_ledger.config.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

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


def test_get_role_endpoints_resolves_default(
    sample_yaml_file: Path,
) -> None:
    """
    get_role_endpoints reads the default project's role map.
    """
    config = load_config(sample_yaml_file)
    eps = config.get_role_endpoints(project="any-project", role="quick")
    assert len(eps) == 1
    assert eps[0].name == "ollama-local"


def test_get_role_endpoints_unknown_role_raises(
    sample_yaml_file: Path,
) -> None:
    """
    Unknown role raises ConfigError, not KeyError, for a single library
    exception hierarchy.
    """
    config = load_config(sample_yaml_file)
    with pytest.raises(ConfigError):
        config.get_role_endpoints(project="any", role="nonexistent")


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
    assert "legacy-local" in message
    assert "ollama" in message
    assert "lmstudio" in message
