"""Unit tests for environment-backed configuration."""

from __future__ import annotations

import pytest

from funding_optimizer.config import (
    Config,
    Settings,
    StellarNetwork,
    load_config,
    load_settings,
)
from funding_optimizer.errors import ValidationError

_REQUIRED = {
    "STELLAR_NETWORK": "testnet",
    "HORIZON_URL": "https://horizon-testnet.stellar.org",
}


def test_load_config_reads_all_values() -> None:
    config = load_config(
        {
            **_REQUIRED,
            "ANALYTICS_API_URL": "https://analytics.example.com",
            "DATABASE_URL": "postgresql://user:password@localhost:5432/funding",
        }
    )

    assert config.stellar_network is StellarNetwork.TESTNET
    assert config.network is StellarNetwork.TESTNET
    assert config.horizon_url == "https://horizon-testnet.stellar.org"
    assert config.analytics_api_url == "https://analytics.example.com"
    assert config.database_url == "postgresql://user:password@localhost:5432/funding"


def test_optional_values_default_to_none() -> None:
    config = load_config(_REQUIRED)

    assert config.analytics_api_url is None
    assert config.database_url is None


def test_settings_aliases() -> None:
    assert Settings is Config
    assert load_settings is load_config


def test_network_value_is_normalized() -> None:
    config = load_config({**_REQUIRED, "STELLAR_NETWORK": "  PUBLIC  "})

    assert config.stellar_network is StellarNetwork.PUBLIC


def test_async_database_driver_is_accepted() -> None:
    config = load_config(
        {**_REQUIRED, "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/funding"}
    )

    assert config.database_url == "postgresql+asyncpg://user:pass@localhost:5432/funding"


@pytest.mark.parametrize("key", ["STELLAR_NETWORK", "HORIZON_URL"])
def test_required_values_are_enforced(key: str) -> None:
    env = dict(_REQUIRED)
    env.pop(key)

    with pytest.raises(ValidationError, match=key):
        load_config(env)


@pytest.mark.parametrize("value", ["mainnet", "devnet", ""])
def test_unknown_network_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="STELLAR_NETWORK"):
        load_config({**_REQUIRED, "STELLAR_NETWORK": value})


@pytest.mark.parametrize(
    "value",
    [
        "ftp://horizon.example.com",
        "horizon.example.com",
        "https://",
        "https://horizon example.com",
    ],
)
def test_invalid_horizon_url_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="HORIZON_URL"):
        load_config({**_REQUIRED, "HORIZON_URL": value})


@pytest.mark.parametrize(
    "value",
    ["not-a-url", "ftp://analytics.example.com", "https://"],
)
def test_invalid_analytics_url_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="ANALYTICS_API_URL"):
        load_config({**_REQUIRED, "ANALYTICS_API_URL": value})


@pytest.mark.parametrize(
    "value",
    ["mysql://localhost/funding", "postgresql://", "sqlite://", "not-a-url"],
)
def test_invalid_database_url_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        load_config({**_REQUIRED, "DATABASE_URL": value})


def test_database_url_is_redacted_from_repr_and_safe_dict() -> None:
    secret = "postgresql://user:supersecret@localhost:5432/funding"
    config = load_config({**_REQUIRED, "DATABASE_URL": secret})

    assert "supersecret" not in repr(config)
    assert config.safe_dict()["database_url"] == "***"
    assert "supersecret" not in str(config.safe_dict())


def test_safe_dict_keeps_non_secret_values() -> None:
    config = load_config({**_REQUIRED, "ANALYTICS_API_URL": "https://analytics.example.com"})

    assert config.safe_dict() == {
        "stellar_network": "testnet",
        "horizon_url": "https://horizon-testnet.stellar.org",
        "analytics_api_url": "https://analytics.example.com",
        "database_url": None,
    }
