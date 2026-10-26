"""Environment-backed runtime configuration.

Configuration is read from the four variables documented in ``README.md``.
Values are validated at startup so a bad network, URL, or database setting
fails before an allocation run begins. ``DATABASE_URL`` is kept out of the
default string representation because it may contain credentials.

The optimizer recommends allocations. It never signs or submits payouts.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlsplit

from funding_optimizer.errors import ValidationError

__all__ = [
    "Config",
    "Settings",
    "StellarNetwork",
    "load_config",
    "load_settings",
]


class StellarNetwork(StrEnum):
    """Stellar networks supported by the project."""

    PUBLIC = "public"
    TESTNET = "testnet"


_HTTP_SCHEMES = frozenset({"http", "https"})
_DATABASE_SCHEMES = frozenset({"postgres", "postgresql", "sqlite"})


def _read(env: Mapping[str, str], key: str) -> str:
    value = env.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValidationError(f"{key} must be a string")
    return value.strip()


def _require(env: Mapping[str, str], key: str) -> str:
    value = _read(env, key)
    if not value:
        raise ValidationError(f"{key} is required")
    return value


def _validate_http_url(key: str, value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValidationError(f"{key} must be a URL without whitespace")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValidationError(f"{key} must be a valid URL") from exc
    if parsed.scheme not in _HTTP_SCHEMES:
        raise ValidationError(f"{key} must use http or https")
    if not parsed.netloc:
        raise ValidationError(f"{key} must include a host")
    return value


def _validate_database_url(value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValidationError("DATABASE_URL must be a URL without whitespace")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValidationError("DATABASE_URL must be a valid URL") from exc
    scheme = parsed.scheme.split("+", 1)[0]
    if scheme not in _DATABASE_SCHEMES:
        allowed = ", ".join(sorted(_DATABASE_SCHEMES))
        raise ValidationError(f"DATABASE_URL must use one of: {allowed}")
    if scheme in {"postgres", "postgresql"} and not parsed.netloc:
        raise ValidationError("DATABASE_URL must include a host")
    if scheme == "sqlite" and not (parsed.path or parsed.netloc):
        raise ValidationError("DATABASE_URL must include a database path")
    return value


@dataclass(frozen=True)
class Config:
    """Validated runtime settings for funding-optimizer.

    ``STELLAR_NETWORK`` and ``HORIZON_URL`` are required.
    ``ANALYTICS_API_URL`` and ``DATABASE_URL`` are optional. When present,
    each URL is validated. ``database_url`` is redacted from ``repr()`` and
    :meth:`safe_dict` because it may contain credentials.
    """

    stellar_network: StellarNetwork
    horizon_url: str
    analytics_api_url: str | None = None
    database_url: str | None = field(default=None, repr=False)

    @property
    def network(self) -> StellarNetwork:
        """Alias for :attr:`stellar_network`."""
        return self.stellar_network

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Build a config from ``env`` (defaults to ``os.environ``)."""
        source = os.environ if env is None else env

        network_value = _require(source, "STELLAR_NETWORK").lower()
        try:
            network = StellarNetwork(network_value)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in StellarNetwork)
            raise ValidationError(f"STELLAR_NETWORK must be one of: {allowed}") from exc

        horizon_url = _validate_http_url("HORIZON_URL", _require(source, "HORIZON_URL"))

        analytics_raw = _read(source, "ANALYTICS_API_URL")
        analytics_api_url = (
            _validate_http_url("ANALYTICS_API_URL", analytics_raw) if analytics_raw else None
        )

        database_raw = _read(source, "DATABASE_URL")
        database_url = _validate_database_url(database_raw) if database_raw else None

        return cls(
            stellar_network=network,
            horizon_url=horizon_url,
            analytics_api_url=analytics_api_url,
            database_url=database_url,
        )

    def safe_dict(self) -> dict[str, str | None]:
        """Return settings with the database URL redacted, for logging."""
        return {
            "stellar_network": self.stellar_network.value,
            "horizon_url": self.horizon_url,
            "analytics_api_url": self.analytics_api_url,
            "database_url": "***" if self.database_url else None,
        }

    def __repr__(self) -> str:
        database_url = "***" if self.database_url else None
        return (
            "Config("
            f"stellar_network={self.stellar_network.value!r}, "
            f"horizon_url={self.horizon_url!r}, "
            f"analytics_api_url={self.analytics_api_url!r}, "
            f"database_url={database_url!r}"
            ")"
        )


Settings = Config


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Load and validate configuration from the environment."""
    return Config.from_env(env)


load_settings = load_config
