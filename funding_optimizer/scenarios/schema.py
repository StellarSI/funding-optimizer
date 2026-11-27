"""Versioned scenario documents for scenario simulation (issue #42).

A scenario is a pure, validated description of an alternative budget and
schedule: which assets and amounts, which assumptions, and which model
versions produced any forecasts it references. Parsing and validating a
scenario never runs a solver and never signs or submits a payout.

Repeated runs with the same input and seed produce the same document and the
same :meth:`ScenarioDocument.fingerprint`, so scenarios can be versioned and
compared deterministically.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from funding_optimizer.models import Budget

SCENARIO_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (1,)

_SCENARIO_FIELDS = frozenset(
    {"version", "id", "name", "budgets", "assumptions", "model_versions", "seed"}
)


class ScenarioValidationError(ValueError):
    """Raised when a scenario document is malformed or unsupported."""


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _scalar_mapping(value: Any, label: str) -> dict[str, str | float | bool | None]:
    if not isinstance(value, Mapping):
        raise ScenarioValidationError(f"{label} must be a mapping")
    result: dict[str, str | float | bool | None] = {}
    for key, item in value.items():
        name = _non_empty_string(key, f"{label} key")
        if isinstance(item, bool) or item is None or isinstance(item, (str, int, float)):
            result[name] = item
            continue
        raise ScenarioValidationError(f"{label}[{name!r}] must be a scalar value")
    return result


@dataclass(frozen=True)
class ScenarioDocument:
    """A validated, versioned scenario description.

    Units: budget amounts are in asset units; ``seed`` is an integer used by
    downstream simulation models for reproducibility.
    """

    version: int
    id: str
    name: str
    budget: Budget
    assumptions: Mapping[str, str | float | bool | None]
    model_versions: Mapping[str, str]
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "id": self.id,
            "name": self.name,
            "budgets": dict(self.budget.amounts),
            "assumptions": dict(self.assumptions),
            "model_versions": dict(self.model_versions),
            "seed": self.seed,
        }

    def fingerprint(self) -> str:
        """Stable SHA-256 of the canonical document, including its seed."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_scenario(document: Mapping[str, Any]) -> ScenarioDocument:
    """Validates a raw scenario mapping and returns a frozen document.

    Unknown fields are rejected so a typo cannot silently change a
    simulation. Only field validation happens here — no allocation is solved.
    """
    if not isinstance(document, Mapping):
        raise ScenarioValidationError("scenario must be a mapping")
    unknown = sorted(set(document) - _SCENARIO_FIELDS)
    if unknown:
        raise ScenarioValidationError(f"unknown scenario field(s): {', '.join(unknown)}")

    version = document.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ScenarioValidationError(
            f"unsupported scenario schema version {version!r}; "
            f"supported versions: {', '.join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)}"
        )
    identifier = _non_empty_string(document.get("id"), "id")
    name = _non_empty_string(document.get("name"), "name")

    budgets = document.get("budgets")
    if not isinstance(budgets, Mapping) or not budgets:
        raise ScenarioValidationError("budgets must be a non-empty mapping of asset to amount")
    try:
        budget = Budget(amounts=dict(budgets))
    except (TypeError, ValueError) as exc:
        raise ScenarioValidationError(f"invalid scenario budget: {exc}") from exc

    assumptions = _scalar_mapping(document.get("assumptions", {}), "assumptions")

    raw_versions = document.get("model_versions", {})
    if not isinstance(raw_versions, Mapping):
        raise ScenarioValidationError("model_versions must be a mapping")
    model_versions: dict[str, str] = {}
    for key, value in raw_versions.items():
        model_versions[_non_empty_string(key, "model_versions key")] = _non_empty_string(
            value, f"model_versions[{key!r}]"
        )

    seed = document.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ScenarioValidationError("seed must be a non-negative integer")

    return ScenarioDocument(
        version=version,
        id=identifier,
        name=name,
        budget=budget,
        assumptions=assumptions,
        model_versions=model_versions,
        seed=seed,
    )


def validate_scenario_document(document: Mapping[str, Any]) -> ScenarioDocument:
    """Alias for :func:`parse_scenario`, for callers that only validate."""
    return parse_scenario(document)


__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ScenarioDocument",
    "ScenarioValidationError",
    "parse_scenario",
    "validate_scenario_document",
]
