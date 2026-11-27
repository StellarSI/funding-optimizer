"""Versioned scenario documents for the optimizer (issue #42).

The optimizer recommends allocations and never signs or submits payouts.
"""

from .schema import (
    SCENARIO_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    ScenarioDocument,
    ScenarioValidationError,
    parse_scenario,
    validate_scenario_document,
)

__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "ScenarioDocument",
    "ScenarioValidationError",
    "parse_scenario",
    "validate_scenario_document",
]
