"""Tests for versioned scenario documents (issue #42)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from funding_optimizer.scenarios.schema import (
    SCENARIO_SCHEMA_VERSION,
    ScenarioValidationError,
    parse_scenario,
    validate_scenario_document,
)


def scenario_input() -> dict[str, object]:
    return {
        "version": SCENARIO_SCHEMA_VERSION,
        "id": "baseline-2026",
        "name": "Baseline 2026",
        "budgets": {"usdc": 150_000, "XLM": 40_000_000},
        "assumptions": {"cashout_rate": 0.82, "horizon": "testnet", "enabled": True},
        "model_versions": {"forecast": "v2.1", "optimizer": "v1.4"},
        "seed": 42,
    }


def test_parse_scenario_happy_path() -> None:
    scenario = parse_scenario(scenario_input())

    assert scenario.version == SCENARIO_SCHEMA_VERSION
    assert scenario.id == "baseline-2026"
    assert scenario.budget.assets == ("USDC", "XLM")
    assert scenario.budget.amount("USDC") == 150_000
    assert scenario.assumptions["cashout_rate"] == pytest.approx(0.82)
    assert scenario.model_versions["forecast"] == "v2.1"
    assert scenario.seed == 42

    as_dict = scenario.as_dict()
    budgets = cast(dict[str, float], as_dict["budgets"])
    assert budgets == {"USDC": 150_000, "XLM": 40_000_000}


def test_fingerprint_is_deterministic_with_same_input_and_seed() -> None:
    first = parse_scenario(scenario_input())
    second = parse_scenario(scenario_input())
    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64

    changed = scenario_input()
    changed["seed"] = 43
    assert parse_scenario(changed).fingerprint() != first.fingerprint()


def test_validation_alias_matches_parse() -> None:
    assert validate_scenario_document(scenario_input()) == parse_scenario(scenario_input())


def test_optional_fields_get_stable_defaults() -> None:
    raw = scenario_input()
    raw.pop("assumptions")
    raw.pop("model_versions")
    raw.pop("seed")
    scenario = parse_scenario(raw)
    assert scenario.assumptions == {}
    assert scenario.model_versions == {}
    assert scenario.seed == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(version=99),
        lambda raw: raw.update(id=" "),
        lambda raw: raw.update(budgets={}),
        lambda raw: raw.update(budgets={"BAD-ASSET": 100}),
        lambda raw: raw.update(seed=-1),
        lambda raw: raw.update(typo="ignored"),
        lambda raw: raw.update(model_versions={"forecast": ""}),
    ],
)
def test_invalid_documents_raise(mutate: Callable[[dict[str, object]], None]) -> None:
    raw = scenario_input()
    mutate(raw)
    with pytest.raises(ScenarioValidationError):
        parse_scenario(raw)


def test_non_mapping_input_raises() -> None:
    with pytest.raises(ScenarioValidationError):
        parse_scenario("not-a-document")  # type: ignore[arg-type]
