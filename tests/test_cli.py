"""Smoke tests for the ``optimize`` command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from funding_optimizer.cli import main

_PROBLEM: dict[str, Any] = {
    "objective": "maximize_reach",
    "fairness": "regional_floor",
    "budget": {"total": 250_000, "asset": "USDC"},
    "programs": [
        {"id": "cash-transfers-ng", "min_share": 0.20, "expected_reach": 12_000},
        {"id": "education-ke", "min_share": 0.10, "expected_reach": 4_500},
        {"id": "health-gh", "min_share": 0.15, "expected_reach": 7_800},
    ],
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_optimize_writes_plan_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "programs.json"
    output_path = tmp_path / "plan.json"
    _write_json(input_path, _PROBLEM)

    assert main(["optimize", "--input", str(input_path), "--output", str(output_path)]) == 0
    plan: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8"))

    assert plan["objective"] == "maximize_reach"
    assert plan["fairness"] == "regional_floor"
    assert plan["asset"] == "USDC"
    assert plan["total"] == 250_000
    assert len(plan["allocations"]) == 3
    assert plan["unallocated"] == 0.0
    assert capsys.readouterr().out == ""


def test_optimize_writes_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "programs.json"
    _write_json(input_path, _PROBLEM)

    assert main(["optimize", "--input", str(input_path)]) == 0
    plan: dict[str, Any] = json.loads(capsys.readouterr().out)

    assert plan["objective"] == "maximize_reach"
    assert len(plan["allocations"]) == 3


def test_program_array_with_budget_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "programs.json"
    _write_json(input_path, _PROBLEM["programs"])

    args = [
        "optimize",
        "--input",
        str(input_path),
        "--total",
        "100",
        "--asset",
        "usdc",
        "--objective",
        "equal_share",
    ]
    assert main(args) == 0
    plan: dict[str, Any] = json.loads(capsys.readouterr().out)

    assert plan["asset"] == "USDC"
    assert plan["objective"] == "equal_share"
    assert len(plan["allocations"]) == 3


def test_array_without_budget_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "programs.json"
    _write_json(input_path, _PROBLEM["programs"])

    assert main(["optimize", "--input", str(input_path)]) == 2
    assert "budget" in capsys.readouterr().err


def test_invalid_json_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "programs.json"
    input_path.write_text("{not json", encoding="utf-8")

    assert main(["optimize", "--input", str(input_path)]) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_validation_error_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "programs.json"
    payload = {
        "budget": {"total": 100, "asset": "USDC"},
        "programs": [{"id": "a", "min_share": 1.5}],
    }
    _write_json(input_path, payload)

    assert main(["optimize", "--input", str(input_path)]) == 2
    assert "min_share" in capsys.readouterr().err


def test_unknown_program_field_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "programs.json"
    payload = {
        "budget": {"total": 100, "asset": "USDC"},
        "programs": [{"id": "a", "unexpected": True}],
    }
    _write_json(input_path, payload)

    assert main(["optimize", "--input", str(input_path)]) == 2
    assert "unknown program field" in capsys.readouterr().err


def test_missing_file_returns_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.json"

    assert main(["optimize", "--input", str(missing)]) == 2
    assert "error:" in capsys.readouterr().err
