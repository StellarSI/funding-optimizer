"""Unit tests for the ``Budget`` and ``Program`` models."""

from __future__ import annotations

import pytest

from funding_optimizer import Budget, Program
from funding_optimizer.errors import ValidationError


def test_budget_normalizes_asset_to_uppercase() -> None:
    budget = Budget(total=250_000, asset="  usdc ")

    assert budget.total == 250_000
    assert budget.asset == "USDC"


def test_budget_accepts_classic_asset_reference() -> None:
    issuer = "G" + "A" * 55
    budget = Budget(total=100, asset=f"usdc:{issuer}")

    assert budget.asset == f"USDC:{issuer}"


def test_budget_rejects_zero_total() -> None:
    with pytest.raises(ValidationError):
        Budget(total=0, asset="USDC")


def test_budget_rejects_negative_total() -> None:
    with pytest.raises(ValidationError):
        Budget(total=-1, asset="USDC")


def test_budget_rejects_nan_total() -> None:
    with pytest.raises(ValidationError):
        Budget(total=float("nan"), asset="USDC")


def test_budget_rejects_infinite_total() -> None:
    with pytest.raises(ValidationError):
        Budget(total=float("inf"), asset="USDC")


def test_budget_rejects_boolean_total() -> None:
    with pytest.raises(ValidationError):
        Budget(total=True, asset="USDC")


def test_budget_rejects_non_string_asset() -> None:
    with pytest.raises(ValidationError):
        Budget(total=100, asset=123)  # type: ignore[arg-type]


def test_budget_rejects_blank_asset() -> None:
    with pytest.raises(ValidationError):
        Budget(total=100, asset="   ")


def test_budget_rejects_invalid_asset_characters() -> None:
    with pytest.raises(ValidationError):
        Budget(total=100, asset="$BAD")


def test_budget_rejects_overlong_asset_code() -> None:
    with pytest.raises(ValidationError):
        Budget(total=100, asset="TOOLONGASSETCODE")


def test_program_defaults() -> None:
    program = Program(id="cash-transfers-ng")

    assert program.min_share == 0.0
    assert program.expected_reach == 0
    assert program.region is None


def test_program_trims_id_and_region() -> None:
    program = Program(id="  cash  ", min_share=0.2, expected_reach=10, region=" east ")

    assert program.id == "cash"
    assert program.region == "east"


def test_program_rejects_blank_id() -> None:
    with pytest.raises(ValidationError):
        Program(id="   ")


def test_program_rejects_min_share_below_zero() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", min_share=-0.01)


def test_program_rejects_min_share_above_one() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", min_share=1.01)


def test_program_rejects_nan_min_share() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", min_share=float("nan"))


def test_program_rejects_infinite_min_share() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", min_share=float("inf"))


def test_program_rejects_boolean_min_share() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", min_share=True)


def test_program_rejects_negative_expected_reach() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", expected_reach=-1)


def test_program_rejects_boolean_expected_reach() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", expected_reach=True)


def test_program_rejects_non_integer_expected_reach() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", expected_reach=1.5)  # type: ignore[arg-type]


def test_program_rejects_blank_region() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", region="   ")
