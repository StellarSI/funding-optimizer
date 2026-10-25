"""Unit tests for the error taxonomy."""

from __future__ import annotations

import pytest

from funding_optimizer import Budget, Program
from funding_optimizer.errors import AllocationError, FundingOptimizerError, ValidationError


def test_error_taxonomy_hierarchy() -> None:
    assert issubclass(ValidationError, FundingOptimizerError)
    assert issubclass(ValidationError, ValueError)
    assert issubclass(AllocationError, FundingOptimizerError)
    assert not issubclass(AllocationError, ValueError)


def test_validation_error_raised_on_invalid_budget_total() -> None:
    with pytest.raises(ValidationError):
        Budget(total=float("nan"), asset="USDC")
    with pytest.raises(ValidationError):
        Budget(total=float("inf"), asset="USDC")
    with pytest.raises(ValidationError):
        Budget(total=-1, asset="USDC")
    with pytest.raises(ValidationError):
        Budget(total=0, asset="USDC")


def test_validation_error_raised_on_invalid_program() -> None:
    with pytest.raises(ValidationError):
        Program(id="a", min_share=1.5, expected_reach=1)
    with pytest.raises(ValidationError):
        Program(id="a", min_share=0.1, expected_reach=-1)


def test_validation_error_raised_on_invalid_asset() -> None:
    with pytest.raises(ValidationError):
        Budget(total=100, asset="$BAD")
    with pytest.raises(ValidationError):
        Budget(total=100, asset="TOOLONGASSETCODE")


def test_allocation_error_is_funding_optimizer_error() -> None:
    err = AllocationError("infeasible plan")
    assert isinstance(err, FundingOptimizerError)
    assert str(err) == "infeasible plan"


def test_validation_error_is_value_error() -> None:
    err = ValidationError("bad input")
    assert isinstance(err, ValueError)
    assert isinstance(err, FundingOptimizerError)
