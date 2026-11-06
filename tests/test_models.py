"""Tests for the multi-asset :class:`Budget` value object (issue #17)."""

from __future__ import annotations

import pytest

from funding_optimizer.models import Budget, validate_asset_code


class TestConstruction:
    def test_single_asset_convenience_shape(self) -> None:
        budget = Budget(total=250_000, asset="USDC")
        assert budget.amounts == {"USDC": 250_000.0}
        assert budget.total == 250_000.0
        assert budget.asset == "USDC"
        assert budget.is_multi_asset is False
        assert budget.assets == ("USDC",)

    def test_multi_asset_amounts_shape(self) -> None:
        budget = Budget(amounts={"USDC": 150_000, "XLM": 40_000_000})
        assert budget.amounts == {"USDC": 150_000.0, "XLM": 40_000_000.0}
        assert budget.is_multi_asset is True
        assert budget.assets == ("USDC", "XLM")
        assert budget.amount("XLM") == 40_000_000.0

    def test_asset_code_normalization(self) -> None:
        budget = Budget(amounts={" usdc ": 10})
        assert budget.assets == ("USDC",)

    def test_issuer_suffix_is_accepted(self) -> None:
        budget = Budget(amounts={"USDC:gaqaaa2222": 1_000})
        assert budget.assets == ("USDC:GAQAAA2222",)

    def test_both_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            Budget(total=1, asset="USDC", amounts={"USDC": 1})

    def test_missing_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="either amounts"):
            Budget()

    def test_empty_amounts_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one asset"):
            Budget(amounts={})

    def test_duplicate_asset_after_normalization_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate asset"):
            Budget(amounts={"USDC": 5, " usdc ": 7})

    def test_invalid_asset_codes_raise(self) -> None:
        for bad in ["", "has space", "A" * 13, "USDC:", ":ISSUER", "USDC:ISSUER:EXTRA"]:
            with pytest.raises(ValueError):
                Budget(amounts={bad: 100})

    def test_non_positive_and_non_finite_amounts_raise(self) -> None:
        for bad in [0, -1, float("nan"), float("inf"), float("-inf")]:
            with pytest.raises(ValueError):
                Budget(amounts={"USDC": bad})

    def test_non_numeric_amounts_raise(self) -> None:
        with pytest.raises(TypeError):
            Budget(amounts={"USDC": "100"})  # type: ignore[dict-item]
        with pytest.raises(TypeError):
            Budget(amounts={"USDC": True})

    def test_total_and_asset_are_ambiguous_for_multi_asset(self) -> None:
        budget = Budget(amounts={"USDC": 10, "XLM": 20})
        with pytest.raises(ValueError, match="ambiguous"):
            _ = budget.total
        with pytest.raises(ValueError, match="ambiguous"):
            _ = budget.asset

    def test_amount_for_missing_asset_raises_key_error(self) -> None:
        budget = Budget(amounts={"USDC": 10})
        with pytest.raises(KeyError, match="not part of this budget"):
            budget.amount("XLM")


class TestValidateAssetCode:
    def test_normalizes_case_and_whitespace(self) -> None:
        assert validate_asset_code(" xlm ") == "XLM"

    def test_rejects_non_string(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            validate_asset_code(5)  # type: ignore[arg-type]
