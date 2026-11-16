"""Tests for cold-start forecasting for new programs and regions (issue #30)."""

from __future__ import annotations

from datetime import date

import pytest

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastKey,
    ForecastRow,
    ForecastTarget,
    ObservationWindow,
    TimeWindow,
)
from funding_optimizer.forecasting.cold_start import cold_start_baseline


def _observation() -> ObservationWindow:
    return ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31))


def _horizon() -> ForecastHorizon:
    return ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")


def _key(program: str, region: str = "Africa", asset: str = "USDC") -> ForecastKey:
    return ForecastKey(program=program, region=region, asset=asset)


def _row(value: float, key: ForecastKey, target: ForecastTarget) -> ForecastRow:
    return ForecastRow(
        key=key,
        window=TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31)),
        target=target,
        value=value,
    )


def _cash_out_dataset() -> ForecastDataset:
    k1 = _key("cash-transfers-ng", "Africa")
    k2 = _key("education-ke", "Africa")
    k3 = _key("health-gh", "Asia")
    return ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=_observation(),
        rows=(
            _row(0.6, k1, ForecastTarget.CASH_OUT_RATE),
            _row(0.8, k1, ForecastTarget.CASH_OUT_RATE),
            _row(0.5, k2, ForecastTarget.CASH_OUT_RATE),
            _row(0.9, k3, ForecastTarget.CASH_OUT_RATE),
        ),
    )


def test_known_key_uses_historical_mean_with_metadata() -> None:
    dataset = _cash_out_dataset()
    key = _key("cash-transfers-ng", "Africa")
    (prediction,) = cold_start_baseline(dataset, _horizon(), [key])

    assert prediction.key == key
    assert prediction.target is ForecastTarget.CASH_OUT_RATE
    assert prediction.horizon == _horizon()
    assert prediction.method == "historical-mean"
    assert prediction.value == pytest.approx(0.7)
    assert prediction.evaluation is not None
    assert prediction.evaluation.n_observations == 2
    assert prediction.evaluation.mae == pytest.approx(0.1)
    assert prediction.evaluation.rmse == pytest.approx(0.1)
    assert prediction.evaluation.evaluated_at == date(2025, 12, 31)


def test_new_program_in_known_region_falls_back_to_region_mean() -> None:
    dataset = _cash_out_dataset()
    new_program = _key("new-program", "Africa")
    (prediction,) = cold_start_baseline(dataset, _horizon(), [new_program])

    # Region Africa rows for asset USDC: 0.6, 0.8, 0.5 -> mean 19/30.
    assert prediction.method == "cold-start-region-mean"
    assert prediction.value == pytest.approx((0.6 + 0.8 + 0.5) / 3.0)
    assert prediction.evaluation is not None
    assert prediction.evaluation.n_observations == 3


def test_new_region_falls_back_to_asset_mean_when_program_unknown() -> None:
    dataset = _cash_out_dataset()
    new_region_program = _key("unknown-program", "Europe")
    (prediction,) = cold_start_baseline(dataset, _horizon(), [new_region_program])

    # No Europe, no program history -> global asset mean over all 4 rows.
    assert prediction.method == "cold-start-asset-mean"
    assert prediction.value == pytest.approx((0.6 + 0.8 + 0.5 + 0.9) / 4.0)
    assert prediction.evaluation is not None
    assert prediction.evaluation.n_observations == 4


def test_known_program_in_new_region_falls_back_to_program_mean() -> None:
    dataset = _cash_out_dataset()
    moved_program = _key("cash-transfers-ng", "Europe")
    (prediction,) = cold_start_baseline(dataset, _horizon(), [moved_program])

    # No Europe region rows, but program exists in Africa -> program mean 0.7.
    assert prediction.method == "cold-start-program-mean"
    assert prediction.value == pytest.approx(0.7)
    assert prediction.evaluation is not None
    assert prediction.evaluation.n_observations == 2


def test_returns_predictions_in_requested_order() -> None:
    dataset = _cash_out_dataset()
    keys = [
        _key("unknown-program", "Europe"),
        _key("cash-transfers-ng", "Africa"),
        _key("new-program", "Africa"),
    ]
    predictions = cold_start_baseline(dataset, _horizon(), keys)

    assert [prediction.key for prediction in predictions] == keys
    assert [prediction.method for prediction in predictions] == [
        "cold-start-asset-mean",
        "historical-mean",
        "cold-start-region-mean",
    ]


def test_supports_non_cash_out_targets() -> None:
    gap_key = _key("cash-transfers-ng", "Africa")
    dataset = ForecastDataset(
        target=ForecastTarget.FUNDING_GAP,
        observation=_observation(),
        rows=(
            _row(120_000.0, gap_key, ForecastTarget.FUNDING_GAP),
            _row(-40_000.0, gap_key, ForecastTarget.FUNDING_GAP),
        ),
    )
    known, cold = cold_start_baseline(
        dataset,
        _horizon(),
        [gap_key, _key("new-program", "Africa")],
    )
    assert known.target is ForecastTarget.FUNDING_GAP
    assert known.value == pytest.approx(40_000.0)
    assert cold.target is ForecastTarget.FUNDING_GAP
    assert cold.value == pytest.approx(40_000.0)
    assert cold.method == "cold-start-region-mean"


def test_rejects_missing_asset_history() -> None:
    dataset = _cash_out_dataset()
    with pytest.raises(ValueError, match="no historical observations for asset 'XLM'"):
        cold_start_baseline(dataset, _horizon(), [_key("new", "Africa", "XLM")])


def test_bad_argument_types_raise_typed_errors() -> None:
    dataset = _cash_out_dataset()
    with pytest.raises(TypeError, match="dataset must be"):
        cold_start_baseline(None, _horizon(), [_key("p")])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="horizon must be"):
        cold_start_baseline(dataset, "next", [_key("p")])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence of ForecastKey values"):
        cold_start_baseline(dataset, _horizon(), "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"requested_keys\[0\] must be a ForecastKey"):
        cold_start_baseline(dataset, _horizon(), ["p"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="at least one ForecastKey"):
        cold_start_baseline(dataset, _horizon(), [])
