"""Tests for the historical cash-out-rate baseline (issue #23).

The baseline assumes tomorrow looks like yesterday: each identifier group
gets a constant prediction (the historical mean of that group's observed
cash-out rates). Every expected value below is computed by hand.
"""

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
    cash_out_rate_baseline,
)


def _observation() -> ObservationWindow:
    return ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31))


def _horizon() -> ForecastHorizon:
    return ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")


def _key(program: str, region: str = "Africa", asset: str = "USDC") -> ForecastKey:
    return ForecastKey(program=program, region=region, asset=asset)


def _row(value: float, key: ForecastKey) -> ForecastRow:
    return ForecastRow(
        key=key,
        window=TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31)),
        target=ForecastTarget.CASH_OUT_RATE,
        value=value,
    )


def _dataset() -> ForecastDataset:
    main_key = _key("cash-transfers-ng")
    edu_key = _key("education-ke")
    return ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=_observation(),
        rows=(
            _row(0.6, main_key),
            _row(0.8, main_key),
            _row(0.5, edu_key),
        ),
    )


def test_baseline_predicts_each_identifier_and_reports_evaluation() -> None:
    predictions = cash_out_rate_baseline(_dataset(), _horizon())

    assert len(predictions) == 2

    main = next(p for p in predictions if p.key.program == "cash-transfers-ng")
    assert main.method == "historical-mean"
    assert main.horizon == _horizon()
    assert main.target is ForecastTarget.CASH_OUT_RATE
    # (0.6 + 0.8) / 2 = 0.7.
    assert main.value == pytest.approx(0.7)
    assert main.evaluation is not None
    assert main.evaluation.n_observations == 2
    # MAE = (|0.6-0.7| + |0.8-0.7|) / 2 = 0.1; RMSE = sqrt(0.1^2) = 0.1.
    assert main.evaluation.mae == pytest.approx(0.1)
    assert main.evaluation.rmse == pytest.approx(0.1)
    # The fit reports the ``evaluated_at`` end of the training window.
    assert main.evaluation.evaluated_at == date(2025, 12, 31)

    edu = next(p for p in predictions if p.key.program == "education-ke")
    assert edu.value == pytest.approx(0.5)
    assert edu.evaluation is not None
    assert edu.evaluation.n_observations == 1
    assert edu.evaluation.mae == pytest.approx(0.0)
    assert edu.evaluation.rmse == pytest.approx(0.0)


def test_group_order_matches_dataset_first_seen_order() -> None:
    predictions = cash_out_rate_baseline(_dataset(), _horizon())
    assert [p.key.program for p in predictions] == ["cash-transfers-ng", "education-ke"]


def test_non_cash_out_dataset_raises() -> None:
    dataset = ForecastDataset(
        target=ForecastTarget.DELIVERY_DELAY,
        observation=_observation(),
        rows=(
            ForecastRow(
                key=_key("p", region="r", asset="XLM"),
                window=TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31)),
                target=ForecastTarget.DELIVERY_DELAY,
                value=14.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="cash_out_rate"):
        cash_out_rate_baseline(dataset, _horizon())


def test_bad_argument_types_raise_typed_errors() -> None:
    with pytest.raises(TypeError, match="dataset must be"):
        cash_out_rate_baseline(None, _horizon())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="horizon must be"):
        cash_out_rate_baseline(_dataset(), "next week")  # type: ignore[arg-type]


def test_prediction_values_stay_inside_unit_interval() -> None:
    dataset = ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=_observation(),
        rows=(
            _row(0.99, _key("p")),
            _row(0.97, _key("p")),
        ),
    )
    predictions = cash_out_rate_baseline(dataset, _horizon())
    assert predictions[0].value == pytest.approx(0.98)
    assert 0.0 <= predictions[0].value <= 1.0
