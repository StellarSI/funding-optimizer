"""Tests for prediction intervals and uncertainty metadata (issue #29)."""

from __future__ import annotations

from datetime import date

import pytest

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastKey,
    ForecastPrediction,
    ForecastRow,
    ForecastTarget,
    ObservationWindow,
    TimeWindow,
)
from funding_optimizer.forecasting.uncertainty import IntervalPrediction, prediction_intervals


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
    main = _key("cash-transfers-ng")
    edu = _key("education-ke")
    return ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=_observation(),
        rows=(
            _row(0.6, main),
            _row(0.8, main),
            _row(1.0, main),
            _row(0.5, edu),
        ),
    )


def test_prediction_intervals_return_typed_predictions_with_metadata() -> None:
    intervals = prediction_intervals(_dataset(), _horizon(), quantiles=(0.25, 0.5, 0.75))

    assert len(intervals) == 6
    assert all(isinstance(item, IntervalPrediction) for item in intervals)

    main = [item for item in intervals if item.prediction.key.program == "cash-transfers-ng"]
    assert [item.quantile for item in main] == [0.25, 0.5, 0.75]
    assert [item.prediction.method for item in main] == [
        "historical-residual-quantile-0.25p",
        "historical-residual-quantile-0.5p",
        "historical-residual-quantile-0.75p",
    ]
    # mean = 0.8, residuals sorted = [-0.2, 0.0, 0.2]; q25 -> -0.1, q50 -> 0.0, q75 -> 0.1
    assert [item.prediction.value for item in main] == pytest.approx([0.7, 0.8, 0.9])
    for item in main:
        pred = item.prediction
        assert isinstance(pred, ForecastPrediction)
        assert pred.target is ForecastTarget.CASH_OUT_RATE
        assert pred.horizon == _horizon()
        assert pred.evaluation is not None
        assert pred.evaluation.n_observations == 3
        assert pred.evaluation.pinball_loss is not None
        assert pred.evaluation.pinball_loss >= 0.0
        assert pred.evaluation.evaluated_at == date(2025, 12, 31)


def test_single_observation_group_has_zero_uncertainty_interval() -> None:
    intervals = prediction_intervals(_dataset(), _horizon(), quantiles=(0.1, 0.9))
    edu = [item for item in intervals if item.prediction.key.program == "education-ke"]
    assert [item.prediction.value for item in edu] == pytest.approx([0.5, 0.5])
    for item in edu:
        assert item.prediction.evaluation is not None
        assert item.prediction.evaluation.pinball_loss == pytest.approx(0.0)


def test_group_order_matches_dataset_first_seen_order() -> None:
    intervals = prediction_intervals(_dataset(), _horizon(), quantiles=(0.5,))
    assert [item.prediction.key.program for item in intervals] == [
        "cash-transfers-ng",
        "education-ke",
    ]


def test_pinball_loss_reflects_quantile_uncertainty() -> None:
    # mean 0.8, values [0.6,0.8,1.0], median prediction 0.8 -> 0.5 * (0.2+0+0.2)/3
    (interval,) = [
        item
        for item in prediction_intervals(_dataset(), _horizon(), quantiles=(0.5,))
        if item.prediction.key.program == "cash-transfers-ng"
    ]
    assert interval.prediction.evaluation is not None
    assert interval.prediction.evaluation.pinball_loss == pytest.approx(0.2 / 3.0)


def test_supports_other_targets() -> None:
    dataset = ForecastDataset(
        target=ForecastTarget.FUNDING_GAP,
        observation=_observation(),
        rows=(
            ForecastRow(
                key=_key("p"),
                window=TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31)),
                target=ForecastTarget.FUNDING_GAP,
                value=100.0,
            ),
            ForecastRow(
                key=_key("p"),
                window=TimeWindow(start=date(2025, 7, 1), end=date(2025, 9, 30)),
                target=ForecastTarget.FUNDING_GAP,
                value=-100.0,
            ),
        ),
    )
    low, high = prediction_intervals(dataset, _horizon(), quantiles=(0.25, 0.75))
    # mean 0, residuals [-100, 100], q25=-50, q75=50
    assert low.prediction.target is ForecastTarget.FUNDING_GAP
    assert low.prediction.value == pytest.approx(-50.0)
    assert high.prediction.value == pytest.approx(50.0)


def test_bad_argument_types_raise_typed_errors() -> None:
    with pytest.raises(TypeError, match="dataset must be"):
        prediction_intervals(None, _horizon())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="horizon must be"):
        prediction_intervals(_dataset(), "next")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence of fractions"):
        prediction_intervals(_dataset(), _horizon(), quantiles=0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"quantiles\[0\] must be a real number"):
        prediction_intervals(_dataset(), _horizon(), quantiles=("0.5",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly inside"):
        prediction_intervals(_dataset(), _horizon(), quantiles=(0.0,))
    with pytest.raises(ValueError, match="at least one value"):
        prediction_intervals(_dataset(), _horizon(), quantiles=())
