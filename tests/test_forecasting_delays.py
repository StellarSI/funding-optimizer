"""Tests for delivery-delay quantile predictions (issue #24).

The model reads the empirical quantiles of the historical delivery delays
per identifier. Expected values below are hand-computed.
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
)
from funding_optimizer.forecasting.delays import delivery_delay_quantiles


def _observation() -> ObservationWindow:
    return ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31))


def _horizon() -> ForecastHorizon:
    return ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 4, 30), frequency="month")


def _key(program: str, asset: str = "XLM") -> ForecastKey:
    return ForecastKey(program=program, region="Africa", asset=asset)


def _row(key: ForecastKey, value: float) -> ForecastRow:
    return ForecastRow(
        key=key,
        window=TimeWindow(start=date(2025, 9, 1), end=date(2025, 12, 31)),
        target=ForecastTarget.DELIVERY_DELAY,
        value=value,
    )


def _dataset() -> ForecastDataset:
    main_key = _key("cash-transfers-ng")
    other_key = _key("education-ke", asset="USDT")
    return ForecastDataset(
        target=ForecastTarget.DELIVERY_DELAY,
        observation=_observation(),
        rows=(
            _row(main_key, 10.0),
            _row(main_key, 20.0),
            _row(main_key, 30.0),
            _row(other_key, 100.0),
        ),
    )


def test_median_and_interpolated_quantiles_are_ordered() -> None:
    predictions = delivery_delay_quantiles(_dataset(), _horizon(), quantiles=(0.25, 0.5, 0.75))
    main = [p for p in predictions if p.key.program == "cash-transfers-ng"]
    assert [p.method for p in main] == [
        "historical-quantile-25p",
        "historical-quantile-50p",
        "historical-quantile-75p",
    ]
    assert main[0].value == pytest.approx(15.0)
    assert main[1].value == pytest.approx(20.0)
    assert main[2].value == pytest.approx(25.0)

    # Each prediction records observations and a finite pinball loss.
    for prediction in main:
        assert prediction.evaluation is not None
        assert prediction.evaluation.n_observations == 3
        assert prediction.evaluation.pinball_loss is not None
        assert prediction.evaluation.pinball_loss >= 0.0


def test_median_pinball_loss_is_half_mean_absolute_error() -> None:
    predictions = delivery_delay_quantiles(_dataset(), _horizon(), quantiles=(0.5,))
    main = next(p for p in predictions if p.key.program == "cash-transfers-ng")
    assert main.value == pytest.approx(20.0)
    # Pinball loss at q=0.5 is 0.5 * mean(|y - yhat|) = 0.5 * (10+0+10)/3 = 10/3.
    assert main.evaluation is not None
    assert main.evaluation.pinball_loss == pytest.approx(10.0 / 3.0)


def test_single_observation_group_predicts_its_value() -> None:
    predictions = delivery_delay_quantiles(_dataset(), _horizon(), quantiles=(0.5, 0.9))
    other = [p for p in predictions if p.key.program == "education-ke"]
    assert len(other) == 2
    assert all(p.value == pytest.approx(100.0) for p in other)


def test_non_delivery_delay_dataset_raises() -> None:
    dataset = ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=_observation(),
        rows=(
            ForecastRow(
                key=_key("p"),
                window=TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31)),
                target=ForecastTarget.CASH_OUT_RATE,
                value=0.5,
            ),
        ),
    )
    with pytest.raises(ValueError, match="delivery_delay"):
        delivery_delay_quantiles(dataset, _horizon())


def test_bad_argument_types_raise_typed_errors() -> None:
    with pytest.raises(TypeError, match="dataset must be"):
        delivery_delay_quantiles(None, _horizon())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="horizon must be"):
        delivery_delay_quantiles(_dataset(), "next week")  # type: ignore[arg-type]


def test_invalid_quantiles_raise_typed_errors() -> None:
    dataset = _dataset()
    with pytest.raises(TypeError, match="iterable of fractions"):
        delivery_delay_quantiles(dataset, _horizon(), quantiles=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly inside"):
        delivery_delay_quantiles(dataset, _horizon(), quantiles=(0.0,))
    with pytest.raises(ValueError, match="strictly inside"):
        delivery_delay_quantiles(dataset, _horizon(), quantiles=(1.0,))
    with pytest.raises(ValueError, match="at least one"):
        delivery_delay_quantiles(dataset, _horizon(), quantiles=())
    with pytest.raises(TypeError, match="real number"):
        delivery_delay_quantiles(
            dataset,
            _horizon(),
            quantiles=("50p",),  # type: ignore[arg-type]
        )
