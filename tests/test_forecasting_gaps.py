"""Tests for funding-gap projections (issue #25).

The baseline assumes tomorrow looks like yesterday: each identifier group
gets a constant prediction (the historical mean of that group's observed
funding gaps). Positive predictions are projected shortfalls, negative ones
projected surpluses. Every expected value below is computed by hand.
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
    funding_gap_baseline,
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
        target=ForecastTarget.FUNDING_GAP,
        value=value,
    )


def _dataset() -> ForecastDataset:
    main_key = _key("cash-transfers-ng")
    edu_key = _key("education-ke")
    return ForecastDataset(
        target=ForecastTarget.FUNDING_GAP,
        observation=_observation(),
        rows=(
            # Shortfall of 120,000 and a surplus of 40,000 for the main program.
            _row(120_000.0, main_key),
            _row(-40_000.0, main_key),
            _row(30_000.0, edu_key),
        ),
    )


def test_baseline_predicts_each_identifier_and_reports_evaluation() -> None:
    predictions = funding_gap_baseline(_dataset(), _horizon())

    assert len(predictions) == 2

    main = next(p for p in predictions if p.key.program == "cash-transfers-ng")
    assert main.method == "historical-mean"
    assert main.horizon == _horizon()
    assert main.target is ForecastTarget.FUNDING_GAP
    # (120,000 + (-40,000)) / 2 = 40,000.
    assert main.value == pytest.approx(40_000.0)
    assert main.evaluation is not None
    assert main.evaluation.n_observations == 2
    # MAE = (|120,000 - 40,000| + |-40,000 - 40,000|) / 2 = 80,000.
    assert main.evaluation.mae == pytest.approx(80_000.0)
    # RMSE = sqrt((80,000^2 + 80,000^2) / 2) = 80,000.
    assert main.evaluation.rmse == pytest.approx(80_000.0)
    # The fit reports the ``evaluated_at`` end of the training window.
    assert main.evaluation.evaluated_at == date(2025, 12, 31)

    edu = next(p for p in predictions if p.key.program == "education-ke")
    assert edu.value == pytest.approx(30_000.0)
    assert edu.evaluation is not None
    assert edu.evaluation.n_observations == 1
    assert edu.evaluation.mae == pytest.approx(0.0)
    assert edu.evaluation.rmse == pytest.approx(0.0)


def test_negative_gap_predicts_a_surplus() -> None:
    key = _key("education-ke")
    dataset = ForecastDataset(
        target=ForecastTarget.FUNDING_GAP,
        observation=_observation(),
        rows=(
            _row(-10_000.0, key),
            _row(-30_000.0, key),
        ),
    )
    predictions = funding_gap_baseline(dataset, _horizon())
    assert predictions[0].value == pytest.approx(-20_000.0)


def test_group_order_matches_dataset_first_seen_order() -> None:
    predictions = funding_gap_baseline(_dataset(), _horizon())
    assert [p.key.program for p in predictions] == [
        "cash-transfers-ng",
        "education-ke",
    ]


def test_non_funding_gap_dataset_raises() -> None:
    dataset = ForecastDataset(
        target=ForecastTarget.DELIVERY_DELAY,
        observation=_observation(),
        rows=(
            ForecastRow(
                key=_key("p", region="r", asset="USDC"),
                window=TimeWindow(start=date(2025, 10, 1), end=date(2025, 12, 31)),
                target=ForecastTarget.DELIVERY_DELAY,
                value=14.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="funding_gap"):
        funding_gap_baseline(dataset, _horizon())


def test_bad_argument_types_raise_typed_errors() -> None:
    with pytest.raises(TypeError, match="dataset must be"):
        funding_gap_baseline(None, _horizon())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="horizon must be"):
        funding_gap_baseline(_dataset(), "next month")  # type: ignore[arg-type]


def test_predictions_are_typed_and_horizon_scoped() -> None:
    predictions = funding_gap_baseline(_dataset(), _horizon())
    for prediction in predictions:
        assert isinstance(prediction.horizon, ForecastHorizon)
        assert prediction.method == "historical-mean"
        assert prediction.evaluation is not None
        assert prediction.evaluation.n_observations >= 1
        assert prediction.evaluation.mae is not None and prediction.evaluation.mae >= 0.0
        assert prediction.evaluation.rmse is not None and prediction.evaluation.rmse >= 0.0
