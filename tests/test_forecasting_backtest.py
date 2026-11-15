"""Tests for the time-based backtesting harness (issue #27).

The harness must evaluate forecast methods over historical windows without
lookahead: each fold trains only on rows ending before the test horizon and
returns typed predictions with horizon, method, and evaluation metadata.
Expected values below are hand-computed.
"""

from __future__ import annotations

from datetime import date
from math import sqrt

import pytest

from funding_optimizer.forecasting import (
    BacktestFold,
    BacktestResult,
    ForecastDataset,
    ForecastHorizon,
    ForecastKey,
    ForecastPrediction,
    ForecastRow,
    ForecastTarget,
    ObservationWindow,
    TimeWindow,
    cash_out_rate_baseline,
    run_backtest,
)


def _key() -> ForecastKey:
    return ForecastKey(program="cash-transfers-ng", region="Africa", asset="USDC")


def _row(start: str, end: str, value: float) -> ForecastRow:
    return ForecastRow(
        key=_key(),
        window=TimeWindow(start=date.fromisoformat(start), end=date.fromisoformat(end)),
        target=ForecastTarget.CASH_OUT_RATE,
        value=value,
    )


def _dataset() -> ForecastDataset:
    # Four consecutive 10-day windows fully inside the observation window.
    return ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 2, 9)),
        rows=(
            _row("2025-01-01", "2025-01-10", 0.6),
            _row("2025-01-11", "2025-01-20", 0.8),
            _row("2025-01-21", "2025-01-30", 0.7),
            _row("2025-01-31", "2025-02-09", 0.9),
        ),
    )


def _horizon_template() -> ForecastHorizon:
    # 10-day horizon template; run_backtest uses its length and frequency.
    return ForecastHorizon(start=date(2025, 1, 11), end=date(2025, 1, 20), frequency="day")


def test_run_backtest_produces_expanding_time_folds_without_lookahead() -> None:
    result = run_backtest(_dataset(), _horizon_template(), cash_out_rate_baseline)

    assert isinstance(result, BacktestResult)
    assert result.target is ForecastTarget.CASH_OUT_RATE
    assert len(result.folds) == 3

    # Fold 1: train on [Jan1-10], test Jan11-20 (actual 0.8), pred mean([0.6])=0.6
    fold1 = result.folds[0]
    assert fold1.train_observation == ObservationWindow(
        start=date(2025, 1, 1), end=date(2025, 1, 10)
    )
    assert fold1.horizon == ForecastHorizon(
        start=date(2025, 1, 11), end=date(2025, 1, 20), frequency="day"
    )
    assert [r.value for r in fold1.train_rows] == [pytest.approx(0.6)]
    assert [r.value for r in fold1.actuals] == [pytest.approx(0.8)]
    assert all(r.window.end < fold1.horizon.start for r in fold1.train_rows)
    assert all(fold1.horizon.covers(r.window) for r in fold1.actuals)

    # Fold 2: train on first two rows, pred mean([0.6,0.8])=0.7, actual 0.7
    fold2 = result.folds[1]
    assert [r.value for r in fold2.train_rows] == [pytest.approx(0.6), pytest.approx(0.8)]
    assert [r.value for r in fold2.actuals] == [pytest.approx(0.7)]
    assert fold2.predictions[0].value == pytest.approx(0.7)

    # Fold 3: train on first three, pred mean([0.6,0.8,0.7])=0.7, actual 0.9
    fold3 = result.folds[2]
    assert [r.value for r in fold3.train_rows] == [
        pytest.approx(0.6),
        pytest.approx(0.8),
        pytest.approx(0.7),
    ]
    assert [r.value for r in fold3.actuals] == [pytest.approx(0.9)]
    assert fold3.predictions[0].value == pytest.approx((0.6 + 0.8 + 0.7) / 3.0)


def test_backtest_predictions_carry_horizon_method_and_evaluation_metadata() -> None:
    result = run_backtest(_dataset(), _horizon_template(), cash_out_rate_baseline)

    assert len(result.predictions) == 3
    for fold in result.folds:
        for prediction in fold.predictions:
            assert isinstance(prediction, ForecastPrediction)
            assert prediction.horizon == fold.horizon
            assert prediction.method == "historical-mean"
            assert prediction.target is ForecastTarget.CASH_OUT_RATE
            assert prediction.evaluation is not None
            assert prediction.evaluation.n_observations >= 1
            assert prediction.evaluation.evaluated_at == fold.train_observation.end


def test_backtest_aggregate_evaluation_matches_matched_pairs() -> None:
    result = run_backtest(_dataset(), _horizon_template(), cash_out_rate_baseline)

    # Matched (actual, predicted): (0.8,0.6), (0.7,0.7), (0.9,0.7)
    assert result.evaluation is not None
    assert result.evaluation.n_observations == 3
    assert result.evaluation.mae == pytest.approx((0.2 + 0.0 + 0.2) / 3.0)
    assert result.evaluation.rmse == pytest.approx(sqrt((0.04 + 0.0 + 0.04) / 3.0))
    assert result.evaluation.pinball_loss is None
    assert result.evaluation.evaluated_at == date(2025, 2, 9)


def test_backtest_returns_no_folds_when_history_is_insufficient() -> None:
    # Only one row: cannot form a fold with a held-out actual horizon.
    dataset = ForecastDataset(
        target=ForecastTarget.CASH_OUT_RATE,
        observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 2, 9)),
        rows=(_row("2025-01-01", "2025-01-10", 0.6),),
    )
    result = run_backtest(dataset, _horizon_template(), cash_out_rate_baseline)

    assert result.folds == ()
    assert result.predictions == ()
    assert result.evaluation is None


def test_backtest_rejects_invalid_arguments() -> None:
    dataset = _dataset()
    horizon = _horizon_template()

    with pytest.raises(TypeError, match="dataset must be a ForecastDataset"):
        run_backtest(None, horizon, cash_out_rate_baseline)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="horizon must be a ForecastHorizon"):
        run_backtest(dataset, "next", cash_out_rate_baseline)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="forecaster must be callable"):
        run_backtest(dataset, horizon, "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_train_rows must be >= 1"):
        run_backtest(dataset, horizon, cash_out_rate_baseline, min_train_rows=0)
    with pytest.raises(ValueError, match="step_days must be >= 1"):
        run_backtest(dataset, horizon, cash_out_rate_baseline, step_days=0)


def test_backtest_rejects_mismatched_forecaster_output() -> None:
    dataset = _dataset()
    horizon = _horizon_template()

    def wrong_target(d: ForecastDataset, h: ForecastHorizon) -> tuple[ForecastPrediction, ...]:
        return (
            ForecastPrediction(
                key=_key(),
                target=ForecastTarget.FUNDING_GAP,
                horizon=h,
                method="bad",
                value=1.0,
            ),
        )

    with pytest.raises(ValueError, match="prediction target"):
        run_backtest(dataset, horizon, wrong_target)

    def wrong_horizon(d: ForecastDataset, h: ForecastHorizon) -> tuple[ForecastPrediction, ...]:
        other = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 1, 10))
        return (
            ForecastPrediction(
                key=_key(),
                target=ForecastTarget.CASH_OUT_RATE,
                horizon=other,
                method="bad",
                value=0.5,
            ),
        )

    with pytest.raises(ValueError, match="different horizon"):
        run_backtest(dataset, horizon, wrong_horizon)

    def wrong_type(d: ForecastDataset, h: ForecastHorizon) -> tuple[object, ...]:
        return ("not-a-prediction",)

    with pytest.raises(TypeError, match="ForecastPrediction"):
        run_backtest(dataset, horizon, wrong_type)  # type: ignore[arg-type]


def test_backtest_fold_and_result_types_are_validated() -> None:
    dataset = _dataset()
    horizon = _horizon_template()
    fold = run_backtest(dataset, horizon, cash_out_rate_baseline).folds[0]

    with pytest.raises(TypeError, match="train_observation"):
        BacktestFold(
            train_observation="nope",  # type: ignore[arg-type]
            horizon=fold.horizon,
            train_rows=fold.train_rows,
            actuals=fold.actuals,
            predictions=fold.predictions,
        )
    with pytest.raises(TypeError, match="predictions\\[0\\]"):
        BacktestFold(
            train_observation=fold.train_observation,
            horizon=fold.horizon,
            train_rows=fold.train_rows,
            actuals=fold.actuals,
            predictions=("bad",),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least one quantile bin"):
        # Ensure ForecastEvaluation remains the schema type in result validation path.
        pass

    with pytest.raises(TypeError, match="evaluation must be"):
        BacktestResult(
            target=ForecastTarget.CASH_OUT_RATE,
            folds=(),
            predictions=(),
            evaluation="bad",  # type: ignore[arg-type]
        )


def test_backtest_respects_min_train_rows() -> None:
    # Require 2 training rows: first possible split has only 1 row, so first fold is skipped.
    result = run_backtest(
        _dataset(), _horizon_template(), cash_out_rate_baseline, min_train_rows=2
    )

    assert len(result.folds) == 2
    assert [len(f.train_rows) for f in result.folds] == [2, 3]
    assert [p.value for f in result.folds for p in f.predictions] == [
        pytest.approx(0.7),
        pytest.approx((0.6 + 0.8 + 0.7) / 3.0),
    ]
