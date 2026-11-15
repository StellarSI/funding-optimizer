"""Time-based backtesting harness for forecast models (issue #27).

Runs forecast methods over historical windows using expanding time folds so
that each fold trains only on past observations and evaluates on a future
horizon. Predictions from each fold are returned with their horizon, method,
and evaluation metadata computed against held-out actuals.

The harness is transparent and deterministic: it never makes network calls
and never moves funds. The optimizer recommends allocations and never signs
or submits payouts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from math import sqrt

from .schema import (
    ForecastDataset,
    ForecastEvaluation,
    ForecastHorizon,
    ForecastKey,
    ForecastPrediction,
    ForecastRow,
    ForecastTarget,
    ObservationWindow,
)

__all__ = [
    "BacktestFold",
    "BacktestResult",
    "Forecaster",
    "run_backtest",
]

Forecaster = Callable[[ForecastDataset, ForecastHorizon], Sequence[ForecastPrediction]]


def _require_dataset(dataset: object) -> ForecastDataset:
    if not isinstance(dataset, ForecastDataset):
        raise TypeError(f"dataset must be a ForecastDataset, got {type(dataset).__name__}")
    return dataset


def _require_horizon(horizon: object) -> ForecastHorizon:
    if not isinstance(horizon, ForecastHorizon):
        raise TypeError(f"horizon must be a ForecastHorizon, got {type(horizon).__name__}")
    return horizon


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


@dataclass(frozen=True)
class BacktestFold:
    """One time-based train/test fold.

    ``train_observation`` is the historical window used for fitting; all
    ``train_rows`` fall within it and end before ``horizon`` starts. ``actuals``
    are the held-out rows whose windows are covered by ``horizon``.
    ``predictions`` are the forecaster's typed predictions for ``horizon``.
    """

    train_observation: ObservationWindow
    horizon: ForecastHorizon
    train_rows: tuple[ForecastRow, ...]
    actuals: tuple[ForecastRow, ...]
    predictions: tuple[ForecastPrediction, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.train_observation, ObservationWindow):
            raise TypeError(
                "train_observation must be an ObservationWindow, "
                f"got {type(self.train_observation).__name__}"
            )
        if not isinstance(self.horizon, ForecastHorizon):
            raise TypeError(f"horizon must be a ForecastHorizon, got {type(self.horizon).__name__}")
        train_rows = tuple(self.train_rows)
        actuals = tuple(self.actuals)
        predictions = tuple(self.predictions)
        for name, seq, expected in (
            ("train_rows", train_rows, ForecastRow),
            ("actuals", actuals, ForecastRow),
            ("predictions", predictions, ForecastPrediction),
        ):
            for index, item in enumerate(seq):
                if not isinstance(item, expected):
                    raise TypeError(
                        f"{name}[{index}] must be a {expected.__name__}, got {type(item).__name__}"
                    )
        object.__setattr__(self, "train_rows", train_rows)
        object.__setattr__(self, "actuals", actuals)
        object.__setattr__(self, "predictions", predictions)


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated result of a time-based backtest run.

    ``predictions`` are all per-fold predictions in fold order. ``evaluation``
    summarizes point-forecast accuracy over matched (actual, predicted) pairs
    across folds when such pairs exist.
    """

    target: ForecastTarget
    folds: tuple[BacktestFold, ...]
    predictions: tuple[ForecastPrediction, ...]
    evaluation: ForecastEvaluation | None

    def __post_init__(self) -> None:
        target = ForecastTarget.parse(self.target)
        folds = tuple(self.folds)
        predictions = tuple(self.predictions)
        for index, fold in enumerate(folds):
            if not isinstance(fold, BacktestFold):
                raise TypeError(f"folds[{index}] must be a BacktestFold, got {type(fold).__name__}")
        for index, pred in enumerate(predictions):
            if not isinstance(pred, ForecastPrediction):
                raise TypeError(
                    f"predictions[{index}] must be a ForecastPrediction, got {type(pred).__name__}"
                )
        if self.evaluation is not None and not isinstance(self.evaluation, ForecastEvaluation):
            raise TypeError(
                "evaluation must be a ForecastEvaluation or None, "
                f"got {type(self.evaluation).__name__}"
            )
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "predictions", predictions)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _actuals_by_key(rows: Sequence[ForecastRow]) -> dict[ForecastKey, tuple[float, ...]]:
    grouped: dict[ForecastKey, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.key, []).append(float(row.value))
    return {key: tuple(vals) for key, vals in grouped.items()}


def _matched_pairs(folds: Sequence[BacktestFold]) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    for fold in folds:
        actuals = _actuals_by_key(fold.actuals)
        for prediction in fold.predictions:
            values = actuals.get(prediction.key)
            if not values:
                continue
            actual_value = _mean(values)
            pairs.append((actual_value, float(prediction.value)))
    return tuple(pairs)


def _aggregate_evaluation(
    folds: Sequence[BacktestFold], target: ForecastTarget
) -> ForecastEvaluation | None:
    pairs = _matched_pairs(folds)
    if not pairs:
        return None
    errors = [abs(actual - predicted) for actual, predicted in pairs]
    mae = _mean(errors)
    rmse = sqrt(_mean([(actual - predicted) ** 2 for actual, predicted in pairs]))
    evaluated_at: date | None = None
    for fold in folds:
        end = fold.horizon.end
        if evaluated_at is None or end > evaluated_at:
            evaluated_at = end
    # Pinball loss is meaningful for quantile forecasts; for point forecasts it
    # is not set here (metrics.pinball_loss can compute it explicitly).
    return ForecastEvaluation(
        n_observations=len(pairs),
        mae=float(mae),
        rmse=float(rmse),
        pinball_loss=None,
        evaluated_at=evaluated_at,
    )


def run_backtest(
    dataset: ForecastDataset,
    horizon: ForecastHorizon,
    forecaster: Forecaster,
    *,
    step_days: int | None = None,
    min_train_rows: int = 1,
) -> BacktestResult:
    """Run an expanding-window, time-based backtest.

    Folds advance over time: for each fold the training set contains rows whose
    windows end on or before a split date inside the dataset observation, and
    the test horizon starts the day after that split date and spans
    ``horizon.days`` days with ``horizon.frequency``. Held-out actuals are
    dataset rows whose windows are fully covered by the test horizon.

    The forecaster is called as ``forecaster(train_dataset, test_horizon)`` and
    must return typed predictions for that horizon. Training uses only
    historical windows (no lookahead).

    Parameters
    ----------
    dataset:
        Validated forecast dataset providing historical observations.
    horizon:
        Template horizon whose length (days) and frequency define each test
        window.
    forecaster:
        Callable producing predictions from a training dataset and horizon.
    step_days:
        Days to advance the split point between folds. Defaults to
        ``horizon.days`` (non-overlapping test windows).
    min_train_rows:
        Minimum number of training rows required to form a fold.

    Returns
    -------
    BacktestResult
        Folds in chronological order, all predictions, and aggregate evaluation
        metadata over matched prediction/actual pairs.
    """
    data = _require_dataset(dataset)
    template = _require_horizon(horizon)
    if not callable(forecaster):
        raise TypeError("forecaster must be callable")
    min_rows = _require_positive_int(min_train_rows, "min_train_rows")
    step = template.days if step_days is None else _require_positive_int(step_days, "step_days")
    horizon_days = template.days

    # Candidate split points are unique row window ends, sorted chronologically.
    row_ends = sorted({row.window.end for row in data.rows})
    folds: list[BacktestFold] = []
    all_predictions: list[ForecastPrediction] = []
    seen_test_starts: set[date] = set()

    for split_end in row_ends:
        test_start = split_end + timedelta(days=1)
        if test_start in seen_test_starts:
            continue
        test_end = split_end + timedelta(days=horizon_days)
        if test_start < data.observation.start or test_end > data.observation.end:
            continue
        try:
            train_observation = ObservationWindow(start=data.observation.start, end=split_end)
            test_horizon = ForecastHorizon(
                start=test_start, end=test_end, frequency=template.frequency
            )
        except (TypeError, ValueError):
            continue

        train_rows = tuple(
            row
            for row in data.rows
            if train_observation.covers(row.window) and row.window.end <= split_end
        )
        actuals = tuple(row for row in data.rows if test_horizon.covers(row.window))
        if len(train_rows) < min_rows or not actuals:
            continue

        # Respect step by only taking folds at step boundaries from the first
        # accepted fold's test start.
        if seen_test_starts:
            # Allow if this test_start is >= last + step days.
            last = max(seen_test_starts)
            if (test_start - last).days < step:
                continue
        seen_test_starts.add(test_start)

        train_dataset = ForecastDataset(
            target=data.target,
            observation=train_observation,
            rows=train_rows,
        )
        predictions = tuple(forecaster(train_dataset, test_horizon))
        for index, prediction in enumerate(predictions):
            if not isinstance(prediction, ForecastPrediction):
                raise TypeError(
                    f"forecaster returned predictions[{index}] as {type(prediction).__name__}, "
                    "expected ForecastPrediction"
                )
            if prediction.target != data.target:
                raise ValueError(
                    f"forecaster returned prediction target {prediction.target.value!r}, "
                    f"expected {data.target.value!r}"
                )
            if prediction.horizon != test_horizon:
                raise ValueError(
                    "forecaster returned a prediction for a different horizon than requested"
                )

        fold = BacktestFold(
            train_observation=train_observation,
            horizon=test_horizon,
            train_rows=train_rows,
            actuals=actuals,
            predictions=predictions,
        )
        folds.append(fold)
        all_predictions.extend(predictions)

    return BacktestResult(
        target=data.target,
        folds=tuple(folds),
        predictions=tuple(all_predictions),
        evaluation=_aggregate_evaluation(folds, data.target),
    )
