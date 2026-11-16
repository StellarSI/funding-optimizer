"""Prediction intervals and uncertainty metadata for forecasts (issue #29).

Attaches prediction intervals to historical-mean forecasts by modeling the
empirical distribution of residuals within each identifier group. For each
requested quantile, the predicted bound is the group mean plus that quantile
of the group's residuals (observed value minus mean), trained only on the
historical observation window.

This is a transparent baseline: no distribution is assumed beyond the
observed residuals, and predictions are deterministic for a fixed dataset.
Each interval bound is returned as a typed :class:`ForecastPrediction` with
its horizon, method, and evaluation metadata (pinball loss at the quantile).

The optimizer recommends allocations and never signs or submits payouts;
nothing in this package moves funds or performs network calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import (
    ForecastDataset,
    ForecastEvaluation,
    ForecastHorizon,
    ForecastPrediction,
    ForecastRow,
)

__all__ = ["IntervalPrediction", "prediction_intervals"]

_METHOD_PREFIX = "historical-residual-quantile-"


def _require_dataset(dataset: object) -> ForecastDataset:
    if not isinstance(dataset, ForecastDataset):
        raise TypeError(f"dataset must be a ForecastDataset, got {type(dataset).__name__}")
    return dataset


def _require_horizon(horizon: object) -> ForecastHorizon:
    if not isinstance(horizon, ForecastHorizon):
        raise TypeError(f"horizon must be a ForecastHorizon, got {type(horizon).__name__}")
    return horizon


def _validate_quantiles(quantiles: object) -> tuple[float, ...]:
    if isinstance(quantiles, (str, bytes)) or not isinstance(quantiles, Sequence):
        raise TypeError(
            "quantiles must be a sequence of fractions in (0, 1), "
            f"got {type(quantiles).__name__}"
        )
    result: list[float] = []
    for index, quantile in enumerate(quantiles):
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
            raise TypeError(
                f"quantiles[{index}] must be a real number, got {type(quantile).__name__}"
            )
        value = float(quantile)
        if value <= 0.0 or value >= 1.0:
            raise ValueError(
                f"quantiles[{index}] must be strictly inside (0, 1), got {value}"
            )
        result.append(value)
    if not result:
        raise ValueError("quantiles must contain at least one value")
    return tuple(result)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation quantile (type 7)."""
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    position = quantile * (n - 1)
    lower = int(position)
    upper = lower + 1
    fraction = position - lower
    if upper >= n:
        return float(sorted_values[n - 1])
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction)


def _pinball_loss(values: Sequence[float], predicted: float, quantile: float) -> float:
    total = 0.0
    for value in values:
        if value >= predicted:
            total += (value - predicted) * quantile
        else:
            total += (predicted - value) * (1.0 - quantile)
    return total / float(len(values))


def _quantile_method(quantile: float) -> str:
    if quantile.is_integer() and 0.0 < quantile < 1.0:
        label = str(int(quantile))
    else:
        label = f"{quantile:g}"
    return f"{_METHOD_PREFIX}{label}p"


@dataclass(frozen=True)
class IntervalPrediction:
    """A prediction interval bound with its uncertainty metadata.

    Wraps the typed :class:`ForecastPrediction` for a single quantile bound so
    callers can access interval-specific uncertainty metadata directly.
    """

    quantile: float
    prediction: ForecastPrediction

    def __post_init__(self) -> None:
        if isinstance(self.quantile, bool) or not isinstance(self.quantile, (int, float)):
            raise TypeError(f"quantile must be a real number, got {type(self.quantile).__name__}")
        value = float(self.quantile)
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"quantile must be strictly inside (0, 1), got {value}")
        if not isinstance(self.prediction, ForecastPrediction):
            raise TypeError(
                "prediction must be a ForecastPrediction, "
                f"got {type(self.prediction).__name__}"
            )
        object.__setattr__(self, "quantile", value)


def prediction_intervals(
    dataset: ForecastDataset,
    horizon: ForecastHorizon,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> tuple[IntervalPrediction, ...]:
    """Predict interval bounds per identifier using historical residual quantiles.

    For each identifier group, the point estimate is the historical mean and the
    residual distribution is ``value - mean`` over the training observations.
    Each requested quantile is predicted as ``mean + quantile(residuals)`` and
    returned as a typed prediction carrying horizon, method, and evaluation
    metadata (pinball loss).

    Parameters
    ----------
    dataset:
        Validated historical dataset for a single forecast target.
    horizon:
        Future horizon the interval predictions apply to.
    quantiles:
        Quantile levels in ``(0, 1)`` defining the interval/uncertainty bounds.

    Returns
    -------
    tuple[IntervalPrediction, ...]
        Interval bounds for each (identifier, quantile) in first-seen key order
        and requested quantile order.
    """
    data = _require_dataset(dataset)
    future = _require_horizon(horizon)
    levels = _validate_quantiles(quantiles)

    results: list[IntervalPrediction] = []
    for key, rows in data.group_by_key().items():
        values = [float(row.value) for row in rows]
        mean_value = _mean(values)
        residuals = sorted(value - mean_value for value in values)

        for level in levels:
            residual_q = _quantile(residuals, level)
            predicted = mean_value + residual_q
            evaluation = ForecastEvaluation(
                n_observations=len(values),
                mae=None,
                rmse=None,
                pinball_loss=_pinball_loss(values, predicted, level),
                evaluated_at=data.observation.end,
            )
            prediction = ForecastPrediction(
                key=key,
                target=data.target,
                horizon=future,
                method=_quantile_method(level),
                value=predicted,
                evaluation=evaluation,
            )
            results.append(IntervalPrediction(quantile=level, prediction=prediction))

    return tuple(results)
