"""Delivery-delay quantile predictions (issue #24).

Models delivery delay by its historical distribution: for each identifier
group, the empirical quantiles of the observed delivery delays are used as
the predicted quantiles. No underlying distribution is assumed, and
predictions are deterministic for a fixed dataset.

The module consumes a validated :class:`ForecastDataset` (delivery-delay
target only) and returns one typed :class:`ForecastPrediction` per
(requested quantile, identifier) with horizon, method, and pinball-loss
evaluation metadata.

The optimizer recommends allocations and never signs or submits payouts;
nothing in this package moves funds or performs network calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .schema import (
    ForecastDataset,
    ForecastEvaluation,
    ForecastHorizon,
    ForecastPrediction,
    ForecastTarget,
)

_METHOD = "historical-quantile"


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    """Linear-interpolation quantile (type 7, R default).

    ``sorted_values`` must be non-empty and sorted ascending.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    position = quantile * (n - 1)
    lower = int(position)
    upper = lower + 1
    fraction = position - lower
    if upper >= n:
        return sorted_values[n - 1]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _pinball_loss(values: Sequence[float], predicted: float, quantile: float) -> float:
    total = 0.0
    for value in values:
        if value >= predicted:
            total += (value - predicted) * quantile
        else:
            total += (predicted - value) * (1.0 - quantile)
    return total / len(values)


def _validate_quantiles(quantiles: object) -> list[float]:
    if isinstance(quantiles, (bool, int, float)):
        raise TypeError(
            f"quantiles must be an iterable of fractions, got a bare {type(quantiles).__name__}"
        )
    if not isinstance(quantiles, Sequence):
        raise TypeError(
            f"quantiles must be an iterable of fractions, got {type(quantiles).__name__}"
        )
    parsed: list[float] = []
    for index, item in enumerate(quantiles):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"quantiles[{index}] must be a real number, got {item!r}")
        value = float(item)
        if not isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"quantiles[{index}] {value} is not strictly inside (0, 1)")
        parsed.append(value)
    if not parsed:
        raise ValueError("quantiles must contain at least one requested quantile")
    return parsed


def delivery_delay_quantiles(
    dataset: ForecastDataset,
    horizon: ForecastHorizon,
    quantiles: Sequence[float] = (0.5,),
) -> tuple[ForecastPrediction, ...]:
    """Predict delivery-delay quantiles per identifier.

    Returns one :class:`ForecastPrediction` per (identifier, quantile) pair,
    in identifier first-seen order and, within each identifier, in the order
    the quantiles were given. Each prediction's ``method`` is
    ``historical-quantile-<q*100>p`` (e.g. ``historical-quantile-50p``) and
    carries the requested horizon plus evaluation metadata: observation
    count and pinball loss at that quantile.

    Raises
    ------
    TypeError
        If ``dataset`` is not a :class:`ForecastDataset`, ``horizon`` is not a
        :class:`ForecastHorizon`, or a requested quantile is not a real number.
    ValueError
        If the dataset targets something other than ``delivery_delay``, no
        quantile was requested, or a requested quantile is outside ``(0, 1)``.
    """
    if not isinstance(dataset, ForecastDataset):
        raise TypeError(f"dataset must be a ForecastDataset, got {type(dataset).__name__}")
    if not isinstance(horizon, ForecastHorizon):
        raise TypeError(f"horizon must be a ForecastHorizon, got {type(horizon).__name__}")
    if dataset.target != ForecastTarget.DELIVERY_DELAY:
        raise ValueError(
            f"delivery_delay_quantiles forecasts delivery_delay targets; "
            f"got {dataset.target.value!r}"
        )
    if not len(dataset.rows):
        raise ValueError("a forecast dataset needs at least one row")
    percentile_levels = _validate_quantiles(quantiles)

    predictions: list[ForecastPrediction] = []
    for key, rows in dataset.group_by_key().items():
        values = sorted(row.value for row in rows)
        for quantile in percentile_levels:
            predicted = _quantile(values, quantile)
            anchor = round(quantile * 100)
            predictions.append(
                ForecastPrediction(
                    key=key,
                    target=ForecastTarget.DELIVERY_DELAY,
                    horizon=horizon,
                    method=f"{_METHOD}-{anchor}p",
                    value=predicted,
                    evaluation=ForecastEvaluation(
                        n_observations=len(values),
                        pinball_loss=_pinball_loss(values, predicted, quantile),
                    ),
                )
            )
    return tuple(predictions)
