"""Forecast evaluation metrics: MAE, pinball loss, and calibration (issue #28).

Pure, reusable metrics over forecast actuals and predictions. They describe
how well a forecast tracked observed outcomes; nothing here moves funds or
performs network calls.

* :func:`mean_absolute_error` �?" mean absolute error for point forecasts.
* :func:`pinball_loss` �?" asymmetric quantile loss for a single quantile.
* :func:`quantile_calibration` �?" empirical coverage per nominal quantile,
  measuring whether predicted quantiles are calibrated to observed outcomes.

Metrics are deterministic and validate their inputs at the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

__all__ = [
    "CalibrationBin",
    "CalibrationResult",
    "mean_absolute_error",
    "pinball_loss",
    "quantile_calibration",
]


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _require_sequence(name: str, values: object) -> Sequence[object]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers, got {type(values).__name__}")
    if not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of numbers, got {type(values).__name__}")
    return values


def _coerce_values(name: str, values: object) -> tuple[float, ...]:
    seq = _require_sequence(name, values)
    coerced: list[float] = []
    for index, item in enumerate(seq):
        coerced.append(_require_finite_number(item, f"{name}[{index}]"))
    return tuple(coerced)


def _require_matching_length(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError(
            f"actual and predicted must have the same length, got {len(actual)} and {len(predicted)}"
        )
    if not actual:
        raise ValueError("at least one observation is required to compute a metric")


def _require_quantile(quantile: object, name: str = "quantile") -> float:
    value = _require_finite_number(quantile, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly inside (0, 1), got {value!r}")
    return value


def mean_absolute_error(actual: Sequence[object], predicted: Sequence[object]) -> float:
    """Mean absolute error between actual and predicted values.

    Parameters
    ----------
    actual, predicted:
        Sequences of finite numbers with identical length (>= 1).

    Returns
    -------
    float
        ``mean(|actual_i - predicted_i|)``.
    """
    y_true = _coerce_values("actual", actual)
    y_pred = _coerce_values("predicted", predicted)
    _require_matching_length(y_true, y_pred)
    total = 0.0
    for truth, guess in zip(y_true, y_pred):
        total += abs(truth - guess)
    return total / float(len(y_true))


def pinball_loss(
    actual: Sequence[object],
    predicted: object,
    quantile: object,
) -> float:
    """Pinball (quantile) loss for a single quantile forecast.

    The pinball loss is asymmetric: under-predictions are penalized by
    ``quantile`` and over-predictions by ``1 - quantile``. It generalizes the
    absolute error (the median, ``quantile=0.5``, yields half the MAE).

    Parameters
    ----------
    actual:
        Observed values (finite numbers, length >= 1).
    predicted:
        Either a single predicted quantile value applied to every observation,
        or a sequence of per-observation predicted quantile values with the
        same length as ``actual``.
    quantile:
        Nominal quantile level in ``(0, 1)``.

    Returns
    -------
    float
        Mean pinball loss over the observations.
    """
    y_true = _coerce_values("actual", actual)
    q = _require_quantile(quantile)
    if isinstance(predicted, Sequence) and not isinstance(predicted, (str, bytes)):
        y_pred = _coerce_values("predicted", predicted)
    else:
        scalar = _require_finite_number(predicted, "predicted")
        y_pred = tuple(scalar for _ in y_true)
    _require_matching_length(y_true, y_pred)
    total = 0.0
    for truth, guess in zip(y_true, y_pred):
        error = truth - guess
        if error >= 0.0:
            total += q * error
        else:
            total += (q - 1.0) * error
    return total / float(len(y_true))


@dataclass(frozen=True)
class CalibrationBin:
    """Calibration for one nominal quantile level.

    A perfectly calibrated quantile forecast has ``empirical_coverage``
    equal to ``nominal_quantile`` (the fraction of actuals at or below the
    predicted quantile equals the nominal level).
    """

    nominal_quantile: float
    empirical_coverage: float
    n_observations: int
    calibration_error: float

    def __post_init__(self) -> None:
        nominal = _require_quantile(self.nominal_quantile, "nominal_quantile")
        coverage = _require_finite_number(self.empirical_coverage, "empirical_coverage")
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(
                f"empirical_coverage must be in [0, 1], got {coverage!r}"
            )
        if isinstance(self.n_observations, bool) or not isinstance(self.n_observations, int):
            raise TypeError(
                f"n_observations must be an integer, got {type(self.n_observations).__name__}"
            )
        if self.n_observations < 1:
            raise ValueError(f"n_observations must be >= 1, got {self.n_observations}")
        error = _require_finite_number(self.calibration_error, "calibration_error")
        if error < 0.0:
            raise ValueError(f"calibration_error must be non-negative, got {error!r}")
        object.__setattr__(self, "nominal_quantile", nominal)
        object.__setattr__(self, "empirical_coverage", coverage)
        object.__setattr__(self, "calibration_error", error)


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration summary across nominal quantile levels."""

    bins: tuple[CalibrationBin, ...]
    mean_calibration_error: float

    def __post_init__(self) -> None:
        bins = tuple(self.bins)
        if not bins:
            raise ValueError("calibration requires at least one quantile bin")
        for index, item in enumerate(bins):
            if not isinstance(item, CalibrationBin):
                raise TypeError(
                    f"bins[{index}] must be a CalibrationBin, got {type(item).__name__}"
                )
        mean_error = _require_finite_number(
            self.mean_calibration_error, "mean_calibration_error"
        )
        if mean_error < 0.0:
            raise ValueError(
                f"mean_calibration_error must be non-negative, got {mean_error!r}"
            )
        object.__setattr__(self, "bins", bins)
        object.__setattr__(self, "mean_calibration_error", mean_error)


def quantile_calibration(
    actual: Sequence[object],
    predicted_quantiles: Mapping[object, object],
) -> CalibrationResult:
    """Empirical calibration of quantile forecasts.

    For each nominal quantile level ``q``, computes the fraction of actual
    observations that are less than or equal to the predicted value at ``q``.
    A well-calibrated forecast has empirical coverage close to ``q``.

    Parameters
    ----------
    actual:
        Observed values (finite numbers, length >= 1).
    predicted_quantiles:
        Mapping of nominal quantile level -> predicted value(s). Each level
        must be in ``(0, 1)``. A value may be a single number (applied to all
        observations) or a sequence of per-observation predictions with the
        same length as ``actual``.

    Returns
    -------
    CalibrationResult
        Per-quantile coverage/error and the mean absolute calibration error.
    """
    y_true = _coerce_values("actual", actual)
    if not y_true:
        raise ValueError("at least one observation is required to compute calibration")
    if not isinstance(predicted_quantiles, Mapping):
        raise TypeError(
            "predicted_quantiles must be a mapping of quantile -> prediction, "
            f"got {type(predicted_quantiles).__name__}"
        )
    if not predicted_quantiles:
        raise ValueError("predicted_quantiles must contain at least one quantile level")

    bins: list[CalibrationBin] = []
    for raw_q, raw_pred in predicted_quantiles.items():
        q = _require_quantile(raw_q, "quantile level")
        if isinstance(raw_pred, Sequence) and not isinstance(raw_pred, (str, bytes)):
            preds = _coerce_values(f"predicted_quantiles[{q!r}]", raw_pred)
        else:
            scalar = _require_finite_number(raw_pred, f"predicted_quantiles[{q!r}]")
            preds = tuple(scalar for _ in y_true)
        _require_matching_length(y_true, preds)
        covered = sum(1 for truth, guess in zip(y_true, preds) if truth <= guess)
        coverage = float(covered) / float(len(y_true))
        error = abs(coverage - q)
        bins.append(
            CalibrationBin(
                nominal_quantile=q,
                empirical_coverage=coverage,
                n_observations=len(y_true),
                calibration_error=error,
            )
        )

    bins_sorted = tuple(sorted(bins, key=lambda item: item.nominal_quantile))
    mean_error = sum(item.calibration_error for item in bins_sorted) / float(len(bins_sorted))
    return CalibrationResult(bins=bins_sorted, mean_calibration_error=mean_error)
