"""Cold-start handling for new programs and regions (issue #30).

When a requested identifier has no historical observations in the training
dataset, this module produces a transparent fallback prediction so funders
can still estimate cash-out rates, delivery delays, or funding gaps before
committing funds.

Fallback is deterministic and trains only on the historical observation
window:
1. If the exact (program, region, asset) key has observations, use that key's
   historical mean.
2. Else, if other observations exist for the same region and asset (new
   program in a known region), use their historical mean.
3. Else, if other observations exist for the same program and asset (known
   program in a new region), use their historical mean.
4. Else, use the historical mean over all observations for the same asset.

If no observations exist for the requested asset, prediction is not possible
and a ValueError is raised naming the missing history.

Predictions are typed and carry horizon, method, and evaluation metadata.
The optimizer recommends allocations and never signs or submits payouts;
nothing in this package moves funds or performs network calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from math import sqrt

from .schema import (
    ForecastDataset,
    ForecastEvaluation,
    ForecastHorizon,
    ForecastKey,
    ForecastPrediction,
    ForecastRow,
)

__all__ = ["cold_start_baseline"]

_METHOD_EXACT = "historical-mean"
_METHOD_REGION = "cold-start-region-mean"
_METHOD_PROGRAM = "cold-start-program-mean"
_METHOD_GLOBAL = "cold-start-asset-mean"


def _require_dataset(dataset: object) -> ForecastDataset:
    if not isinstance(dataset, ForecastDataset):
        raise TypeError(f"dataset must be a ForecastDataset, got {type(dataset).__name__}")
    return dataset


def _require_horizon(horizon: object) -> ForecastHorizon:
    if not isinstance(horizon, ForecastHorizon):
        raise TypeError(f"horizon must be a ForecastHorizon, got {type(horizon).__name__}")
    return horizon


def _require_requested_keys(keys: object) -> tuple[ForecastKey, ...]:
    if keys is None:
        raise TypeError("requested_keys must be a sequence of ForecastKey values")
    if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
        raise TypeError(
            "requested_keys must be a sequence of ForecastKey values, " f"got {type(keys).__name__}"
        )
    result: list[ForecastKey] = []
    for index, key in enumerate(keys):
        if not isinstance(key, ForecastKey):
            raise TypeError(
                f"requested_keys[{index}] must be a ForecastKey, got {type(key).__name__}"
            )
        result.append(key)
    if not result:
        raise ValueError("requested_keys must contain at least one ForecastKey")
    return tuple(result)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values))


def _evaluation(
    rows: Sequence[ForecastRow], mean_value: float, evaluated_at: date
) -> ForecastEvaluation:
    values = [float(row.value) for row in rows]
    n = len(values)
    abs_errors = [abs(value - mean_value) for value in values]
    sq_errors = [(value - mean_value) ** 2 for value in values]
    return ForecastEvaluation(
        n_observations=n,
        mae=_mean(abs_errors),
        rmse=sqrt(_mean(sq_errors)),
        pinball_loss=None,
        evaluated_at=evaluated_at,
    )


def _pool(
    rows: Sequence[ForecastRow],
    *,
    region: str | None = None,
    program: str | None = None,
    asset: str,
) -> tuple[ForecastRow, ...]:
    selected: list[ForecastRow] = []
    for row in rows:
        if row.key.asset != asset:
            continue
        if region is not None and row.key.region != region:
            continue
        if program is not None and row.key.program != program:
            continue
        selected.append(row)
    return tuple(selected)


def cold_start_baseline(
    dataset: ForecastDataset,
    horizon: ForecastHorizon,
    requested_keys: Sequence[ForecastKey],
) -> tuple[ForecastPrediction, ...]:
    """Predict each requested key using historical means with cold-start fallback.

    For keys not present in ``dataset``, fallback prefers same-region/same-asset
    history (new program), then same-program/same-asset history (new region),
    then all history for the same asset. This handles cold starts for new
    programs and regions while training only on the dataset observation window.

    Parameters
    ----------
    dataset:
        Validated historical dataset for a single forecast target.
    horizon:
        Future horizon the predictions apply to.
    requested_keys:
        Identifiers to predict. May include keys absent from ``dataset`` to
        request cold-start predictions.

    Returns
    -------
    tuple[ForecastPrediction, ...]
        One typed prediction per requested key, in requested order.
    """
    data = _require_dataset(dataset)
    future = _require_horizon(horizon)
    keys = _require_requested_keys(requested_keys)

    by_key = data.group_by_key()
    evaluated_at = data.observation.end
    predictions: list[ForecastPrediction] = []

    for key in keys:
        exact = by_key.get(key)
        if exact is not None:
            rows = tuple(exact)
            method = _METHOD_EXACT
        else:
            region_rows = _pool(data.rows, region=key.region, asset=key.asset)
            if region_rows:
                rows = region_rows
                method = _METHOD_REGION
            else:
                program_rows = _pool(data.rows, program=key.program, asset=key.asset)
                if program_rows:
                    rows = program_rows
                    method = _METHOD_PROGRAM
                else:
                    global_rows = _pool(data.rows, asset=key.asset)
                    if not global_rows:
                        raise ValueError(
                            "cannot produce a cold-start prediction for "
                            f"{key.label!r}: no historical observations for asset "
                            f"{key.asset!r} in the dataset observation window"
                        )
                    rows = global_rows
                    method = _METHOD_GLOBAL

        mean_value = _mean([float(row.value) for row in rows])
        predictions.append(
            ForecastPrediction(
                key=key,
                target=data.target,
                horizon=future,
                method=method,
                value=mean_value,
                evaluation=_evaluation(rows, mean_value, evaluated_at),
            )
        )

    return tuple(predictions)
