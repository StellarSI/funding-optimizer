"""Funding-gap projections (issue #25).

A transparent, history-only baseline for :data:`~.schema.ForecastTarget.FUNDING_GAP`.
It predicts each (program, region, asset) key as the mean of that key's observed
funding gaps in the training dataset. The model trains only on the dataset's
observation window and never makes network calls or moves funds.

Sign convention (per the schema): a *funding gap* is the shortfall of funds
for a window — positive when commitments exceed disbursements available to
the program, negative when the program has a surplus (funds available above
what was drawn). Predictions therefore carry the target's sign: a positive
prediction is a projected shortfall, a negative one a projected surplus.

Predictions are returned as :class:`~.schema.ForecastPrediction` values carrying
the forecast horizon, the ``"historical-mean"`` method, and in-sample evaluation
metadata (MAE/RMSE over the observations used for that key's fit).
"""

from __future__ import annotations

from math import sqrt

from .schema import (
    ForecastDataset,
    ForecastEvaluation,
    ForecastHorizon,
    ForecastPrediction,
    ForecastTarget,
)

METHOD = "historical-mean"


def _require_funding_gap_dataset(dataset: object) -> ForecastDataset:
    """Validate that ``dataset`` is a funding-gap dataset."""
    if not isinstance(dataset, ForecastDataset):
        raise TypeError(f"dataset must be a ForecastDataset, got {type(dataset).__name__}")
    if dataset.target != ForecastTarget.FUNDING_GAP:
        raise ValueError(
            f"gaps projection model requires target {ForecastTarget.FUNDING_GAP.value}, "
            f"got {dataset.target.value}"
        )
    return dataset


def _require_horizon(horizon: object) -> ForecastHorizon:
    """Validate that ``horizon`` is a forecast horizon."""
    if not isinstance(horizon, ForecastHorizon):
        raise TypeError(f"horizon must be a ForecastHorizon, got {type(horizon).__name__}")
    return horizon


def funding_gap_baseline(
    dataset: ForecastDataset,
    horizon: ForecastHorizon,
) -> tuple[ForecastPrediction, ...]:
    """Predict funding gaps as the historical mean per identifier.

    For each identifier group in ``dataset`` (in first-seen order), returns
    one :class:`~.schema.ForecastPrediction` whose value is the arithmetic
    mean of that group's observed funding gaps (positive values are
    shortfalls, negative values surpluses). Each prediction carries the
    requested horizon, the ``"historical-mean"`` method, and evaluation
    metadata: observation count, MAE, and RMSE of the constant predictor
    against the group's historical observations (computed on the training
    window only).

    Predictions describe expected outcomes only; they never move funds.

    Raises
    ------
    TypeError
        If ``dataset`` is not a :class:`~.schema.ForecastDataset` or
        ``horizon`` is not a :class:`~.schema.ForecastHorizon`.
    ValueError
        If ``dataset`` does not target ``funding_gap``.
    """
    data = _require_funding_gap_dataset(dataset)
    forecast_horizon = _require_horizon(horizon)

    predictions: list[ForecastPrediction] = []
    for key, rows in data.group_by_key().items():
        values = [float(row.value) for row in rows]
        mean = sum(values) / float(len(values))
        mae = sum(abs(value - mean) for value in values) / float(len(values))
        rmse = sqrt(sum((value - mean) ** 2 for value in values) / float(len(values)))
        predictions.append(
            ForecastPrediction(
                key=key,
                target=ForecastTarget.FUNDING_GAP,
                horizon=forecast_horizon,
                method=METHOD,
                value=float(mean),
                evaluation=ForecastEvaluation(
                    n_observations=len(values),
                    mae=float(mae),
                    rmse=float(rmse),
                    evaluated_at=data.observation.end,
                ),
            )
        )
    return tuple(predictions)
