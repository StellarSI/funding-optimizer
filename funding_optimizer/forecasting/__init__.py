"""Disbursement forecasting: schema, baselines, delay, and gap predictions.

The forecasting package describes what funders need before committing
funds: observed cash-out rates, delivery delays, and funding gaps keyed by
program, region, asset and time window, plus the typed predictions those
observations produce.

The optimizer recommends allocations and never signs or submits payouts;
nothing in this package moves funds or performs network calls.
"""

from .baseline import cash_out_rate_baseline
from .delays import delivery_delay_quantiles
from .gaps import funding_gap_baseline
from .schema import (
    ForecastDataset,
    ForecastEvaluation,
    ForecastHorizon,
    ForecastKey,
    ForecastPrediction,
    ForecastRow,
    ForecastTarget,
    ObservationWindow,
    TimeWindow,
    missing_identifiers,
)

__all__ = [
    "ForecastDataset",
    "ForecastEvaluation",
    "ForecastHorizon",
    "ForecastKey",
    "ForecastPrediction",
    "ForecastRow",
    "ForecastTarget",
    "ObservationWindow",
    "TimeWindow",
    "cash_out_rate_baseline",
    "delivery_delay_quantiles",
    "funding_gap_baseline",
    "missing_identifiers",
]
