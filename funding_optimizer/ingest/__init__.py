"""Ingestion paths for the funding optimizer.

Ingestion turns raw operator and upstream input into the typed structures the
optimizer and forecasting stack consume:

- :mod:`funding_optimizer.ingest.csv_loader` (issue #31) loads programs,
  cohorts, and region-floor constraints from CSV exports.
- :mod:`funding_optimizer.ingest.horizon` (issue #32) reads on-chain
  delivery data from a Stellar Horizon endpoint with pagination, retries,
  and rate limiting.
- :mod:`funding_optimizer.ingest.analytics_api` (issue #33) reads typed flow
  data from a transaction-analytics-api endpoint with the same strict,
  deterministic transport contract.
- :mod:`funding_optimizer.ingest.normalize` (issue #34) maps Horizon and
  analytics records onto one frozen event model.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

from funding_optimizer.ingest.analytics_api import (
    AccountFlow,
    AnalyticsApiClient,
    AnalyticsApiError,
    AnalyticsClientError,
    AnalyticsEnvelopeError,
    AnalyticsRateLimitError,
    AnalyticsRecordError,
    AnalyticsServerError,
    AnalyticsTransportError,
    AssetFlow,
    FlowBucket,
    parse_account_flow,
    parse_asset_flow,
    parse_flow_bucket,
)
from funding_optimizer.ingest.csv_loader import (
    AllocationInputs,
    CsvLoaderError,
    load_allocation_inputs,
    load_cohorts,
    load_programs,
    load_region_floors,
)
from funding_optimizer.ingest.horizon import (
    HorizonClient,
    HorizonClientError,
    HorizonEnvelopeError,
    HorizonError,
    HorizonPage,
    HorizonPayment,
    HorizonRateLimitError,
    HorizonRecordError,
    HorizonServerError,
    HorizonTransportError,
    parse_payment_record,
)
from funding_optimizer.ingest.normalize import (
    EventDirection,
    EventSource,
    NormalizationError,
    NormalizedEvent,
    normalize_analytics_record,
    normalize_analytics_records,
    normalize_horizon_record,
    normalize_horizon_records,
    normalize_record,
    normalize_records,
)

__all__ = [
    "AccountFlow",
    "AllocationInputs",
    "AnalyticsApiClient",
    "AnalyticsApiError",
    "AnalyticsClientError",
    "AnalyticsEnvelopeError",
    "AnalyticsRateLimitError",
    "AnalyticsRecordError",
    "AnalyticsServerError",
    "AnalyticsTransportError",
    "AssetFlow",
    "CsvLoaderError",
    "EventDirection",
    "EventSource",
    "FlowBucket",
    "HorizonClient",
    "HorizonClientError",
    "HorizonEnvelopeError",
    "HorizonError",
    "HorizonPage",
    "HorizonPayment",
    "HorizonRateLimitError",
    "HorizonRecordError",
    "HorizonServerError",
    "HorizonTransportError",
    "NormalizationError",
    "NormalizedEvent",
    "load_allocation_inputs",
    "load_cohorts",
    "load_programs",
    "load_region_floors",
    "normalize_analytics_record",
    "normalize_analytics_records",
    "normalize_horizon_record",
    "normalize_horizon_records",
    "normalize_record",
    "normalize_records",
    "parse_account_flow",
    "parse_asset_flow",
    "parse_flow_bucket",
    "parse_payment_record",
]
