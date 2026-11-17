"""Ingestion paths for the funding optimizer.

Ingestion turns raw operator input into the typed structures the optimizer
and forecasting stack consume:

- :mod:`funding_optimizer.ingest.csv_loader` (issue #31) loads programs,
  cohorts, and region-floor constraints from CSV exports.
- :mod:`funding_optimizer.ingest.horizon` (issue #32) reads on-chain
  delivery data from a Stellar Horizon endpoint with pagination, retries,
  and rate limiting.

The optimizer recommends allocations and never signs or submits payouts.
"""

from __future__ import annotations

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

__all__ = [
    "AllocationInputs",
    "CsvLoaderError",
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
    "load_allocation_inputs",
    "load_cohorts",
    "load_programs",
    "load_region_floors",
    "parse_payment_record",
]
