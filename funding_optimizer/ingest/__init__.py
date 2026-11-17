"""Ingestion paths for the funding optimizer.

Ingestion turns raw operator input into the typed structures the optimizer
and forecasting stack consume:

- :mod:`funding_optimizer.ingest.csv_loader` (issue #31) loads programs,
  cohorts, and region-floor constraints from CSV exports.

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

__all__ = [
    "AllocationInputs",
    "CsvLoaderError",
    "load_allocation_inputs",
    "load_cohorts",
    "load_programs",
    "load_region_floors",
]
