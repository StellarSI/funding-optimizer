"""Delivery analytics.

The optimizer recommends allocations and never signs or submits payouts.
"""

from .trustlines import (
    CohortReadiness,
    ObservationWindow,
    TrustlineReadinessError,
    TrustlineReadinessReport,
    compute_trustline_readiness_rates,
)

__all__ = [
    "CohortReadiness",
    "ObservationWindow",
    "TrustlineReadinessError",
    "TrustlineReadinessReport",
    "compute_trustline_readiness_rates",
]
