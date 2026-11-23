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

from .delivery import (
    DeliveryAnalyticsError,
    ObservationWindow,
    ReceiptMatch,
    TimeToReceiptMetrics,
    compute_time_to_receipt_metrics,
)

__all__ = [
    "DeliveryAnalyticsError",
    "ObservationWindow",
    "ReceiptMatch",
    "TimeToReceiptMetrics",
    "compute_time_to_receipt_metrics",
    "CohortReadiness",
    "ObservationWindow",
    "TrustlineReadinessError",
    "TrustlineReadinessReport",
    "compute_trustline_readiness_rates",
]
