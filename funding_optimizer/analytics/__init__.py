"""Delivery analytics.

The optimizer recommends allocations and never signs or submits payouts.
"""

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
]
