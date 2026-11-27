"""Delivery analytics.

The optimizer recommends allocations and never signs or submits payouts.
"""

from .aggregates import (
    DEFAULT_COHORT_KEY,
    DEFAULT_PROGRAM_KEY,
    DEFAULT_REGION_KEY,
    AggregateMetrics,
    AggregateReport,
    DeliveryAggregatesError,
    compute_delivery_aggregates,
)
from .aggregates import (
    ObservationWindow as AggregateObservationWindow,
)
from .cashout import (
    AssetCashoutRate,
    CashoutAnalyticsError,
    CashoutReport,
    compute_realized_cashout_rates,
)
from .delays import (
    DEFAULT_CORRIDOR_KEY,
    DEFAULT_DELAY_BUCKETS_SECONDS,
    UNKNOWN_DIMENSION,
    AssetDelaySummary,
    DeliveryDelayDistribution,
    DeliveryDelayError,
    DeliveryDelayReport,
    compute_delivery_delay_distributions,
)
from .delivery import (
    DeliveryAnalyticsError,
    ObservationWindow,
    ReceiptMatch,
    TimeToReceiptMetrics,
    compute_time_to_receipt_metrics,
)
from .quality import (
    DEFAULT_MAX_LAG_SECONDS,
    CompletenessCheck,
    DataQualityError,
    DataQualityReport,
    FreshnessCheck,
    compute_data_quality,
)
from .trustlines import (
    CohortReadiness,
    TrustlineReadinessError,
    TrustlineReadinessReport,
    compute_trustline_readiness_rates,
)
from .trustlines import (
    ObservationWindow as TrustlineObservationWindow,
)
from .unclaimed import (
    AgingBucketSummary,
    UnclaimedBalance,
    UnclaimedFundsError,
    UnclaimedFundsReport,
    compute_unclaimed_funds,
)

__all__ = [
    "DEFAULT_COHORT_KEY",
    "DEFAULT_CORRIDOR_KEY",
    "DEFAULT_DELAY_BUCKETS_SECONDS",
    "DEFAULT_MAX_LAG_SECONDS",
    "DEFAULT_PROGRAM_KEY",
    "DEFAULT_REGION_KEY",
    "UNKNOWN_DIMENSION",
    "AggregateMetrics",
    "AggregateObservationWindow",
    "AggregateReport",
    "AgingBucketSummary",
    "AssetCashoutRate",
    "AssetDelaySummary",
    "CashoutAnalyticsError",
    "CashoutReport",
    "CohortReadiness",
    "CompletenessCheck",
    "DataQualityError",
    "DataQualityReport",
    "DeliveryAggregatesError",
    "DeliveryAnalyticsError",
    "DeliveryDelayDistribution",
    "DeliveryDelayError",
    "DeliveryDelayReport",
    "FreshnessCheck",
    "ObservationWindow",
    "ReceiptMatch",
    "TimeToReceiptMetrics",
    "TrustlineObservationWindow",
    "TrustlineReadinessError",
    "TrustlineReadinessReport",
    "UnclaimedBalance",
    "UnclaimedFundsError",
    "UnclaimedFundsReport",
    "compute_data_quality",
    "compute_delivery_aggregates",
    "compute_delivery_delay_distributions",
    "compute_realized_cashout_rates",
    "compute_time_to_receipt_metrics",
    "compute_trustline_readiness_rates",
    "compute_unclaimed_funds",
]
