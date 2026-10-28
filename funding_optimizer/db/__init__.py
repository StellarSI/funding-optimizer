"""Persistence boundaries for optimization runs and plans (issue #9)."""

from funding_optimizer.db.models import (
    AllocationRecord,
    Base,
    OptimizationRunRecord,
    plan_from_run,
)

__all__ = ["AllocationRecord", "Base", "OptimizationRunRecord", "plan_from_run"]
