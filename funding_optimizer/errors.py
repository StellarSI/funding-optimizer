"""Exceptions raised by funding-optimizer.

The optimizer recommends allocations. It never signs or submits payouts.
"""

from __future__ import annotations

__all__ = ["AllocationError", "FundingOptimizerError", "ValidationError"]


class FundingOptimizerError(Exception):
    """Base class for all funding-optimizer errors."""


class ValidationError(FundingOptimizerError, ValueError):
    """Raised when an input fails validation.

    Inherits from :class:`ValueError` so callers can catch either the
    SDK-specific type or the standard exception.
    """


class AllocationError(FundingOptimizerError):
    """Raised when a valid input cannot be turned into a feasible plan."""
