"""funding-optimizer: predictive analytics for grant allocation and disbursement.

The optimizer recommends allocations; it never signs or submits payouts.
Callers retain control over execution.

The public API matches the quick example in ``README.md``::

    from funding_optimizer import Budget, Program, Optimizer
"""

from __future__ import annotations

from funding_optimizer.models import Budget, Program
from funding_optimizer.optimizer import Optimizer

__all__ = ["Budget", "Program", "Optimizer"]
__version__ = "0.1.0"
