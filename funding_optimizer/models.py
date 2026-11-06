"""Core value objects shared across the funding optimizer.

Currently provides :class:`Budget`, the (possibly multi-asset) budget model
used by the optimizer. The optimizer recommends allocations and never signs
or submits payouts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

_ASSET_CODE = re.compile(r"^[A-Za-z0-9]{1,12}(:[A-Za-z0-9]{1,56})?$")


def validate_asset_code(asset: str) -> str:
    """Trim, uppercase and validate a Stellar asset code.

    Accepts 1-12 alphanumeric characters (case-insensitive input is
    normalized), optionally suffixed with ``:ISSUER`` where the issuer is
    1-56 alphanumeric characters.
    """
    if not isinstance(asset, str):
        raise TypeError(f"Asset code must be a string, got {type(asset).__name__}")
    code = asset.strip().upper()
    if not _ASSET_CODE.match(code):
        raise ValueError(
            f"Asset code {asset!r} is not valid; expected 1-12 alphanumeric "
            "characters with an optional ':ISSUER' suffix"
        )
    return code


@dataclass(frozen=True)
class Budget:
    """A budget the optimizer may allocate, in one or more Stellar assets.

    Two construction shapes are supported:

    .. code-block:: python

        Budget(total=250_000, asset="USDC")              # single asset
        Budget(amounts={"USDC": 150_000, "XLM": 40_000_000})  # multi-asset

    Exactly one shape may be used. Asset codes are normalized with
    :func:`validate_asset_code`; amounts must be finite, strictly positive
    real numbers. Amounts are kept in insertion order so allocation layouts
    are deterministic across runs.
    """

    amounts: Mapping[str, float]

    def __init__(
        self,
        total: float | None = None,
        asset: str | None = None,
        *,
        amounts: Mapping[str, float] | None = None,
    ) -> None:
        if amounts is not None:
            if total is not None or asset is not None:
                raise ValueError(
                    "Budget accepts either amounts=... or total=... with asset=..., not both"
                )
            raw: Mapping[str, object] = dict(amounts)
        elif total is not None and asset is not None:
            raw = {asset: total}
        else:
            raise ValueError("Budget requires either amounts=... or both total=... and asset=...")

        parsed: dict[str, float] = {}
        for code, amount in raw.items():
            asset_code = validate_asset_code(code)
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise TypeError(
                    f"Budget amount for asset {asset_code!r} must be a real number, "
                    f"got {amount!r}"
                )
            value = float(amount)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Budget amount for asset {asset_code!r} must be a positive, finite "
                    f"number, got {amount!r}"
                )
            if asset_code in parsed:
                raise ValueError(f"Duplicate asset {asset_code!r} in budget")
            parsed[asset_code] = value

        if not parsed:
            raise ValueError("Budget must contain at least one asset")

        object.__setattr__(self, "amounts", parsed)

    @property
    def assets(self) -> tuple[str, ...]:
        """Asset codes in insertion order."""
        return tuple(self.amounts)

    @property
    def is_multi_asset(self) -> bool:
        """True when the budget spans more than one asset."""
        return len(self.amounts) > 1

    def amount(self, asset: str) -> float:
        """Amount budgeted in ``asset``; raises :class:`KeyError` when absent."""
        code = validate_asset_code(asset)
        try:
            return self.amounts[code]
        except KeyError:
            raise KeyError(f"Asset {code!r} is not part of this budget") from None

    @property
    def total(self) -> float:
        """Total budgeted amount; only defined for single-asset budgets.

        Raises :class:`ValueError` for multi-asset budgets because summing
        distinct assets is ambiguous; use ``amounts`` instead.
        """
        if self.is_multi_asset:
            raise ValueError(
                "total is ambiguous for multi-asset budgets; use amounts or amount(asset)"
            )
        return next(iter(self.amounts.values()))

    @property
    def asset(self) -> str:
        """The asset code; only defined for single-asset budgets.

        Raises :class:`ValueError` for multi-asset budgets; use ``assets``
        instead.
        """
        if self.is_multi_asset:
            raise ValueError("asset is ambiguous for multi-asset budgets; use assets")
        return next(iter(self.amounts))
