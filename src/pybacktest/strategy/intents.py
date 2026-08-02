"""Typed order-intent union exposed to strategy implementations."""

from typing import TypeAlias

from pybacktest.domain.orders import (
    CancelOrderIntent,
    DecisionReason,
    LimitOrderIntent,
    MarketOrderIntent,
    TargetQuantity,
    TargetWeight,
)

OrderIntent: TypeAlias = (
    TargetWeight
    | TargetQuantity
    | MarketOrderIntent
    | LimitOrderIntent
    | CancelOrderIntent
)

__all__ = [
    "CancelOrderIntent",
    "DecisionReason",
    "LimitOrderIntent",
    "MarketOrderIntent",
    "OrderIntent",
    "TargetQuantity",
    "TargetWeight",
]
