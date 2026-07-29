"""Read-only strategy protocol and validated output boundary."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from pybacktest.data.features import FeatureBuilder, FeaturePlan, FeatureView
from pybacktest.domain.errors import AdapterContractError, ConfigurationError
from pybacktest.domain.identifiers import OrderId, RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.money import Quantity
from pybacktest.domain.orders import (
    CancelOrderIntent,
    LimitOrderIntent,
    MarketOrderIntent,
    Order,
    TargetQuantity,
    TargetWeight,
)
from pybacktest.strategy.intents import OrderIntent

_INTENT_TYPES = (
    TargetWeight,
    TargetQuantity,
    MarketOrderIntent,
    LimitOrderIntent,
    CancelOrderIntent,
)
_INSTRUMENT_INTENT_TYPES = (
    TargetWeight,
    TargetQuantity,
    MarketOrderIntent,
    LimitOrderIntent,
)


@runtime_checkable
class PortfolioSnapshot(Protocol):
    """Smallest read-only portfolio shape exposed to strategies."""

    @property
    def positions(self) -> Mapping[InstrumentId, Quantity]:
        """Current signed quantities keyed by instrument."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Immutable state supplied to one strategy decision."""

    timestamp: np.datetime64
    portfolio: PortfolioSnapshot
    active_orders: Sequence[Order]
    features: FeatureView
    run_id: RunId

    def __post_init__(self) -> None:
        try:
            timestamp = self.timestamp.astype("datetime64[ns]")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                "strategy timestamp must be valid."
            ) from exc
        if np.isnat(timestamp):
            raise ConfigurationError("strategy timestamp cannot be NaT.")
        if not isinstance(self.portfolio, PortfolioSnapshot):
            raise ConfigurationError(
                "portfolio must implement PortfolioSnapshot."
            )
        if (
            isinstance(self.active_orders, (str, bytes, bytearray))
            or not isinstance(self.active_orders, Sequence)
        ):
            raise ConfigurationError(
                "active_orders must be a sequence of Order values."
            )
        active_orders = tuple(self.active_orders)
        if not all(isinstance(order, Order) for order in active_orders):
            raise ConfigurationError(
                "active_orders must contain only Order values."
            )
        if not isinstance(self.features, FeatureView):
            raise ConfigurationError("features must be a FeatureView.")
        if not isinstance(self.run_id, RunId):
            raise ConfigurationError("run_id must be a RunId.")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "active_orders", active_orders)


@runtime_checkable
class Strategy(Protocol):
    """A strategy that declares features before receiving market values."""

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        """Declare a causal feature plan without reading market values."""
        raise NotImplementedError

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> Sequence[OrderIntent]:
        """Return intents without mutating portfolio or market state."""
        raise NotImplementedError


def validate_strategy_output(
    intents: Sequence[OrderIntent],
    *,
    universe: Collection[InstrumentId],
) -> tuple[OrderIntent, ...]:
    """Return an immutable validated strategy-output sequence."""
    if (
        isinstance(intents, (str, bytes, bytearray))
        or not isinstance(intents, Sequence)
    ):
        raise AdapterContractError(
            "strategy output must be a sequence of order intents."
        )
    if (
        isinstance(universe, (str, bytes, bytearray))
        or not isinstance(universe, Collection)
        or not all(isinstance(item, InstrumentId) for item in universe)
    ):
        raise AdapterContractError(
            "strategy universe must contain InstrumentId values."
        )

    allowed = frozenset(universe)
    validated: list[OrderIntent] = []
    cancelled: set[OrderId] = set()
    for intent in intents:
        if not isinstance(intent, _INTENT_TYPES):
            raise AdapterContractError(
                "strategy output contains a non-intent value."
            )
        if (
            isinstance(intent, _INSTRUMENT_INTENT_TYPES)
            and intent.instrument not in allowed
        ):
            raise AdapterContractError(
                f"strategy intent instrument {intent.instrument} "
                "is outside the universe."
            )
        if isinstance(intent, CancelOrderIntent):
            if intent.order_id in cancelled:
                raise AdapterContractError(
                    "strategy output contains a duplicate cancel request."
                )
            cancelled.add(intent.order_id)
        validated.append(intent)
    return tuple(validated)
