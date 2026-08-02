"""Read-only strategy protocol and validated output boundary."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
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


class _PortfolioSnapshotSource(Protocol):
    @property
    def positions(self) -> object:
        """Positions projected from an engine-owned portfolio state."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True, init=False)
class PortfolioSnapshot:
    """Explicit immutable portfolio quantities exposed to strategies."""

    positions: Mapping[InstrumentId, Quantity]

    def __init__(self, positions: object) -> None:
        if not isinstance(positions, Mapping):
            raise ConfigurationError(
                "portfolio positions must be a mapping."
            )
        copied = dict(positions)
        if not all(
            isinstance(instrument, InstrumentId)
            and isinstance(quantity, Quantity)
            for instrument, quantity in copied.items()
        ):
            raise ConfigurationError(
                "portfolio positions must map InstrumentId to Quantity."
            )
        object.__setattr__(
            self,
            "positions",
            MappingProxyType(copied),
        )


@dataclass(frozen=True, slots=True, init=False)
class StrategyContext:
    """Immutable state supplied to one strategy decision."""

    timestamp: np.datetime64
    portfolio: PortfolioSnapshot
    active_orders: tuple[Order, ...]
    features: FeatureView
    run_id: RunId

    def __init__(
        self,
        timestamp: np.datetime64,
        portfolio: PortfolioSnapshot | _PortfolioSnapshotSource,
        active_orders: Sequence[Order],
        features: FeatureView,
        run_id: RunId,
    ) -> None:
        if not isinstance(timestamp, np.datetime64):
            raise ConfigurationError(
                "strategy timestamp must be a scalar np.datetime64."
            )
        try:
            normalized_timestamp = timestamp.astype("datetime64[ns]")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                "strategy timestamp must be valid."
            ) from exc
        if np.isnat(normalized_timestamp):
            raise ConfigurationError("strategy timestamp cannot be NaT.")
        try:
            positions = portfolio.positions
        except AttributeError as exc:
            raise ConfigurationError(
                "portfolio must expose positions."
            ) from exc
        normalized_portfolio = PortfolioSnapshot(positions=positions)
        if (
            isinstance(active_orders, (str, bytes, bytearray))
            or not isinstance(active_orders, Sequence)
        ):
            raise ConfigurationError(
                "active_orders must be a sequence of Order values."
            )
        normalized_orders = tuple(active_orders)
        if not all(isinstance(order, Order) for order in normalized_orders):
            raise ConfigurationError(
                "active_orders must contain only Order values."
            )
        if not isinstance(features, FeatureView):
            raise ConfigurationError("features must be a FeatureView.")
        if features.timestamp != normalized_timestamp:
            raise ConfigurationError(
                "feature view timestamp must match strategy timestamp."
            )
        if not isinstance(run_id, RunId):
            raise ConfigurationError("run_id must be a RunId.")
        object.__setattr__(self, "timestamp", normalized_timestamp)
        object.__setattr__(self, "portfolio", normalized_portfolio)
        object.__setattr__(self, "active_orders", normalized_orders)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "run_id", run_id)


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
                    "strategy output contains a duplicate cancel request.",
                    code="duplicate_cancel_order",
                )
            cancelled.add(intent.order_id)
        validated.append(intent)
    return tuple(validated)
