"""Typed broker, execution-model, and run-context boundaries."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.market import BarView, MarketSlice
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import Fill, Order, OrderStatus


def _require_aware_timestamp(value: object) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ConfigurationError("timestamp must be timezone-aware.")


def _require_execution(
    fill: object,
    order: object,
    expected_status: OrderStatus,
) -> None:
    if not isinstance(fill, Fill):
        raise ConfigurationError("fill must be a Fill.")
    if not isinstance(order, Order):
        raise ConfigurationError("order must be an Order.")
    if fill.order_id != order.id or fill.instrument != order.instrument:
        raise ConfigurationError("fill and order identities must match.")
    if order.status is not expected_status:
        raise ConfigurationError(f"order must have status {expected_status.value}.")


@dataclass(frozen=True, slots=True)
class OrderPartiallyFilledEvent:
    """One complete fill and the immutable order that remains active."""

    fill: Fill
    order: Order

    def __post_init__(self) -> None:
        _require_execution(self.fill, self.order, OrderStatus.PARTIALLY_FILLED)


@dataclass(frozen=True, slots=True)
class OrderFilledEvent:
    """One complete fill and the immutable terminal order."""

    fill: Fill
    order: Order

    def __post_init__(self) -> None:
        _require_execution(self.fill, self.order, OrderStatus.FILLED)


@dataclass(frozen=True, slots=True)
class OrderCancelledEvent:
    """An explicit cancellation and the immutable terminal order."""

    order: Order
    timestamp: datetime
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.order, Order):
            raise ConfigurationError("order must be an Order.")
        if self.order.status is not OrderStatus.CANCELLED:
            raise ConfigurationError("cancelled event order must be CANCELLED.")
        _require_aware_timestamp(self.timestamp)
        if not isinstance(self.message, str) or not self.message.strip():
            raise ConfigurationError("message must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class OrderExpiredEvent:
    """A DAY expiry and the immutable cancelled order."""

    order: Order
    timestamp: datetime
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.order, Order):
            raise ConfigurationError("order must be an Order.")
        if self.order.status is not OrderStatus.CANCELLED:
            raise ConfigurationError("expired event order must be CANCELLED.")
        _require_aware_timestamp(self.timestamp)
        if not isinstance(self.message, str) or not self.message.strip():
            raise ConfigurationError("message must be a non-empty string.")


BrokerEvent: TypeAlias = (
    OrderPartiallyFilledEvent
    | OrderFilledEvent
    | OrderCancelledEvent
    | OrderExpiredEvent
)


@runtime_checkable
class FillModel(Protocol):
    """Choose an eligible, pre-slippage reference price for one bar."""

    def reference_price(self, order: Order, market: BarView) -> Money | None:
        """Return a reference price, or ``None`` when the order did not trade."""
        raise NotImplementedError


@runtime_checkable
class CommissionModel(Protocol):
    """Calculate a nonnegative fee in the execution quote currency."""

    def calculate(
        self,
        order: Order,
        quantity: Quantity,
        price: Money,
    ) -> Money:
        """Return the commission for the actual filled quantity."""
        raise NotImplementedError


@runtime_checkable
class SlippageModel(Protocol):
    """Apply price impact using only the explicitly supplied engine RNG."""

    def apply(
        self,
        order: Order,
        quantity: Quantity,
        reference_price: Money,
        market: BarView,
        rng: np.random.Generator,
    ) -> Money:
        """Return the post-impact execution price."""
        raise NotImplementedError


@runtime_checkable
class LiquidityModel(Protocol):
    """Supply shared per-instrument capacity for the current bar."""

    def available_quantity(
        self,
        order: Order,
        market: BarView,
    ) -> Quantity | None:
        """Return bar capacity, or ``None`` for unlimited positive-volume capacity."""
        raise NotImplementedError


@runtime_checkable
class BorrowCostModel(Protocol):
    """Calculate an explicit borrow charge in the execution quote currency."""

    def calculate(
        self,
        order: Order,
        quantity: Quantity,
        price: Money,
    ) -> Money:
        """Return a nonnegative borrow cost for the supplied position value."""
        raise NotImplementedError


@runtime_checkable
class SessionBoundaryProvider(Protocol):
    """Decide DAY expiry from an injected venue/session calendar."""

    def day_order_expired(self, order: Order, timestamp: datetime) -> bool:
        """Return whether the order's explicit trading session has ended."""
        raise NotImplementedError


@runtime_checkable
class FillIdSource(Protocol):
    """Issue deterministic fill identities owned by the current run."""

    def next_fill_id(self) -> FillId:
        """Return the next identity in the run-scoped fill sequence."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BrokerRunContext:
    """Immutable metadata and optional run-owned broker services.

    A context with no ``session_boundary`` deliberately disables automatic
    DAY expiry. No UTC or instrument-local date is inferred.
    """

    instruments: Mapping[InstrumentId, Instrument]
    session_boundary: SessionBoundaryProvider | None = None
    fill_ids: FillIdSource | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instruments, Mapping):
            raise ConfigurationError("instruments must be a mapping.")
        instruments = dict(self.instruments)
        if not all(
            isinstance(instrument_id, InstrumentId)
            and isinstance(instrument, Instrument)
            and instrument.id == instrument_id
            for instrument_id, instrument in instruments.items()
        ):
            raise ConfigurationError(
                "instruments must map matching InstrumentId to Instrument."
            )
        if self.session_boundary is not None and not isinstance(
            self.session_boundary, SessionBoundaryProvider
        ):
            raise ConfigurationError(
                "session_boundary must implement SessionBoundaryProvider."
            )
        if self.fill_ids is not None and not isinstance(
            self.fill_ids,
            FillIdSource,
        ):
            raise ConfigurationError("fill_ids must implement FillIdSource.")
        object.__setattr__(
            self,
            "instruments",
            MappingProxyType(instruments),
        )


@runtime_checkable
class Broker(Protocol):
    """Consume active immutable orders and emit typed immutable events."""

    @property
    def active_orders(self) -> Mapping[OrderId, Order]:
        """Return an immutable point-in-time snapshot of active orders."""
        raise NotImplementedError

    def submit(self, order: Order) -> Sequence[BrokerEvent]:
        """Submit an already accepted or partially filled order."""
        raise NotImplementedError

    def cancel(
        self,
        order_id: OrderId,
        timestamp: datetime,
    ) -> Sequence[BrokerEvent]:
        """Cancel one known active order."""
        raise NotImplementedError

    def process(
        self,
        market: MarketSlice,
        rng: np.random.Generator,
    ) -> Sequence[BrokerEvent]:
        """Process one strictly increasing market timestamp atomically."""
        raise NotImplementedError


__all__ = [
    "BorrowCostModel",
    "Broker",
    "BrokerEvent",
    "BrokerRunContext",
    "CommissionModel",
    "FillIdSource",
    "FillModel",
    "LiquidityModel",
    "OrderCancelledEvent",
    "OrderExpiredEvent",
    "OrderFilledEvent",
    "OrderPartiallyFilledEvent",
    "SessionBoundaryProvider",
    "SlippageModel",
]
