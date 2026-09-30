"""Immutable risk context values and runtime-checkable risk ports."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, TypeAlias, overload, runtime_checkable

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.events import OrderRejected
from pybacktest.domain.identifiers import OrderId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    CancelOrderIntent,
    LimitOrderIntent,
    MarketOrderIntent,
    Order,
    OrderStatus,
    TargetQuantity,
    TargetWeight,
)
from pybacktest.domain.portfolio import PortfolioSnapshot

SizedOrderIntent: TypeAlias = (
    TargetWeight | TargetQuantity | MarketOrderIntent | LimitOrderIntent
)


def _aware_datetime(value: object, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ConfigurationError(f"{field_name} must be timezone-aware.")
    return value


@dataclass(frozen=True, slots=True)
class RiskContext:
    """A deterministic immutable view used by sizing and risk evaluation."""

    snapshot: PortfolioSnapshot
    prices: Mapping[InstrumentId, Money]
    instruments: Mapping[InstrumentId, Instrument]
    tradable: frozenset[InstrumentId]
    order_id: OrderId
    submitted_at: datetime
    active_from: datetime
    active_orders: tuple[Order, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, PortfolioSnapshot):
            raise ConfigurationError("snapshot must be a domain PortfolioSnapshot.")
        if not isinstance(self.prices, Mapping):
            raise ConfigurationError("prices must be a mapping.")
        copied_prices = dict(self.prices)
        if not all(
            isinstance(instrument_id, InstrumentId) and isinstance(price, Money)
            for instrument_id, price in copied_prices.items()
        ):
            raise ConfigurationError("prices must map InstrumentId to Money.")
        if not isinstance(self.instruments, Mapping):
            raise ConfigurationError("instruments must be a mapping.")
        copied_instruments = dict(self.instruments)
        if not all(
            isinstance(instrument_id, InstrumentId)
            and isinstance(instrument, Instrument)
            and instrument.id == instrument_id
            for instrument_id, instrument in copied_instruments.items()
        ):
            raise ConfigurationError(
                "instruments must map matching InstrumentId to Instrument."
            )
        for instrument_id, price in copied_prices.items():
            instrument = copied_instruments.get(instrument_id)
            if instrument is None:
                raise ConfigurationError(
                    "price instrument must exist in the instrument catalog."
                )
            if price.amount <= Decimal("0"):
                raise ConfigurationError("prices must contain positive Money values.")
            if price.currency != instrument.quote_currency:
                raise ConfigurationError(
                    "price currency must match the instrument quote currency."
                )
        if isinstance(self.tradable, (str, bytes, bytearray)) or not isinstance(
            self.tradable, Collection
        ):
            raise ConfigurationError(
                "tradable must be a collection of InstrumentId values."
            )
        copied_tradable = frozenset(self.tradable)
        if not all(
            isinstance(instrument_id, InstrumentId) for instrument_id in copied_tradable
        ):
            raise ConfigurationError("tradable must contain InstrumentId values.")
        if not copied_tradable.issubset(copied_instruments):
            raise ConfigurationError(
                "tradable instruments must exist in the instrument catalog."
            )
        if not isinstance(self.order_id, OrderId):
            raise ConfigurationError("order_id must be an OrderId.")
        submitted_at = _aware_datetime(self.submitted_at, "submitted_at")
        active_from = _aware_datetime(self.active_from, "active_from")
        if active_from < submitted_at:
            raise ConfigurationError("active_from cannot precede submitted_at.")
        if (
            self.snapshot.timestamp is not None
            and self.snapshot.timestamp > submitted_at
        ):
            raise ConfigurationError("snapshot timestamp cannot follow submitted_at.")
        if isinstance(self.active_orders, (str, bytes, bytearray)) or not isinstance(
            self.active_orders, Sequence
        ):
            raise ConfigurationError(
                "active_orders must be a sequence of active Order values."
            )
        active_orders = tuple(self.active_orders)
        if not all(
            isinstance(order, Order)
            and order.status
            in {
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
            }
            and order.instrument in copied_instruments
            and order.quote_currency
            == copied_instruments[order.instrument].quote_currency
            for order in active_orders
        ):
            raise ConfigurationError(
                "active_orders must contain catalog-backed active orders."
            )
        if len({order.id for order in active_orders}) != len(active_orders):
            raise ConfigurationError("active_orders must contain unique order IDs.")
        object.__setattr__(
            self,
            "prices",
            MappingProxyType(copied_prices),
        )
        object.__setattr__(
            self,
            "instruments",
            MappingProxyType(copied_instruments),
        )
        object.__setattr__(self, "tradable", copied_tradable)
        object.__setattr__(self, "active_orders", active_orders)


class RiskStatus(StrEnum):
    """The observable outcome of evaluating one proposed order."""

    PASSED = "passed"
    ADJUSTED = "adjusted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """An immutable, explicit quantity decision from a risk policy."""

    status: RiskStatus
    original_quantity: Quantity
    final_quantity: Quantity
    codes: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, RiskStatus):
            raise ConfigurationError("status must be a RiskStatus.")
        if not isinstance(
            self.original_quantity, Quantity
        ) or self.original_quantity.value <= Decimal("0"):
            raise ConfigurationError("original_quantity must be a positive Quantity.")
        if not isinstance(
            self.final_quantity, Quantity
        ) or self.final_quantity.value < Decimal("0"):
            raise ConfigurationError("final_quantity must be a non-negative Quantity.")
        if isinstance(self.codes, (str, bytes, bytearray)) or not isinstance(
            self.codes, Sequence
        ):
            raise ConfigurationError("codes must be a sequence of strings.")
        copied_codes = tuple(self.codes)
        if not all(
            isinstance(code, str) and code.strip() for code in copied_codes
        ) or len(set(copied_codes)) != len(copied_codes):
            raise ConfigurationError("codes must contain unique non-empty strings.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ConfigurationError("message must be a non-empty string.")

        original = self.original_quantity.value
        final = self.final_quantity.value
        if self.status is RiskStatus.PASSED and (final != original or copied_codes):
            raise ConfigurationError(
                "PASSED decisions must preserve quantity without codes."
            )
        if self.status is RiskStatus.ADJUSTED and (
            not Decimal("0") < final < original or not copied_codes
        ):
            raise ConfigurationError(
                "ADJUSTED decisions require a smaller positive quantity and codes."
            )
        if self.status is RiskStatus.REJECTED and (
            final != Decimal("0") or not copied_codes
        ):
            raise ConfigurationError(
                "REJECTED decisions require zero final quantity and codes."
            )
        object.__setattr__(self, "codes", copied_codes)


@runtime_checkable
class OrderSizer(Protocol):
    """Convert strategy intents into deterministic proposed orders."""

    @overload
    def size(
        self,
        intent: CancelOrderIntent,
        context: RiskContext,
    ) -> CancelOrderIntent: ...

    @overload
    def size(
        self,
        intent: SizedOrderIntent,
        context: RiskContext,
    ) -> Order | OrderRejected: ...

    def size(
        self,
        intent: SizedOrderIntent | CancelOrderIntent,
        context: RiskContext,
    ) -> Order | OrderRejected | CancelOrderIntent:
        """Size one intent or pass a cancellation through unchanged."""
        raise NotImplementedError


@runtime_checkable
class RiskPolicy(Protocol):
    """Approve, reduce, or reject one proposed immutable order."""

    def evaluate(
        self,
        order: Order,
        context: RiskContext,
    ) -> RiskDecision:
        """Return an explicit risk decision without mutating the order."""
        raise NotImplementedError


__all__ = [
    "OrderSizer",
    "RiskContext",
    "RiskDecision",
    "RiskPolicy",
    "RiskStatus",
    "SizedOrderIntent",
]
