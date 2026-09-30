"""Immutable order intents, orders, and fills."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import NoReturn, TypeAlias

from .errors import ConfigurationError
from .identifiers import FillId, OrderId
from .instruments import InstrumentId
from .money import Money, Quantity, decimal_from, normalize_currency

JSONScalar: TypeAlias = str | int | float | bool | None


class OrderSide(StrEnum):
    """The direction of an order or fill."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Supported execution instructions."""

    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    """How long an order may remain active."""

    DAY = "day"
    GOOD_TIL_CANCELLED = "good_til_cancelled"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"
    FILL_OR_KILL = "fill_or_kill"


class OrderStatus(StrEnum):
    """The lifecycle state of an order."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


def _require_instance(value: object, expected_type: type[object], field: str) -> None:
    if not isinstance(value, expected_type):
        raise ConfigurationError(f"{field} must be a {expected_type.__name__}.")


def _require_aware_datetime(value: object, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ConfigurationError(f"{field} must be timezone-aware.")
    return value


def _require_positive_quantity(value: object, field: str) -> Quantity:
    if not isinstance(value, Quantity) or value.value <= Decimal("0"):
        raise ConfigurationError(f"{field} must be a positive Quantity.")
    return value


def _require_reason(value: object) -> "DecisionReason":
    if not isinstance(value, DecisionReason):
        raise ConfigurationError("reason must be a DecisionReason.")
    return value


@dataclass(frozen=True, slots=True)
class DecisionReason:
    """A stable decision code with immutable JSON-scalar context."""

    code: str
    details: Mapping[str, JSONScalar]

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ConfigurationError("code must be a non-empty string.")
        if not isinstance(self.details, Mapping):
            raise ConfigurationError("details must be a mapping.")

        copied_details: dict[str, JSONScalar] = {}
        for key, value in self.details.items():
            if not isinstance(key, str):
                raise ConfigurationError("DecisionReason detail keys must be strings.")
            if type(value) is float and not isfinite(value):
                raise ConfigurationError(
                    "DecisionReason float detail values must be finite."
                )
            if type(value) not in {str, int, float, bool, type(None)}:
                raise ConfigurationError(
                    "DecisionReason detail values must be JSON scalars."
                )
            copied_details[key] = value
        object.__setattr__(self, "details", MappingProxyType(copied_details))

    @classmethod
    def of(cls, code: str, **details: JSONScalar) -> "DecisionReason":
        return cls(code=code, details=MappingProxyType(dict(details)))


@dataclass(frozen=True, slots=True)
class TargetWeight:
    """Intent to set an instrument to a signed portfolio weight."""

    instrument: InstrumentId
    weight: Decimal
    reason: DecisionReason

    def __post_init__(self) -> None:
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_reason(self.reason)
        object.__setattr__(self, "weight", decimal_from(self.weight, "weight"))


@dataclass(frozen=True, slots=True)
class TargetQuantity:
    """Intent to set an instrument to a signed absolute position."""

    instrument: InstrumentId
    quantity: Quantity
    reason: DecisionReason

    def __post_init__(self) -> None:
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_instance(self.quantity, Quantity, "quantity")
        _require_reason(self.reason)


@dataclass(frozen=True, slots=True)
class MarketOrderIntent:
    """Intent to submit a market order."""

    instrument: InstrumentId
    side: OrderSide
    quantity: Quantity
    time_in_force: TimeInForce
    reason: DecisionReason

    def __post_init__(self) -> None:
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_instance(self.side, OrderSide, "side")
        _require_positive_quantity(self.quantity, "quantity")
        _require_instance(self.time_in_force, TimeInForce, "time_in_force")
        _require_reason(self.reason)


@dataclass(frozen=True, slots=True)
class LimitOrderIntent:
    """Intent to submit a limit order."""

    instrument: InstrumentId
    side: OrderSide
    quantity: Quantity
    limit_price: Money
    time_in_force: TimeInForce
    reason: DecisionReason

    def __post_init__(self) -> None:
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_instance(self.side, OrderSide, "side")
        _require_positive_quantity(self.quantity, "quantity")
        _require_instance(self.limit_price, Money, "limit_price")
        if self.limit_price.amount <= Decimal("0"):
            raise ConfigurationError("limit_price must be positive.")
        _require_instance(self.time_in_force, TimeInForce, "time_in_force")
        _require_reason(self.reason)


@dataclass(frozen=True, slots=True)
class CancelOrderIntent:
    """Intent to cancel an identified order."""

    order_id: OrderId
    reason: DecisionReason

    def __post_init__(self) -> None:
        _require_instance(self.order_id, OrderId, "order_id")
        _require_reason(self.reason)


@dataclass(frozen=True, slots=True)
class Fill:
    """An immutable execution reported for an order."""

    id: FillId
    order_id: OrderId
    instrument: InstrumentId
    side: OrderSide
    quantity: Quantity
    price: Money
    fee: Money
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_instance(self.id, FillId, "id")
        _require_instance(self.order_id, OrderId, "order_id")
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_instance(self.side, OrderSide, "side")
        _require_positive_quantity(self.quantity, "quantity")
        _require_instance(self.price, Money, "price")
        _require_instance(self.fee, Money, "fee")
        if self.price.amount <= Decimal("0"):
            raise ConfigurationError("price must be positive.")
        if self.fee.currency != self.price.currency:
            raise ConfigurationError("Fill fee and price currencies must match.")
        _require_aware_datetime(self.timestamp, "timestamp")


@dataclass(frozen=True, slots=True)
class Order:
    """An immutable order whose lifecycle advances through replacements."""

    id: OrderId
    instrument: InstrumentId
    side: OrderSide
    type: OrderType
    quantity: Quantity
    quote_currency: str
    limit_price: Money | None
    time_in_force: TimeInForce
    submitted_at: datetime
    active_from: datetime
    reason: DecisionReason
    status: OrderStatus
    filled_quantity: Quantity

    def __post_init__(self) -> None:
        _require_instance(self.id, OrderId, "id")
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_instance(self.side, OrderSide, "side")
        _require_instance(self.type, OrderType, "type")
        _require_positive_quantity(self.quantity, "quantity")
        object.__setattr__(
            self, "quote_currency", normalize_currency(self.quote_currency)
        )
        _require_instance(self.time_in_force, TimeInForce, "time_in_force")
        submitted_at = _require_aware_datetime(self.submitted_at, "submitted_at")
        active_from = _require_aware_datetime(self.active_from, "active_from")
        if active_from < submitted_at:
            raise ConfigurationError("active_from cannot precede submitted_at.")
        _require_reason(self.reason)
        _require_instance(self.status, OrderStatus, "status")
        if not isinstance(self.filled_quantity, Quantity):
            raise ConfigurationError("filled_quantity must be a Quantity.")
        if self.filled_quantity.value < Decimal("0"):
            raise ConfigurationError("filled_quantity cannot be negative.")
        if self.filled_quantity.value > self.quantity.value:
            raise ConfigurationError("filled_quantity cannot exceed quantity.")
        if self.type is OrderType.LIMIT:
            if not isinstance(self.limit_price, Money):
                raise ConfigurationError("limit_price must be a Money.")
            if self.limit_price.amount <= Decimal("0"):
                raise ConfigurationError("limit_price must be positive.")
            if self.limit_price.currency != self.quote_currency:
                raise ConfigurationError(
                    "limit_price currency must match quote_currency."
                )
        elif self.limit_price is not None:
            raise ConfigurationError("Market orders cannot have a limit_price.")
        self._validate_status_quantity()

    @classmethod
    def pending(
        cls,
        *,
        id: OrderId,
        instrument: InstrumentId,
        side: OrderSide,
        type: OrderType,
        quantity: Quantity,
        quote_currency: str,
        limit_price: Money | None,
        time_in_force: TimeInForce,
        submitted_at: datetime,
        active_from: datetime,
        reason: DecisionReason,
    ) -> "Order":
        return cls(
            id=id,
            instrument=instrument,
            side=side,
            type=type,
            quantity=quantity,
            quote_currency=quote_currency,
            limit_price=limit_price,
            time_in_force=time_in_force,
            submitted_at=submitted_at,
            active_from=active_from,
            reason=reason,
            status=OrderStatus.PENDING,
            filled_quantity=Quantity.of("0"),
        )

    @property
    def remaining_quantity(self) -> Quantity:
        """The positive unfilled quantity, or zero once fully filled."""
        return Quantity.of(self.quantity.value - self.filled_quantity.value)

    def accept(self) -> "Order":
        """Accept a pending order."""
        if self.status is OrderStatus.PENDING:
            return replace(self, status=OrderStatus.ACCEPTED)
        self._raise_invalid_transition("accept")

    def apply_fill(self, fill: Fill) -> "Order":
        """Apply a matching fill to an active order."""
        if self.status not in {
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            self._raise_invalid_transition("fill")
        if not isinstance(fill, Fill):
            raise ConfigurationError("fill must be a Fill.")
        if fill.order_id != self.id or fill.instrument != self.instrument:
            raise ConfigurationError("Fill identity does not match the order.")
        if fill.side is not self.side:
            raise ConfigurationError("Fill side does not match the order.")
        if (
            fill.price.currency != self.quote_currency
            or fill.fee.currency != self.quote_currency
        ):
            raise ConfigurationError("Fill currency does not match the order.")
        if fill.timestamp < self.active_from:
            raise ConfigurationError("Fill timestamp cannot precede active_from.")
        if self.limit_price is not None:
            if (
                self.side is OrderSide.BUY
                and fill.price.amount > self.limit_price.amount
            ):
                raise ConfigurationError("Fill price exceeds the buy limit.")
            if (
                self.side is OrderSide.SELL
                and fill.price.amount < self.limit_price.amount
            ):
                raise ConfigurationError("Fill price falls below the sell limit.")

        next_filled = self.filled_quantity.value + fill.quantity.value
        if next_filled > self.quantity.value:
            raise ConfigurationError("Fill quantity exceeds remaining order quantity.")
        next_status = (
            OrderStatus.FILLED
            if next_filled == self.quantity.value
            else OrderStatus.PARTIALLY_FILLED
        )
        return replace(
            self,
            status=next_status,
            filled_quantity=Quantity.of(next_filled),
        )

    def cancel(self, message: str) -> "Order":
        """Cancel an order that is still eligible for cancellation."""
        if self.status in {
            OrderStatus.PENDING,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            if not isinstance(message, str) or not message.strip():
                raise ConfigurationError("message must be a non-empty string.")
            return replace(self, status=OrderStatus.CANCELLED)
        self._raise_invalid_transition("cancel")

    def reject(self) -> "Order":
        """Reject an order before it has received a fill."""
        if self.status in {OrderStatus.PENDING, OrderStatus.ACCEPTED}:
            return replace(self, status=OrderStatus.REJECTED)
        self._raise_invalid_transition("reject")

    def _validate_status_quantity(self) -> None:
        filled = self.filled_quantity.value
        quantity = self.quantity.value
        if self.status in {OrderStatus.PENDING, OrderStatus.ACCEPTED} and filled != 0:
            raise ConfigurationError("Unfilled orders cannot have filled_quantity.")
        if self.status is OrderStatus.PARTIALLY_FILLED and not 0 < filled < quantity:
            raise ConfigurationError(
                "Partially filled orders require a partial quantity."
            )
        if self.status is OrderStatus.FILLED and filled != quantity:
            raise ConfigurationError("Filled orders require the full quantity.")
        if self.status is OrderStatus.REJECTED and filled != 0:
            raise ConfigurationError("Rejected orders cannot have filled_quantity.")
        if self.status is OrderStatus.CANCELLED and filled == quantity:
            raise ConfigurationError("Cancelled orders cannot be fully filled.")

    def _raise_invalid_transition(self, action: str) -> NoReturn:
        if self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }:
            raise ConfigurationError(f"Cannot {action} a terminal order.")
        raise ConfigurationError(f"Cannot {action} an order outside an active state.")
