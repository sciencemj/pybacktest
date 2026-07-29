"""Structured domain events for order execution and decision tracing."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .errors import ConfigurationError
from .identifiers import FillId, OrderId
from .instruments import InstrumentId
from .money import Quantity
from .orders import DecisionReason, OrderSide


def _require_instance(value: object, expected_type: type[object], field: str) -> None:
    if not isinstance(value, expected_type):
        raise ConfigurationError(f"{field} must be a {expected_type.__name__}.")


def _require_code(code: object) -> None:
    if not isinstance(code, str) or not code.strip():
        raise ConfigurationError("code must be a non-empty string.")


def _require_timestamp(timestamp: object) -> None:
    if (
        not isinstance(timestamp, datetime)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise ConfigurationError("timestamp must be timezone-aware.")


def _require_message(message: object) -> None:
    if message is not None and (not isinstance(message, str) or not message.strip()):
        raise ConfigurationError("message must be a non-empty string when provided.")


def _require_positive_quantity(quantity: object, field: str) -> None:
    if not isinstance(quantity, Quantity) or quantity.value <= Decimal("0"):
        raise ConfigurationError(f"{field} must be a positive Quantity.")


@dataclass(frozen=True, slots=True)
class OrderAccepted:
    """An order was accepted for execution."""

    order_id: OrderId
    instrument: InstrumentId
    code: str
    timestamp: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.order_id, OrderId, "order_id")
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_code(self.code)
        _require_timestamp(self.timestamp)
        _require_message(self.message)


@dataclass(frozen=True, slots=True)
class OrderAdjusted:
    """An accepted order's requested quantity changed."""

    order_id: OrderId
    instrument: InstrumentId
    requested_quantity: Quantity
    adjusted_quantity: Quantity
    code: str
    timestamp: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.order_id, OrderId, "order_id")
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_positive_quantity(self.requested_quantity, "requested_quantity")
        _require_positive_quantity(self.adjusted_quantity, "adjusted_quantity")
        _require_code(self.code)
        _require_timestamp(self.timestamp)
        _require_message(self.message)


@dataclass(frozen=True, slots=True)
class OrderRejected:
    """An order was rejected before it could execute."""

    order_id: OrderId
    instrument: InstrumentId
    code: str
    timestamp: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.order_id, OrderId, "order_id")
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_code(self.code)
        _require_timestamp(self.timestamp)
        _require_message(self.message)


@dataclass(frozen=True, slots=True)
class OrderExpired:
    """An order expired according to its time-in-force instruction."""

    order_id: OrderId
    instrument: InstrumentId
    code: str
    timestamp: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.order_id, OrderId, "order_id")
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_code(self.code)
        _require_timestamp(self.timestamp)
        _require_message(self.message)


@dataclass(frozen=True, slots=True)
class PartialFill:
    """A fill that leaves part of an order open."""

    order_id: OrderId
    fill_id: FillId
    instrument: InstrumentId
    side: OrderSide
    quantity: Quantity
    code: str
    timestamp: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.order_id, OrderId, "order_id")
        _require_instance(self.fill_id, FillId, "fill_id")
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_instance(self.side, OrderSide, "side")
        _require_positive_quantity(self.quantity, "quantity")
        _require_code(self.code)
        _require_timestamp(self.timestamp)
        _require_message(self.message)


@dataclass(frozen=True, slots=True)
class DataUnavailable:
    """A decision could not use data required for the specified instrument."""

    instrument: InstrumentId
    code: str
    timestamp: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_code(self.code)
        _require_timestamp(self.timestamp)
        _require_message(self.message)


@dataclass(frozen=True, slots=True)
class DecisionTraceEntry:
    """A structured record of a decision and its immutable reason."""

    instrument: InstrumentId
    reason: DecisionReason
    code: str
    timestamp: datetime
    message: str | None = None

    def __post_init__(self) -> None:
        _require_instance(self.instrument, InstrumentId, "instrument")
        _require_instance(self.reason, DecisionReason, "reason")
        _require_code(self.code)
        _require_timestamp(self.timestamp)
        _require_message(self.message)
