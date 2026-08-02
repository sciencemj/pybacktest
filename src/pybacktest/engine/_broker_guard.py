"""Validate broker ownership, call origin, and immutable order identity."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import cast
from weakref import ref

from pybacktest.domain.errors import AdapterContractError, ConfigurationError
from pybacktest.domain.identifiers import CashEventId, FillId, OrderId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import Order, OrderSide, OrderStatus, TimeInForce
from pybacktest.domain.portfolio import CashEvent, CashEventCode
from pybacktest.ports.broker import (
    BorrowCostSource,
    Broker,
    BrokerEvent,
    FillIdSource,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    SessionBoundaryProvider,
)
from pybacktest.results._decimal import ExactDecimalError, exact_add

_BROKER_REGISTRY: dict[int, ref[Broker]] = {}
_BROKER_REGISTRY_LOCK = threading.Lock()


class _BrokerCallOrigin(StrEnum):
    """Identify the engine call that produced one broker event batch."""

    SUBMIT = "submit"
    CANCEL = "cancel"
    PROCESS = "process"


_STABLE_ORDER_FIELDS = (
    "instrument",
    "side",
    "type",
    "quantity",
    "quote_currency",
    "limit_price",
    "time_in_force",
    "submitted_at",
    "active_from",
    "reason",
)

_ALLOWED_BROKER_TRANSITIONS = {
    OrderStatus.ACCEPTED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
}


def _claim_fresh_broker(broker: Broker) -> None:
    """Require every session to own a broker no prior session has used."""
    key = id(broker)
    with _BROKER_REGISTRY_LOCK:
        existing = _BROKER_REGISTRY.get(key)
        if existing is not None and existing() is broker:
            raise AdapterContractError(
                "broker factory returned an instance already used by another session.",
                code="reused_broker_instance",
            )

        def _discard(discarded: ref[Broker], key: int = key) -> None:
            with _BROKER_REGISTRY_LOCK:
                if _BROKER_REGISTRY.get(key) is discarded:
                    del _BROKER_REGISTRY[key]

        try:
            _BROKER_REGISTRY[key] = ref(broker, _discard)
        except TypeError as exc:
            raise AdapterContractError(
                "broker instance must support weak references so the "
                "engine can prove it is fresh.",
                code="unsupported_broker_instance",
            ) from exc


def _require_broker_call_origin(
    broker_events: Sequence[BrokerEvent],
    *,
    origin: _BrokerCallOrigin,
    cancelled_order_id: OrderId | None,
    at: datetime,
    session_boundary: SessionBoundaryProvider | None,
) -> None:
    """Reject events the engine's specific broker call cannot produce."""
    if origin is _BrokerCallOrigin.SUBMIT:
        if broker_events:
            raise AdapterContractError(
                "broker submit() reported an event for an order that is "
                "not active yet.",
                code="invalid_broker_event_origin",
            )
        return

    if origin is _BrokerCallOrigin.CANCEL:
        if len(broker_events) != 1:
            raise AdapterContractError(
                "broker cancel() must report exactly one cancellation.",
                code="invalid_broker_event_origin",
            )
        event = broker_events[0]
        if (
            not isinstance(event, OrderCancelledEvent)
            or cancelled_order_id is None
            or event.order.id != cancelled_order_id
        ):
            raise AdapterContractError(
                "broker cancel() must report a cancellation for exactly "
                "the requested order.",
                code="invalid_broker_event_origin",
            )
        return

    for event in broker_events:
        if isinstance(event, OrderCancelledEvent):
            raise AdapterContractError(
                "broker process() cannot cancel an order; only the engine "
                "requests cancellation.",
                code="invalid_broker_event_origin",
            )
        if not isinstance(event, OrderExpiredEvent):
            continue
        if event.order.time_in_force is not TimeInForce.DAY:
            raise AdapterContractError(
                "broker process() expired an order that carries no DAY "
                "time-in-force instruction.",
                code="invalid_broker_event_origin",
            )
        if session_boundary is None or not session_boundary.day_order_expired(
            event.order,
            at,
        ):
            raise AdapterContractError(
                "broker process() expired a DAY order whose session has not "
                "ended on the engine-owned calendar.",
                code="invalid_broker_event_origin",
            )


def _require_stable_order_identity(previous: Order, current: Order) -> None:
    """Reject a broker order whose immutable identity fields changed."""
    if any(
        getattr(previous, field_name) != getattr(current, field_name)
        for field_name in _STABLE_ORDER_FIELDS
    ):
        raise AdapterContractError(
            "broker event changed an immutable order identity field.",
            code="invalid_broker_order_state",
        )


def _snapshot_broker_events(broker_events: object) -> tuple[BrokerEvent, ...]:
    """Materialize one untrusted adapter batch exactly once."""
    if isinstance(broker_events, (str, bytes, bytearray)) or not isinstance(
        broker_events, Sequence
    ):
        raise AdapterContractError(
            "broker events must be a sequence.",
            code="invalid_broker_events",
        )
    try:
        return cast("tuple[BrokerEvent, ...]", tuple(broker_events))
    except Exception as exc:
        raise AdapterContractError(
            "broker events could not be read as a fixed sequence.",
            code="invalid_broker_events",
        ) from exc


def _plan_borrow_cash_events(
    broker_events: Sequence[BrokerEvent],
    *,
    broker: Broker,
    position_quantities: Mapping[InstrumentId, Decimal],
    submitted_orders: Mapping[OrderId, Order],
    cash_event_ordinal: int,
    cash_event_id: Callable[[int], CashEventId],
) -> dict[FillId, CashEvent]:
    """Project validated fills into incremental-short cash events."""
    if not isinstance(broker, BorrowCostSource):
        return {}

    staged_positions = dict(position_quantities)
    staged_orders = dict(submitted_orders)
    planned: dict[FillId, CashEvent] = {}
    for broker_event in broker_events:
        previous_order = staged_orders[broker_event.order.id]
        staged_orders[broker_event.order.id] = broker_event.order
        if not isinstance(
            broker_event,
            (OrderFilledEvent, OrderPartiallyFilledEvent),
        ):
            continue

        fill = broker_event.fill
        before = staged_positions.get(fill.instrument, Decimal("0"))
        signed_fill = (
            fill.quantity.value
            if fill.side is OrderSide.BUY
            else fill.quantity.value.copy_negate()
        )
        try:
            after = exact_add(before, signed_fill)
            short_before = (
                before.copy_negate() if before < Decimal("0") else Decimal("0")
            )
            short_after = after.copy_negate() if after < Decimal("0") else Decimal("0")
            short_increase = exact_add(
                short_after,
                short_before.copy_negate(),
            )
        except ExactDecimalError as error:
            raise ConfigurationError(
                "borrow cost quantity exceeds the exact numeric range."
            ) from error
        staged_positions[fill.instrument] = after
        if short_increase <= Decimal("0"):
            continue

        cost = broker.calculate_borrow_cost(
            previous_order,
            Quantity.of(short_increase),
            fill.price,
        )
        _validate_borrow_cost(previous_order, cost)
        if cost.amount == Decimal("0"):
            continue
        event = CashEvent(
            id=cash_event_id(cash_event_ordinal + len(planned)),
            timestamp=fill.timestamp,
            amount=Money.of(cost.amount.copy_negate(), cost.currency),
            code=CashEventCode.BORROW_FEE,
        )
        planned[fill.id] = event
    return planned


def _validate_borrow_cost(order: Order, cost: object) -> None:
    if (
        not isinstance(cost, Money)
        or cost.currency != order.quote_currency
        or not cost.amount.is_finite()
        or cost.amount < Decimal("0")
    ):
        raise ConfigurationError(
            "borrow cost must be nonnegative finite Money in the order quote currency."
        )


def _validate_broker_events(
    broker_events: Sequence[BrokerEvent],
    *,
    submitted_orders: Mapping[OrderId, Order],
    fill_ordinal: int,
    fill_ids: FillIdSource,
    origin: _BrokerCallOrigin,
    cancelled_order_id: OrderId | None,
    at: datetime,
    session_boundary: SessionBoundaryProvider | None,
) -> dict[OrderId, Order]:
    """Project a complete batch while rejecting before engine mutation."""
    _require_broker_call_origin(
        broker_events,
        origin=origin,
        cancelled_order_id=cancelled_order_id,
        at=at,
        session_boundary=session_boundary,
    )
    staged = dict(submitted_orders)
    staged_ordinal = fill_ordinal
    for broker_event in broker_events:
        if not isinstance(
            broker_event,
            (
                OrderFilledEvent,
                OrderPartiallyFilledEvent,
                OrderCancelledEvent,
                OrderExpiredEvent,
            ),
        ):
            raise AdapterContractError(
                "broker returned an unsupported event value.",
                code="invalid_broker_event",
            )
        order = broker_event.order
        previous = staged.get(order.id)
        if previous is None:
            raise AdapterContractError(
                "broker event references an order the engine never submitted.",
                code="unknown_broker_order",
            )
        _require_stable_order_identity(previous, order)
        if order.status not in _ALLOWED_BROKER_TRANSITIONS.get(
            previous.status,
            frozenset(),
        ):
            raise AdapterContractError(
                "broker event moved an order through an invalid lifecycle transition.",
                code="invalid_broker_order_state",
            )
        if isinstance(
            broker_event,
            (OrderFilledEvent, OrderPartiallyFilledEvent),
        ):
            fill = broker_event.fill
            if (
                fill.order_id != order.id
                or fill.instrument != order.instrument
                or fill.side is not order.side
                or fill.price.currency != order.quote_currency
                or fill.fee.currency != order.quote_currency
            ):
                raise AdapterContractError(
                    "broker fill disagrees with the engine-owned order "
                    "on identity, side, or quote currency.",
                    code="invalid_broker_order_state",
                )
            if fill.quantity.value > previous.remaining_quantity.value:
                raise AdapterContractError(
                    "broker fill quantity exceeds the order's remaining quantity.",
                    code="invalid_broker_order_state",
                )
            if fill.timestamp != at or fill.timestamp < order.active_from:
                raise AdapterContractError(
                    "broker fill timestamp does not match the timestamp "
                    "the broker was asked to process.",
                    code="invalid_broker_event_timestamp",
                )
            if fill.id != fill_ids.fill_id(staged_ordinal):
                raise AdapterContractError(
                    "broker fill identity is not the next run-scoped "
                    "UUID5 fill ordinal.",
                    code="invalid_fill_identity",
                )
            try:
                expected_filled = exact_add(
                    previous.filled_quantity.value,
                    fill.quantity.value,
                )
            except ExactDecimalError as error:
                raise AdapterContractError(
                    "broker fill quantity exceeds the exact numeric range.",
                    code="invalid_broker_order_state",
                ) from error
            if order.filled_quantity.value != expected_filled:
                raise AdapterContractError(
                    "broker order filled quantity does not equal the prior "
                    "quantity plus this fill.",
                    code="invalid_broker_order_state",
                )
            staged_ordinal += 1
        elif broker_event.timestamp != at:
            raise AdapterContractError(
                "broker lifecycle timestamp does not match the timestamp "
                "the broker was asked to process.",
                code="invalid_broker_event_timestamp",
            )
        elif order.filled_quantity != previous.filled_quantity:
            raise AdapterContractError(
                "broker terminated an order while changing its filled quantity.",
                code="invalid_broker_order_state",
            )
        staged[order.id] = order
    return staged


def _validate_active_orders(
    active_orders: object,
    *,
    instruments: Mapping[InstrumentId, Instrument],
) -> dict[OrderId, Order]:
    """Copy and validate the adapter's active-order mapping exactly once."""
    if not isinstance(active_orders, Mapping):
        raise AdapterContractError(
            "broker active_orders must be a mapping.",
            code="invalid_active_orders",
        )
    copied = dict(cast("Mapping[object, object]", active_orders))
    if not all(
        isinstance(order_id, OrderId)
        and isinstance(order, Order)
        and order.id == order_id
        and order.status in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}
        for order_id, order in copied.items()
    ):
        raise AdapterContractError(
            "broker active_orders must map matching IDs to active orders.",
            code="invalid_active_orders",
        )
    validated = cast("dict[OrderId, Order]", copied)
    for order in validated.values():
        instrument = instruments.get(order.instrument)
        if instrument is None:
            raise AdapterContractError(
                "broker active_orders contain an instrument outside the run catalog.",
                code="invalid_active_orders",
            )
        if order.quote_currency != instrument.quote_currency:
            raise AdapterContractError(
                "broker active_orders contain an order whose quote currency "
                "disagrees with its instrument.",
                code="invalid_active_orders",
            )
    return validated


def _require_broker_state_agreement(
    projected_orders: Mapping[OrderId, Order],
    active_orders: Mapping[OrderId, Order],
) -> None:
    """Require adapter active state to equal the validated engine projection."""
    expected = {
        order_id: order
        for order_id, order in projected_orders.items()
        if order.status in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}
    }
    if active_orders != expected:
        raise AdapterContractError(
            "broker active_orders disagree with the engine-owned order "
            "lifecycle after the call.",
            code="broker_state_disagreement",
        )


def _engine_active_orders(
    submitted_orders: Mapping[OrderId, Order],
) -> tuple[Order, ...]:
    """Project active orders in engine insertion/submission order."""
    return tuple(
        order
        for order in submitted_orders.values()
        if order.status in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}
    )
