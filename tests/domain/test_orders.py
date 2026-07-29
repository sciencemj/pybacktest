from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.events import (
    DataUnavailable,
    DecisionTraceEntry,
    OrderAccepted,
    OrderAdjusted,
    OrderExpired,
    OrderRejected,
    PartialFill,
)
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    CancelOrderIntent,
    DecisionReason,
    Fill,
    LimitOrderIntent,
    MarketOrderIntent,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TargetQuantity,
    TargetWeight,
    TimeInForce,
)

NOW = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
AAPL = InstrumentId.parse("XNAS:AAPL")


def make_order() -> Order:
    return Order.pending(
        id=OrderId.new(),
        instrument=AAPL,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Quantity.of("10"),
        limit_price=Money.usd("100"),
        time_in_force=TimeInForce.DAY,
        submitted_at=NOW,
        active_from=NOW,
        reason=DecisionReason.of(
            "ma_cross",
            signal="fast_crossed_above_slow",
        ),
    )


def test_order_state_machine_accepts_partial_then_full_fill():
    accepted = make_order().accept()
    first = Fill(
        id=FillId.new(),
        order_id=accepted.id,
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("4"),
        price=Money.usd("99"),
        fee=Money.usd("1"),
        timestamp=NOW,
    )
    partial = accepted.apply_fill(first)
    assert partial.status is OrderStatus.PARTIALLY_FILLED
    assert partial.remaining_quantity.value == Decimal("6")

    second = Fill(
        id=FillId.new(),
        order_id=accepted.id,
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("6"),
        price=Money.usd("100"),
        fee=Money.usd("1"),
        timestamp=NOW,
    )
    filled = partial.apply_fill(second)
    assert filled.status is OrderStatus.FILLED
    assert filled.remaining_quantity.value == Decimal("0")


def test_filled_order_cannot_be_cancelled():
    order = make_order().accept()
    fill = Fill(
        id=FillId.new(),
        order_id=order.id,
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("10"),
        price=Money.usd("100"),
        fee=Money.usd("0"),
        timestamp=NOW,
    )
    with pytest.raises(ConfigurationError, match="terminal"):
        order.apply_fill(fill).cancel("too late")


def test_order_transitions_return_replacements_without_mutating_prior_orders():
    pending = make_order()
    accepted = pending.accept()
    cancelled = accepted.cancel("strategy invalidated")

    assert pending.status is OrderStatus.PENDING
    assert accepted.status is OrderStatus.ACCEPTED
    assert cancelled.status is OrderStatus.CANCELLED
    assert pending is not accepted
    assert accepted is not cancelled


def test_partially_filled_order_cannot_be_rejected():
    accepted = make_order().accept()
    partial = accepted.apply_fill(
        Fill(
            id=FillId.new(),
            order_id=accepted.id,
            instrument=AAPL,
            side=OrderSide.BUY,
            quantity=Quantity.of("1"),
            price=Money.usd("100"),
            fee=Money.usd("0"),
            timestamp=NOW,
        )
    )

    with pytest.raises(ConfigurationError, match="outside an active state"):
        partial.reject()


def test_rejected_order_cannot_carry_a_partial_fill():
    with pytest.raises(ConfigurationError, match="Rejected orders cannot have filled"):
        replace(
            make_order(),
            status=OrderStatus.REJECTED,
            filled_quantity=Quantity.of("1"),
        )


@pytest.mark.parametrize(
    ("order_id", "instrument", "side", "price", "fee", "expected_message"),
    [
        (
            OrderId.new(),
            AAPL,
            OrderSide.BUY,
            Money.usd("100"),
            Money.usd("0"),
            "identity",
        ),
        (
            None,
            InstrumentId.parse("XNYS:MSFT"),
            OrderSide.BUY,
            Money.usd("100"),
            Money.usd("0"),
            "identity",
        ),
        (None, AAPL, OrderSide.SELL, Money.usd("100"), Money.usd("0"), "side"),
        (
            None,
            AAPL,
            OrderSide.BUY,
            Money.of("100", "KRW"),
            Money.of("0", "KRW"),
            "currency",
        ),
    ],
)
def test_order_rejects_mismatched_fill_fields(
    order_id: OrderId | None,
    instrument: InstrumentId,
    side: OrderSide,
    price: Money,
    fee: Money,
    expected_message: str,
):
    order = make_order().accept()
    fill = Fill(
        id=FillId.new(),
        order_id=order.id if order_id is None else order_id,
        instrument=instrument,
        side=side,
        quantity=Quantity.of("1"),
        price=price,
        fee=fee,
        timestamp=NOW,
    )

    with pytest.raises(ConfigurationError, match=expected_message):
        order.apply_fill(fill)


def test_order_rejects_fill_larger_than_remaining_quantity():
    order = make_order().accept()
    too_large = Fill(
        id=FillId.new(),
        order_id=order.id,
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("11"),
        price=Money.usd("100"),
        fee=Money.usd("0"),
        timestamp=NOW,
    )

    with pytest.raises(ConfigurationError, match="exceeds remaining"):
        order.apply_fill(too_large)


@pytest.mark.parametrize("quantity", [Quantity.of("0"), Quantity.of("-1")])
def test_order_boundary_models_reject_nonpositive_requested_quantities(
    quantity: Quantity,
):
    reason = DecisionReason.of("risk_check")
    constructors = [
        lambda: TargetQuantity(AAPL, quantity, reason),
        lambda: MarketOrderIntent(
            AAPL, OrderSide.BUY, quantity, TimeInForce.DAY, reason
        ),
        lambda: LimitOrderIntent(
            AAPL, OrderSide.BUY, quantity, Money.usd("100"), TimeInForce.DAY, reason
        ),
        lambda: Order.pending(
            id=OrderId.new(),
            instrument=AAPL,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=Money.usd("100"),
            time_in_force=TimeInForce.DAY,
            submitted_at=NOW,
            active_from=NOW,
            reason=reason,
        ),
        lambda: Fill(
            id=FillId.new(),
            order_id=OrderId.new(),
            instrument=AAPL,
            side=OrderSide.BUY,
            quantity=quantity,
            price=Money.usd("100"),
            fee=Money.usd("0"),
            timestamp=NOW,
        ),
    ]

    for construct in constructors:
        with pytest.raises(ConfigurationError, match="positive Quantity"):
            construct()


def test_decision_reason_copies_immutable_scalar_details():
    details = {"signal": "cross", "count": 2, "ratio": 0.5, "confirmed": True}
    reason = DecisionReason(code="ma_cross", details=details)
    details["signal"] = "changed"

    assert reason.details["signal"] == "cross"
    with pytest.raises(TypeError):
        reason.details["signal"] = "mutate"  # type: ignore[index]


@pytest.mark.parametrize(
    "details",
    [
        {"nested": {"value": "no"}},
        {"list": ["no"]},
        {"nan": float("nan")},
        {"infinity": float("inf")},
    ],
)
def test_decision_reason_rejects_non_json_scalar_details(details: dict[str, object]):
    with pytest.raises(ConfigurationError):
        DecisionReason(code="invalid", details=details)


def test_intents_require_a_structured_decision_reason():
    with pytest.raises(ConfigurationError, match="DecisionReason"):
        TargetWeight(AAPL, Decimal("0.5"), "buy signal")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="DecisionReason"):
        CancelOrderIntent(OrderId.new(), {"code": "cancel"})  # type: ignore[arg-type]


def test_structured_events_have_codes_timestamps_and_typed_identities():
    order_id = OrderId.new()
    fill_id = FillId.new()
    reason = DecisionReason.of("ma_cross", signal="cross")
    events = [
        OrderAccepted(order_id, AAPL, "order.accepted", NOW),
        OrderAdjusted(
            order_id,
            AAPL,
            Quantity.of("10"),
            Quantity.of("8"),
            "order.adjusted",
            NOW,
        ),
        OrderRejected(order_id, AAPL, "order.rejected", NOW, "broker denied"),
        OrderExpired(order_id, AAPL, "order.expired", NOW),
        PartialFill(
            order_id,
            fill_id,
            AAPL,
            OrderSide.BUY,
            Quantity.of("3"),
            "order.partial_fill",
            NOW,
        ),
        DataUnavailable(AAPL, "data.unavailable", NOW, "missing bar"),
        DecisionTraceEntry(AAPL, reason, "decision.trace", NOW),
    ]

    assert [event.code for event in events] == [
        "order.accepted",
        "order.adjusted",
        "order.rejected",
        "order.expired",
        "order.partial_fill",
        "data.unavailable",
        "decision.trace",
    ]
    assert all(event.timestamp == NOW for event in events)
    with pytest.raises(FrozenInstanceError):
        events[0].code = "order.changed"  # type: ignore[misc]


def test_order_accepted_event_rejects_untyped_order_identity():
    with pytest.raises(ConfigurationError, match="order_id"):
        OrderAccepted("order_" + "a" * 32, AAPL, "order.accepted", NOW)  # type: ignore[arg-type]
