from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
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
DEFAULT_FILL_PRICE = Money.usd("100")
DEFAULT_FILL_FEE = Money.usd("0")


def make_order(
    *,
    side: OrderSide = OrderSide.BUY,
    active_from: datetime = NOW,
) -> Order:
    return Order.pending(
        id=OrderId.new(),
        instrument=AAPL,
        side=side,
        type=OrderType.LIMIT,
        quantity=Quantity.of("10"),
        quote_currency="USD",
        limit_price=Money.usd("100"),
        time_in_force=TimeInForce.DAY,
        submitted_at=NOW,
        active_from=active_from,
        reason=DecisionReason.of(
            "ma_cross",
            signal="fast_crossed_above_slow",
        ),
    )


def pending_order_arguments(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "id": OrderId.new(),
        "instrument": AAPL,
        "side": OrderSide.BUY,
        "type": OrderType.LIMIT,
        "quantity": Quantity.of("10"),
        "limit_price": Money.usd("100"),
        "time_in_force": TimeInForce.DAY,
        "submitted_at": NOW,
        "active_from": NOW,
        "reason": DecisionReason.of("ma_cross"),
    }
    arguments.update(overrides)
    return arguments


def matching_fill(
    order: Order,
    *,
    price: Money = DEFAULT_FILL_PRICE,
    fee: Money = DEFAULT_FILL_FEE,
    timestamp: datetime = NOW,
) -> Fill:
    return Fill(
        id=FillId.new(),
        order_id=order.id,
        instrument=order.instrument,
        side=order.side,
        quantity=Quantity.of("1"),
        price=price,
        fee=fee,
        timestamp=timestamp,
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
            quote_currency="USD",
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
        OrderAccepted(order_id, AAPL, NOW),
        OrderAdjusted(
            order_id,
            AAPL,
            Quantity.of("10"),
            Quantity.of("8"),
            NOW,
        ),
        OrderRejected(order_id, AAPL, NOW, "broker denied"),
        OrderExpired(order_id, AAPL, NOW),
        PartialFill(
            order_id,
            fill_id,
            AAPL,
            OrderSide.BUY,
            Quantity.of("3"),
            NOW,
        ),
        DataUnavailable(AAPL, NOW, "missing bar"),
        DecisionTraceEntry(AAPL, reason, NOW),
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


def test_order_rejected_carries_a_typed_machine_readable_reason():
    rejection_reason = DecisionReason.of(
        "missing_price",
        instrument="XNAS:AAPL",
    )

    event = OrderRejected(
        OrderId.new(),
        AAPL,
        NOW,
        "current mark is unavailable",
        reason=rejection_reason,
    )

    assert event.reason == rejection_reason
    assert event.reason.code == "missing_price"


def test_order_accepted_event_rejects_untyped_order_identity():
    with pytest.raises(ConfigurationError, match="order_id"):
        OrderAccepted("order_" + "a" * 32, AAPL, NOW)  # type: ignore[arg-type]


def test_order_pending_requires_and_normalizes_explicit_quote_currency():
    with pytest.raises(TypeError, match="quote_currency"):
        Order.pending(**pending_order_arguments())  # type: ignore[arg-type]

    order = Order.pending(
        **pending_order_arguments(quote_currency="usd")  # type: ignore[arg-type]
    )
    assert order.quote_currency == "USD"


def test_limit_order_requires_quote_currency_matching_its_limit_price():
    with pytest.raises(ConfigurationError, match="quote_currency"):
        Order.pending(
            **pending_order_arguments(quote_currency="KRW")  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("side", "execution_price"),
    [
        (OrderSide.BUY, Money.usd("100.01")),
        (OrderSide.SELL, Money.usd("99.99")),
    ],
)
def test_limit_order_rejects_worse_execution_prices(
    side: OrderSide,
    execution_price: Money,
):
    order = make_order(side=side).accept()

    with pytest.raises(ConfigurationError, match="limit"):
        order.apply_fill(matching_fill(order, price=execution_price))


def test_market_order_rejects_fill_currency_outside_explicit_quote_currency():
    order = Order.pending(
        **pending_order_arguments(
            type=OrderType.MARKET,
            limit_price=None,
            quote_currency="USD",
        )  # type: ignore[arg-type]
    ).accept()

    with pytest.raises(ConfigurationError, match="currency"):
        order.apply_fill(
            matching_fill(
                order,
                price=Money.of("100", "KRW"),
                fee=Money.of("0", "KRW"),
            )
        )


def test_fill_rejects_naive_timestamp():
    order = make_order()

    with pytest.raises(ConfigurationError, match="timezone-aware"):
        matching_fill(order, timestamp=datetime(2024, 1, 2, 14, 30))


def test_order_rejects_fill_before_active_from():
    order = make_order(active_from=NOW + timedelta(minutes=1)).accept()

    with pytest.raises(ConfigurationError, match="active_from"):
        order.apply_fill(matching_fill(order, timestamp=NOW))


def order_with_status(status: OrderStatus) -> Order:
    pending = make_order()
    if status is OrderStatus.PENDING:
        return pending
    accepted = pending.accept()
    if status is OrderStatus.ACCEPTED:
        return accepted
    partial = accepted.apply_fill(matching_fill(accepted))
    if status is OrderStatus.PARTIALLY_FILLED:
        return partial
    if status is OrderStatus.FILLED:
        return partial.apply_fill(
            Fill(
                id=FillId.new(),
                order_id=partial.id,
                instrument=partial.instrument,
                side=partial.side,
                quantity=Quantity.of("9"),
                price=Money.usd("100"),
                fee=Money.usd("0"),
                timestamp=NOW,
            )
        )
    if status is OrderStatus.CANCELLED:
        return partial.cancel("cancelled")
    if status is OrderStatus.REJECTED:
        return accepted.reject()
    raise AssertionError(f"Unhandled status: {status}")


def apply_matrix_action(order: Order, action: str) -> Order:
    if action == "accept":
        return order.accept()
    if action == "fill":
        return order.apply_fill(matching_fill(order))
    if action == "cancel":
        return order.cancel("matrix")
    if action == "reject":
        return order.reject()
    raise AssertionError(f"Unhandled action: {action}")


@pytest.mark.parametrize(
    ("source_status", "action", "expected_status"),
    [
        (OrderStatus.PENDING, "accept", OrderStatus.ACCEPTED),
        (OrderStatus.PENDING, "fill", None),
        (OrderStatus.PENDING, "cancel", OrderStatus.CANCELLED),
        (OrderStatus.PENDING, "reject", OrderStatus.REJECTED),
        (OrderStatus.ACCEPTED, "accept", None),
        (OrderStatus.ACCEPTED, "fill", OrderStatus.PARTIALLY_FILLED),
        (OrderStatus.ACCEPTED, "cancel", OrderStatus.CANCELLED),
        (OrderStatus.ACCEPTED, "reject", OrderStatus.REJECTED),
        (OrderStatus.PARTIALLY_FILLED, "accept", None),
        (OrderStatus.PARTIALLY_FILLED, "fill", OrderStatus.PARTIALLY_FILLED),
        (OrderStatus.PARTIALLY_FILLED, "cancel", OrderStatus.CANCELLED),
        (OrderStatus.PARTIALLY_FILLED, "reject", None),
        (OrderStatus.FILLED, "accept", None),
        (OrderStatus.FILLED, "fill", None),
        (OrderStatus.FILLED, "cancel", None),
        (OrderStatus.FILLED, "reject", None),
        (OrderStatus.CANCELLED, "accept", None),
        (OrderStatus.CANCELLED, "fill", None),
        (OrderStatus.CANCELLED, "cancel", None),
        (OrderStatus.CANCELLED, "reject", None),
        (OrderStatus.REJECTED, "accept", None),
        (OrderStatus.REJECTED, "fill", None),
        (OrderStatus.REJECTED, "cancel", None),
        (OrderStatus.REJECTED, "reject", None),
    ],
)
def test_order_transition_matrix(
    source_status: OrderStatus,
    action: str,
    expected_status: OrderStatus | None,
):
    order = order_with_status(source_status)

    if expected_status is None:
        with pytest.raises(ConfigurationError):
            apply_matrix_action(order, action)
    else:
        assert apply_matrix_action(order, action).status is expected_status


def test_events_fix_codes_and_disallow_mismatched_code_override():
    accepted = OrderAccepted(OrderId.new(), AAPL, NOW)

    assert accepted.code == "order.accepted"
    with pytest.raises(TypeError, match="code"):
        OrderAccepted(
            order_id=OrderId.new(),
            instrument=AAPL,
            timestamp=NOW,
            code="order.rejected",
        )


@pytest.mark.parametrize(
    "construct",
    [
        lambda: OrderAccepted(OrderId.new(), AAPL, datetime(2024, 1, 2, 14, 30)),
        lambda: OrderRejected(OrderId.new(), AAPL, NOW, ""),
        lambda: OrderAccepted("order_" + "a" * 32, AAPL, NOW),
        lambda: PartialFill(
            OrderId.new(),
            "fill_" + "a" * 32,
            AAPL,
            OrderSide.BUY,
            Quantity.of("1"),
            NOW,
        ),
        lambda: OrderAdjusted(
            OrderId.new(),
            AAPL,
            Quantity.of("0"),
            Quantity.of("1"),
            NOW,
        ),
        lambda: PartialFill(
            OrderId.new(),
            FillId.new(),
            AAPL,
            OrderSide.BUY,
            Quantity.of("-1"),
            NOW,
        ),
    ],
)
def test_event_constructors_enforce_shared_invariants(construct):
    with pytest.raises(ConfigurationError):
        construct()


@pytest.mark.parametrize(
    ("construct", "field"),
    [
        (lambda: DataUnavailable("XNAS:AAPL", NOW), "instrument"),
        (
            lambda: PartialFill(
                OrderId.new(),
                FillId.new(),
                AAPL,
                "buy",
                Quantity.of("1"),
                NOW,
            ),
            "side",
        ),
        (lambda: DecisionTraceEntry(AAPL, "ma_cross", NOW), "reason"),
        (
            lambda: OrderAdjusted(
                OrderId.new(),
                AAPL,
                Quantity.of("1"),
                Quantity.of("0"),
                NOW,
            ),
            "adjusted_quantity",
        ),
    ],
)
def test_event_constructors_reject_remaining_typed_field_mutations(
    construct,
    field: str,
):
    with pytest.raises(ConfigurationError, match=field):
        construct()
