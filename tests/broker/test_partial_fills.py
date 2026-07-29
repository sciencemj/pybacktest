from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest

from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    PerShareCommission,
    SimulatedBroker,
    VolumeParticipationLimit,
    VolumeShareSlippage,
)
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import OrderId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import OrderSide, OrderStatus
from tests.factories import accepted_order, instrument, market_slice, rng


def test_volume_limit_creates_partial_fill_and_keeps_order_active():
    order = accepted_order(quantity="200")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    events = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="100",
            volume="1000",
        ),
        rng(),
    )

    assert events[0].fill.quantity.value == Decimal("100")
    assert sim.active_orders[order.id].status is OrderStatus.PARTIALLY_FILLED


def test_negative_volume_ratio_is_rejected():
    with pytest.raises(ConfigurationError, match="max_volume_ratio"):
        VolumeParticipationLimit(
            max_volume_ratio=Decimal("-0.01"),
        )


def test_partial_order_fills_across_multiple_bars_then_becomes_terminal():
    order = accepted_order(quantity="150")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    first = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="100",
            volume="1000",
        ),
        rng(),
    )[0]
    second = sim.process(
        market_slice(
            "2024-01-04T14:30:00Z",
            open="101",
            volume="1000",
        ),
        rng(),
    )[0]

    assert first.order.status is OrderStatus.PARTIALLY_FILLED
    assert second.fill.quantity == Quantity.of("50")
    assert second.order.status is OrderStatus.FILLED
    assert order.id not in sim.active_orders


def test_shared_bar_capacity_is_consumed_in_submission_order():
    first = accepted_order(
        order_id=OrderId.parse("order_" + "1" * 32),
        quantity="80",
    )
    second = accepted_order(
        order_id=OrderId.parse("order_" + "2" * 32),
        quantity="80",
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(first)
    sim.submit(second)

    events = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            volume="1000",
        ),
        rng(),
    )

    assert [event.order.id for event in events] == [first.id, second.id]
    assert [event.fill.quantity.value for event in events] == [
        Decimal("80"),
        Decimal("20"),
    ]
    assert sim.active_orders[second.id].remaining_quantity == Quantity.of("60")


def test_liquidity_is_floored_to_explicit_instrument_lot_size():
    item = instrument()
    item = item.__class__(
        id=item.id,
        quote_currency=item.quote_currency,
        tick_size=item.tick_size,
        lot_size=Decimal("25"),
        timezone=item.timezone,
    )
    order = accepted_order(item=item, quantity="100")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.33")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )
    sim.submit(order)

    event = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            item=item,
            volume="100",
        ),
        rng(),
    )[0]

    assert event.fill.quantity == Quantity.of("25")


def test_unit_lot_fallback_is_deterministic_under_tiny_decimal_context():
    order = accepted_order(quantity="10")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.333333333333333333")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    with localcontext() as context:
        context.prec = 2
        event = sim.process(
            market_slice(
                "2024-01-03T14:30:00Z",
                volume="10",
            ),
            rng(),
        )[0]

    assert event.fill.quantity == Quantity.of("3")


@pytest.mark.parametrize(
    ("side", "expected_price"),
    [
        (OrderSide.BUY, Decimal("100.1")),
        (OrderSide.SELL, Decimal("99.9")),
    ],
)
def test_volume_share_slippage_uses_actual_participation_and_side(
    side: OrderSide,
    expected_price: Decimal,
):
    order = accepted_order(side=side, quantity="100")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=VolumeShareSlippage(impact_bps=Decimal("100")),
        liquidity=VolumeParticipationLimit(Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    event = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="100",
            volume="1000",
        ),
        rng(),
    )[0]

    assert event.fill.price.amount == expected_price


def test_per_share_commission_uses_fill_quantity_and_minimum():
    order = accepted_order(quantity="100")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=PerShareCommission(
            rate_per_share=Decimal("0.005"),
            minimum_fee=Decimal("1"),
        ),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    event = sim.process(
        market_slice("2024-01-03T14:30:00Z"),
        rng(),
    )[0]

    assert event.fill.fee == Money.usd("1")


@pytest.mark.parametrize(
    "construct",
    [
        lambda: VolumeParticipationLimit(Decimal("1.01")),
        lambda: VolumeParticipationLimit(Decimal("NaN")),
        lambda: VolumeParticipationLimit(True),
        lambda: VolumeShareSlippage(Decimal("-1")),
        lambda: VolumeShareSlippage(Decimal("10000.01")),
        lambda: VolumeShareSlippage(True),
        lambda: PerShareCommission(Decimal("-0.01")),
        lambda: PerShareCommission(Decimal("0.01"), Decimal("-1")),
        lambda: PerShareCommission(True),
    ],
)
def test_cost_and_liquidity_models_reject_invalid_parameters(construct):
    with pytest.raises(ConfigurationError):
        construct()


def test_model_values_are_frozen():
    commission = PerShareCommission(Decimal("0.01"))

    with pytest.raises(FrozenInstanceError):
        commission.rate_per_share = Decimal("0.02")  # type: ignore[misc]


def test_no_borrow_cost_is_explicit_and_currency_safe():
    order = accepted_order()

    assert NoBorrowCost().calculate(
        order,
        Quantity.of("10"),
        Money.usd("100"),
    ) == Money.usd("0")


class SecondCalculationFailsOnce:
    def __init__(self) -> None:
        self.calls = 0
        self.failed = False

    def calculate(
        self,
        order,
        quantity,
        price,
    ) -> Money:
        self.calls += 1
        if self.calls == 2 and not self.failed:
            self.failed = True
            raise RuntimeError("commission failure")
        return Money.of("0", price.currency)


def test_model_failure_is_atomic_for_orders_capacity_and_fill_ids():
    first = accepted_order(
        order_id=OrderId.parse("order_" + "1" * 32),
        quantity="80",
    )
    second = accepted_order(
        order_id=OrderId.parse("order_" + "2" * 32),
        quantity="80",
    )
    commission = SecondCalculationFailsOnce()
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=commission,
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(first)
    sim.submit(second)
    current = market_slice(
        "2024-01-03T14:30:00Z",
        volume="1000",
    )

    with pytest.raises(RuntimeError, match="commission failure"):
        sim.process(current, rng())

    assert sim.active_orders[first.id] is first
    assert sim.active_orders[second.id] is second

    events = sim.process(current, rng())
    assert str(events[0].fill.id) == "fill_" + "0" * 32
    assert [event.fill.quantity.value for event in events] == [
        Decimal("80"),
        Decimal("20"),
    ]


class NegativeCommission:
    def calculate(self, order, quantity, price) -> Money:
        return Money.of("-1", price.currency)


def test_negative_fee_is_rejected_before_broker_state_changes():
    order = accepted_order()
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NegativeCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    with pytest.raises(ConfigurationError, match="fee"):
        sim.process(market_slice("2024-01-03T14:30:00Z"), rng())

    assert sim.active_orders[order.id] is order
