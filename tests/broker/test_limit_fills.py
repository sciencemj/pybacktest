from decimal import Decimal

import pytest

from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoSlippage,
    SimulatedBroker,
    VolumeParticipationLimit,
)
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.money import Money
from pybacktest.domain.orders import OrderSide
from tests.factories import accepted_order, market_slice, rng


def broker() -> SimulatedBroker:
    return SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )


def test_buy_limit_opening_below_limit_gets_open_price_improvement():
    order = accepted_order(
        order_type="limit",
        quantity="10",
        limit_price="100",
    )
    sim = broker()
    sim.submit(order)

    event = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="98",
            high="102",
            low="97",
            close="101",
        ),
        rng(),
    )[0]

    assert event.fill.price.amount == Decimal("98")


def test_buy_limit_touched_intrabar_fills_at_the_limit():
    order = accepted_order(
        order_type="limit",
        limit_price="100",
    )
    sim = broker()
    sim.submit(order)

    event = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="102",
            high="103",
            low="99",
            close="101",
        ),
        rng(),
    )[0]

    assert event.fill.price.amount == Decimal("100")


def test_sell_limit_opening_above_limit_gets_open_price_improvement():
    order = accepted_order(
        side="sell",
        order_type="limit",
        limit_price="100",
    )
    sim = broker()
    sim.submit(order)

    event = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="102",
            high="103",
            low="99",
            close="101",
        ),
        rng(),
    )[0]

    assert event.fill.price.amount == Decimal("102")


def test_sell_limit_touched_intrabar_fills_at_the_limit():
    order = accepted_order(
        side="sell",
        order_type="limit",
        limit_price="100",
    )
    sim = broker()
    sim.submit(order)

    event = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="98",
            high="101",
            low="97",
            close="99",
        ),
        rng(),
    )[0]

    assert event.fill.price.amount == Decimal("100")


def test_untouched_buy_and_sell_limits_remain_active():
    buy = accepted_order(
        order_id=None,
        order_type="limit",
        limit_price="95",
    )
    sell = accepted_order(
        order_id=buy.id.__class__.parse("order_" + "9" * 32),
        side="sell",
        order_type="limit",
        limit_price="105",
    )
    sim = broker()
    sim.submit(buy)
    sim.submit(sell)

    events = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="100",
            high="104",
            low="96",
            close="101",
        ),
        rng(),
    )

    assert events == ()
    assert tuple(sim.active_orders) == (buy.id, sell.id)


class FixedAdverseSlippage:
    def apply(
        self,
        order,
        quantity,
        reference_price,
        market,
        rng,
    ) -> Money:
        adjustment = Decimal("5")
        amount = (
            reference_price.amount + adjustment
            if order.side is OrderSide.BUY
            else reference_price.amount - adjustment
        )
        return Money.of(amount, reference_price.currency)


def test_adverse_slippage_is_clamped_to_buy_and_sell_limits():
    buy = accepted_order(
        order_type="limit",
        limit_price="100",
    )
    sell = accepted_order(
        order_id=buy.id.__class__.parse("order_" + "9" * 32),
        side="sell",
        order_type="limit",
        limit_price="100",
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=FixedAdverseSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(buy)
    sim.submit(sell)

    events = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="100",
            high="100",
            low="100",
            close="100",
        ),
        rng(),
    )

    assert [event.fill.price.amount for event in events] == [
        Decimal("100"),
        Decimal("100"),
    ]


class InvalidPriceSlippage:
    def apply(
        self,
        order,
        quantity,
        reference_price,
        market,
        rng,
    ) -> Money:
        return Money.of("0", reference_price.currency)


def test_nonpositive_model_price_is_rejected_without_changing_order_state():
    order = accepted_order()
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=InvalidPriceSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    with pytest.raises(ConfigurationError, match="positive"):
        sim.process(
            market_slice("2024-01-03T14:30:00Z"),
            rng(),
        )

    assert sim.active_orders[order.id] is order
