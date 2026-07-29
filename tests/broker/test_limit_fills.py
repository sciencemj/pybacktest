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
from pybacktest.domain.instruments import Instrument
from pybacktest.domain.money import Money
from pybacktest.domain.orders import OrderSide
from pybacktest.engine.accounting import PortfolioLedger
from tests.factories import accepted_order, market_slice, rng
from tests.factories import instrument as make_instrument


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


def test_limit_order_never_fills_on_its_submission_bar_even_when_active():
    order = accepted_order(
        order_type="limit",
        limit_price="100",
        submitted_at="2024-01-03T14:30:00Z",
        active_from="2024-01-03T14:30:00Z",
    )
    sim = broker()
    sim.submit(order)

    assert (
        sim.process(
            market_slice(
                "2024-01-03T14:30:00Z",
                open="99",
                high="101",
                low="98",
                close="100",
            ),
            rng(),
        )
        == ()
    )
    event = sim.process(
        market_slice(
            "2024-01-04T14:30:00Z",
            open="98",
            high="100",
            low="97",
            close="99",
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


def priced_instrument(
    *,
    tick_size: str = "0.01",
    lot_size: str = "1",
) -> Instrument:
    item = make_instrument()
    return Instrument(
        id=item.id,
        quote_currency=item.quote_currency,
        tick_size=Decimal(tick_size),
        lot_size=Decimal(lot_size),
        timezone=item.timezone,
    )


class FractionalTickSlippage:
    def apply(
        self,
        order,
        quantity,
        reference_price,
        market,
        rng,
    ) -> Money:
        del quantity, reference_price, market, rng
        amount = (
            Decimal("100.003") if order.side is OrderSide.BUY else Decimal("100.007")
        )
        return Money.of(amount, order.quote_currency)


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (OrderSide.BUY, Decimal("100.01")),
        (OrderSide.SELL, Decimal("100.00")),
    ],
)
def test_post_slippage_price_uses_side_aware_adverse_tick_rounding(
    side: OrderSide,
    expected: Decimal,
):
    item = priced_instrument()
    order = accepted_order(item=item, side=side)
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=FractionalTickSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )
    sim.submit(order)

    event = sim.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0]

    assert event.fill.price.amount == expected


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (OrderSide.BUY, Decimal("100.02")),
        (OrderSide.SELL, Decimal("99.99")),
    ],
)
def test_adverse_tick_rounding_supports_non_power_of_ten_ticks(
    side: OrderSide,
    expected: Decimal,
):
    item = priced_instrument(tick_size="0.03")
    order = accepted_order(item=item, side=side)
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )
    sim.submit(order)

    event = sim.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0]

    assert event.fill.price.amount == expected


class LimitCrossingSlippage:
    def apply(
        self,
        order,
        quantity,
        reference_price,
        market,
        rng,
    ) -> Money:
        del quantity, reference_price, market, rng
        amount = (
            Decimal("100.003") if order.side is OrderSide.BUY else Decimal("99.997")
        )
        return Money.of(amount, order.quote_currency)


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_tick_rounding_then_limit_protection_keeps_limit_price(
    side: OrderSide,
):
    item = priced_instrument()
    order = accepted_order(
        item=item,
        side=side,
        order_type="limit",
        limit_price="100.00",
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=LimitCrossingSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )
    sim.submit(order)

    event = sim.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0]

    assert event.fill.price.amount == Decimal("100.00")


class TinyPositiveSlippage:
    def apply(
        self,
        order,
        quantity,
        reference_price,
        market,
        rng,
    ) -> Money:
        del order, quantity, reference_price, market, rng
        return Money.usd("0.005")


def test_sell_limit_protects_a_positive_price_that_tick_floors_to_zero():
    item = priced_instrument()
    order = accepted_order(
        item=item,
        side=OrderSide.SELL,
        order_type="limit",
        limit_price="0.01",
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=TinyPositiveSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )
    sim.submit(order)

    fill = sim.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0].fill
    ledger = PortfolioLedger(
        base_currency="USD",
        initial_cash=Money.usd("10000"),
        instruments={item.id: item},
    )

    snapshot = ledger.apply_fill(fill)

    assert fill.price == Money.usd("0.01")
    assert snapshot.positions[item.id].quantity.value == Decimal("-10")


def test_market_sell_still_rejects_a_price_that_tick_floors_to_zero():
    item = priced_instrument()
    order = accepted_order(
        item=item,
        side=OrderSide.SELL,
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=TinyPositiveSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )
    sim.submit(order)

    with pytest.raises(ConfigurationError, match="execution price"):
        sim.process(
            market_slice("2024-01-03T14:30:00Z", item=item),
            rng(),
        )


def test_tick_rounded_broker_fill_is_accepted_by_portfolio_ledger():
    item = priced_instrument()
    order = accepted_order(item=item)
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=FractionalTickSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )
    sim.submit(order)
    fill = sim.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0].fill
    ledger = PortfolioLedger(
        base_currency="USD",
        initial_cash=Money.usd("10000"),
        instruments={item.id: item},
    )

    snapshot = ledger.apply_fill(fill)

    assert snapshot.positions[item.id].quantity.value == Decimal("10")


def test_catalog_rejects_a_limit_price_off_tick():
    item = priced_instrument()
    order = accepted_order(
        item=item,
        order_type="limit",
        limit_price="100.005",
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )

    with pytest.raises(ConfigurationError, match=r"limit price.*tick"):
        sim.submit(order)
