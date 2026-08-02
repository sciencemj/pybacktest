from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal
from types import MappingProxyType

import numpy as np
import pytest

from pybacktest.adapters.broker import (
    BrokerRunContext,
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    SimulatedBroker,
    SimulatedBrokerFactory,
    VolumeParticipationLimit,
)
from pybacktest.domain.errors import (
    ClockRegressionError,
    ConfigurationError,
    DataValidationError,
)
from pybacktest.domain.identifiers import OrderId
from pybacktest.domain.market import BarView, MarketSlice
from pybacktest.domain.money import Money
from pybacktest.domain.orders import OrderStatus
from tests.factories import (
    accepted_order,
    instrument,
    market_slice,
    rng,
)


def broker() -> SimulatedBroker:
    return SimulatedBroker(
        fill_model=NextBarOpenFill(
            intrabar_policy=IntrabarPolicy.CONSERVATIVE,
        ),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(
            max_volume_ratio=Decimal("0.10"),
        ),
        borrow_cost=NoBorrowCost(),
    )


def test_market_order_does_not_fill_before_its_active_bar():
    order = accepted_order(
        quantity="10",
        submitted_at="2024-01-02T21:00:00Z",
        active_from="2024-01-03T14:30:00Z",
    )
    sim = broker()
    sim.submit(order)

    assert (
        sim.process(
            market_slice("2024-01-02T21:00:00Z", open="100"),
            rng(),
        )
        == ()
    )
    events = sim.process(
        market_slice("2024-01-03T14:30:00Z", open="101"),
        rng(),
    )

    assert events[0].fill.price.amount == Decimal("101")


def test_market_order_never_fills_on_its_submission_bar_even_when_active():
    order = accepted_order(
        submitted_at="2024-01-03T14:30:00Z",
        active_from="2024-01-03T14:30:00Z",
    )
    sim = broker()
    sim.submit(order)

    assert (
        sim.process(
            market_slice("2024-01-03T14:30:00Z"),
            rng(),
        )
        == ()
    )
    event = sim.process(
        market_slice("2024-01-04T14:30:00Z", open="101"),
        rng(),
    )[0]

    assert event.fill.price.amount == Decimal("101")


def test_market_order_uses_the_eligible_bar_open_for_a_sell():
    order = accepted_order(side="sell")
    sim = broker()
    sim.submit(order)

    event = sim.process(
        market_slice("2024-01-03T14:30:00Z", open="99"),
        rng(),
    )[0]

    assert event.fill.price.amount == Decimal("99")
    assert event.order.status is OrderStatus.FILLED


def test_submit_consumes_only_accepted_or_partially_filled_orders():
    pending = accepted_order()
    pending = pending.cancel("make terminal")

    with pytest.raises(ConfigurationError, match="ACCEPTED or PARTIALLY_FILLED"):
        broker().submit(pending)


def test_partially_filled_order_can_be_resubmitted_to_a_fresh_broker():
    original = accepted_order(quantity="20")
    first_broker = broker()
    first_broker.submit(original)
    partial = first_broker.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="100",
            volume="100",
        ),
        rng(),
    )[0].order
    fresh = broker()

    fresh.submit(partial)
    final = fresh.process(
        market_slice(
            "2024-01-04T14:30:00Z",
            open="101",
            volume="100",
        ),
        rng(),
    )[0]

    assert final.order.status is OrderStatus.FILLED


def test_duplicate_order_id_is_rejected_across_the_broker_lifetime():
    order = accepted_order()
    sim = broker()
    sim.submit(order)
    sim.process(
        market_slice("2024-01-03T14:30:00Z"),
        rng(),
    )

    with pytest.raises(ConfigurationError, match="duplicate order id"):
        sim.submit(order)


def test_active_orders_is_an_immutable_snapshot():
    order = accepted_order()
    sim = broker()
    sim.submit(order)
    snapshot = sim.active_orders

    assert isinstance(snapshot, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot[order.id] = order  # type: ignore[index]

    sim.process(market_slice("2024-01-03T14:30:00Z"), rng())
    assert order.id in snapshot
    assert order.id not in sim.active_orders


def test_process_requires_an_explicit_numpy_generator():
    sim = broker()

    with pytest.raises(ConfigurationError, match=r"numpy\.random\.Generator"):
        sim.process(
            market_slice("2024-01-03T14:30:00Z"),
            42,  # type: ignore[arg-type]
        )


def test_repeated_or_regressing_market_timestamps_are_rejected():
    sim = broker()
    sim.process(market_slice("2024-01-03T14:30:00Z"), rng())

    with pytest.raises(ClockRegressionError, match="strictly increasing"):
        sim.process(market_slice("2024-01-03T14:30:00Z"), rng())
    with pytest.raises(ClockRegressionError, match="strictly increasing"):
        sim.process(market_slice("2024-01-02T14:30:00Z"), rng())


def test_submicrosecond_market_timestamp_is_rejected_without_precision_loss():
    item = instrument()
    observed_at = np.datetime64("2024-01-03T14:30:00.000000001", "ns")
    bar = BarView(
        timestamp=observed_at,
        open=100,
        high=100,
        low=100,
        close=100,
        volume=1000,
    )
    market = MarketSlice(timestamp=observed_at, bars={item.id: bar})

    with pytest.raises(DataValidationError, match="microsecond"):
        broker().process(market, rng())


def test_missing_instrument_bar_keeps_the_order_active():
    requested = instrument("AAPL")
    available = instrument("MSFT")
    order = accepted_order(item=requested)
    sim = broker()
    sim.submit(order)

    events = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            item=available,
        ),
        rng(),
    )

    assert events == ()
    assert sim.active_orders[order.id] is order


def test_zero_volume_bar_is_untradable_even_without_a_liquidity_cap():
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )
    order = accepted_order()
    sim.submit(order)

    events = sim.process(
        market_slice("2024-01-03T14:30:00Z", volume="0"),
        rng(),
    )

    assert events == ()
    assert order.id in sim.active_orders


def test_direct_brokers_replay_the_same_deterministic_fill_ids():
    first = broker()
    second = broker()
    first.submit(accepted_order())
    second.submit(accepted_order())

    first_fill = first.process(
        market_slice("2024-01-03T14:30:00Z"),
        rng(1),
    )[0].fill
    second_fill = second.process(
        market_slice("2024-01-03T14:30:00Z"),
        rng(999),
    )[0].fill

    assert first_fill.id == second_fill.id
    assert str(first_fill.id) == "fill_" + "0" * 32


def test_factory_configuration_is_frozen_and_each_broker_is_isolated():
    item = instrument()
    factory = SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )
    context = BrokerRunContext(instruments={item.id: item})

    with pytest.raises(FrozenInstanceError):
        factory.commission = NoCommission()  # type: ignore[misc]

    first = factory.create(context)
    second = factory.create(context)
    order = accepted_order(item=item)
    first.submit(order)

    assert order.id in first.active_orders
    assert order.id not in second.active_orders
    second.submit(order)
    first_fill = first.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0].fill
    second_fill = second.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0].fill
    assert first_fill.id == second_fill.id


class CountingCommission:
    def __init__(self) -> None:
        self.calls = 0

    def calculate(self, order, quantity, price) -> Money:
        self.calls += 1
        return Money.of(self.calls, price.currency)


def test_factory_clones_stateful_models_for_each_broker():
    item = instrument()
    factory = SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=CountingCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )
    context = BrokerRunContext(instruments={item.id: item})
    first = factory.create(context)
    second = factory.create(context)
    order = accepted_order(item=item)
    first.submit(order)
    second.submit(order)

    first_fee = first.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0].fill.fee
    second_fee = second.process(
        market_slice("2024-01-03T14:30:00Z", item=item),
        rng(),
    )[0].fill.fee

    assert first_fee == Money.usd("1")
    assert second_fee == Money.usd("1")


class UncloneableCommission:
    def __deepcopy__(self, memo):
        del memo
        raise TypeError("cannot clone")

    def calculate(self, order, quantity, price) -> Money:
        return Money.of("0", price.currency)


def test_factory_rejects_an_uncloneable_model_extension():
    item = instrument()
    factory = SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=UncloneableCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )

    with pytest.raises(ConfigurationError, match=r"commission.*clone"):
        factory.create(BrokerRunContext(instruments={item.id: item}))


def test_broker_run_context_copies_and_freezes_instrument_metadata():
    item = instrument()
    caller_catalog = {item.id: item}
    context = BrokerRunContext(instruments=caller_catalog)
    caller_catalog.clear()

    assert context.instruments[item.id] is item
    with pytest.raises(TypeError):
        context.instruments[item.id] = item  # type: ignore[index]


def test_cancel_rejects_an_untyped_order_id():
    with pytest.raises(ConfigurationError, match="OrderId"):
        broker().cancel(
            "order_" + "0" * 32,  # type: ignore[arg-type]
            datetime(2024, 1, 3, tzinfo=UTC),
        )


def test_duplicate_check_uses_typed_identity_not_object_identity():
    order = accepted_order(
        order_id=OrderId.parse("order_" + "a" * 32),
    )
    duplicate = accepted_order(
        order_id=OrderId.parse("order_" + "a" * 32),
    )
    sim = broker()
    sim.submit(order)

    with pytest.raises(ConfigurationError, match="duplicate"):
        sim.submit(duplicate)
