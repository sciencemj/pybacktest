from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext

import numpy as np
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
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import Instrument
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


class FixedCapacity:
    def __init__(self, quantity: Decimal) -> None:
        self.quantity = quantity

    def available_quantity(self, order, market) -> Quantity:
        del order, market
        return Quantity.of(self.quantity)


def test_shared_capacity_subtraction_is_exact_across_large_exponent_gap():
    first = accepted_order(
        order_id=OrderId.parse("order_" + "1" * 32),
        quantity="1",
    )
    second = accepted_order(
        order_id=OrderId.parse("order_" + "2" * 32),
        quantity="1E+100",
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=FixedCapacity(Decimal("1E+100")),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(first)
    sim.submit(second)

    events = sim.process(
        market_slice("2024-01-03T14:30:00Z"),
        rng(),
    )

    assert [event.fill.quantity.value for event in events] == [
        Decimal("1"),
        Decimal("9" * 100),
    ]


def test_capacity_subtraction_accepts_exact_4096_digit_boundary():
    capacity = Decimal((0, (1,), 4095))
    expected = Decimal((0, (9,) * 4095, 0))
    first = accepted_order(
        order_id=OrderId.parse("order_" + "1" * 32),
        quantity="1",
    )
    second = accepted_order(
        order_id=OrderId.parse("order_" + "2" * 32),
        quantity=capacity,
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=FixedCapacity(capacity),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(first)
    sim.submit(second)

    events = sim.process(
        market_slice("2024-01-03T14:30:00Z"),
        rng(),
    )

    assert events[1].fill.quantity.value == expected


def test_capacity_subtraction_rejects_work_beyond_documented_bound():
    capacity = Decimal((0, (1,), 4096))
    order = accepted_order(quantity="1")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=FixedCapacity(capacity),
        borrow_cost=NoBorrowCost(),
    )
    sim.submit(order)

    with pytest.raises(
        ConfigurationError,
        match=r"capacity subtraction.*4096",
    ):
        sim.process(
            market_slice("2024-01-03T14:30:00Z"),
            rng(),
        )

    assert sim.active_orders[order.id] is order


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


def lot_instrument() -> Instrument:
    item = instrument()
    return Instrument(
        id=item.id,
        quote_currency=item.quote_currency,
        tick_size=item.tick_size,
        lot_size=Decimal("10"),
        timezone=item.timezone,
    )


def test_catalog_rejects_total_quantity_that_would_strand_a_residual_lot():
    item = lot_instrument()
    order = accepted_order(item=item, quantity="15")
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )

    with pytest.raises(ConfigurationError, match=r"order quantity.*lot"):
        sim.submit(order)


def test_catalog_rejects_misaligned_existing_partial_fill_and_remainder():
    item = lot_instrument()
    accepted = accepted_order(item=item, quantity="30")
    partial = replace(
        accepted,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=Quantity.of("15"),
    )
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
        instruments={item.id: item},
    )

    with pytest.raises(ConfigurationError, match=r"filled quantity.*lot"):
        sim.submit(partial)


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


class RandomDrawSlippage:
    def apply(
        self,
        order,
        quantity,
        reference_price,
        market,
        rng: np.random.Generator,
    ) -> Money:
        del order, quantity, market
        return Money.of(
            reference_price.amount + Decimal(str(rng.random())),
            reference_price.currency,
        )


def two_orders() -> tuple:
    return (
        accepted_order(
            order_id=OrderId.parse("order_" + "1" * 32),
            quantity="10",
        ),
        accepted_order(
            order_id=OrderId.parse("order_" + "2" * 32),
            quantity="10",
        ),
    )


def random_broker(commission, *, fill_ids=None) -> SimulatedBroker:
    return SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=commission,
        slippage=RandomDrawSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
        fill_ids=fill_ids,
    )


def test_later_model_failure_rolls_back_caller_rng_for_identical_retry():
    first, second = two_orders()
    failing = SecondCalculationFailsOnce()
    sim = random_broker(failing)
    clean = random_broker(NoCommission())
    for target in (sim, clean):
        target.submit(first)
        target.submit(second)
    caller_rng = rng(123)
    original_state = deepcopy(caller_rng.bit_generator.state)
    current = market_slice("2024-01-03T14:30:00Z")

    with pytest.raises(RuntimeError, match="commission failure"):
        sim.process(current, caller_rng)

    assert caller_rng.bit_generator.state == original_state

    retried = sim.process(current, caller_rng)
    expected = clean.process(current, rng(123))
    assert [(event.fill.id, event.fill.price) for event in retried] == [
        (event.fill.id, event.fill.price) for event in expected
    ]


class RepairableIndexedFillIds:
    def __init__(self, second_value: object) -> None:
        self.values: list[object] = [
            FillId.parse("fill_" + "a" * 32),
            second_value,
        ]
        self.indexed_calls: list[int] = []
        self.incremental_calls = 0

    def fill_id(self, sequence: int) -> object:
        self.indexed_calls.append(sequence)
        return self.values[sequence]

    def next_fill_id(self) -> object:
        value = self.values[self.incremental_calls]
        self.incremental_calls += 1
        return value


@pytest.mark.parametrize(
    "invalid_second",
    [
        object(),
        FillId.parse("fill_" + "a" * 32),
    ],
)
def test_later_external_fill_id_failure_is_idempotent_and_retryable(
    invalid_second: object,
):
    first, second = two_orders()
    source = RepairableIndexedFillIds(invalid_second)
    sim = random_broker(NoCommission(), fill_ids=source)
    for order in (first, second):
        sim.submit(order)
    caller_rng = rng(321)
    original_state = deepcopy(caller_rng.bit_generator.state)
    current = market_slice("2024-01-03T14:30:00Z")

    with pytest.raises(ConfigurationError, match=r"FillId|fill id"):
        sim.process(current, caller_rng)

    assert caller_rng.bit_generator.state == original_state
    assert sim.active_orders[first.id] is first
    assert sim.active_orders[second.id] is second

    source.values[1] = FillId.parse("fill_" + "b" * 32)
    retried = sim.process(current, caller_rng)

    assert source.indexed_calls == [0, 1, 0, 1]
    assert [event.fill.id for event in retried] == [
        FillId.parse("fill_" + "a" * 32),
        FillId.parse("fill_" + "b" * 32),
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
