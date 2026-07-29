from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from pybacktest.adapters.broker import (
    BrokerRunContext,
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoSlippage,
    OrderCancelledEvent,
    OrderExpiredEvent,
    SimulatedBroker,
    SimulatedBrokerFactory,
    VolumeParticipationLimit,
)
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.orders import TimeInForce
from tests.factories import accepted_order, instrument, market_slice, rng


def test_ioc_is_rejected_explicitly_instead_of_behaving_like_gtc():
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(max_volume_ratio="0.10"),
        borrow_cost=NoBorrowCost(),
    )
    order = accepted_order(time_in_force=TimeInForce.IMMEDIATE_OR_CANCEL)

    with pytest.raises(ConfigurationError, match="IOC and FOK are unsupported"):
        sim.submit(order)


@pytest.mark.parametrize(
    "time_in_force",
    [TimeInForce.IMMEDIATE_OR_CANCEL, TimeInForce.FILL_OR_KILL],
)
def test_unsupported_immediate_tif_values_are_rejected_at_submission(
    time_in_force: TimeInForce,
):
    sim = SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )

    with pytest.raises(ConfigurationError, match="IOC and FOK are unsupported"):
        sim.submit(accepted_order(time_in_force=time_in_force))


@dataclass(frozen=True)
class FixedSessionBoundary:
    boundary: datetime

    def day_order_expired(self, order, timestamp: datetime) -> bool:
        return timestamp >= self.boundary


def configured_broker(
    *,
    boundary: FixedSessionBoundary | None = None,
) -> SimulatedBroker:
    return SimulatedBroker(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
        session_boundary=boundary,
    )


def test_day_order_expires_only_at_an_explicit_session_boundary():
    order = accepted_order(
        order_type="limit",
        limit_price="90",
        time_in_force=TimeInForce.DAY,
    )
    sim = configured_broker(
        boundary=FixedSessionBoundary(
            datetime(2024, 1, 3, 21, tzinfo=UTC),
        )
    )
    sim.submit(order)

    assert (
        sim.process(
            market_slice("2024-01-03T20:59:00Z"),
            rng(),
        )
        == ()
    )
    event = sim.process(
        market_slice("2024-01-03T21:00:00Z"),
        rng(),
    )[0]

    assert isinstance(event, OrderExpiredEvent)
    assert event.order.status.value == "cancelled"
    assert order.id not in sim.active_orders


def test_direct_broker_without_session_provider_does_not_infer_day_expiry():
    order = accepted_order(
        order_type="limit",
        limit_price="90",
        time_in_force=TimeInForce.DAY,
    )
    sim = configured_broker()
    sim.submit(order)

    events = sim.process(
        market_slice("2024-01-10T14:30:00Z"),
        rng(),
    )

    assert events == ()
    assert order.id in sim.active_orders
    assert sim.expires_day_orders is False


def test_gtc_persists_across_explicit_session_boundaries():
    order = accepted_order(
        order_type="limit",
        limit_price="90",
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
    )
    sim = configured_broker(
        boundary=FixedSessionBoundary(
            datetime(2024, 1, 3, 21, tzinfo=UTC),
        )
    )
    sim.submit(order)

    events = sim.process(
        market_slice("2024-01-04T14:30:00Z"),
        rng(),
    )

    assert events == ()
    assert order.id in sim.active_orders


def test_cancel_emits_typed_event_and_preserves_prior_order_value():
    order = accepted_order()
    sim = configured_broker()
    sim.submit(order)
    cancelled_at = datetime(2024, 1, 3, 14, tzinfo=UTC)

    event = sim.cancel(order.id, cancelled_at)[0]

    assert isinstance(event, OrderCancelledEvent)
    assert event.order.status.value == "cancelled"
    assert event.timestamp == cancelled_at
    assert order.status.value == "accepted"
    assert order.id not in sim.active_orders
    with pytest.raises(FrozenInstanceError):
        event.timestamp = datetime.now(UTC)  # type: ignore[misc]


def test_unknown_and_repeated_cancellations_have_distinct_errors():
    order = accepted_order()
    sim = configured_broker()
    now = datetime(2024, 1, 3, 14, tzinfo=UTC)

    with pytest.raises(ConfigurationError, match="unknown order"):
        sim.cancel(order.id, now)

    sim.submit(order)
    sim.cancel(order.id, now)
    with pytest.raises(ConfigurationError, match="no longer active"):
        sim.cancel(order.id, now)


def test_factory_injects_the_explicit_session_boundary():
    item = instrument()
    boundary = FixedSessionBoundary(
        datetime(2024, 1, 3, 21, tzinfo=UTC),
    )
    factory = SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("1")),
        borrow_cost=NoBorrowCost(),
    )
    context = BrokerRunContext(
        instruments={item.id: item},
        session_boundary=boundary,
    )

    sim = factory.create(context)

    assert sim.expires_day_orders is True
