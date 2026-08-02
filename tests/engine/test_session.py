from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType

import numpy as np
import pytest

from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    SimulatedBrokerFactory,
    VolumeParticipationLimit,
)
from pybacktest.application.requests import SimulationRequest
from pybacktest.data.calendar import CalendarPolicy
from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.data.features import FeatureBuilder
from pybacktest.domain.errors import (
    AdapterContractError,
    ConfigurationError,
    DataValidationError,
)
from pybacktest.domain.events import OrderRejected
from pybacktest.domain.identifiers import FillId, OrderId, RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.market import MarketSlice
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
    TimeInForce,
)
from pybacktest.domain.time import DateRange, Timeframe
from pybacktest.engine import BacktestEngine, SessionStateError
from pybacktest.engine.session import (
    _as_datetime,
    _FixedDatasetSessionBoundary,
    _RunIdSequence,
)
from pybacktest.ports.broker import (
    BrokerRunContext,
    FillIdSource,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
)
from pybacktest.ports.components import ComponentDescriptor
from pybacktest.ports.risk import RiskContext, RiskDecision, RiskStatus
from pybacktest.results.metrics import MetricsConfig
from pybacktest.risk import DefaultOrderSizer, LongShortRisk

from .test_engine import (
    _simulation,
    _StaticSource,
    _three_bar_dataset,
    _two_bar_dataset,
)


def test_empty_universe_has_stable_configuration_error_code() -> None:
    with pytest.raises(ConfigurationError) as raised:
        SimulationRequest(
            universe=(),
            period=DateRange(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 5, tzinfo=UTC),
            ),
            timeframe=Timeframe.days(1),
            calendar=CalendarPolicy.union(),
            initial_cash=Money.usd("10000"),
            seed=7,
            metrics=MetricsConfig(
                risk_free_rate=0,
                annualization_periods=252,
            ),
        )

    assert raised.value.code == "empty_universe"


def test_simulation_request_rejects_duplicate_universe_and_invalid_seed() -> None:
    dataset = _two_bar_dataset()
    baseline = _simulation(dataset)
    instrument_id = baseline.universe[0]

    with pytest.raises(ConfigurationError, match="unique"):
        replace(
            baseline,
            universe=(instrument_id, instrument_id),
        )
    for invalid_seed in (-1, 2**63, True):
        with pytest.raises(ConfigurationError, match="seed"):
            replace(baseline, seed=invalid_seed)


def _broker_factory():
    return SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(
            intrabar_policy=IntrabarPolicy.CONSERVATIVE
        ),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )


def _risk():
    return LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    )


def _session(
    *,
    dataset=None,
    simulation=None,
    broker_factory=None,
    order_sizer=None,
    risk_policy=None,
):
    resolved_dataset = dataset or _two_bar_dataset()
    engine = BacktestEngine(
        data_source=_StaticSource(resolved_dataset),
        broker_factory=broker_factory or _broker_factory(),
        order_sizer=order_sizer or DefaultOrderSizer(),
        risk_policy=risk_policy or _risk(),
    )
    return engine.create_session(
        simulation or _simulation(resolved_dataset),
        feature_plan=FeatureBuilder().plan(),
        run_id=RunId.parse("run_" + "2" * 32),
    )


@pytest.mark.parametrize(
    "timestamps",
    [
        np.asarray(
            ["2024-01-02T14:30:00.000000001"],
            dtype="datetime64[ns]",
        ),
        np.asarray(
            [
                "2024-01-02T14:30:00.000000001",
                "2024-01-03T14:30:00.000000999",
            ],
            dtype="datetime64[ns]",
        ),
    ],
    ids=("one-sub-microsecond-bar", "two-sub-microsecond-bars"),
)
def test_reset_rejects_calendars_that_lose_nanosecond_precision(
    timestamps: np.ndarray,
) -> None:
    baseline = _two_bar_dataset()
    item = next(iter(baseline.instruments.values()))
    count = len(timestamps)
    dataset = MarketDataSet(
        series={
            item.id: BarSeries(
                timestamps=timestamps,
                open=np.full(count, 100.0),
                high=np.full(count, 101.0),
                low=np.full(count, 99.0),
                close=np.full(count, 100.0),
                volume=np.full(count, 1_000.0),
            )
        },
        instruments={item.id: item},
        timeframe=Timeframe.days(1),
    )
    session = _session(dataset=dataset)

    with pytest.raises(DataValidationError) as raised:
        session.reset()

    assert raised.value.code == "timestamp_precision_loss"
    assert "nanosecond" in str(raised.value)
    with pytest.raises(SessionStateError, match="failed"):
        session.reset()


def test_session_rejects_out_of_order_and_stale_observation_use() -> None:
    session = _session()

    with pytest.raises(SessionStateError, match="reset"):
        session.advance((), observation=object())  # type: ignore[arg-type]

    first = session.reset()
    session.strategy_context(first)
    with pytest.raises(SessionStateError, match="already"):
        session.strategy_context(first)

    step = session.advance((), observation=first)
    assert step.observation is not None
    second = step.observation
    with pytest.raises(SessionStateError, match="stale"):
        session.strategy_context(first)

    final = session.advance((), observation=second)
    assert final.done
    with pytest.raises(SessionStateError, match="completed"):
        session.advance((), observation=second)


def test_advance_requires_the_exact_current_observation() -> None:
    session = _session(dataset=_three_bar_dataset())
    first = session.reset()

    step = session.advance((), observation=first)
    assert step.observation is not None
    current = step.observation

    with pytest.raises(SessionStateError, match="stale"):
        session.advance((), observation=first)

    retry = session.advance((), observation=current)
    assert retry.observation is not None


def test_invalid_cancel_makes_the_whole_intent_batch_atomic() -> None:
    session = _session(dataset=_three_bar_dataset())
    first = session.reset()
    instrument_id = next(iter(first.market.bars))
    resting = LimitOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        limit_price=Money.usd("50"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("resting"),
    )
    step = session.advance((resting,), observation=first)
    assert step.observation is not None
    current = step.observation
    active_before = current.active_orders
    another = replace(
        resting,
        reason=DecisionReason.of("must_not_be_submitted"),
    )
    foreign_cancel = CancelOrderIntent(
        order_id=OrderId.parse("order_" + "f" * 32),
        reason=DecisionReason.of("foreign_cancel"),
    )

    with pytest.raises(AdapterContractError) as raised:
        session.advance(
            (another, foreign_cancel),
            observation=current,
        )

    assert raised.value.code == "unknown_cancel_order"
    assert current.active_orders == active_before
    retry = session.advance((), observation=current)
    assert retry.observation is not None
    assert retry.observation.active_orders == active_before


def test_repeated_cancel_of_one_order_never_partially_applies() -> None:
    session = _session(dataset=_three_bar_dataset())
    first = session.reset()
    instrument_id = next(iter(first.market.bars))
    resting = LimitOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        limit_price=Money.usd("50"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("resting"),
    )
    step = session.advance((resting,), observation=first)
    assert step.observation is not None
    current = step.observation
    active_before = current.active_orders
    order_id = active_before[0].id
    cancel = CancelOrderIntent(
        order_id=order_id,
        reason=DecisionReason.of("repeated_cancel"),
    )

    with pytest.raises(AdapterContractError) as raised:
        session.advance((cancel, cancel), observation=current)

    assert raised.value.code == "duplicate_cancel_order"
    retry = session.advance((), observation=current)
    assert retry.observation is not None
    assert retry.observation.active_orders == active_before


def test_prior_gtc_remainder_reserves_capacity_for_a_later_step() -> None:
    session = _session(dataset=_three_bar_dataset())
    first = session.reset()
    instrument_id = next(iter(first.market.bars))
    resting = LimitOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("40"),
        limit_price=Money.usd("50"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("resting_reservation"),
    )
    step = session.advance((resting,), observation=first)
    assert step.observation is not None
    assert len(step.observation.active_orders) == 1

    second = session.advance(
        (
            MarketOrderIntent(
                instrument=instrument_id,
                side=OrderSide.BUY,
                quantity=Quantity.of("100"),
                time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
                reason=DecisionReason.of("later_step_buy"),
            ),
        ),
        observation=step.observation,
    )
    assert second.observation is not None
    session.advance((), observation=second.observation)
    result = session.result()

    assert result.orders[1].quantity == Quantity.of("60")
    adjusted = [
        event
        for event in result.events
        if event.code.value == "order.adjusted"
    ]
    assert len(adjusted) == 1
    assert adjusted[0].details["codes"] == (
        "max_leverage",
        "available_cash",
    )


def test_observation_rejects_a_portfolio_from_another_timestamp() -> None:
    session = _session()
    observation = session.reset()
    mismatched = replace(
        observation.portfolio,
        timestamp=observation.portfolio.timestamp
        + timedelta(seconds=1),
    )

    with pytest.raises(ConfigurationError, match="portfolio"):
        replace(observation, portfolio=mismatched)


def test_final_bar_intent_is_rejected_without_inventing_future_time() -> None:
    dataset = _two_bar_dataset().prefix(1)
    session = _session(dataset=dataset)
    final_observation = session.reset()
    instrument_id = next(iter(final_observation.market.bars))

    final = session.advance(
        (
            MarketOrderIntent(
                instrument=instrument_id,
                side=OrderSide.BUY,
                quantity=Quantity.of("1"),
                time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
                reason=DecisionReason.of("terminal_request"),
            ),
        ),
        observation=final_observation,
    )
    result = session.result()

    assert final.done
    assert len(result.orders) == 1
    assert result.orders[0].status is OrderStatus.REJECTED
    assert result.orders[0].active_from == result.orders[0].submitted_at
    assert result.orders[0].submitted_at == result.market_timestamps[-1]
    assert result.fills == ()
    assert result.events[-1].code.value == "terminal.no_next_bar"


class _InvalidSizer:
    def size(self, intent, context: RiskContext):
        del intent, context
        return None


class _InvalidRisk:
    def evaluate(self, order, context: RiskContext):
        del order, context
        return None


class _MismatchedQuantityRisk:
    def evaluate(self, order, context: RiskContext):
        del order, context
        return RiskDecision(
            status=RiskStatus.PASSED,
            original_quantity=Quantity.of("2"),
            final_quantity=Quantity.of("2"),
            codes=(),
            message="Mismatched quantity.",
        )


class _MisalignedAdjustmentRisk:
    def evaluate(self, order, context: RiskContext):
        del context
        return RiskDecision(
            status=RiskStatus.ADJUSTED,
            original_quantity=order.quantity,
            final_quantity=Quantity.of("0.5"),
            codes=("test_adjustment",),
            message="Misaligned adjustment.",
        )


def _buy(observation) -> MarketOrderIntent:
    return MarketOrderIntent(
        instrument=next(iter(observation.market.bars)),
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("contract_test"),
    )


def test_sizer_result_is_validated_at_the_adapter_boundary() -> None:
    session = _session(order_sizer=_InvalidSizer())
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == "invalid_sizer_result"


def test_risk_result_is_validated_at_the_adapter_boundary() -> None:
    session = _session(risk_policy=_InvalidRisk())
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == "invalid_risk_decision"


@pytest.mark.parametrize(
    "risk_policy",
    [_MismatchedQuantityRisk(), _MisalignedAdjustmentRisk()],
)
def test_risk_decision_must_match_the_sized_order_and_lot(
    risk_policy,
) -> None:
    session = _session(risk_policy=risk_policy)
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == "invalid_risk_decision"


def test_same_step_orders_share_deterministic_cash_reservations() -> None:
    session = _session(dataset=_three_bar_dataset())
    observation = session.reset()
    instrument_id = next(iter(observation.market.bars))
    buy = MarketOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("100"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("full_cash"),
    )

    step = session.advance(
        (buy, buy),
        observation=observation,
    )

    assert step.observation is not None
    assert (
        step.observation.portfolio.positions[instrument_id].quantity
        == Quantity.of("100")
    )
    second = session.advance((), observation=step.observation)
    assert second.observation is not None
    session.advance((), observation=second.observation)
    result = session.result()
    assert [order.status for order in result.orders] == [
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
    ]


def test_successful_trade_explanation_records_every_causal_stage() -> None:
    session = _session(dataset=_three_bar_dataset())
    observation = session.reset()
    instrument_id = next(iter(observation.market.bars))
    intent = MarketOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("100"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("full_size_buy", horizon="next_bar"),
    )

    step = session.advance((intent,), observation=observation)
    assert step.observation is not None
    second = session.advance((), observation=step.observation)
    assert second.observation is not None
    session.advance((), observation=second.observation)
    result = session.result()

    explanation = result.explain_trade(result.orders[0].id)
    assert [entry.stage.value for entry in explanation.entries] == [
        "intent",
        "sizing",
        "risk",
        "scheduling",
        "broker",
        "broker",
        "accounting",
    ]
    assert [entry.code.value for entry in explanation.entries] == [
        "intent.received",
        "order.sized",
        "risk.passed",
        "order.scheduled",
        "order.accepted",
        "order.filled",
        "ledger.applied",
    ]
    assert explanation.entries[0].details["reason"] == "full_size_buy"
    assert explanation.entries[0].details["reason.horizon"] == "next_bar"
    assert explanation.entries[1].details["quantity"] == "100"
    assert explanation.entries[2].details["codes"] == ()
    assert explanation.entries[-1].details["position_quantity"] == "100"


def test_adjusted_trade_explanation_keeps_the_risk_decision_codes() -> None:
    session = _session(dataset=_three_bar_dataset())
    observation = session.reset()
    instrument_id = next(iter(observation.market.bars))
    intent = MarketOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("200"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("oversized_buy"),
    )

    step = session.advance((intent,), observation=observation)
    assert step.observation is not None
    second = session.advance((), observation=step.observation)
    assert second.observation is not None
    session.advance((), observation=second.observation)
    result = session.result()

    explanation = result.explain_trade(result.orders[0].id)
    risk_entries = [
        entry
        for entry in explanation.entries
        if entry.stage.value == "risk"
    ]
    assert len(risk_entries) == 1
    assert risk_entries[0].code.value == "order.adjusted"
    assert risk_entries[0].details["codes"] == (
        "max_leverage",
        "available_cash",
    )
    assert Decimal(risk_entries[0].details["requested_quantity"]) == (
        Decimal("200")
    )
    assert Decimal(risk_entries[0].details["adjusted_quantity"]) == (
        Decimal("100")
    )
    assert [entry.stage.value for entry in explanation.entries] == [
        "intent",
        "sizing",
        "risk",
        "scheduling",
        "broker",
        "broker",
        "accounting",
    ]


class _TamperingBroker:
    """Delegate to a real broker and replace only its emitted events."""

    def __init__(self, inner, factory) -> None:
        self._inner = inner
        self._factory = factory

    @property
    def active_orders(self):
        return self._inner.active_orders

    def submit(self, order):
        return self._inner.submit(order)

    def cancel(self, order_id: OrderId, timestamp: datetime):
        return self._inner.cancel(order_id, timestamp)

    def process(self, market: MarketSlice, rng):
        return self._factory.tamper(
            tuple(self._inner.process(market, rng))
        )


class _TamperingBrokerFactory:
    """Stateless factory whose subclasses rewrite broker events."""

    def create(self, run_context: BrokerRunContext):
        return _TamperingBroker(
            _broker_factory().create(run_context),
            self,
        )

    def tamper(self, events):
        return events


def _foreign_order(order):
    return replace(
        order,
        id=OrderId.parse("order_" + "e" * 32),
    )


class _ForeignOrderBrokerFactory(_TamperingBrokerFactory):
    def tamper(self, events):
        if not events:
            return events
        first = events[0]
        return (
            *events,
            replace(
                first,
                order=_foreign_order(first.order),
                fill=replace(
                    first.fill,
                    order_id=_foreign_order(first.order).id,
                ),
            ),
        )


class _MutatedOrderBrokerFactory(_TamperingBrokerFactory):
    def tamper(self, events):
        if not events:
            return events
        first = events[0]
        return (
            replace(
                first,
                order=replace(
                    first.order,
                    reason=DecisionReason.of("rewritten_by_broker"),
                ),
            ),
        )


class _ForeignFillIdBrokerFactory(_TamperingBrokerFactory):
    def tamper(self, events):
        if not events:
            return events
        first = events[0]
        return (
            replace(
                first,
                fill=replace(
                    first.fill,
                    id=FillId.parse("fill_" + "c" * 32),
                ),
            ),
        )


class _FutureFillTimestampBrokerFactory(_TamperingBrokerFactory):
    def tamper(self, events):
        if not events:
            return events
        first = events[0]
        shifted = first.fill.timestamp + timedelta(seconds=1)
        return (
            replace(first, fill=replace(first.fill, timestamp=shifted)),
        )


class _SilentlyResurrectingBroker(_TamperingBroker):
    def process(self, market: MarketSlice, rng):
        events = tuple(self._inner.process(market, rng))
        self._resurrected = {
            order.id: order
            for event in events
            for order in (event.order,)
        }
        return events

    @property
    def active_orders(self):
        current = dict(self._inner.active_orders)
        for order_id, order in getattr(
            self,
            "_resurrected",
            {},
        ).items():
            if order.status is OrderStatus.FILLED:
                current[order_id] = replace(
                    order,
                    status=OrderStatus.ACCEPTED,
                    filled_quantity=Quantity.of("0"),
                )
        return MappingProxyType(current)


class _SilentlyResurrectingBrokerFactory(_TamperingBrokerFactory):
    def create(self, run_context: BrokerRunContext):
        return _SilentlyResurrectingBroker(
            _broker_factory().create(run_context),
            self,
        )


class _FlippedFillSideBrokerFactory(_TamperingBrokerFactory):
    def tamper(self, events):
        if not events:
            return events
        first = events[0]
        flipped = (
            OrderSide.SELL
            if first.fill.side is OrderSide.BUY
            else OrderSide.BUY
        )
        return (replace(first, fill=replace(first.fill, side=flipped)),)


class _ForeignFillCurrencyBrokerFactory(_TamperingBrokerFactory):
    def tamper(self, events):
        if not events:
            return events
        first = events[0]
        return (
            replace(
                first,
                fill=replace(
                    first.fill,
                    price=Money.of(first.fill.price.amount, "EUR"),
                    fee=Money.of(first.fill.fee.amount, "EUR"),
                ),
            ),
        )


class _ForeignFillInstrumentBrokerFactory(_TamperingBrokerFactory):
    def tamper(self, events):
        if not events:
            return events
        first = events[0]
        other = replace(first.order.instrument, symbol="MSFT")
        return (
            replace(
                first,
                order=replace(first.order, instrument=other),
                fill=replace(first.fill, instrument=other),
            ),
        )


@pytest.mark.parametrize(
    ("broker_factory", "code"),
    [
        (_FlippedFillSideBrokerFactory(), "invalid_broker_order_state"),
        (_ForeignFillCurrencyBrokerFactory(), "invalid_broker_order_state"),
        (
            _ForeignFillInstrumentBrokerFactory(),
            "invalid_broker_order_state",
        ),
    ],
    ids=("flipped-side", "foreign-currency", "foreign-instrument"),
)
def test_fill_order_disagreement_is_rejected_at_the_engine_boundary(
    broker_factory,
    code: str,
) -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=broker_factory,
    )
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == code


def _engine_owned_state(session):
    """Snapshot every engine-owned mutable structure a fill would change."""
    recorder = session._required_recorder()
    ledger_snapshot = session._required_ledger().snapshot()
    return {
        "fill_ordinal": session._fill_ordinal,
        "submitted_orders": dict(session._submitted_orders),
        "recorder_fills": tuple(recorder._fills),
        "recorder_orders": dict(recorder._orders),
        "recorder_events": tuple(recorder._events),
        "ledger_cash": ledger_snapshot.cash,
        "ledger_positions": {
            instrument_id: position.quantity
            for instrument_id, position in (
                ledger_snapshot.positions.items()
            )
        },
    }


@pytest.mark.parametrize(
    "broker_factory",
    [
        _FlippedFillSideBrokerFactory(),
        _ForeignFillCurrencyBrokerFactory(),
        _SilentlyResurrectingBrokerFactory(),
    ],
    ids=("side-mismatch", "currency-mismatch", "broker-state-mismatch"),
)
def test_rejected_broker_batch_mutates_no_engine_owned_state(
    broker_factory,
) -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=broker_factory,
    )
    observation = session.reset()
    before = _engine_owned_state(session)

    with pytest.raises(AdapterContractError):
        session.advance((_buy(observation),), observation=observation)

    after = _engine_owned_state(session)
    # Sizing/risk/scheduling before the broker call is legitimate; nothing a
    # broker *event* would have changed may have moved.
    assert after["fill_ordinal"] == before["fill_ordinal"]
    assert after["recorder_fills"] == before["recorder_fills"]
    assert after["ledger_cash"] == before["ledger_cash"]
    assert after["ledger_positions"] == before["ledger_positions"]
    broker_driven = {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    }
    assert not [
        order
        for order in after["recorder_orders"].values()
        if order.status in broker_driven
    ]
    assert not [
        order
        for order in after["submitted_orders"].values()
        if order.status in broker_driven
    ]


@pytest.mark.parametrize(
    ("broker_factory", "code"),
    [
        (_ForeignOrderBrokerFactory(), "unknown_broker_order"),
        (_MutatedOrderBrokerFactory(), "invalid_broker_order_state"),
        (_ForeignFillIdBrokerFactory(), "invalid_fill_identity"),
        (
            _FutureFillTimestampBrokerFactory(),
            "invalid_broker_event_timestamp",
        ),
        (
            _SilentlyResurrectingBrokerFactory(),
            "broker_state_disagreement",
        ),
    ],
    ids=(
        "foreign-order",
        "mutated-stable-field",
        "unexpected-fill-identity",
        "invalid-fill-timestamp",
        "active-orders-disagreement",
    ),
)
def test_broker_events_cross_the_engine_owned_trust_boundary(
    broker_factory,
    code: str,
) -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=broker_factory,
    )
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == code


def test_broker_trust_boundary_rejects_before_recorder_mutation() -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=_ForeignOrderBrokerFactory(),
    )
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == "unknown_broker_order"
    with pytest.raises(SessionStateError, match="failed"):
        session.result()


class _SubmitFabricatesFillBroker(_TamperingBroker):
    def submit(self, order):
        self._inner.submit(order)
        return (
            OrderFilledEvent(
                fill=Fill(
                    id=FillId.parse("fill_" + "d" * 32),
                    order_id=order.id,
                    instrument=order.instrument,
                    side=order.side,
                    quantity=order.quantity,
                    price=Money.usd("100"),
                    fee=Money.usd("0"),
                    timestamp=order.active_from,
                ),
                order=replace(
                    order,
                    status=OrderStatus.FILLED,
                    filled_quantity=order.quantity,
                ),
            ),
        )


class _CancelReturnsExpiryBroker(_TamperingBroker):
    def cancel(self, order_id: OrderId, timestamp: datetime):
        events = tuple(self._inner.cancel(order_id, timestamp))
        return tuple(
            OrderExpiredEvent(
                order=event.order,
                timestamp=event.timestamp,
                message="fabricated expiry from cancel",
            )
            for event in events
        )


class _LateFabricatingBroker(_TamperingBroker):
    """Behave normally until the order is resting, then fabricate one event."""

    _calls = 0

    def process(self, market: MarketSlice, rng):
        self._calls += 1
        active = tuple(self._inner.active_orders.values())
        events = tuple(self._inner.process(market, rng))
        if events or not active or self._calls < 2:
            return events
        return (self.fabricate(active[0], _as_datetime(market.timestamp)),)

    def fabricate(self, order, timestamp):
        raise NotImplementedError


class _ProcessCancelsBroker(_LateFabricatingBroker):
    def fabricate(self, order, timestamp):
        return OrderCancelledEvent(
            order=order.cancel("fabricated cancel from process"),
            timestamp=timestamp,
            message="fabricated cancel from process",
        )


class _ProcessExpiresGtcBroker(_LateFabricatingBroker):
    def fabricate(self, order, timestamp):
        return OrderExpiredEvent(
            order=order.cancel("fabricated gtc expiry"),
            timestamp=timestamp,
            message="fabricated gtc expiry",
        )


def _origin_factory(broker_type):
    class _OriginFactory(_TamperingBrokerFactory):
        def create(self, run_context: BrokerRunContext):
            return broker_type(
                _broker_factory().create(run_context),
                self,
            )

    return _OriginFactory()


def test_submit_may_not_fabricate_an_event_for_an_inactive_order() -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=_origin_factory(_SubmitFabricatesFillBroker),
    )
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == "invalid_broker_event_origin"


def _resting_limit(observation) -> LimitOrderIntent:
    return LimitOrderIntent(
        instrument=next(iter(observation.market.bars)),
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        limit_price=Money.usd("50"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("resting"),
    )


def test_cancel_may_only_return_a_cancellation_for_that_order() -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=_origin_factory(_CancelReturnsExpiryBroker),
    )
    first = session.reset()
    step = session.advance((_resting_limit(first),), observation=first)
    assert step.observation is not None
    current = step.observation

    with pytest.raises(AdapterContractError) as raised:
        session.advance(
            (
                CancelOrderIntent(
                    order_id=current.active_orders[0].id,
                    reason=DecisionReason.of("cancel_resting"),
                ),
            ),
            observation=current,
        )

    assert raised.value.code == "invalid_broker_event_origin"


@pytest.mark.parametrize(
    "broker_type",
    [_ProcessCancelsBroker, _ProcessExpiresGtcBroker],
    ids=("process-cancels", "process-expires-gtc"),
)
def test_process_may_not_cancel_or_expire_a_gtc_order(broker_type) -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=_origin_factory(broker_type),
    )
    first = session.reset()
    step = session.advance((_resting_limit(first),), observation=first)
    assert step.observation is not None

    with pytest.raises(AdapterContractError) as raised:
        session.advance((), observation=step.observation)

    assert raised.value.code == "invalid_broker_event_origin"


class _ShapeShiftingBatch(Sequence):
    """Serve benign events for N element reads, then a side-flipped batch."""

    def __init__(self, events, swap_after: int) -> None:
        self._events = tuple(events)
        self._swapped = tuple(
            replace(
                event,
                fill=replace(
                    event.fill,
                    side=(
                        OrderSide.SELL
                        if event.fill.side is OrderSide.BUY
                        else OrderSide.BUY
                    ),
                ),
            )
            for event in self._events
        )
        self._swap_after = swap_after
        self.reads = 0
        self.traversals = 0

    def __len__(self) -> int:
        return len(self._events)

    def __getitem__(self, index):
        if index == 0:
            self.traversals += 1
        current = (
            self._swapped if self.reads >= self._swap_after else self._events
        )
        self.reads += 1
        return current[index]


class _ShapeShiftingBroker(_TamperingBroker):
    swap_after = 0

    def __init__(self, inner, factory) -> None:
        super().__init__(inner, factory)
        self.batches: list[_ShapeShiftingBatch] = []

    def process(self, market: MarketSlice, rng):
        events = tuple(self._inner.process(market, rng))
        if not events:
            return events
        batch = _ShapeShiftingBatch(events, self.swap_after)
        self.batches.append(batch)
        return batch


def _shape_shifting_session(swap_after: int):
    class _Factory(_TamperingBrokerFactory):
        def __init__(self) -> None:
            self.broker = None

        def component_descriptor(self) -> ComponentDescriptor:
            return ComponentDescriptor(
                identity="tests.shape_shifting_factory",
                version="1",
                configuration={},
            )

        def create(self, run_context: BrokerRunContext):
            self.broker = _ShapeShiftingBroker(
                _broker_factory().create(run_context),
                self,
            )
            self.broker.swap_after = swap_after
            return self.broker

    factory = _Factory()
    return _session(dataset=_three_bar_dataset(), broker_factory=factory), factory


def test_shape_shifting_batch_is_rejected_when_it_lies_immediately() -> None:
    session, _ = _shape_shifting_session(0)
    observation = session.reset()
    before = _engine_owned_state(session)

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == "invalid_broker_order_state"
    after = _engine_owned_state(session)
    assert after["fill_ordinal"] == before["fill_ordinal"]
    assert after["recorder_fills"] == before["recorder_fills"]
    assert after["ledger_cash"] == before["ledger_cash"]
    assert after["ledger_positions"] == before["ledger_positions"]
    assert not [
        order
        for order in after["recorder_orders"].values()
        if order.status
        in {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ]


@pytest.mark.parametrize("swap_after", [1, 2, 3], ids=("one", "two", "three"))
def test_batch_read_once_makes_later_substitution_unreachable(
    swap_after: int,
) -> None:
    """The engine must act on one snapshot, never on a re-read of the adapter.

    A batch that turns malicious only after the engine's first element read
    can no longer influence anything: the engine reads each element exactly
    once and acts on that snapshot alone.
    """
    session, factory = _shape_shifting_session(swap_after)
    observation = session.reset()

    step = session.advance((_buy(observation),), observation=observation)

    assert step.observation is not None
    second = session.advance((), observation=step.observation)
    assert second.observation is not None
    session.advance((), observation=second.observation)
    result = session.result()

    assert [fill.side for fill in result.fills] == [OrderSide.BUY]
    assert result.orders[0].status is OrderStatus.FILLED
    assert factory.broker.batches
    assert [batch.traversals for batch in factory.broker.batches] == [
        1 for _ in factory.broker.batches
    ]


class _EarlyDayExpiryBroker(_TamperingBroker):
    """Expire a resting DAY order on its own anchor bar, consistently."""

    def __init__(self, inner, factory) -> None:
        super().__init__(inner, factory)
        self._expired: dict[OrderId, Order] = {}

    @property
    def active_orders(self):
        current = dict(self._inner.active_orders)
        for order_id in self._expired:
            current.pop(order_id, None)
        return MappingProxyType(current)

    def process(self, market: MarketSlice, rng):
        events = tuple(self._inner.process(market, rng))
        active = tuple(self._inner.active_orders.values())
        if events or not active:
            return events
        order = active[0]
        expired = order.cancel("premature day expiry")
        self._expired[order.id] = expired
        return (
            OrderExpiredEvent(
                order=expired,
                timestamp=_as_datetime(market.timestamp),
                message="premature day expiry",
            ),
        )


def test_process_may_not_expire_a_day_order_before_its_session_ends() -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=_origin_factory(_EarlyDayExpiryBroker),
    )
    first = session.reset()
    resting = replace(
        _resting_limit(first),
        time_in_force=TimeInForce.DAY,
        reason=DecisionReason.of("resting_day"),
    )

    with pytest.raises(AdapterContractError) as raised:
        session.advance((resting,), observation=first)

    assert raised.value.code == "invalid_broker_event_origin"


def test_legitimate_day_expiry_still_passes_origin_validation() -> None:
    session = _session(dataset=_three_bar_dataset())
    first = session.reset()
    resting = replace(
        _resting_limit(first),
        time_in_force=TimeInForce.DAY,
        reason=DecisionReason.of("resting_day"),
    )

    step = session.advance((resting,), observation=first)
    assert step.observation is not None
    expired = session.advance((), observation=step.observation)

    assert expired.observation is not None
    assert expired.observation.active_orders == ()
    session.advance((), observation=expired.observation)
    result = session.result()
    assert result.orders[0].status is OrderStatus.CANCELLED
    assert "order.expired" in {event.code.value for event in result.events}


_PHANTOM_ORDER_ID = OrderId.parse("order_" + "b" * 32)


class _PhantomOnLaterReadBroker(_TamperingBroker):
    """Inject a phantom active order from a configured property read on."""

    inject_from = 1

    def __init__(self, inner, factory) -> None:
        super().__init__(inner, factory)
        self.reads = 0

    @property
    def active_orders(self):
        self.reads += 1
        current = dict(self._inner.active_orders)
        if self.reads >= self.inject_from and current:
            genuine = next(iter(current.values()))
            current[_PHANTOM_ORDER_ID] = replace(
                genuine,
                id=_PHANTOM_ORDER_ID,
                quantity=Quantity.of("99"),
            )
        return MappingProxyType(current)


@pytest.mark.parametrize("inject_from", [1, 2, 3, 4, 5, 6, 7, 8])
def test_phantom_active_order_cannot_enter_an_observation(
    inject_from: int,
) -> None:
    """No adapter read may put an order the engine never submitted into an
    Observation, whichever read the adapter chooses to lie on."""

    class _Factory(_TamperingBrokerFactory):
        def create(self, run_context: BrokerRunContext):
            broker = _PhantomOnLaterReadBroker(
                _broker_factory().create(run_context),
                self,
            )
            broker.inject_from = inject_from
            return broker

    session = _session(dataset=_three_bar_dataset(), broker_factory=_Factory())
    observations = []
    try:
        first = session.reset()
        observations.append(first)
        step = session.advance((_resting_limit(first),), observation=first)
        if step.observation is not None:
            observations.append(step.observation)
    except AdapterContractError as error:
        assert error.code in {
            "broker_state_disagreement",
            "invalid_active_orders",
        }

    for observation in observations:
        assert _PHANTOM_ORDER_ID not in {
            order.id for order in observation.active_orders
        }


class _FalsyRejectingSizer:
    def __bool__(self) -> bool:
        return False

    def size(self, intent, context: RiskContext):
        return OrderRejected(
            order_id=context.order_id,
            instrument=intent.instrument,
            timestamp=context.submitted_at,
            message="Falsy sizer rejection.",
            reason=DecisionReason.of("falsy_sizer"),
        )


class _FalsyRejectingRisk:
    def __bool__(self) -> bool:
        return False

    def evaluate(self, order, context: RiskContext):
        del context
        return RiskDecision(
            status=RiskStatus.REJECTED,
            original_quantity=order.quantity,
            final_quantity=Quantity.of("0"),
            codes=("falsy_risk",),
            message="Falsy risk rejection.",
        )


@pytest.mark.parametrize(
    ("order_sizer", "risk_policy", "stage"),
    [
        (_FalsyRejectingSizer(), _risk(), "sizing"),
        (DefaultOrderSizer(), _FalsyRejectingRisk(), "risk"),
    ],
)
def test_falsy_valid_run_overrides_are_not_replaced_by_defaults(
    order_sizer,
    risk_policy,
    stage: str,
) -> None:
    dataset = _two_bar_dataset()
    engine = BacktestEngine(
        data_source=_StaticSource(dataset),
        broker_factory=_broker_factory(),
        order_sizer=DefaultOrderSizer(),
        risk_policy=_risk(),
    )
    session = engine.create_session(
        _simulation(dataset),
        feature_plan=FeatureBuilder().plan(),
        order_sizer=order_sizer,
        risk_policy=risk_policy,
    )
    observation = session.reset()
    first = session.advance(
        (_buy(observation),),
        observation=observation,
    )
    assert first.observation is not None
    session.advance((), observation=first.observation)
    result = session.result()

    assert result.fills == ()
    rejections = [
        event
        for event in result.events
        if event.code.value == "order.rejected"
    ]
    assert [event.stage.value for event in rejections] == [stage]


class _NoneSubmitBroker:
    @property
    def active_orders(self):
        return MappingProxyType({})

    def submit(self, order):
        del order
        return None

    def cancel(self, order_id: OrderId, timestamp: datetime):
        del order_id, timestamp
        return ()

    def process(self, market: MarketSlice, rng):
        del market, rng
        return ()


class _NoneSubmitBrokerFactory:
    def create(self, run_context: BrokerRunContext):
        del run_context
        return _NoneSubmitBroker()


class _InvalidActiveOrdersBroker(_NoneSubmitBroker):
    @property
    def active_orders(self):
        return ()


class _InvalidActiveOrdersBrokerFactory:
    def create(self, run_context: BrokerRunContext):
        del run_context
        return _InvalidActiveOrdersBroker()


def test_broker_active_orders_is_validated_before_observation() -> None:
    session = _session(
        broker_factory=_InvalidActiveOrdersBrokerFactory()
    )

    with pytest.raises(AdapterContractError) as raised:
        session.reset()

    assert raised.value.code == "invalid_active_orders"


def test_empty_or_invalid_submit_results_always_cross_event_validation() -> None:
    session = _session(broker_factory=_NoneSubmitBrokerFactory())
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),), observation=observation)

    assert raised.value.code == "invalid_broker_events"


def test_held_position_beyond_union_staleness_has_typed_policy_failure() -> None:
    dataset = _two_bar_dataset()
    item = next(iter(dataset.instruments.values()))
    other = replace(item, id=replace(item.id, symbol="MSFT"))
    first_series = dataset.series[item.id]
    timestamps = np.asarray(
        [
            "2024-01-02T14:30:00",
            "2024-01-03T14:30:00",
            "2024-01-04T14:30:00",
        ],
        dtype="datetime64[ns]",
    )
    other_series = BarSeries(
        timestamps=timestamps,
        open=np.asarray([100.0, 100.0, 100.0]),
        high=np.asarray([101.0, 101.0, 101.0]),
        low=np.asarray([99.0, 99.0, 99.0]),
        close=np.asarray([100.0, 100.0, 100.0]),
        volume=np.asarray([1_000.0, 1_000.0, 1_000.0]),
    )
    union_dataset = MarketDataSet(
        series={item.id: first_series, other.id: other_series},
        instruments={item.id: item, other.id: other},
        timeframe=Timeframe.days(1),
    )
    simulation = replace(
        _simulation(union_dataset),
        calendar=CalendarPolicy.union(max_staleness_bars=0),
    )
    session = _session(
        dataset=union_dataset,
        simulation=simulation,
    )
    first = session.reset()
    step = session.advance((_buy(first),), observation=first)
    assert step.observation is not None
    second = step.observation

    with pytest.raises(DataValidationError, match="staleness"):
        session.advance((), observation=second)
    with pytest.raises(SessionStateError, match="failed"):
        session.advance((), observation=second)


@pytest.mark.parametrize(
    ("time_in_force", "expected_status", "expected_event"),
    [
        (TimeInForce.DAY, OrderStatus.CANCELLED, "order.expired"),
        (
            TimeInForce.GOOD_TIL_CANCELLED,
            OrderStatus.ACCEPTED,
            None,
        ),
    ],
)
def test_run_scoped_dataset_boundary_distinguishes_day_from_gtc(
    time_in_force: TimeInForce,
    expected_status: OrderStatus,
    expected_event: str | None,
) -> None:
    dataset = _three_bar_dataset()
    session = _session(dataset=dataset)
    observation = session.reset()
    instrument_id = next(iter(observation.market.bars))
    limit = LimitOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        limit_price=Money.usd("50"),
        time_in_force=time_in_force,
        reason=DecisionReason.of("resting_limit"),
    )

    first = session.advance((limit,), observation=observation)
    assert first.observation is not None
    assert len(first.observation.active_orders) == 1
    second = session.advance((), observation=first.observation)
    assert second.observation is not None
    session.advance((), observation=second.observation)
    result = session.result()

    assert result.orders[0].status is expected_status
    codes = tuple(event.code.value for event in result.events)
    if expected_event is None:
        assert "order.expired" not in codes
    else:
        assert expected_event in codes


def test_daily_day_order_waits_for_its_instruments_first_eligible_bar() -> None:
    dataset, _, sparse_id = _asymmetric_dataset()
    simulation = replace(
        _simulation(dataset),
        calendar=CalendarPolicy.union(max_staleness_bars=1),
    )
    session = _session(dataset=dataset, simulation=simulation)
    first = session.reset()
    resting = LimitOrderIntent(
        instrument=sparse_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        limit_price=Money.usd("50"),
        time_in_force=TimeInForce.DAY,
        reason=DecisionReason.of("sparse_daily"),
    )

    first_step = session.advance((resting,), observation=first)
    assert first_step.observation is not None
    second_step = session.advance(
        (),
        observation=first_step.observation,
    )

    assert second_step.observation is not None
    assert len(second_step.observation.active_orders) == 1
    session.advance((), observation=second_step.observation)
    result = session.result()
    assert result.orders[0].status is OrderStatus.ACCEPTED
    assert "order.expired" not in {
        event.code.value for event in result.events
    }


def test_intraday_day_order_partially_fills_at_anchor_then_expires() -> None:
    baseline = _two_bar_dataset()
    first = next(iter(baseline.instruments.values()))
    second = replace(first, id=replace(first.id, symbol="MSFT"))
    first_timestamps = np.asarray(
        [
            "2024-01-02T14:30:00",
            "2024-01-03T14:30:00",
            "2024-01-04T14:30:00",
        ],
        dtype="datetime64[ns]",
    )
    second_timestamps = np.asarray(
        ["2024-01-02T15:00:00"],
        dtype="datetime64[ns]",
    )
    dataset = MarketDataSet(
        series={
            first.id: BarSeries(
                timestamps=first_timestamps,
                open=np.full(3, 100.0),
                high=np.full(3, 101.0),
                low=np.full(3, 99.0),
                close=np.full(3, 100.0),
                volume=np.full(3, 2.0),
            ),
            second.id: BarSeries(
                timestamps=second_timestamps,
                open=np.asarray([200.0]),
                high=np.asarray([201.0]),
                low=np.asarray([199.0]),
                close=np.asarray([200.0]),
                volume=np.asarray([1_000.0]),
            ),
        },
        instruments={first.id: first, second.id: second},
        timeframe=Timeframe.minutes(30),
    )
    simulation = replace(
        _simulation(dataset),
        timeframe=Timeframe.minutes(30),
        calendar=CalendarPolicy.union(max_staleness_bars=2),
    )
    broker_factory = SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(
            intrabar_policy=IntrabarPolicy.CONSERVATIVE
        ),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(Decimal("0.5")),
        borrow_cost=NoBorrowCost(),
    )
    session = _session(
        dataset=dataset,
        simulation=simulation,
        broker_factory=broker_factory,
    )
    observation = session.reset()
    order = MarketOrderIntent(
        instrument=first.id,
        side=OrderSide.BUY,
        quantity=Quantity.of("2"),
        time_in_force=TimeInForce.DAY,
        reason=DecisionReason.of("intraday_anchor"),
    )

    step = session.advance((order,), observation=observation)
    assert step.observation is not None
    anchor_step = session.advance((), observation=step.observation)

    assert anchor_step.observation is not None
    assert len(anchor_step.observation.active_orders) == 1
    assert (
        anchor_step.observation.active_orders[0].filled_quantity
        == Quantity.of("1")
    )
    expired_step = session.advance(
        (),
        observation=anchor_step.observation,
    )
    assert expired_step.observation is not None
    assert expired_step.observation.active_orders == ()
    session.advance((), observation=expired_step.observation)
    result = session.result()
    assert result.orders[0].status is OrderStatus.CANCELLED
    assert [fill.quantity for fill in result.fills] == [Quantity.of("1")]


def test_instrument_day_calendar_uses_exact_nanosecond_membership() -> None:
    baseline = _two_bar_dataset()
    item = next(iter(baseline.instruments.values()))
    series_timestamps = np.asarray(
        [
            "2024-01-02T14:30:00",
            "2024-01-03T14:30:00.000000001",
            "2024-01-04T14:30:00",
        ],
        dtype="datetime64[ns]",
    )
    dataset = MarketDataSet(
        series={
            item.id: BarSeries(
                timestamps=series_timestamps,
                open=np.full(3, 100.0),
                high=np.full(3, 101.0),
                low=np.full(3, 99.0),
                close=np.full(3, 100.0),
                volume=np.full(3, 1_000.0),
            )
        },
        instruments={item.id: item},
        timeframe=Timeframe.days(1),
    )
    calendar = np.asarray(
        [
            "2024-01-02T14:30:00",
            "2024-01-03T14:30:00",
            "2024-01-04T14:30:00",
        ],
        dtype="datetime64[ns]",
    )
    boundary = _FixedDatasetSessionBoundary(dataset, calendar)
    anchored = Order.pending(
        id=OrderId.parse("order_" + "1" * 32),
        instrument=item.id,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Quantity.of("1"),
        quote_currency=item.quote_currency,
        limit_price=Money.usd("50"),
        time_in_force=TimeInForce.DAY,
        submitted_at=datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        active_from=datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
        reason=DecisionReason.of("phantom_anchor_probe"),
    ).accept()

    assert (
        boundary.day_order_expired(
            anchored,
            datetime(2024, 1, 4, 14, 30, tzinfo=UTC),
        )
        is False
    )


class _RunContextRecordingFactory:
    """Capture the run context the engine hands to a broker factory."""

    def __init__(self) -> None:
        self.run_context = None

    def component_descriptor(self) -> ComponentDescriptor:
        return ComponentDescriptor(
            identity="tests.run_context_recording_factory",
            version="1",
            configuration={},
        )

    def create(self, run_context: BrokerRunContext):
        self.run_context = run_context
        return _broker_factory().create(run_context)


def test_broker_receives_only_a_narrowed_fill_id_capability() -> None:
    factory = _RunContextRecordingFactory()
    session = _session(broker_factory=factory)

    session.reset()

    fill_ids = factory.run_context.fill_ids
    assert callable(fill_ids.fill_id)
    assert not hasattr(fill_ids, "next_order_id")
    assert str(fill_ids.fill_id(0)).startswith("fill_")
    assert fill_ids.fill_id(0) == fill_ids.fill_id(0)
    assert fill_ids.fill_id(0) != fill_ids.fill_id(1)


def _reachable_state(value) -> list[object]:
    """Every object the facade retains, via __dict__ and every MRO slot."""
    retained: list[object] = list(vars(value).values()) if hasattr(
        value, "__dict__"
    ) else []
    for klass in type(value).__mro__:
        declared = klass.__dict__.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        for slot in declared:
            if slot not in {"__dict__", "__weakref__"} and hasattr(
                value, slot
            ):
                retained.append(getattr(value, slot))
    return retained


def test_fill_id_facade_retains_no_order_id_allocator() -> None:
    factory = _RunContextRecordingFactory()
    session = _session(broker_factory=factory)

    session.reset()

    fill_ids = factory.run_context.fill_ids
    retained = _reachable_state(fill_ids)
    assert retained, "the facade must retain the data it derives IDs from"
    for item in retained:
        assert not hasattr(item, "next_order_id"), item
        assert not isinstance(item, _RunIdSequence), item
    assert isinstance(fill_ids, FillIdSource)
    assert fill_ids.fill_id(3) == _RunIdSequence(
        RunId.parse("run_" + "2" * 32)
    ).fill_id(3)


class _ForeignActiveOrderBroker(_TamperingBroker):
    @property
    def active_orders(self):
        current = dict(self._inner.active_orders)
        for order_id, order in tuple(current.items()):
            current[order_id] = replace(
                order,
                instrument=replace(order.instrument, symbol="MSFT"),
            )
        return MappingProxyType(current)


def test_active_orders_outside_the_catalog_are_a_coded_adapter_error() -> None:
    session = _session(
        dataset=_three_bar_dataset(),
        broker_factory=_origin_factory(_ForeignActiveOrderBroker),
    )
    first = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_resting_limit(first),), observation=first)

    assert raised.value.code == "invalid_active_orders"


def test_engine_and_session_constructors_perform_no_data_io() -> None:
    dataset = _two_bar_dataset()
    source = _StaticSource(dataset)
    engine = BacktestEngine(
        data_source=source,
        broker_factory=_broker_factory(),
        order_sizer=DefaultOrderSizer(),
        risk_policy=_risk(),
    )

    session = engine.create_session(
        _simulation(dataset),
        feature_plan=FeatureBuilder().plan(),
    )

    assert source.loads == 0
    session.reset()
    assert source.loads == 1


class _InvalidDatasetSource:
    def load(self, universe, period, timeframe):
        del universe, period, timeframe
        return object()


def test_data_adapter_result_type_is_validated_explicitly() -> None:
    dataset = _two_bar_dataset()
    engine = BacktestEngine(
        data_source=_InvalidDatasetSource(),
        broker_factory=_broker_factory(),
        order_sizer=DefaultOrderSizer(),
        risk_policy=_risk(),
    )
    session = engine.create_session(
        _simulation(dataset),
        feature_plan=FeatureBuilder().plan(),
    )

    with pytest.raises(AdapterContractError) as raised:
        session.reset()

    assert raised.value.code == "invalid_dataset_type"


def test_dataset_currency_is_validated_immediately_after_load() -> None:
    dataset = _two_bar_dataset()
    item = next(iter(dataset.instruments.values()))
    euro = replace(item, quote_currency="EUR")
    euro_dataset = MarketDataSet(
        series={euro.id: dataset.series[item.id]},
        instruments={euro.id: euro},
        timeframe=dataset.timeframe,
    )
    session = _session(
        dataset=euro_dataset,
        simulation=_simulation(dataset),
    )

    with pytest.raises(DataValidationError, match="currency"):
        session.reset()


def test_dataset_timeframe_must_match_the_explicit_request() -> None:
    dataset = _two_bar_dataset()
    simulation = replace(
        _simulation(dataset),
        timeframe=Timeframe.minutes(30),
    )
    session = _session(dataset=dataset, simulation=simulation)

    with pytest.raises(DataValidationError, match="timeframe"):
        session.reset()


def _asymmetric_dataset() -> tuple[
    MarketDataSet,
    InstrumentId,
    InstrumentId,
]:
    dataset = _three_bar_dataset()
    first = next(iter(dataset.instruments.values()))
    second = replace(first, id=replace(first.id, symbol="MSFT"))
    full = dataset.series[first.id]
    sparse = BarSeries(
        timestamps=full.timestamps[[0, 2]],
        open=np.asarray([200.0, 202.0]),
        high=np.asarray([201.0, 203.0]),
        low=np.asarray([199.0, 201.0]),
        close=np.asarray([200.0, 202.0]),
        volume=full.volume[[0, 2]],
    )
    return (
        MarketDataSet(
            series={first.id: full, second.id: sparse},
            instruments={first.id: first, second.id: second},
            timeframe=dataset.timeframe,
        ),
        first.id,
        second.id,
    )


def test_union_market_is_current_only_while_marks_obey_staleness() -> None:
    dataset, first_id, second_id = _asymmetric_dataset()
    simulation = replace(
        _simulation(dataset),
        calendar=CalendarPolicy.union(max_staleness_bars=1),
    )
    session = _session(dataset=dataset, simulation=simulation)
    first = session.reset()

    step = session.advance((), observation=first)

    assert step.observation is not None
    assert set(step.observation.market.bars) == {first_id}
    assert set(step.observation.portfolio.valuation_prices) == {
        first_id,
        second_id,
    }
    assert (
        step.observation.portfolio.valuation_prices[
            second_id
        ].amount
        == Decimal("200.0")
    )


def test_intersection_calendar_skips_nonshared_timestamps() -> None:
    dataset, first_id, second_id = _asymmetric_dataset()
    simulation = replace(
        _simulation(dataset),
        calendar=CalendarPolicy.intersection(),
    )
    session = _session(dataset=dataset, simulation=simulation)
    first = session.reset()

    step = session.advance((), observation=first)

    assert step.observation is not None
    assert step.observation.timestamp == dataset.timestamps[-1]
    assert set(first.market.bars) == {first_id, second_id}
    assert set(step.observation.market.bars) == {
        first_id,
        second_id,
    }
