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
from pybacktest.domain.identifiers import OrderId, RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    LimitOrderIntent,
    MarketOrderIntent,
    OrderSide,
    OrderStatus,
    TimeInForce,
)
from pybacktest.domain.time import DateRange, Timeframe
from pybacktest.engine import BacktestEngine, SessionStateError
from pybacktest.ports.broker import BrokerRunContext
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


def test_session_rejects_out_of_order_and_stale_observation_use() -> None:
    session = _session()

    with pytest.raises(SessionStateError, match="reset"):
        session.advance(())

    first = session.reset()
    session.strategy_context(first)
    with pytest.raises(SessionStateError, match="already"):
        session.strategy_context(first)

    step = session.advance(())
    assert step.observation is not None
    with pytest.raises(SessionStateError, match="stale"):
        session.strategy_context(first)

    final = session.advance(())
    assert final.done
    with pytest.raises(SessionStateError, match="completed"):
        session.advance(())


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
        )
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
        session.advance((_buy(observation),))

    assert raised.value.code == "invalid_sizer_result"


def test_risk_result_is_validated_at_the_adapter_boundary() -> None:
    session = _session(risk_policy=_InvalidRisk())
    observation = session.reset()

    with pytest.raises(AdapterContractError) as raised:
        session.advance((_buy(observation),))

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
        session.advance((_buy(observation),))

    assert raised.value.code == "invalid_risk_decision"


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
    session.advance((_buy(observation),))
    session.advance(())
    result = session.result()

    assert result.fills == ()
    assert result.events[0].stage.value == stage


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
        session.advance((_buy(observation),))

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
    session.advance((_buy(first),))

    with pytest.raises(DataValidationError, match="staleness"):
        session.advance(())
    with pytest.raises(SessionStateError, match="failed"):
        session.advance(())


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

    first = session.advance((limit,))
    assert first.observation is not None
    assert len(first.observation.active_orders) == 1
    second = session.advance(())
    assert second.observation is not None
    session.advance(())
    result = session.result()

    assert result.orders[0].status is expected_status
    codes = tuple(event.code.value for event in result.events)
    if expected_event is None:
        assert "order.expired" not in codes
    else:
        assert expected_event in codes


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
    session.reset()

    step = session.advance(())

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

    step = session.advance(())

    assert step.observation is not None
    assert step.observation.timestamp == dataset.timestamps[-1]
    assert set(first.market.bars) == {first_id, second_id}
    assert set(step.observation.market.bars) == {
        first_id,
        second_id,
    }
