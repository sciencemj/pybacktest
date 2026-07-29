from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    SimulatedBrokerFactory,
)
from pybacktest.application.requests import BacktestRequest, SimulationRequest
from pybacktest.application.service import BacktestService
from pybacktest.data.calendar import CalendarPolicy
from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    CancelOrderIntent,
    DecisionReason,
    LimitOrderIntent,
    MarketOrderIntent,
    OrderSide,
    OrderStatus,
    TimeInForce,
)
from pybacktest.domain.time import DateRange, Timeframe
from pybacktest.engine import BacktestEngine
from pybacktest.ports.strategy import StrategyContext
from pybacktest.results.metrics import MetricsConfig
from pybacktest.risk import DefaultOrderSizer, LongShortRisk
from tests.factories import instrument


@dataclass(frozen=True)
class _BuyWhenFlat:
    quantity: Quantity

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[MarketOrderIntent, ...]:
        instrument_id = next(iter(market.bars))
        if (
            context.portfolio.positions.get(
                instrument_id,
                Quantity.of("0"),
            ).value
            or context.active_orders
        ):
            return ()
        return (
            MarketOrderIntent(
                instrument=instrument_id,
                side=OrderSide.BUY,
                quantity=self.quantity,
                time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
                reason=DecisionReason.of("buy_when_flat"),
            ),
        )


class _StaticSource:
    def __init__(self, dataset: MarketDataSet) -> None:
        self.dataset = dataset
        self.loads = 0

    def load(
        self,
        universe: tuple,
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        self.loads += 1
        return self.dataset


def _two_bar_dataset() -> MarketDataSet:
    item = instrument()
    timestamps = np.asarray(
        ["2024-01-02T14:30:00", "2024-01-03T14:30:00"],
        dtype="datetime64[ns]",
    )
    return MarketDataSet(
        series={
            item.id: BarSeries(
                timestamps=timestamps,
                open=np.asarray([100.0, 105.0]),
                high=np.asarray([101.0, 107.0]),
                low=np.asarray([99.0, 104.0]),
                close=np.asarray([100.0, 106.0]),
                volume=np.asarray([1_000.0, 1_000.0]),
            )
        },
        instruments={item.id: item},
        timeframe=Timeframe.days(1),
    )


def _three_bar_dataset() -> MarketDataSet:
    dataset = _two_bar_dataset()
    item = next(iter(dataset.instruments.values()))
    timestamps = np.asarray(
        [
            "2024-01-02T14:30:00",
            "2024-01-03T14:30:00",
            "2024-01-04T14:30:00",
        ],
        dtype="datetime64[ns]",
    )
    series = BarSeries(
        timestamps=timestamps,
        open=np.asarray([100.0, 100.0, 100.0]),
        high=np.asarray([101.0, 101.0, 101.0]),
        low=np.asarray([99.0, 99.0, 99.0]),
        close=np.asarray([100.0, 100.0, 100.0]),
        volume=np.asarray([1_000.0, 1_000.0, 1_000.0]),
    )
    return MarketDataSet(
        series={item.id: series},
        instruments={item.id: item},
        timeframe=Timeframe.days(1),
    )


def _simulation(dataset: MarketDataSet) -> SimulationRequest:
    return SimulationRequest(
        universe=tuple(dataset.instruments),
        period=DateRange(
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 5, tzinfo=UTC),
        ),
        timeframe=Timeframe.days(1),
        calendar=CalendarPolicy.union(),
        initial_cash=Money.usd("10000"),
        seed=11,
        metrics=MetricsConfig(
            risk_free_rate=0,
            annualization_periods=252,
        ),
    )


def _engine(dataset: MarketDataSet) -> BacktestEngine:
    return BacktestEngine(
        data_source=_StaticSource(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(
                intrabar_policy=IntrabarPolicy.CONSERVATIVE
            ),
            commission=NoCommission(),
            slippage=NoSlippage(),
            liquidity=NoLiquidityLimit(),
            borrow_cost=NoBorrowCost(),
        ),
        order_sizer=DefaultOrderSizer(),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=False,
        ),
    )


def _manual_session(dataset: MarketDataSet):
    return _engine(dataset).create_session(
        _simulation(dataset),
        feature_plan=FeatureBuilder().plan(),
        run_id=RunId.parse("run_" + "6" * 32),
    )


def test_close_observed_order_fills_at_next_bar_open() -> None:
    dataset = _two_bar_dataset()
    result = _engine(dataset).run(
        BacktestRequest(
            strategy=_BuyWhenFlat(Quantity.of("1")),
            simulation=_simulation(dataset),
        )
    )

    assert len(result.fills) == 1
    assert result.fills[0].price.amount == Decimal("105")
    assert result.fills[0].timestamp == result.market_timestamps[1]


def test_run_and_manual_session_produce_identical_event_stream() -> None:
    dataset = _two_bar_dataset()
    strategy = _BuyWhenFlat(Quantity.of("1"))
    request = BacktestRequest(
        strategy=strategy,
        simulation=_simulation(dataset),
        run_id=RunId.parse(f"run_{'1' * 32}"),
    )
    engine = _engine(dataset)
    run_result = engine.run(request)

    builder = FeatureBuilder()
    feature_plan = strategy.build_features(builder)
    session = engine.create_session(
        request.simulation,
        feature_plan=feature_plan,
        run_id=request.run_id,
        order_sizer=None,
        risk_policy=None,
    )
    observation = session.reset()
    while not session.done:
        intents = strategy.on_bar(
            session.strategy_context(observation),
            observation.market,
        )
        step = session.advance(intents)
        if step.observation is not None:
            observation = step.observation
    manual_result = session.result()

    assert manual_result.events == run_result.events
    assert manual_result.snapshots == run_result.snapshots


def test_risk_adjustment_flows_through_broker_ledger_and_recorder() -> None:
    session = _manual_session(_two_bar_dataset())
    observation = session.reset()
    instrument_id = next(iter(observation.market.bars))
    intent = MarketOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("200"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("oversized_buy"),
    )

    step = session.advance((intent,))
    assert step.observation is not None
    session.advance(())
    result = session.result()

    assert result.orders[0].quantity == Quantity.of("100")
    assert result.orders[0].filled_quantity == Quantity.of("100")
    assert result.fills[0].quantity == Quantity.of("100")
    assert (
        result.snapshots[-1].positions[instrument_id].quantity
        == Quantity.of("100")
    )
    assert "order.adjusted" in {
        event.code.value for event in result.events
    }
    assert "order.filled" in {
        event.code.value for event in result.events
    }


def test_cancel_intent_flows_through_sizer_broker_and_recorder() -> None:
    dataset = _three_bar_dataset()
    session = _manual_session(dataset)
    observation = session.reset()
    instrument_id = next(iter(observation.market.bars))
    resting = LimitOrderIntent(
        instrument=instrument_id,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        limit_price=Money.usd("50"),
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        reason=DecisionReason.of("resting"),
    )
    first = session.advance((resting,))
    assert first.observation is not None
    order_id = first.observation.active_orders[0].id

    second = session.advance(
        (
            CancelOrderIntent(
                order_id=order_id,
                reason=DecisionReason.of("cancel_resting"),
            ),
        )
    )
    assert second.observation is not None
    assert second.observation.active_orders == ()
    session.advance(())
    result = session.result()

    assert result.fills == ()
    assert result.orders[0].status is OrderStatus.CANCELLED
    assert result.events[-1].code.value == "order.cancelled"


def test_backtest_service_uses_the_engine_execution_path() -> None:
    dataset = _two_bar_dataset()
    run_id = RunId.parse("run_" + "5" * 32)
    request = BacktestRequest(
        strategy=_BuyWhenFlat(Quantity.of("1")),
        simulation=_simulation(dataset),
        run_id=run_id,
    )
    engine = _engine(dataset)

    direct = engine.run(request)
    through_service = BacktestService(engine).run_python(request)

    assert through_service == direct
