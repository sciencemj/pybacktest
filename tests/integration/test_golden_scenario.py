"""A five-bar, hand-computed reference run for the whole core pipeline.

Every expected value below is a literal taken from the hand computation in the
module docstring table, never from a production helper. This is the behavioral
reference future adapters are checked against, so a regression in same-bar
execution, lookahead, partial-fill ordering, commission, slippage, order
replacement, ledger reconciliation, or causal attribution must break it.

Hand computation (``fast = close``; ``slow`` = mean of the last two closes;
``*_prev`` = one-bar lag)::

    #  date        open  high  low  close  volume   fast   slow  fast_prev slow_prev
    0  2024-01-02   100   101   99    100    1000    100    nan        nan       nan
    1  2024-01-03   100   101   97     98    1000     98     99        100       nan
    2  2024-01-04    98   103   97    102    1000    102    100         98        99
    3  2024-01-05   104   105  103    104     400    104    103        102       100
    4  2024-01-08   106   107  105    106    2000    106    105        104       103

    bar 2: 98 <= 99 and 102 > 100                      -> cross_up
           equity 10000 x weight 0.5 = 5000 notional
           floor(5000 / 102) = 49 lots, flat position  -> BUY 49 MARKET
           49 x 102 = 4998 <= 10000 cash and <= equity -> risk PASSED
           scheduled active_from 2024-01-05            -> no same-bar fill
    bar 3: capacity 400 x 0.05 = 20                    -> 20 @ 104.00, fee 0.100
    bar 4: capacity 2000 x 0.05 = 100 >= 29            -> 29 @ 106.00, fee 0.145

    cash   = 10000 - 20x104 - 0.100 - 29x106 - 0.145   = 4845.755
    equity = 4845.755 + 49 x 106                       = 10039.755
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    BacktestResult,
    BarSeries,
    CalendarPolicy,
    DateRange,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LongShortRisk,
    MarketDataSet,
    MetricName,
    MetricsConfig,
    Money,
    MovingAverageCross,
    NextBarOpenFill,
    NoBorrowCost,
    NoSlippage,
    OrderSide,
    OrderStatus,
    OrderType,
    PerShareCommission,
    Quantity,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
    TimeInForce,
    VolumeParticipationLimit,
    WarningCode,
)
from pybacktest.domain.events import OrderRejected
from pybacktest.domain.orders import CancelOrderIntent, Order
from pybacktest.ports.risk import RiskContext, SizedOrderIntent
from pybacktest.risk.sizing import DefaultOrderSizer

AAPL = InstrumentId.parse("XNAS:AAPL")

BAR_TIMESTAMPS = (
    datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
    datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
    datetime(2024, 1, 4, 14, 30, tzinfo=UTC),
    datetime(2024, 1, 5, 14, 30, tzinfo=UTC),
    datetime(2024, 1, 8, 14, 30, tzinfo=UTC),
)
OPENS = (100.0, 100.0, 98.0, 104.0, 106.0)
HIGHS = (101.0, 101.0, 103.0, 105.0, 107.0)
LOWS = (99.0, 97.0, 97.0, 103.0, 105.0)
CLOSES = (100.0, 98.0, 102.0, 104.0, 106.0)
VOLUMES = (1000.0, 1000.0, 1000.0, 400.0, 2000.0)


@dataclass(frozen=True, slots=True)
class _GoodTilCancelledSizer:
    """Delegate to the production sizer and only widen DAY targets to GTC.

    ``DefaultOrderSizer`` gives every ``TargetWeight`` a DAY time in force, so
    a partially filled target expires on the next session boundary instead of
    completing. This test-local override is passed through the existing public
    ``engine.run(..., order_sizer=...)`` parameter; production default sizing
    semantics are unchanged.
    """

    delegate: DefaultOrderSizer = field(default_factory=DefaultOrderSizer)

    def size(
        self,
        intent: SizedOrderIntent | CancelOrderIntent,
        context: RiskContext,
    ) -> Order | OrderRejected | CancelOrderIntent:
        sized = self.delegate.size(intent, context)
        if isinstance(sized, Order) and sized.time_in_force is TimeInForce.DAY:
            return replace(sized, time_in_force=TimeInForce.GOOD_TIL_CANCELLED)
        return sized


class _FixedSource:
    """Serve one already-constructed dataset without any I/O."""

    def __init__(self, dataset: MarketDataSet) -> None:
        self._dataset = dataset

    def load(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        del universe, period, timeframe
        return self._dataset


def _dataset() -> MarketDataSet:
    return MarketDataSet(
        series={
            AAPL: BarSeries(
                timestamps=np.asarray(
                    [
                        np.datetime64(moment.replace(tzinfo=None), "ns")
                        for moment in BAR_TIMESTAMPS
                    ],
                    dtype="datetime64[ns]",
                ),
                open=np.asarray(OPENS, dtype=np.float64),
                high=np.asarray(HIGHS, dtype=np.float64),
                low=np.asarray(LOWS, dtype=np.float64),
                close=np.asarray(CLOSES, dtype=np.float64),
                volume=np.asarray(VOLUMES, dtype=np.float64),
            )
        },
        instruments={
            AAPL: Instrument(
                id=AAPL,
                quote_currency="USD",
                tick_size=Decimal("0.01"),
                lot_size=Decimal("1"),
                timezone=UTC,
            )
        },
        timeframe=Timeframe.days(1),
    )


def _golden_result() -> BacktestResult:
    dataset = _dataset()
    engine = BacktestEngine(
        data_source=_FixedSource(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
            commission=PerShareCommission(rate_per_share=Decimal("0.005")),
            slippage=NoSlippage(),
            liquidity=VolumeParticipationLimit(max_volume_ratio=Decimal("0.05")),
            borrow_cost=NoBorrowCost(),
        ),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=False,
        ),
    )
    request = BacktestRequest(
        strategy=MovingAverageCross(
            fast=1,
            slow=2,
            long_weight=Decimal("0.5"),
            flat_weight=Decimal("0"),
            instrument=AAPL,
        ),
        simulation=SimulationRequest(
            universe=(AAPL,),
            period=DateRange(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 9, tzinfo=UTC),
            ),
            timeframe=Timeframe.days(1),
            calendar=CalendarPolicy.union(),
            initial_cash=Money.usd("10000"),
            seed=7,
            metrics=MetricsConfig(
                risk_free_rate=Decimal("0"),
                annualization_periods=252,
            ),
        ),
    )
    return engine.run(request, order_sizer=_GoodTilCancelledSizer())


def test_market_clock_is_exactly_the_five_bars() -> None:
    assert _golden_result().market_timestamps == BAR_TIMESTAMPS


def test_one_replaced_order_is_recorded_as_a_filled_gtc_market_buy() -> None:
    (order,) = _golden_result().orders

    assert order.instrument == AAPL
    assert order.side is OrderSide.BUY
    assert order.type is OrderType.MARKET
    assert order.time_in_force is TimeInForce.GOOD_TIL_CANCELLED
    assert order.quantity == Quantity.of("49")
    assert order.filled_quantity == Quantity.of("49")
    assert order.status is OrderStatus.FILLED
    assert order.limit_price is None
    assert order.reason.code == "ma_cross"
    assert order.reason.details["outcome"] == "cross_up"
    assert order.submitted_at == datetime(2024, 1, 4, 14, 30, tzinfo=UTC)
    assert order.active_from == datetime(2024, 1, 5, 14, 30, tzinfo=UTC)


def test_the_crossover_bar_does_not_execute_the_order() -> None:
    result = _golden_result()

    assert all(
        fill.timestamp > datetime(2024, 1, 4, 14, 30, tzinfo=UTC)
        for fill in result.fills
    )
    assert all(
        fill.price.amount not in (Decimal("102"), Decimal("98"))
        for fill in result.fills
    )


def test_liquidity_capped_partial_then_remaining_fill_at_the_bar_opens() -> None:
    result = _golden_result()
    first, second = result.fills

    assert first.timestamp == datetime(2024, 1, 5, 14, 30, tzinfo=UTC)
    assert first.quantity == Quantity.of("20")
    assert first.price == Money.usd("104.00")
    assert first.fee == Money.usd("0.100")
    assert first.side is OrderSide.BUY

    assert second.timestamp == datetime(2024, 1, 8, 14, 30, tzinfo=UTC)
    assert second.quantity == Quantity.of("29")
    assert second.price == Money.usd("106.00")
    assert second.fee == Money.usd("0.145")
    assert second.side is OrderSide.BUY

    assert first.order_id == second.order_id == result.orders[0].id


def test_final_ledger_reconciles_to_the_hand_computed_literals() -> None:
    final = _golden_result().snapshots[-1]

    assert final.cash == Money.usd("4845.755")
    assert final.positions[AAPL].quantity == Quantity.of("49")
    assert final.equity == Money.usd("10039.755")
    assert final.total_fees == Money.usd("0.245")


def test_pinned_metrics_and_the_unavailable_win_rate_warning() -> None:
    result = _golden_result()
    summary = result.summary

    assert summary.total_return == Decimal("0.0039755")
    assert summary.maximum_drawdown == Decimal("-0.00001")
    assert summary.win_rate is None
    (warning,) = summary.warnings
    assert warning.code == WarningCode.of("metric.unavailable_closing_legs")
    assert warning.metric is MetricName.WIN_RATE
    assert result.warnings == summary.warnings


def test_the_causal_chain_for_the_order_is_exactly_nine_attributed_entries() -> None:
    result = _golden_result()
    order_id = result.orders[0].id
    explanation = result.explain_trade(order_id)

    assert explanation.order_id == order_id
    assert tuple(
        (entry.stage.value, entry.code.value, entry.timestamp)
        for entry in explanation.entries
    ) == (
        ("intent", "intent.received", datetime(2024, 1, 4, 14, 30, tzinfo=UTC)),
        ("sizing", "order.sized", datetime(2024, 1, 4, 14, 30, tzinfo=UTC)),
        ("risk", "risk.passed", datetime(2024, 1, 4, 14, 30, tzinfo=UTC)),
        ("scheduling", "order.scheduled", datetime(2024, 1, 4, 14, 30, tzinfo=UTC)),
        ("broker", "order.accepted", datetime(2024, 1, 4, 14, 30, tzinfo=UTC)),
        ("broker", "order.partial_fill", datetime(2024, 1, 5, 14, 30, tzinfo=UTC)),
        ("accounting", "ledger.applied", datetime(2024, 1, 5, 14, 30, tzinfo=UTC)),
        ("broker", "order.filled", datetime(2024, 1, 8, 14, 30, tzinfo=UTC)),
        ("accounting", "ledger.applied", datetime(2024, 1, 8, 14, 30, tzinfo=UTC)),
    )
    assert all(entry.order_id == order_id for entry in explanation.entries)
    assert [entry.sequence for entry in explanation.entries] == sorted(
        entry.sequence for entry in explanation.entries
    )


def test_the_replay_fingerprint_is_stable_across_identical_runs() -> None:
    assert _golden_result().replay_fingerprint() == (
        _golden_result().replay_fingerprint()
    )
