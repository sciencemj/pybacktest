"""Measured performance gate for one million engine observations.

The dataset is built outside the timed region, only ``engine.run()`` is timed,
and neither artifact serialization nor replay fingerprinting is executed here.
``resource.getrusage(RUSAGE_SELF)`` is a whole-process high-water mark, so this
module is meaningful only when run in its own process; see ``benchmarks/README.md``.
"""

import platform
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from benchmarks._rss import normalized_peak_rss_bytes
from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    BarSeries,
    CalendarPolicy,
    DateRange,
    DecisionReason,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LongShortRisk,
    MarketDataSet,
    MarketOrderIntent,
    MetricsConfig,
    Money,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    OrderSide,
    Quantity,
    SimulatedBrokerFactory,
    SimulationRequest,
    StrategyContext,
    Timeframe,
    TimeInForce,
)
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.market import MarketSlice

INSTRUMENT_COUNT = 100
BAR_COUNT = 10_000
OBSERVATION_COUNT = INSTRUMENT_COUNT * BAR_COUNT
ORDER_INTERVAL = 500
EXPECTED_ORDER_COUNT = 19
FAST_WINDOW = 10
SLOW_WINDOW = 50
RUNTIME_BUDGET_SECONDS = 30.0
PEAK_RSS_BUDGET_BYTES = 1_500_000_000

_FIRST_BAR = np.datetime64("2000-01-03T14:30:00", "ns")


def _timestamps() -> np.ndarray:
    return _FIRST_BAR + np.arange(BAR_COUNT, dtype="int64") * np.timedelta64(1, "D")


@dataclass(frozen=True, slots=True)
class _SparseMarketOrders:
    """Buy one share on a fixed, sparse schedule and hold everything else.

    The schedule is an immutable tuple of ``(nanoseconds, instrument)`` pairs
    rather than a remembered bar counter, so the strategy stays stateless and
    its configuration stays canonically serializable for provenance.
    """

    feature_symbol: str
    schedule: tuple[tuple[int, str], ...]

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        close = builder.source(
            "close",
            InstrumentId.parse(self.feature_symbol),
            "close",
        )
        builder.sma("fast", close, FAST_WINDOW)
        builder.sma("slow", close, SLOW_WINDOW)
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[MarketOrderIntent, ...]:
        del market
        moment = int(context.timestamp.astype("int64"))
        for scheduled, symbol in self.schedule:
            if scheduled == moment:
                return (
                    MarketOrderIntent(
                        instrument=InstrumentId.parse(symbol),
                        side=OrderSide.BUY,
                        quantity=Quantity.of("1"),
                        time_in_force=TimeInForce.DAY,
                        reason=DecisionReason.of("sparse_benchmark_order"),
                    ),
                )
        return ()


class _InMemorySource:
    """Return an already-constructed dataset so no ingestion is timed."""

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


def _instrument_ids() -> tuple[InstrumentId, ...]:
    return tuple(
        InstrumentId(venue="XNAS", symbol=f"S{index:04d}")
        for index in range(INSTRUMENT_COUNT)
    )


def _dataset(instrument_ids: tuple[InstrumentId, ...]) -> MarketDataSet:
    """Build tick-aligned synthetic bars on one shared daily grid."""
    generator = np.random.default_rng(11)
    timestamps = _timestamps()
    volume = np.full(BAR_COUNT, 1_000_000.0)
    series: dict[InstrumentId, BarSeries] = {}
    instruments: dict[InstrumentId, Instrument] = {}
    for instrument_id in instrument_ids:
        steps = generator.normal(0.0, 0.005, BAR_COUNT)
        # Rounding to the instrument tick is not cosmetic: an unaligned mark
        # is rejected by the risk policy, and the benchmark would then measure
        # a no-fill path. The fill assertion below guards that regression.
        close = np.round(100.0 * np.exp(np.cumsum(steps)), 2)
        series[instrument_id] = BarSeries(
            timestamps=timestamps,
            open=close,
            high=np.round(close * 1.01, 2),
            low=np.round(close * 0.99, 2),
            close=close,
            volume=volume,
        )
        instruments[instrument_id] = Instrument(
            id=instrument_id,
            quote_currency="USD",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("1"),
            timezone=UTC,
        )
    return MarketDataSet(
        series=series,
        instruments=instruments,
        timeframe=Timeframe.days(1),
    )


def _request(
    instrument_ids: tuple[InstrumentId, ...],
) -> BacktestRequest:
    timestamps = _timestamps()
    schedule = tuple(
        (
            int(timestamps[step * ORDER_INTERVAL].astype("int64")),
            str(instrument_ids[(step - 1) % INSTRUMENT_COUNT]),
        )
        for step in range(1, EXPECTED_ORDER_COUNT + 1)
    )
    return BacktestRequest(
        strategy=_SparseMarketOrders(
            feature_symbol=str(instrument_ids[0]),
            schedule=schedule,
        ),
        simulation=SimulationRequest(
            universe=instrument_ids,
            period=DateRange(
                datetime(2000, 1, 1, tzinfo=UTC),
                datetime(2030, 1, 1, tzinfo=UTC),
            ),
            timeframe=Timeframe.days(1),
            calendar=CalendarPolicy.union(max_staleness_bars=0),
            initial_cash=Money.usd("10000000"),
            seed=11,
            metrics=MetricsConfig(
                risk_free_rate=Decimal("0"),
                annualization_periods=252,
            ),
        ),
    )


def _engine(dataset: MarketDataSet) -> BacktestEngine:
    return BacktestEngine(
        data_source=_InMemorySource(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
            commission=NoCommission(),
            slippage=NoSlippage(),
            liquidity=NoLiquidityLimit(),
            borrow_cost=NoBorrowCost(),
        ),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("2"),
            max_position_weight=None,
            allow_short=False,
        ),
    )


@pytest.mark.performance
def test_one_million_observations_stay_within_the_runtime_and_memory_budget() -> None:
    assert INSTRUMENT_COUNT == 100
    assert BAR_COUNT == 10_000
    assert OBSERVATION_COUNT == 1_000_000

    instrument_ids = _instrument_ids()
    dataset = _dataset(instrument_ids)
    assert len(dataset.series) == 100
    assert all(len(series.timestamps) == 10_000 for series in dataset.series.values())
    assert (
        sum(len(series.timestamps) for series in dataset.series.values()) == 1_000_000
    )

    engine = _engine(dataset)
    request = _request(instrument_ids)

    started = time.perf_counter()
    result = engine.run(request)
    elapsed_seconds = time.perf_counter() - started

    peak_rss_bytes = normalized_peak_rss_bytes(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform.system(),
    )
    print(
        f"\nobservations={OBSERVATION_COUNT} "
        f"instruments={INSTRUMENT_COUNT} bars={BAR_COUNT} "
        f"orders={len(result.orders)} fills={len(result.fills)} "
        f"elapsed_seconds={elapsed_seconds:.2f} "
        f"peak_rss_bytes={peak_rss_bytes} "
        f"platform={platform.system()}"
    )

    assert len(dataset.timestamps) == BAR_COUNT
    assert len(result.market_timestamps) == BAR_COUNT
    assert len(result.orders) == EXPECTED_ORDER_COUNT
    assert len(result.fills) > 0
    assert elapsed_seconds <= RUNTIME_BUDGET_SECONDS
    assert peak_rss_bytes <= PEAK_RSS_BUDGET_BYTES
