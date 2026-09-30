"""Run a demo configuration through the core engine and shape the results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd

import pybacktest
from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    BacktestResult,
    CalendarPolicy,
    DateRange,
    IntrabarPolicy,
    LongShortRisk,
    MarketDataSet,
    MetricName,
    MetricsConfig,
    Money,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    PerShareCommission,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
)
from streamlit_ui.errors import DemoInputError
from streamlit_ui.market_data import TickerHistory, instrument_id, to_market_dataset
from streamlit_ui.strategies import (
    BuyAndHold,
    DemoStrategy,
    MovingAverageCrossAll,
    RsiReversion,
)

STRATEGY_NAMES = ("buy_and_hold", "ma_cross", "rsi")
# A held instrument may miss up to this many union bars (a local holiday, a
# trading halt, or a bar yfinance dropped) before the core rejects its stale
# mark. The limit only bounds how long a held position keeps its last price.
MAX_STALENESS_BARS = 30
METRIC_ROWS = (
    MetricName.TOTAL_RETURN,
    MetricName.CAGR,
    MetricName.VOLATILITY,
    MetricName.SHARPE,
    MetricName.SORTINO,
    MetricName.MAXIMUM_DRAWDOWN,
    MetricName.WIN_RATE,
    MetricName.TURNOVER,
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything the sidebar collects for one run."""

    tickers: tuple[str, ...]
    start: date
    end: date
    strategy: str
    params: Mapping[str, int] = field(default_factory=dict)
    initial_cash: Decimal = Decimal("10000")
    commission_per_share: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class RunReport:
    """Display-ready results for the chosen strategy and its benchmark."""

    metrics: pd.DataFrame
    equity: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    warnings: pd.DataFrame
    info: dict[str, object]


class _InMemoryData:
    """A data source that owns one already-validated dataset."""

    def __init__(self, dataset: MarketDataSet) -> None:
        self._dataset = dataset

    def load(
        self,
        universe: object,
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        del universe, period, timeframe
        return self._dataset


def build_strategy(config: RunConfig) -> DemoStrategy:
    """Construct the configured demo strategy."""
    instruments = tuple(instrument_id(ticker) for ticker in config.tickers)
    params = dict(config.params)
    if config.strategy == "buy_and_hold":
        return BuyAndHold(instruments=instruments)
    if config.strategy == "ma_cross":
        return MovingAverageCrossAll(
            instruments=instruments,
            fast=params["fast"],
            slow=params["slow"],
        )
    if config.strategy == "rsi":
        return RsiReversion(
            instruments=instruments,
            period=params["period"],
            lower=params["lower"],
            upper=params["upper"],
        )
    raise ValueError(f"unknown strategy: {config.strategy!r}")


def _engine(dataset: MarketDataSet, config: RunConfig) -> BacktestEngine:
    commission = (
        NoCommission()
        if config.commission_per_share == 0
        else PerShareCommission(rate_per_share=config.commission_per_share)
    )
    return BacktestEngine(
        data_source=_InMemoryData(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
            commission=commission,
            slippage=NoSlippage(),
            liquidity=NoLiquidityLimit(),
            borrow_cost=NoBorrowCost(),
        ),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=False,
        ),
    )


def _simulation(config: RunConfig, currency: str) -> SimulationRequest:
    start = datetime(
        config.start.year, config.start.month, config.start.day, tzinfo=UTC
    )
    end = datetime(config.end.year, config.end.month, config.end.day, tzinfo=UTC)
    return SimulationRequest(
        universe=tuple(instrument_id(ticker) for ticker in config.tickers),
        period=DateRange(start, end + timedelta(days=1)),
        timeframe=Timeframe.days(1),
        calendar=CalendarPolicy.union(max_staleness_bars=MAX_STALENESS_BARS),
        initial_cash=Money.of(config.initial_cash, currency),
        seed=0,
        metrics=MetricsConfig(
            risk_free_rate=Decimal("0"),
            annualization_periods=252,
        ),
    )


def require_affordable(dataset: MarketDataSet, config: RunConfig) -> None:
    """Reject a budget too small to ever buy one share of an instrument.

    Each instrument targets ``1/N`` of the cash and shares are whole, so an
    instrument whose lowest close exceeds that budget can never be bought and
    would silently produce an empty result.
    """
    budget = config.initial_cash / len(config.tickers)
    unaffordable = []
    for ticker in config.tickers:
        lowest = Decimal(repr(float(dataset.series[instrument_id(ticker)].close.min())))
        if lowest > budget:
            unaffordable.append(f"{ticker}: {lowest:f} > {budget:.2f}")
    if unaffordable:
        raise DemoInputError("insufficient_cash", "; ".join(unaffordable))


def run_backtest(
    config: RunConfig,
    histories: Mapping[str, TickerHistory],
) -> RunReport:
    """Run ``config`` and a buy-and-hold benchmark on the same dataset."""
    strategy = build_strategy(config)
    benchmark = BuyAndHold(instruments=strategy.instruments)
    dataset, currency = to_market_dataset(histories, min_bars=strategy.warmup_bars)
    require_affordable(dataset, config)
    engine = _engine(dataset, config)
    simulation = _simulation(config, currency)
    result = engine.run(BacktestRequest(strategy=strategy, simulation=simulation))
    base = engine.run(BacktestRequest(strategy=benchmark, simulation=simulation))
    return RunReport(
        metrics=metrics_table(result, base),
        equity=equity_frame(result, base),
        orders=orders_table(result),
        fills=fills_table(result),
        warnings=warnings_table(result, base),
        info={
            "pybacktest": pybacktest.__version__,
            "currency": currency,
            "bars": {str(i): len(s.timestamps) for i, s in dataset.series.items()},
            "strategy": config.strategy,
            "params": dict(config.params),
            "period": f"{config.start.isoformat()} to {config.end.isoformat()}",
        },
    )


def _metric(result: BacktestResult, name: MetricName) -> float | None:
    value = result.summary.result_for(name).value
    return None if value is None else float(value)


def metrics_table(result: BacktestResult, benchmark: BacktestResult) -> pd.DataFrame:
    """One row per metric with ``strategy`` and ``benchmark`` columns."""
    return pd.DataFrame(
        {
            "strategy": [_metric(result, name) for name in METRIC_ROWS],
            "benchmark": [_metric(benchmark, name) for name in METRIC_ROWS],
        },
        index=[name.value for name in METRIC_ROWS],
    )


def _equity(result: BacktestResult) -> pd.Series:
    points = {
        snapshot.timestamp: float(snapshot.equity.amount)
        for snapshot in result.snapshots
        if snapshot.timestamp is not None
    }
    series = pd.Series(points, dtype="float64")
    series.index = pd.DatetimeIndex(series.index)
    return series


def equity_frame(result: BacktestResult, benchmark: BacktestResult) -> pd.DataFrame:
    """Equity per timestamp for the strategy and the benchmark."""
    frame = pd.DataFrame(
        {"strategy": _equity(result), "benchmark": _equity(benchmark)}
    ).sort_index()
    frame.index.name = "timestamp"
    return frame


def orders_table(result: BacktestResult) -> pd.DataFrame:
    """Orders with their final status."""
    return pd.DataFrame(
        [
            {
                "submitted_at": order.submitted_at,
                "instrument": str(order.instrument),
                "side": order.side.value,
                "quantity": float(order.quantity.value),
                "filled": float(order.filled_quantity.value),
                "status": order.status.value,
            }
            for order in result.orders
        ],
        columns=["submitted_at", "instrument", "side", "quantity", "filled", "status"],
    )


def fills_table(result: BacktestResult) -> pd.DataFrame:
    """Executions with price and fee."""
    return pd.DataFrame(
        [
            {
                "timestamp": fill.timestamp,
                "instrument": str(fill.instrument),
                "side": fill.side.value,
                "quantity": float(fill.quantity.value),
                "price": float(fill.price.amount),
                "fee": float(fill.fee.amount),
            }
            for fill in result.fills
        ],
        columns=["timestamp", "instrument", "side", "quantity", "price", "fee"],
    )


def warnings_table(result: BacktestResult, benchmark: BacktestResult) -> pd.DataFrame:
    """Warnings from both runs, tagged with their source."""
    rows = [
        {"run": run, "code": warning.code.value, "message": warning.message}
        for run, source in (("strategy", result), ("benchmark", benchmark))
        for warning in source.warnings
    ]
    return pd.DataFrame(rows, columns=["run", "code", "message"])
