"""Integration smoke tests exercising Backtest with combined config kwargs.

These tests cover the README's recommended usage pattern of passing
``rebalance``, ``execution``, and ``projection`` together to ``Backtest``.
"""

from __future__ import annotations

import pandas as pd

from pybacktest.backtest import Backtest
from pybacktest.models import Stock
from pybacktest.strategy import StrategyManager, StrategyWrapper


def _build_stock(ticker: str, start_price: float, drift: float) -> Stock:
    """Build a Stock with deterministic business-day OHLCV data."""
    dates = pd.bdate_range(start="2024-01-01", end="2024-02-29")
    closes = [start_price + drift * i for i in range(len(dates))]
    data = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [100_000] * len(dates),
        },
        index=dates,
    )
    data["Change"] = data["Close"] - data["Close"].shift(1)
    data["Change_Pct"] = data["Close"].pct_change()

    stock = Stock(ticker, start="2024-01-01", end="2024-02-29", fetch=False)
    stock.data = data
    stock.dates = data.index.to_list()
    return stock


def test_backtest_combined_config_smoke():
    stock_a = _build_stock("AAA", start_price=100.0, drift=0.5)
    stock_b = _build_stock("BBB", start_price=200.0, drift=0.75)

    strategy_json = {
        "AAA": {
            "buy": {
                "ticker": "AAA",
                "indicator": ["current", "Close"],
                "window": False,
                "threshold": ["point", 0],
                "quantity": ["count", 5],
                "price_point": "Close",
            },
            "sell": {
                "ticker": "AAA",
                "indicator": ["current", "Change_Pct"],
                "window": False,
                "threshold": ["percent-change", -0.5],
                "quantity": ["percent", 100],
                "price_point": "Close",
            },
            "portfolio_weight": 0.5,
        },
        "BBB": {
            "buy": {
                "ticker": "BBB",
                "indicator": ["current", "Close"],
                "window": False,
                "threshold": ["point", 0],
                "quantity": ["count", 3],
                "price_point": "Close",
            },
            "sell": {
                "ticker": "BBB",
                "indicator": ["current", "Change_Pct"],
                "window": False,
                "threshold": ["percent-change", -0.5],
                "quantity": ["percent", 100],
                "price_point": "Close",
            },
            "portfolio_weight": 0.5,
        },
    }

    strategy_name = "AAA+BBB combined"
    strategy = StrategyManager(
        strategy_name,
        StrategyWrapper.model_validate(strategy_json),
    )

    backtest = Backtest(
        [stock_a, stock_b],
        [strategy],
        initial_capital=100_000.0,
        rebalance={
            "enabled": True,
            "frequency": "monthly",
            "day": 15,
            "mode": "full",
        },
        execution={"liquidity_limit": 0.5},
        projection={
            "enabled": True,
            "days": 10,
            "scenarios": 25,
            "lookback": 20,
            "random_seed": 11,
        },
    )

    expected_day_count = len(backtest.dates)
    result = backtest.run()

    assert strategy_name in result.strategies
    strategy_result = result.strategies[strategy_name]

    projection = strategy_result.projection
    assert projection is not None
    assert isinstance(projection, pd.DataFrame)
    assert list(projection.columns) == ["low", "median", "high"]
    assert len(projection) == 10

    assert len(strategy_result.daily_snapshots) == expected_day_count

    assert isinstance(strategy_result.warnings, list)

    rebalance_date = pd.Timestamp("2024-01-15")
    trades_on_rebalance = [
        trade
        for trade in strategy_result.trades
        if pd.Timestamp(trade["date"]) == rebalance_date
    ]
    assert trades_on_rebalance, (
        "Expected at least one trade on 2024-01-15 from the rebalance hook."
    )
