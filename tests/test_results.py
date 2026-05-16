import pandas as pd

from pybacktest.backtest import Backtest
from pybacktest.models import Action, Portfolio, Stock
from pybacktest.strategy import Strategy


def _stock(ticker: str) -> Stock:
    stock = Stock(ticker, "2024-01-01", "2024-01-03", fetch=False)
    stock.data = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 100.0, 100.0],
            "Low": [100.0, 100.0, 100.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1000, 1000, 1000],
            "Change": [None, 0.0, 0.0],
            "Change_Pct": [None, 0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    return stock


def test_run_returns_results_by_strategy_without_mixing_snapshots():
    first = Strategy("first", lambda p, s, d: [Action(ticker="A", type="buy", quantity=1, price=100.0)] if d.day == 1 else [])
    second = Strategy("second", lambda p, s, d: [])
    backtest = Backtest([_stock("A")], [first, second], initial_capital=1000.0)

    results = backtest.run()

    assert set(results.strategies.keys()) == {"first", "second"}
    assert len(results.strategies["first"].daily_snapshots) == 3
    assert len(results.strategies["second"].daily_snapshots) == 3
    assert results.strategies["first"].daily_snapshots[-1]["Stock_Amount_A"] == 1
    assert results.strategies["second"].daily_snapshots[-1]["Stock_Amount_A"] == 0
