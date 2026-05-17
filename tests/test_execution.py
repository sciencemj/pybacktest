import pandas as pd

from pybacktest.execution import ExecutionConfig, execute_actions
from pybacktest.models import Action, Portfolio, Stock


def _stock(ticker: str, close: float = 100.0, volume: int = 1000) -> Stock:
    stock = Stock(ticker, "2024-01-01", "2024-01-01", fetch=False)
    stock.data = pd.DataFrame(
        {"Close": [close], "Volume": [volume]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    return stock


def test_liquidity_limit_scales_buy_quantity():
    portfolio = Portfolio(100000.0, ["A"])
    trades, warnings = execute_actions(
        portfolio=portfolio,
        actions=[Action(ticker="A", type="buy", quantity=100, price=100.0)],
        stocks=[_stock("A", volume=1000)],
        date=pd.Timestamp("2024-01-01"),
        config=ExecutionConfig(liquidity_limit=0.05),
    )

    assert portfolio.stock_count["A"] == 50
    assert trades[0]["quantity"] == 50
    assert warnings == ["Scaled buy A from 100 to 50 due to liquidity limit."]


def test_sell_executes_before_buy_after_liquidity_scaling():
    portfolio = Portfolio(0.0, ["A", "B"])
    portfolio.stock_count["A"] = 100
    trades, warnings = execute_actions(
        portfolio=portfolio,
        actions=[
            Action(ticker="B", type="buy", quantity=100, price=100.0),
            Action(ticker="A", type="sell", quantity=100, price=100.0),
        ],
        stocks=[_stock("A", volume=10000), _stock("B", volume=10000)],
        date=pd.Timestamp("2024-01-01"),
        config=ExecutionConfig(),
    )

    assert [trade["type"] for trade in trades] == ["sell", "buy"]
    assert portfolio.stock_count == {"A": 0, "B": 100}
    assert warnings == []
