import pandas as pd

from pybacktest.models import Portfolio, Stock
from pybacktest.strategy import (
    StrategyConfig,
    StrategyManager,
    StrategyWrapper,
    TradeAction,
)


def test_split_purchase():
    # Setup
    initial_capital = 10000.0
    portfolio = Portfolio(initial_capital, ["AAPL"])

    # Strategy setup
    buy_action = TradeAction(
        ticker="AAPL",
        indicator=["current", "Close"],
        window=False,
        threshold=["point", 0],  # Always buy
        quantity=["split", 10],
        price_point="Close",
    )
    sell_action = TradeAction(
        ticker="AAPL",
        indicator=["current", "Close"],
        window=False,
        threshold=["point", 10000],
        quantity=["percent", 100],
        price_point="Close",
    )
    config = StrategyConfig(buy=buy_action, sell=sell_action, portfolio_weight=0.5)

    wrapper = StrategyWrapper(root={"AAPL": config})
    manager = StrategyManager("Test", wrapper)

    # Mock Stock Data
    stock = Stock("AAPL", start="2023-01-01", end="2023-01-02", fetch=False)
    stock.data = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2023-01-01"]))

    # Apply
    actions = manager.apply(portfolio, [stock], pd.to_datetime("2023-01-01"))

    # Validate
    # Initial Capital = 10000
    # Split = 10 -> 1000 per split base
    # weight = 0.5 -> 500 target value
    # Price = 100
    # Expected quantity (shares) = 500 // 100 = 5

    assert len(actions) == 1
    assert actions[0].type == "buy"
    print(f"Action Quantity: {actions[0].quantity}")
    assert actions[0].quantity == 5


def test_rebalancing():
    # Setup
    initial_capital = 10000.0
    portfolio = Portfolio(initial_capital, ["A", "B"])

    # Initial State:
    # A: 80 shares @ 100 = 8000
    # B: 20 shares @ 100 = 2000
    # Total = 10000 (assuming cash 0)
    portfolio.stock_count["A"] = 80
    portfolio.stock_count["B"] = 20
    portfolio.cash = 0

    # Mock Stocks
    stock_a = Stock("A", "2023-01-01", "2023-01-31", False)
    stock_a.data = pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2023-01-31"])
    )
    stock_b = Stock("B", "2023-01-01", "2023-01-31", False)
    stock_b.data = pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2023-01-31"])
    )

    # Strategies with weights 70:30
    config_a = StrategyConfig(
        buy=TradeAction(
            ticker="A",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 0],
            quantity=["percent", 100],
        ),
        sell=TradeAction(
            ticker="A",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 9999],
            quantity=["percent", 100],
        ),
        portfolio_weight=0.7,
    )
    config_b = StrategyConfig(
        buy=TradeAction(
            ticker="B",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 0],
            quantity=["percent", 100],
        ),
        sell=TradeAction(
            ticker="B",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 9999],
            quantity=["percent", 100],
        ),
        portfolio_weight=0.3,
    )

    wrapper = StrategyWrapper(root={"A": config_a, "B": config_b})
    manager = StrategyManager("Test", wrapper)

    # Apply Rebalancing (Use month end)
    date = pd.to_datetime("2023-01-31")  # Is month end

    rebalance_actions = manager.rebalance(portfolio, [stock_a, stock_b], date)

    # Analysis
    # Total Value = 10000
    # A Target: 7000. Current: 8000. Diff: -1000 (Sell 10 shares)
    # B Target: 3000. Current: 2000. Diff: +1000 (Buy 10 shares)

    print("Rebalance Actions:", rebalance_actions)

    # Should be [Sell A, Buy B] (Sell sorted first)
    assert len(rebalance_actions) == 2
    assert rebalance_actions[0].ticker == "A"
    assert rebalance_actions[0].type == "sell"
    # value 1000 -> quantity 10
    assert rebalance_actions[0].quantity == 10

    assert rebalance_actions[1].ticker == "B"
    assert rebalance_actions[1].type == "buy"
    assert rebalance_actions[1].quantity == 10


def test_monthly_snapshots():
    import pandas as pd

    from pybacktest.backtest import Backtest
    from pybacktest.models import Stock
    from pybacktest.strategy import StrategyManager, StrategyWrapper

    # Mock Data: 35 days of data (covering 2 months)
    dates = pd.date_range(start="2023-01-01", periods=40, freq="D")
    data = pd.DataFrame({"Close": [100.0] * 40}, index=dates)

    stock = Stock("AAPL", "2023-01-01", "2023-02-09", fetch=False)
    stock.data = data

    # Empty Strategy
    strategy = StrategyManager("Test", StrategyWrapper(root={}))

    backtest = Backtest([stock], [strategy], initial_capital=10000.0)
    backtest.run(end_date="2023-02-09")

    monthly_df = backtest.get_monthly_snapshots()

    # Should have 2 rows (Jan 31, Feb 28 - if Feb has data)
    # Our data goes to Feb 09.
    # resample('ME') will bucket Jan 1-31 to Jan 31.
    # Feb 1-09 will be bucketed to Feb 28 (even if date is future relative to data).
    # So we expect 2 rows.

    assert len(monthly_df) >= 2
    assert "Cash" in monthly_df.columns
    assert "Total_Value" in monthly_df.columns
    assert "Stock_Amount_AAPL" in monthly_df.columns
    assert "Stock_Value_AAPL" in monthly_df.columns
