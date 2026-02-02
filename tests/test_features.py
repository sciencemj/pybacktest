import pandas as pd

from pybacktest.backtest import Backtest
from pybacktest.models import Action, Portfolio, Stock
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

    # Should be [Sell A] (Buy B is disabled)
    assert len(rebalance_actions) == 1
    assert rebalance_actions[0].ticker == "A"
    assert rebalance_actions[0].type == "sell"
    # value 1000 -> quantity 10
    assert rebalance_actions[0].quantity == 10

    # Buy B expectation removed


def test_monthly_snapshots():
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


def test_fair_cash_allocation():
    # Scenario 1: Proportional Allocation
    # Cash 10,000. Buy A, B, C (Price 100). Qty 50 each. Total Cost 15,000.
    # Ratio = 10000/15000 = 0.666. New Qty = 33.

    portfolio = Portfolio(10000.0, ["A", "B", "C"])
    date = pd.to_datetime("2023-01-01")

    # Mocking Backtest instance partially or just testing execute_action via Backtest logic?
    # Better to test logic directly via a strategy or manual invocation if possible.
    # However, execute_action is method of Backtest. We need Backtest instance.

    stock_a = Stock("A", "2023-01-01", "2023-01-02", fetch=False)
    stock_b = Stock("B", "2023-01-01", "2023-01-02", fetch=False)
    stock_c = Stock("C", "2023-01-01", "2023-01-02", fetch=False)

    dummy_strategy = StrategyManager("Test", StrategyWrapper(root={}))

    backtest = Backtest(
        [stock_a, stock_b, stock_c], [dummy_strategy], initial_capital=10000.0
    )
    # Inject portfolio
    backtest.portfolio = portfolio

    actions = [
        Action(ticker="A", type="buy", quantity=50, price=100.0),
        Action(ticker="B", type="buy", quantity=50, price=100.0),
        Action(ticker="C", type="buy", quantity=50, price=100.0),
    ]

    backtest.execute_action(actions, date, dummy_strategy)

    # Check trades or portfolio
    # A, B, C should have 33 shares each
    print(backtest.portfolio.stock_count)
    assert backtest.portfolio.stock_count["A"] == 33
    assert backtest.portfolio.stock_count["B"] == 33
    assert backtest.portfolio.stock_count["C"] == 33

    # Scenario 2: Sell Priority
    # Cash 0. Hold 100 A (Price 100). Sell 100 A. Buy 100 B (Price 100).
    portfolio_2 = Portfolio(0.0, ["A", "B"])
    portfolio_2.stock_count["A"] = 100
    backtest.portfolio = portfolio_2

    actions_2 = [
        Action(ticker="B", type="buy", quantity=100, price=100.0),
        Action(ticker="A", type="sell", quantity=100, price=100.0),
    ]

    backtest.execute_action(actions_2, date, dummy_strategy)

    assert backtest.portfolio.stock_count["A"] == 0
    assert backtest.portfolio.stock_count["B"] == 100
    assert backtest.portfolio.cash == 0.0


def test_rebalancing_mid_month():
    # Setup
    initial_capital = 10000.0
    portfolio = Portfolio(initial_capital, ["A", "B"])

    # Initial State:
    # A: 100 shares @ 100 = 10000 (100% allocation)
    # B: 0 shares
    # Cash: 0
    portfolio.stock_count["A"] = 100
    portfolio.stock_count["B"] = 0
    portfolio.cash = 0

    # Mock Stocks
    stock_a = Stock("A", "2023-01-01", "2023-01-31", fetch=False)
    stock_a.data = pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2023-01-15"])
    )
    stock_b = Stock("B", "2023-01-01", "2023-01-31", fetch=False)
    stock_b.data = pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2023-01-15"])
    )

    # Strategy with 50-50 weights
    config_a = StrategyConfig(
        buy=TradeAction(
            ticker="A",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 0],
            quantity=["percent", 0],
        ),  # No op
        sell=TradeAction(
            ticker="A",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 9999],
            quantity=["percent", 0],
        ),  # No op
        portfolio_weight=0.5,
    )
    config_b = StrategyConfig(
        buy=TradeAction(
            ticker="B",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 0],
            quantity=["percent", 0],
        ),
        sell=TradeAction(
            ticker="B",
            indicator=["current", "Close"],
            window=False,
            threshold=["point", 9999],
            quantity=["percent", 0],
        ),
        portfolio_weight=0.5,
    )

    wrapper = StrategyWrapper(root={"A": config_a, "B": config_b})
    manager = StrategyManager("Test", wrapper)

    # Test 1: Mid-month Trigger (15th)
    date_15th = pd.to_datetime("2023-01-15")

    actions_15th = manager.apply(portfolio, [stock_a, stock_b], date_15th)

    # Expectation:
    # A is 10000, target 5000. Diff -5000. Should Sell 50 shares of A.
    # B is 0, target 5000. Diff +5000. Buy is DISABLED -> No Buy B.
    # Result: [Sell A] (plus any strategy actions, but here we discard 0 qty)

    valid_actions = [a for a in actions_15th if a.quantity > 0]

    print(f"Actions on 15th: {actions_15th}")
    print(f"Valid Actions: {valid_actions}")

    assert len(valid_actions) == 1
    assert valid_actions[0].ticker == "A"
    assert valid_actions[0].type == "sell"
    assert valid_actions[0].quantity == 50

    # Test 2: Month-End Trigger (should NOT trigger rebalance anymore)
    date_end = pd.to_datetime("2023-01-31")
    # Need data for 31st for apply to work without error if strategy checks indicators
    stock_a.data = pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2023-01-31"])
    )
    stock_b.data = pd.DataFrame(
        {"Close": [100.0]}, index=pd.to_datetime(["2023-01-31"])
    )

    actions_end = manager.apply(portfolio, [stock_a, stock_b], date_end)

    valid_actions_end = [a for a in actions_end if a.quantity > 0]

    print(f"Actions on 31st: {actions_end}")
    # Should be empty if no normal strategy triggers
    assert len(valid_actions_end) == 0
