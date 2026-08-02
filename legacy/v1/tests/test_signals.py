import pandas as pd

from pybacktest.models import Portfolio, Stock
from pybacktest.signals import evaluate_trade_action
from pybacktest.strategy import StrategyConfig, TradeAction


def _stock() -> Stock:
    stock = Stock("AAPL", "2024-01-01", "2024-01-03", fetch=False)
    stock.data = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000, 1100, 3000],
            "Change": [None, 1.0, 1.0],
            "Change_Pct": [None, 1.0, 0.990099],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    return stock


def _flat_stock() -> Stock:
    stock = Stock("AAPL", "2024-01-01", "2024-01-03", fetch=False)
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


def test_volume_ratio_signal_triggers_when_current_volume_exceeds_average():
    action = TradeAction(
        ticker="AAPL",
        indicator=["average", "Volume"],
        window=2,
        threshold=["volume-ratio", 1.5],
        quantity=["count", 3],
        price_point="Close",
    )
    portfolio = Portfolio(10000.0, ["AAPL"])

    triggered = evaluate_trade_action(
        action=action,
        target_ticker="AAPL",
        order_type="buy",
        portfolio=portfolio,
        stocks=[_stock()],
    )

    assert triggered is not None
    assert triggered.ticker == "AAPL"
    assert triggered.type == "buy"
    assert triggered.quantity == 3
    assert triggered.price == 102.0


def test_volume_ratio_signal_does_not_trigger_when_ratio_is_low():
    action = TradeAction(
        ticker="AAPL",
        indicator=["average", "Volume"],
        window=3,
        threshold=["volume-ratio", 10.0],
        quantity=["count", 3],
        price_point="Close",
    )
    portfolio = Portfolio(10000.0, ["AAPL"])

    triggered = evaluate_trade_action(
        action=action,
        target_ticker="AAPL",
        order_type="buy",
        portfolio=portfolio,
        stocks=[_stock()],
    )

    assert triggered is None


def test_split_quantity_uses_portfolio_weight_for_buy_signal():
    action = TradeAction(
        ticker="AAPL",
        indicator=["current", "Close"],
        window=False,
        threshold=["point", 0],
        quantity=["split", 10],
        price_point="Close",
    )
    portfolio = Portfolio(10000.0, ["AAPL"])

    triggered = evaluate_trade_action(
        action=action,
        target_ticker="AAPL",
        order_type="buy",
        portfolio=portfolio,
        stocks=[_flat_stock()],
        portfolio_weight=0.5,
    )

    assert triggered is not None
    assert triggered.quantity == 5
