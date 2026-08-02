import pandas as pd
import pytest

from pybacktest.models import Portfolio, Stock
from pybacktest.rebalancing import RebalanceConfig, generate_rebalance_actions, validate_weights


def _stock(ticker: str, close: float = 100.0) -> Stock:
    stock = Stock(ticker, "2024-01-15", "2024-01-15", fetch=False)
    stock.data = pd.DataFrame(
        {"Close": [close], "Volume": [10000]},
        index=pd.to_datetime(["2024-01-15"]),
    )
    return stock


def test_full_rebalance_is_default_and_generates_sell_and_buy_actions():
    portfolio = Portfolio(0.0, ["A", "B"])
    portfolio.stock_count["A"] = 100
    portfolio.stock_count["B"] = 0
    config = RebalanceConfig(enabled=True)

    actions, warnings = generate_rebalance_actions(
        portfolio=portfolio,
        stocks=[_stock("A"), _stock("B")],
        weights={"A": 0.5, "B": 0.5},
        date=pd.Timestamp("2024-01-15"),
        config=config,
    )

    assert warnings == []
    assert [(a.ticker, a.type, a.quantity) for a in actions] == [
        ("A", "sell", 50),
        ("B", "buy", 50),
    ]


def test_sell_only_rebalance_keeps_existing_trim_behavior():
    portfolio = Portfolio(0.0, ["A", "B"])
    portfolio.stock_count["A"] = 100
    portfolio.stock_count["B"] = 0
    config = RebalanceConfig(enabled=True, mode="sell_only")

    actions, warnings = generate_rebalance_actions(
        portfolio=portfolio,
        stocks=[_stock("A"), _stock("B")],
        weights={"A": 0.5, "B": 0.5},
        date=pd.Timestamp("2024-01-15"),
        config=config,
    )

    assert warnings == []
    assert [(a.ticker, a.type, a.quantity) for a in actions] == [("A", "sell", 50)]


def test_weights_over_one_raise_error():
    with pytest.raises(ValueError, match="Target weights sum to 1.20"):
        validate_weights({"A": 0.7, "B": 0.5})
