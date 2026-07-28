from pybacktest.strategy import StrategyWrapper


def test_readme_strategy_schema_validates():
    strategy_json = {
        "AAPL": {
            "buy": {
                "ticker": "AAPL",
                "indicator": ["current", "Change_Pct"],
                "window": False,
                "threshold": ["percent-change", 0.5],
                "quantity": ["count", 10],
                "price_point": "Close",
            },
            "sell": {
                "ticker": "AAPL",
                "indicator": ["current", "Close"],
                "window": False,
                "threshold": ["profit-rate", 10],
                "quantity": ["percent", 100],
                "price_point": "Close",
            },
            "portfolio_weight": 0.5,
        }
    }

    strategy = StrategyWrapper.model_validate(strategy_json)

    assert strategy["AAPL"].buy.indicator == ["current", "Change_Pct"]
    assert strategy["AAPL"].buy.price_point == "Close"
    assert strategy["AAPL"].portfolio_weight == 0.5
