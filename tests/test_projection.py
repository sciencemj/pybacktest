import pandas as pd

from pybacktest.models import Portfolio, Stock
from pybacktest.projection import ProjectionConfig, project_portfolio


def _stock(ticker: str) -> Stock:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    stock = Stock(ticker, "2024-01-01", "2024-01-08", fetch=False)
    stock.data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 99.0, 102.0, 103.0, 104.0, 103.0, 105.0],
            "Volume": [1000] * 8,
        },
        index=dates,
    )
    return stock


def test_projection_returns_value_bands_for_future_dates():
    portfolio = Portfolio(100.0, ["A"])
    portfolio.stock_count["A"] = 10

    projection, warnings = project_portfolio(
        portfolio=portfolio,
        stocks=[_stock("A")],
        config=ProjectionConfig(enabled=True, days=5, scenarios=25, lookback=5, random_seed=7),
    )

    assert warnings == []
    assert list(projection.columns) == ["low", "median", "high"]
    assert len(projection) == 5
    assert projection.index[0] > pd.Timestamp("2024-01-08")
    assert (projection["low"] <= projection["median"]).all()
    assert (projection["median"] <= projection["high"]).all()


def test_projection_skips_with_too_little_history():
    portfolio = Portfolio(100.0, ["A"])
    portfolio.stock_count["A"] = 10
    stock = _stock("A")
    stock.data = stock.data.iloc[:2]

    projection, warnings = project_portfolio(
        portfolio=portfolio,
        stocks=[stock],
        config=ProjectionConfig(enabled=True, days=5, scenarios=25, lookback=5),
    )

    assert projection is None
    assert warnings == ["Projection skipped: at least 3 return observations are required."]
