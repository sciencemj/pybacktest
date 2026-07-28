from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pybacktest.models import Portfolio, Stock


@dataclass
class ProjectionConfig:
    enabled: bool = False
    days: int = 30
    scenarios: int = 500
    lookback: int = 60
    random_seed: int | None = None


def project_portfolio(
    portfolio: Portfolio,
    stocks: list[Stock],
    config: ProjectionConfig,
) -> tuple[pd.DataFrame | None, list[str]]:
    if not config.enabled:
        return None, []

    returns_by_ticker = {}
    latest_prices = {}
    for stock in stocks:
        if stock.ticker not in portfolio.stock_count or portfolio.stock_count[stock.ticker] <= 0:
            continue
        returns = stock.data["Close"].pct_change().dropna().tail(config.lookback)
        if len(returns) >= 3:
            returns_by_ticker[stock.ticker] = returns.to_numpy()
            latest_prices[stock.ticker] = float(stock.data["Close"].iloc[-1])

    if not returns_by_ticker:
        return None, ["Projection skipped: at least 3 return observations are required."]

    rng = np.random.default_rng(config.random_seed)
    paths = np.zeros((config.scenarios, config.days))
    tickers = list(returns_by_ticker)

    for scenario in range(config.scenarios):
        prices = latest_prices.copy()
        for day in range(config.days):
            value = portfolio.cash
            for ticker in tickers:
                sampled_return = rng.choice(returns_by_ticker[ticker])
                prices[ticker] = prices[ticker] * (1 + sampled_return)
                value += portfolio.stock_count[ticker] * prices[ticker]
            paths[scenario, day] = value

    start_date = max(stock.data.index.max() for stock in stocks if not stock.data.empty)
    index = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=config.days)
    projection = pd.DataFrame(
        {
            "low": np.percentile(paths, 10, axis=0),
            "median": np.percentile(paths, 50, axis=0),
            "high": np.percentile(paths, 90, axis=0),
        },
        index=index,
    )
    return projection, []
