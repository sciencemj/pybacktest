from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pybacktest.models import Action, Portfolio, Stock


@dataclass
class RebalanceConfig:
    enabled: bool = False
    frequency: Literal["monthly"] = "monthly"
    day: int = 15
    mode: Literal["full", "sell_only"] = "full"


def validate_weights(weights: dict[str, float]) -> list[str]:
    total = round(sum(weights.values()), 10)
    if total > 1.0:
        raise ValueError(f"Target weights sum to {total:.2f}; must be <= 1.00.")
    if total < 1.0 and total > 0:
        return [f"Target weights sum to {total:.2f}; remaining {1.0 - total:.2f} stays as cash."]
    return []


def should_rebalance(date: pd.Timestamp, config: RebalanceConfig) -> bool:
    return config.enabled and config.frequency == "monthly" and date.day == config.day


def _current_prices(stocks: list[Stock]) -> dict[str, float]:
    return {stock.ticker: float(stock.data["Close"].iloc[-1]) for stock in stocks if not stock.data.empty}


def generate_rebalance_actions(
    portfolio: Portfolio,
    stocks: list[Stock],
    weights: dict[str, float],
    date: pd.Timestamp,
    config: RebalanceConfig,
) -> tuple[list[Action], list[str]]:
    if not should_rebalance(date, config):
        return [], []

    warnings = validate_weights(weights)
    prices = _current_prices(stocks)
    total_value = portfolio.cash + sum(
        portfolio.stock_count.get(ticker, 0) * prices.get(ticker, 0.0)
        for ticker in portfolio.tickers
    )

    actions: list[Action] = []
    for ticker, weight in weights.items():
        price = prices.get(ticker, 0.0)
        if price <= 0:
            continue
        current_value = portfolio.stock_count.get(ticker, 0) * price
        target_value = total_value * weight
        diff = target_value - current_value
        quantity = int(abs(diff) // price)
        if quantity <= 0:
            continue
        if diff < 0:
            actions.append(Action(ticker=ticker, type="sell", quantity=quantity, price=price))
        elif config.mode == "full":
            actions.append(Action(ticker=ticker, type="buy", quantity=quantity, price=price))

    actions.sort(key=lambda action: 0 if action.type == "sell" else 1)
    return actions, warnings
