from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from pybacktest.models import Action, Portfolio, Stock


@dataclass
class ExecutionConfig:
    liquidity_limit: float | None = None


def _volume_by_ticker(stocks: list[Stock]) -> dict[str, int]:
    volumes: dict[str, int] = {}
    for stock in stocks:
        if not stock.data.empty and "Volume" in stock.data:
            volumes[stock.ticker] = int(stock.data["Volume"].iloc[-1])
    return volumes


def _apply_liquidity_limit(
    action: Action, volumes: dict[str, int], limit: float | None
) -> tuple[Action, str | None]:
    if limit is None:
        return action, None
    max_quantity = math.floor(volumes.get(action.ticker, 0) * limit)
    if action.quantity <= max_quantity:
        return action, None
    scaled = action.model_copy(update={"quantity": max_quantity})
    warning = (
        f"Scaled {action.type} {action.ticker} from {action.quantity} "
        f"to {max_quantity} due to liquidity limit."
    )
    return scaled, warning


def execute_actions(
    portfolio: Portfolio,
    actions: list[Action],
    stocks: list[Stock],
    date: pd.Timestamp,
    config: ExecutionConfig,
) -> tuple[list[dict], list[str]]:
    volumes = _volume_by_ticker(stocks)
    warnings: list[str] = []
    trades: list[dict] = []
    buys: list[Action] = []
    sells: list[Action] = []

    for raw_action in actions:
        if raw_action.quantity <= 0:
            continue
        action, warning = _apply_liquidity_limit(
            raw_action, volumes, config.liquidity_limit
        )
        if warning:
            warnings.append(warning)
        if action.quantity <= 0:
            continue
        if action.type == "sell":
            sells.append(action)
        elif action.type == "buy":
            buys.append(action)

    for action in sells:
        if portfolio.stock_count[action.ticker] < action.quantity:
            raise ValueError(
                f"Not enough shares to sell {action.quantity} of {action.ticker} "
                f"on {date}! Check your strategy."
            )
        portfolio.update(action.ticker, -action.quantity, action.price)
        trades.append(
            {
                "date": date,
                "ticker": action.ticker,
                "type": "sell",
                "quantity": action.quantity,
                "price": action.price,
            }
        )

    total_buy_cost = sum(action.price * action.quantity for action in buys)
    ratio = 1.0
    if total_buy_cost > portfolio.cash and portfolio.cash > 0:
        ratio = portfolio.cash / total_buy_cost
        warnings.append(
            f"Insufficient cash on {date}. Scaling buy orders by ratio {ratio:.4f}."
        )
    elif total_buy_cost > portfolio.cash and portfolio.cash <= 0:
        warnings.append(f"No cash available on {date} to process buy orders.")
        return trades, warnings

    for action in buys:
        quantity = math.floor(action.quantity * ratio)
        cost = quantity * action.price
        if quantity > 0 and portfolio.cash >= cost:
            portfolio.update(action.ticker, quantity, action.price)
            trades.append(
                {
                    "date": date,
                    "ticker": action.ticker,
                    "type": "buy",
                    "quantity": quantity,
                    "price": action.price,
                }
            )

    return trades, warnings
