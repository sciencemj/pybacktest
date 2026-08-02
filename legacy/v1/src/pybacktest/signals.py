from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from pybacktest.models import Action, Portfolio, Stock

if TYPE_CHECKING:
    from pybacktest.strategy import TradeAction


def _find_stock(ticker: str, stocks: list[Stock]) -> Stock:
    for stock in stocks:
        if stock.ticker == ticker:
            return stock
    raise KeyError(f"No Stock Data for {ticker}")


def _indicator_value(action: TradeAction, stock: Stock) -> float:
    method, field = action.indicator
    if method == "average":
        if isinstance(action.window, int):
            return float(stock.data[field].rolling(window=action.window, min_periods=1).mean().iloc[-1])
        return float(stock.data[field].mean())
    if method == "current":
        return float(stock.data[field].iloc[-1])
    raise ValueError(f"Unsupported indicator method: {method}")


def _volume_ratio(action: TradeAction, stock: Stock) -> float:
    if not isinstance(action.window, int):
        raise ValueError("volume-ratio threshold requires an integer window")
    prior_volume = stock.data["Volume"].shift(1)
    average_volume = float(prior_volume.rolling(window=action.window, min_periods=1).mean().iloc[-1])
    if average_volume == 0 or pd.isna(average_volume):
        return 0.0
    return float(stock.data["Volume"].iloc[-1]) / average_volume


def _threshold_triggered(action: TradeAction, compare_value: float, portfolio: Portfolio, target_ticker: str, stock: Stock) -> bool:
    threshold_type, threshold_value = action.threshold
    if threshold_type == "volume-ratio":
        return _volume_ratio(action, stock) >= float(threshold_value)
    if threshold_type == "percent-change":
        threshold = float(threshold_value)
    elif threshold_type == "point":
        threshold = portfolio.buy_value[target_ticker] + float(threshold_value)
    elif threshold_type == "profit-rate":
        threshold = portfolio.buy_value[target_ticker] * (100 + float(threshold_value)) / 100
    else:
        raise ValueError(f"Unsupported threshold type: {threshold_type}")
    return compare_value <= threshold if float(threshold_value) <= 0 else compare_value >= threshold


def evaluate_trade_action(
    action: TradeAction,
    target_ticker: str,
    order_type: Literal["buy", "sell"],
    portfolio: Portfolio,
    stocks: list[Stock],
    portfolio_weight: float = 1.0,
) -> Action | None:
    target_stock = _find_stock(target_ticker, stocks)
    indicator_stock = _find_stock(action.ticker, stocks)
    price = float(target_stock.data[action.price_point].iloc[-1])
    compare_value = _indicator_value(action, indicator_stock)

    if (
        action.threshold[0] not in ("percent-change", "volume-ratio")
        and portfolio.buy_value[target_ticker] == 0
        and order_type == "buy"
    ):
        triggered = True
    else:
        triggered = _threshold_triggered(action, compare_value, portfolio, target_ticker, indicator_stock)

    if not triggered:
        return None

    from pybacktest.strategy import StrategyManager

    quantity_type = action.quantity[0]
    quantity_value = action.quantity[1]
    if order_type == "buy" and quantity_type == "split":
        quantity_type = "value"
        quantity_value = (portfolio.initial_capital / quantity_value) * portfolio_weight

    return StrategyManager.create_action(
        order_type,
        target_ticker,
        price,
        quantity_type,
        quantity_value,
        portfolio,
    )
