import math
from typing import Callable, Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, RootModel

from pybacktest.models import Action, Portfolio, Stock
from pybacktest.rebalancing import RebalanceConfig, generate_rebalance_actions
from pybacktest.signals import evaluate_trade_action


class TradeAction(BaseModel):
    ticker: str  # index ticker
    indicator: List[
        Union[
            Literal["average", "current", "percentage"],
            Literal["Close", "Open", "Low", "High", "Change", "Change_Pct", "Volume"],
        ]
    ]
    window: Union[int, bool]
    threshold: List[
        Union[
            Literal["point", "profit-rate", "percent-change", "volume-ratio"], float
        ]
    ]
    quantity: Optional[List[Union[str, float | int]]] = ["percent", 100]
    price_point: Optional[Literal["Close", "Open", "Low", "High"]] = "Close"


class StrategyConfig(BaseModel):
    buy: TradeAction
    sell: TradeAction
    portfolio_weight: float = 0.0


class StrategyWrapper(RootModel):
    root: Dict[str, StrategyConfig]

    def __getitem__(self, item):
        return self.root[item]

    def items(self):
        return self.root.items()


class StrategyManager:
    def __init__(self, name: str, strategies: StrategyWrapper):
        self.name = name
        self.strategies: StrategyWrapper = strategies

    def apply_signals(
        self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp
    ) -> List[Action]:
        actions = []
        for ticker, strategy in self.strategies.items():
            actions.extend(
                self.apply_strategy(ticker, strategy, portfolio, stocks, date)
            )
        return actions

    def apply(
        self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp
    ) -> List[Action]:
        actions = self.apply_signals(portfolio, stocks, date)
        if date.day == 15:
            actions.extend(self.rebalance(portfolio, stocks, date))
        return actions

    def rebalance(
        self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp
    ) -> List[Action]:
        weights = {
            ticker: strategy.portfolio_weight
            for ticker, strategy in self.strategies.items()
            if strategy.portfolio_weight > 0
        }
        actions, _warnings = generate_rebalance_actions(
            portfolio,
            stocks,
            weights,
            date,
            RebalanceConfig(enabled=True, day=date.day, mode="sell_only"),
        )
        return actions

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def apply_strategy(
        ticker: str,
        strategy: StrategyConfig,
        portfolio: Portfolio,
        stocks: List[Stock],
        date: pd.Timestamp,
    ) -> List[Action]:
        actions = []
        portfolio_weight = strategy.portfolio_weight or 1.0
        buy_action = evaluate_trade_action(
            strategy.buy,
            ticker,
            "buy",
            portfolio,
            stocks,
            portfolio_weight=portfolio_weight,
        )
        if buy_action is not None:
            actions.append(buy_action)
        sell_action = evaluate_trade_action(
            strategy.sell, ticker, "sell", portfolio, stocks
        )
        if sell_action is not None:
            actions.append(sell_action)
        return actions

    @staticmethod
    def create_action(
        type: Literal["buy", "sell"],
        ticker,
        price,
        quantity_type: Literal["count", "percent", "value"],
        quantity,
        portfolio: Portfolio,
    ):
        if quantity_type == "count":
            pass
        elif quantity_type == "percent":
            quantity = max(
                math.floor(portfolio.stock_count[ticker] * (quantity / 100)), 1
            )
        elif quantity_type == "value":
            quantity = quantity // price
        else:
            raise ValueError("wrong value for quantity_type!")
        if type == "buy":
            over_quantity = math.ceil((price * quantity - portfolio.cash) / price)
            return Action(
                ticker=ticker,
                type=type,
                quantity=min(quantity, quantity - over_quantity),
                price=price,
            )
        elif type == "sell":
            return Action(
                ticker=ticker,
                type=type,
                quantity=min(quantity, portfolio.stock_count[ticker]),
                price=price,
            )
        else:
            raise ValueError("wrong value for type!")


# for only testing
class Strategy:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func: Callable = func

    def apply(
        self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp
    ) -> List[Action]:
        return self.func(portfolio, stocks, date)

    def get_name(self) -> str:
        return self.name
