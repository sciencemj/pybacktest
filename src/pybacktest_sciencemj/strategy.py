from pydantic import BaseModel, RootModel
from pybacktest_sciencemj.models import Action, Stock, Portfolio
import pandas as pd
from typing import Callable, List, Union, Optional, Literal, Dict
import numpy as np

class TradeAction(BaseModel):
    ticker: str # index ticker
    by: List[Union[Literal["average", "current", "percentage"], Literal["Close", "Open", "Low", "High", "Change", "Change_Pct"]]]
    period: Union[int, bool]
    criteria: List[Union[str, float]]
    amount: Optional[List[Union[str, float | int]]] = None

class StrategyConfig(BaseModel):
    buy: TradeAction
    sell: TradeAction

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
    
    def apply(self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp) -> List[Action]:
        actions = []
        for ticker, strategy in self.strategies.items():
            actions.append()
    
    def get_name(self) -> str:
        return self.name
    
    @staticmethod
    def run_action(ticker: str, strategy: StrategyConfig, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp) -> Action:
        target_data, buy_index_data, sell_index_data = None
        for s in stocks: 
            if s.ticker == ticker: target_data = s
            if s.ticker == strategy.buy.ticker: buy_index_data = s
            if s.ticker == strategy.sell.ticker: sell_index_data = s
        if not target_data or not buy_index_data or not sell_index_data: raise KeyError("No Stock Data for Strategy")
        # buy part
        buy: TradeAction = strategy.buy
        by = buy.by
        available_cash = portfolio.money
        compare_value = None
        if by[0] == "average":
            if isinstance(buy.period, int):
                compare_value = buy_index_data.data[by[1]].rolling(window=buy.period, min_periods=1).mean()
            else: compare_value = buy_index_data.data[by[1]].mean()
        elif by[0] == "current":
            compare_value = buy_index_data.data[by[1]].loc[-1]
        elif not compare_value: raise ValueError("Error While setting compare value")
        # TODO: set criteria
        criteria = None


# for only testing
class Strategy:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func: Callable = func
    
    def apply(self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp) -> List[Action]:
        return self.func(portfolio, stocks, date)
    
    def get_name(self) -> str:
        return self.name


        