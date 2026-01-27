from pydantic import BaseModel, RootModel
from pybacktest_sciencemj.models import Action, Stock, Portfolio
import pandas as pd
from typing import Callable, List, Union, Optional, Literal, Dict, Tuple
import math

class TradeAction(BaseModel):
    ticker: str # index ticker
    by: List[Union[Literal["average", "current", "percentage"], Literal["Close", "Open", "Low", "High", "Change", "Change_Pct"]]]
    period: Union[int, bool]
    criteria: List[Union[Literal["point", "profit-rate", "percent-change"], float]]
    quantity: Optional[List[Union[str, float | int]]] = ["percent", 100]
    trade_as: Optional[Literal["Close", "Open", "Low", "High"]] = "Close"

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
            actions.extend(self.apply_strategy(ticker, strategy, portfolio, stocks, date))
        #print(f"actions: {actions}")
        return actions
    
    def get_name(self) -> str:
        return self.name
    
    @staticmethod
    def apply_strategy(ticker: str, strategy: StrategyConfig, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp) -> List[Action]:
        target_data, buy_index_data, sell_index_data = None, None, None
        actions = []
        for s in stocks: 
            if s.ticker == ticker: target_data = s
            if s.ticker == strategy.buy.ticker: buy_index_data = s
            if s.ticker == strategy.sell.ticker: sell_index_data = s
        if not target_data or not buy_index_data or not sell_index_data: raise KeyError("No Stock Data for Strategy")
        # buy part ---------------------------------------------------------
        buy: TradeAction = strategy.buy
        price = target_data.data[buy.trade_as].iloc[-1]
        by = buy.by
        if by[0] == "average":
            if isinstance(buy.period, int):
                compare_value = buy_index_data.data[by[1]].rolling(window=buy.period, min_periods=1).mean()
            else: compare_value = buy_index_data.data[by[1]].mean()
            compare_value = float(compare_value.to_numpy()[-1])
        elif by[0] == "current":
            compare_value = buy_index_data.data[by[1]].iloc[-1]
        else: raise ValueError("Error While setting compare value")
        criteria = portfolio.buy_value[ticker]
        #if criteria <= 0: criteria = price
        crit = buy.criteria
        if crit[0] == "percent-change":
            criteria = crit[1]
        elif crit[0] == "point":
            criteria += crit[1]
        elif crit[0] == "profit-rate":
            criteria *= (100 + crit[1])/100
        else: raise ValueError(f"you got wrong criteria {crit[0]}")
        if crit[1] <= 0:
            if compare_value <= criteria:
                actions.append(StrategyManager.create_action("buy", ticker, price, buy.quantity[0], buy.quantity[1], portfolio))
        else:
            if compare_value >= criteria:
                actions.append(StrategyManager.create_action("buy", ticker, price, buy.quantity[0], buy.quantity[1], portfolio))
        # sell part ---------------------------------------------------------
        sell: TradeAction = strategy.sell
        by = sell.by
        if by[0] == "average":
            if isinstance(sell.period, int):
                compare_value = sell_index_data.data[by[1]].rolling(window=sell.period, min_periods=1).mean()
            else: compare_value = sell_index_data.data[by[1]].mean()
            compare_value = float(compare_value.to_numpy()[-1])
        elif by[0] == "current":
            compare_value = sell_index_data.data[by[1]].iloc[-1]
        else: raise ValueError("Error While setting compare value")
        criteria = portfolio.buy_value[ticker]
        #if criteria <= 0: criteria = price
        crit = sell.criteria
        if crit[0] == "percent-change":
            criteria = crit[1]
        elif crit[0] == "point":
            criteria += crit[1]
        elif crit[0] == "profit-rate":
            criteria *= (100 + crit[1])/100
        else: raise ValueError(f"you got wrong criteria {crit[0]}")
        if crit[1] <= 0:
            if compare_value <= criteria:
                actions.append(StrategyManager.create_action("sell", ticker, price, sell.quantity[0], sell.quantity[1], portfolio))
        else:
            if compare_value >= criteria:
                actions.append(StrategyManager.create_action("sell", ticker, price, sell.quantity[0], sell.quantity[1], portfolio))
        return actions

    @staticmethod
    def create_action(type: Literal["buy", "sell"], ticker, price, quantity_type: Literal["count", "percent", "value"], quantity, portfolio: Portfolio):
        if quantity_type == "count": pass
        elif quantity_type == "percent": quantity =max(math.floor(portfolio.stock_count[ticker] * (quantity/100)), 1)
        elif quantity_type == "value": quantity = quantity // price
        else: raise ValueError("wrong value for quantity_type!")
        if type == "buy":
            over_quantity = math.ceil((price*quantity - portfolio.cash) / price)
            return Action(ticker=ticker, type=type, quantity=min(quantity, quantity - over_quantity), price=price)
        elif type == "sell":
            return Action(ticker=ticker, type=type, quantity=min(quantity, portfolio.stock_count[ticker]), price=price)
        else:
            raise ValueError("wrong value for type!")


        

# for only testing
class Strategy:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func: Callable = func
    
    def apply(self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp) -> List[Action]:
        return self.func(portfolio, stocks, date)
    
    def get_name(self) -> str:
        return self.name


        