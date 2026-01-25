import yfinance as yf
from pybacktest_sciencemj.models import Stock, Action, Portfolio
from pybacktest_sciencemj.strategy import Strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable, Tuple, Optional, List

class Backtest:
    def __init__(self, stocks: List[Stock], strategies: List[Strategy], initial_capital: float = 10000.0):
        self.stocks = stocks
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.trades = defaultdict(list)
        self.dates = self.get_common_dates()
        self.value_over_time = defaultdict(dict)
        self.portfolio:Portfolio = Portfolio(initial_capital, [stock.ticker for stock in stocks])

    def get_protfolio_value(self, date: str) -> float:
        '''
        get total portfolio value at a specific date
        
        :param date: date in 'YYYY-MM-DD' format
        :type date: str
        :return: total portfolio value at the given date
        :rtype: float
        '''
        total_value = self.portfolio.money
        for stock in self.stocks:
            if stock.ticker in self.portfolio.tickers:
                total_value += self.portfolio.tickers[stock.ticker] * stock.data.loc[pd.to_datetime(date), 'Close']
        return total_value

    def get_common_dates(self) -> pd.DatetimeIndex:
        '''
        get common dates across all stocks
        
        :param self: 설명
        :return: common dates across all stocks
        :rtype: DatetimeIndex
        '''
        common_dates = set(self.stocks[0].data.index)
        for stock in self.stocks[1:]:
            common_dates = common_dates.intersection(set(stock.data.index))
        return pd.DatetimeIndex(sorted(common_dates))
    
    def run(self, end_date: str = None):
        '''
        Run the backtest.
        
        :param self: self
        :param end_date: ending date in 'YYYY-MM-DD' format if specified, otherwise runs till the last date available
        :type end_date: str
        '''
        if end_date: run_dates = self.dates[self.dates <= pd.to_datetime(end_date)]
        else: run_dates = self.dates
        for strategy in self.strategies:
            self.portfolio =  Portfolio(self.initial_capital, [stock.ticker for stock in self.stocks])
            for date in run_dates:
                stock_data = [stock.cut_data(stock.start, date) for stock in self.stocks]
                action = strategy.apply(self.portfolio, self.stocks, date)
                self.execute_action(action, date, strategy)
                self.value_over_time[strategy][date] = self.get_protfolio_value(date)
    
    def execute_action(self, actions: list[Action], date: pd.Timestamp, strategy: Strategy):
        '''
        action: {
            'AAPL': {'type': 'buy', 'quantity': 10, price: 150.0},
            'MSFT': {'type': 'sell', 'quantity': 5, price: 250.0}
        }
        '''
        for action in actions:
            if action.type == 'buy':
                cost = action.price * action.quantity
                if self.portfolio.money >= cost:
                    self.portfolio.money -= cost
                    self.portfolio.tickers[action.ticker] += action.quantity
                    self.trades[strategy].append({'date': date, 'ticker': action.ticker, 'type': 'buy', 'quantity': action.quantity, 'price': action.price})
                else:
                    raise ValueError(f"Not enough money to buy {action.quantity} shares of {action.ticker} at {action.price} on {date}! Check your strategy.")
            elif action.type == 'sell':
                if self.portfolio.tickers[action.ticker] >= action.quantity:
                    revenue = action.price * action.quantity
                    self.portfolio.money += revenue
                    self.portfolio.tickers[action.ticker] -= action.quantity
                    self.trades[strategy].append({'date': date, 'ticker': action.ticker, 'type': 'sell', 'quantity': action.quantity, 'price': action.price})
                else:
                    raise ValueError(f"Not enough shares to sell {action.quantity} of {action.ticker} on {date}! Check your strategy.")

    def plot_performance(self, figsize: Tuple[int, int]=(14, 7), show_trades: bool=True, subplot: Optional[Tuple[int, int]]=None, instance_show=True):
        '''
        plot_performance의 Docstring
        
        :param self: 설명
        :param figsize: Size of the figure
        :type figsize: Tuple[int, int]
        :param show_trades: show trade points on the plot (buy/sell)
        :type show_trades: bool
        :param subplot: If specified, creates subplots for each strategy with given (rows, cols)
        :type subplot: Optional[Tuple[int, int]]
        :param instance_show: If False do not show plot when function ended
        :type instance_show: bool
        '''
        if not subplot:
            if instance_show: plt.figure(figsize=figsize)
            for strategy in self.strategies:
                dates = list(self.value_over_time[strategy].keys())
                values = list(self.value_over_time[strategy].values())
                plt.plot(dates, values, label=strategy.get_name())
                
                if show_trades:
                    for trade in self.trades[strategy]:
                        color = 'g' if trade['type'] == 'buy' else 'r'
                        plt.scatter(trade['date'], self.value_over_time[strategy][trade['date']], color=color, marker='^' if trade['type'] == 'buy' else 'v')
            
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.grid()
            if instance_show: plt.show()
        else:
            if instance_show: plt.figure(figsize=figsize)
            for i, strategy in enumerate(self.strategies):
                plt.subplot(subplot[0], subplot[1], i+1)
                dates = list(self.value_over_time[strategy].keys())
                values = list(self.value_over_time[strategy].values())
                plt.plot(dates, values, label=strategy.get_name())
                
                if show_trades:
                    for trade in self.trades[strategy]:
                        color = 'g' if trade['type'] == 'buy' else 'r'
                        plt.scatter(trade['date'], self.value_over_time[strategy][trade['date']], color=color, marker='^' if trade['type'] == 'buy' else 'v')
                
                plt.title(f'Portfolio Value Over Time - {strategy.get_name()}')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value')
                plt.legend()
                plt.grid()
                if instance_show: plt.show()