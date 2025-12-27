import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable

class Stock:
    def __init__(self, ticker: str, start: str, end: str):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = self.fetch_data()
        self.dates = self.data.index.to_list()
    
    def fetch_data(self) -> pd.DataFrame:
        data = yf.download(self.ticker, start=self.start, end=self.end)
        data = self.data_processing(data=data)
        return data
    
    def data_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        data.columns = ['Close', 'High', 'Low', 'Open', 'Volume'] # Rename
        data['Change'] = data['Close'] - data['Close'].shift(1) # Daily Change
        data['Change_Pct'] = data['Change'] / data['Close'].shift(1) * 100 # Daily Change Percentage
        return data
    
    def plot_data(self, figsize: tuple[int, int]=(14, 7)):
        plt.figure(figsize=figsize)
        plt.plot(self.data.index, self.data['Close'], label='Close Price')
        plt.title(f'{self.ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

class Strategy:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func
    
    def apply(self, portfolio: dict, stocks: list[Stock], date: pd.Timestamp) -> dict:
        return self.func(portfolio, stocks, date)
    
    def get_name(self) -> str:
        return self.name

class Backtest:
    def __init__(self, stocks: list[Stock], strategies: list[Strategy], initial_capital: float = 10000.0):
        self.stocks = stocks
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.trades = defaultdict(list)
        self.dates = self.get_common_dates()
        self.value_over_time = defaultdict(dict)
        self.portfolio = {'money': initial_capital} | {stock.ticker: 0 for stock in stocks} 

    def get_protfolio_value(self, date: str) -> float:
        '''
        get total portfolio value at a specific date
        
        :param date: date in 'YYYY-MM-DD' format
        :type date: str
        :return: total portfolio value at the given date
        :rtype: float
        '''
        total_value = self.portfolio['money']
        for stock in self.stocks:
            if stock.ticker in self.portfolio:
                total_value += self.portfolio[stock.ticker] * stock.data.loc[pd.to_datetime(date), 'Close']
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
            self.portfolio = {'money': self.initial_capital} | {stock.ticker: 0 for stock in self.stocks}
            for date in run_dates:
                action = strategy.apply(self.portfolio, self.stocks, date)
                self.execute_action(action, date, strategy)
                self.value_over_time[strategy][date] = self.get_protfolio_value(date)
    
    def execute_action(self, action: dict, date: pd.Timestamp, strategy: Strategy):
        '''
        action: {
            'AAPL': {'type': 'buy', 'quantity': 10, price: 150.0},
            'MSFT': {'type': 'sell', 'quantity': 5, price: 250.0}
        }
        '''
        for stock_ticker, trade in action.items():
            stock = next((s for s in self.stocks if s.ticker == stock_ticker), None)
            if stock is None:
                continue
            
            price = trade.get('price', stock.data.loc[date, 'Close'])
            if trade['type'] == 'buy':
                quantity = trade['quantity']
                cost = price * quantity
                if self.portfolio['money'] >= cost:
                    self.portfolio['money'] -= cost
                    self.portfolio[stock_ticker] += quantity
                    self.trades[strategy].append({'date': date, 'ticker': stock_ticker, 'type': 'buy', 'quantity': quantity, 'price': price})
                else:
                    raise ValueError(f"Not enough money to buy {quantity} shares of {stock_ticker} at {price} on {date}! Check your strategy.")
            elif trade['type'] == 'sell':
                quantity = trade['quantity']
                if self.portfolio[stock_ticker] >= quantity:
                    revenue = price * quantity
                    self.portfolio['money'] += revenue
                    self.portfolio[stock_ticker] -= quantity
                    self.trades[strategy].append({'date': date, 'ticker': stock_ticker, 'type': 'sell', 'quantity': quantity, 'price': price})
                else:
                    raise ValueError(f"Not enough shares to sell {quantity} of {stock_ticker} on {date}! Check your strategy.")

    def plot_performance(self, figsize: tuple[int, int]=(14, 7), show_trades: bool=True, subplot: tuple[int, int]=None):
        '''
        plot_performance의 Docstring
        
        :param self: 설명
        :param figsize: Size of the figure
        :type figsize: (int | int)
        :param show_trades: show trade points on the plot (buy/sell)
        :type show_trades: bool
        :param subplot: If specified, creates subplots for each strategy with given (rows, cols)
        :type subplot: (int | int) | None
        '''
        if not subplot:
            plt.figure(figsize=figsize)
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
            plt.show()
        else:
            plt.figure(figsize=figsize)
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
                plt.show()