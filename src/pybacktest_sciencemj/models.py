import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable, Union, Tuple, List
from pydantic import BaseModel, Field

class Stock:
    def __init__(self, ticker: str, start: str, end: str, fetch: bool = True):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = self.fetch_data() if fetch else pd.DataFrame()
        self.dates = self.data.index.to_list() if fetch else []
    
    def fetch_data(self) -> pd.DataFrame:
        data = yf.download(self.ticker, start=self.start, end=self.end, progress=False, group_by='ticker')
        data = self.data_processing(data=data)
        return data
    
    def data_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        #data.columns = ['Close', 'High', 'Low', 'Open', 'Volume'] # Rename
        data['Change'] = data['Close'] - data['Close'].shift(1) # Daily Change
        data['Change_Pct'] = data['Change'] / data['Close'].shift(1) * 100 # Daily Change Percentage
        return data
    
    def cut_data(self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]) -> 'Stock':
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)
        data = self.data.loc[start:end]
        stock = Stock(self.ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), fetch=False)
        stock.data = data
        stock.dates = data.index.to_list()
        return stock
    
    def plot_data(self, figsize: Tuple[int, int]=(14, 7)):
        plt.figure(figsize=figsize)
        plt.plot(self.data.index, self.data['Close'], label='Close Price')
        plt.title(f'{self.ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

class Action(BaseModel):
    ticker: str
    type: str # 'buy' or 'sell'
    quantity: int
    price: float

class Portfolio:
    def __init__(self, cash: float, tickers: List[str]):
        self.cash = cash
        self.tickers = tickers
        self.stock_count = {ticker: 0 for ticker in tickers}
        self.buy_value = {ticker: 0 for ticker in tickers}

    def update(self, ticker: str, amount: int, price: float) -> None:
        '''
        update for stock
        
        :param self: 설명
        :param ticker: ticker string
        :type ticker: str
        :param amount: +- amount of stock
        :type amount: int
        :param price: buy price of stock
        :type price: float
        '''
        if amount > 0:
            current_value = self.stock_count[ticker] * self.buy_value[ticker]
            cost = amount * price
            updated_value = current_value + cost
            self.cash -= cost
            self.stock_count[ticker] += amount
            self.buy_value[ticker] = updated_value/self.stock_count[ticker]
        if amount < 0:
            rev = -amount * price
            self.cash += rev
            self.stock_count[ticker] += amount
            if self.stock_count[ticker] <= 0: self.buy_value[ticker] = 0
        