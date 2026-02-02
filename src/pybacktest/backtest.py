import math
import warnings
from collections import defaultdict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from pybacktest.models import Action, Portfolio, Stock
from pybacktest.strategy import StrategyManager


class Backtest:
    def __init__(
        self,
        stocks: List[Stock],
        strategies: List[StrategyManager],
        initial_capital: float = 10000.0,
    ):
        self.stocks = stocks
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.trades = defaultdict(list)
        self.dates = self.get_common_dates()
        self.value_over_time = defaultdict(dict)
        self.daily_snapshots = []  # To store daily portfolio state
        self.portfolio: Portfolio = Portfolio(
            initial_capital, [stock.ticker for stock in stocks]
        )

    def get_protfolio_value(self, date: str) -> float:
        """
        get total portfolio value at a specific date

        :param date: date in 'YYYY-MM-DD' format
        :type date: str
        :return: total portfolio value at the given date
        :rtype: float
        """
        total_value = self.portfolio.cash
        for stock in self.stocks:
            if stock.ticker in self.portfolio.tickers:
                if pd.to_datetime(date) not in stock.data.index.to_list():
                    stock.data.loc[pd.to_datetime(date)] = None
                    stock.data.sort_index(inplace=True, ascending=True)
                total_value += (
                    self.portfolio.stock_count[stock.ticker]
                    * stock.data.asof(pd.to_datetime(date))["Close"]
                )
        return total_value

    def get_common_dates(self) -> pd.DatetimeIndex:
        """
        get common dates across all stocks

        :param self: 설명
        :return: common dates across all stocks
        :rtype: DatetimeIndex
        """
        common_dates = set(self.stocks[0].data.index)
        for stock in self.stocks[1:]:
            common_dates = common_dates.intersection(set(stock.data.index))
        return pd.DatetimeIndex(sorted(common_dates))

    def run(self, end_date: str = None):
        """
            :param self: self
        :param end_date: ending date in 'YYYY-MM-DD' format if specified, otherwise runs till the last date available
        :type end_date: str
        '''"""
        print("Start Runing Backtest!")
        if end_date:
            run_dates = self.dates[self.dates <= pd.to_datetime(end_date)]
        else:
            run_dates = self.dates
        for strategy in self.strategies:
            self.portfolio = Portfolio(
                self.initial_capital, [stock.ticker for stock in self.stocks]
            )
            for date in run_dates:
                stock_data = [
                    stock.cut_data(stock.start, date) for stock in self.stocks
                ]
                actions = strategy.apply(self.portfolio, stock_data, date)
                self.execute_action(actions, date, strategy)
                self.value_over_time[strategy][date] = self.get_protfolio_value(date)
                self.record_daily_snapshot(date)
        print("Ended Running Backtest!")

    def record_daily_snapshot(self, date: pd.Timestamp):
        snapshot = {
            "date": date,
            "Cash": self.portfolio.cash,
            "Total_Value": self.get_protfolio_value(date.strftime("%Y-%m-%d")),
        }
        for ticker in self.portfolio.tickers:
            snapshot[f"Stock_Amount_{ticker}"] = self.portfolio.stock_count[ticker]
            # Calculate stock value. Need to get current price.
            # Assuming get_protfolio_value logic or similar can be used,
            # but simpler here since we are inside loop or can access stock data.
            # Using stock.data directly might be slow if we search it every time.
            # Optimizing for now: reusing price fetching logic or cache?
            # Re-using logic from get_protfolio_value essentially.
            for stock in self.stocks:
                if stock.ticker == ticker:
                    if pd.to_datetime(date) in stock.data.index:
                        price = stock.data.loc[pd.to_datetime(date)]["Close"]
                        snapshot[f"Stock_Value_{ticker}"] = (
                            self.portfolio.stock_count[ticker] * price
                        )
                    else:
                        snapshot[f"Stock_Value_{ticker}"] = 0  # Or prev close?
        self.daily_snapshots.append(snapshot)

    def get_monthly_snapshots(self) -> pd.DataFrame:
        if not self.daily_snapshots:
            return pd.DataFrame()
        df = pd.DataFrame(self.daily_snapshots)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        # Resample to month end, taking the last value
        monthly_df = df.resample("ME").last()
        return monthly_df

    def execute_action(
        self, actions: list[Action], date: pd.Timestamp, strategy: StrategyManager
    ):
        """
        Executes a list of actions with fair cash allocation.
        1. Sells are executed first to release cash.
        2. Buys are executed second.
           If total cost of buys > available cash, buy quantities are scaled down proportionally.
        """
        # separate actions
        buys = []
        sells = []
        for action in actions:
            if action.quantity <= 0:
                continue
            if action.type == "sell":
                sells.append(action)
            elif action.type == "buy":
                buys.append(action)

        # 1. Execute Sells first
        for action in sells:
            if self.portfolio.stock_count[action.ticker] >= action.quantity:
                self.portfolio.update(action.ticker, -action.quantity, action.price)
                self.trades[strategy].append(
                    {
                        "date": date,
                        "ticker": action.ticker,
                        "type": "sell",
                        "quantity": action.quantity,
                        "price": action.price,
                    }
                )
            else:
                raise ValueError(
                    f"Not enough shares to sell {action.quantity} of {action.ticker} on {date}! Check your strategy."
                )

        # 2. Execute Buys with Proportional Allocation
        if not buys:
            return

        total_buy_cost = sum(action.price * action.quantity for action in buys)
        available_cash = self.portfolio.cash

        ratio = 1.0
        if total_buy_cost > available_cash and available_cash > 0:
            ratio = available_cash / total_buy_cost
            warnings.warn(
                f"Insufficient cash on {date}. scaling down buy orders by ratio {ratio:.4f}"
            )
        elif total_buy_cost > available_cash and available_cash <= 0:
            warnings.warn(f"No cash available on {date} to process buy orders.")
            return

        for action in buys:
            # Scale quantity if needed
            quantity_to_buy = math.floor(action.quantity * ratio)

            if quantity_to_buy > 0:
                cost = quantity_to_buy * action.price
                # Double check cash (floating point issues or floor might leave tiny gap, usually fine since we floored)
                if self.portfolio.cash >= cost:
                    self.portfolio.update(action.ticker, quantity_to_buy, action.price)
                    self.trades[strategy].append(
                        {
                            "date": date,
                            "ticker": action.ticker,
                            "type": "buy",
                            "quantity": quantity_to_buy,
                            "price": action.price,
                        }
                    )
                else:
                    # Should rarely happen with proportional logic unless price is huge relative to cash residue
                    warnings.warn(
                        f"Skipping buy for {action.ticker}: Cash check failed after scaling."
                    )

    def plot_performance(
        self,
        figsize: Tuple[int, int] = (14, 7),
        show_trades: bool = True,
        subplot: Optional[Tuple[int, int]] = None,
        instance_show=True,
    ):
        """
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
        """
        fig = plt.figure(figsize=figsize)
        if not subplot:
            for strategy in self.strategies:
                dates = list(self.value_over_time[strategy].keys())
                values = list(self.value_over_time[strategy].values())
                plt.plot(dates, values, label=strategy.get_name())

                if show_trades:
                    for trade in self.trades[strategy]:
                        color = "g" if trade["type"] == "buy" else "r"
                        plt.scatter(
                            trade["date"],
                            self.value_over_time[strategy][trade["date"]],
                            color=color,
                            marker="^" if trade["type"] == "buy" else "v",
                        )

            plt.title("Portfolio Value Over Time")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value")
            plt.legend()
            plt.grid()
        else:
            for i, strategy in enumerate(self.strategies):
                plt.subplot(subplot[0], subplot[1], i + 1)
                dates = list(self.value_over_time[strategy].keys())
                values = list(self.value_over_time[strategy].values())
                plt.plot(dates, values, label=strategy.get_name())

                if show_trades:
                    for trade in self.trades[strategy]:
                        color = "g" if trade["type"] == "buy" else "r"
                        plt.scatter(
                            trade["date"],
                            self.value_over_time[strategy][trade["date"]],
                            color=color,
                            marker="^" if trade["type"] == "buy" else "v",
                        )

                plt.title(f"Portfolio Value Over Time - {strategy.get_name()}")
                plt.xlabel("Date")
                plt.ylabel("Portfolio Value")
                plt.legend()
                plt.grid()
        if instance_show:
            plt.show()
        return fig
