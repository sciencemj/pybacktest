from __future__ import annotations

import warnings as warnings_module
from collections import defaultdict
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from pybacktest.execution import ExecutionConfig, execute_actions
from pybacktest.models import Action, Portfolio, Stock
from pybacktest.projection import ProjectionConfig, project_portfolio
from pybacktest.rebalancing import RebalanceConfig, generate_rebalance_actions
from pybacktest.results import BacktestResult, StrategyResult
from pybacktest.strategy import StrategyManager


class Backtest:
    def __init__(
        self,
        stocks: List[Stock],
        strategies: List[StrategyManager],
        initial_capital: float = 10000.0,
        rebalance: dict | None = None,
        execution: dict | None = None,
        projection: dict | None = None,
    ):
        self.stocks = stocks
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.trades = defaultdict(list)
        self.dates = self.get_common_dates()
        self.value_over_time = defaultdict(dict)
        self.daily_snapshots = []  # To store daily portfolio state
        self.result: Optional[BacktestResult] = None
        self.portfolio: Portfolio = Portfolio(
            initial_capital, [stock.ticker for stock in stocks]
        )
        self.rebalance_config = RebalanceConfig(**rebalance) if rebalance else None
        self.execution_config = (
            ExecutionConfig(**execution) if execution else ExecutionConfig()
        )
        self.projection_config = (
            ProjectionConfig(**projection) if projection else ProjectionConfig()
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
        result = BacktestResult()
        for strategy in self.strategies:
            strategy_result = StrategyResult(name=strategy.get_name())
            self.portfolio = Portfolio(
                self.initial_capital, [stock.ticker for stock in self.stocks]
            )
            self.daily_snapshots = []
            self.value_over_time[strategy] = {}
            self.trades[strategy] = []
            for date in run_dates:
                stock_data = [
                    stock.cut_data(stock.start, date) for stock in self.stocks
                ]
                if self.rebalance_config is None:
                    actions = strategy.apply(self.portfolio, stock_data, date)
                else:
                    actions = strategy.apply_signals(self.portfolio, stock_data, date)
                if self.rebalance_config is not None:
                    weights = {
                        ticker: config.portfolio_weight
                        for ticker, config in strategy.strategies.items()
                        if config.portfolio_weight > 0
                    }
                    rebalance_actions, rebalance_warnings = generate_rebalance_actions(
                        self.portfolio,
                        stock_data,
                        weights,
                        date,
                        self.rebalance_config,
                    )
                    actions.extend(rebalance_actions)
                    strategy_result.warnings.extend(rebalance_warnings)
                self.current_stock_data = stock_data
                execution_warnings = self.execute_action(actions, date, strategy)
                strategy_result.warnings.extend(execution_warnings)
                value = self.get_protfolio_value(date)
                self.value_over_time[strategy][date] = value
                strategy_result.equity_curve[date] = value
                snapshot = self.record_daily_snapshot(date)
                strategy_result.daily_snapshots.append(snapshot)
            strategy_result.trades = list(self.trades[strategy])
            strategy_result.final_cash = self.portfolio.cash
            strategy_result.final_holdings = dict(self.portfolio.stock_count)
            projection_df, projection_warnings = project_portfolio(
                self.portfolio,
                self.stocks,
                self.projection_config,
            )
            strategy_result.projection = projection_df
            strategy_result.warnings.extend(projection_warnings)
            result.strategies[strategy.get_name()] = strategy_result
        self.result = result
        print("Ended Running Backtest!")
        return result

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
        return snapshot

    def get_monthly_snapshots(self):
        if self.result is None:
            raise RuntimeError("Run backtest first.")
        strategy_results = self.result.strategies
        if len(strategy_results) == 1:
            (only_result,) = strategy_results.values()
            return only_result.monthly_snapshots()
        return {
            name: strategy_result.monthly_snapshots()
            for name, strategy_result in strategy_results.items()
        }

    def execute_action(
        self, actions: list[Action], date: pd.Timestamp, strategy: StrategyManager
    ):
        """
        Execute a list of actions for ``strategy`` on ``date``.

        Delegates to :func:`pybacktest.execution.execute_actions` to perform
        sells-before-buys ordering, optional liquidity-limit scaling and
        proportional cash allocation. Warnings produced by the execution
        module are emitted via ``warnings.warn`` and also returned to the
        caller so they can be captured into structured results.
        """
        trades, warnings = execute_actions(
            portfolio=self.portfolio,
            actions=actions,
            stocks=getattr(self, "current_stock_data", self.stocks),
            date=date,
            config=self.execution_config,
        )
        self.trades[strategy].extend(trades)
        for warning in warnings:
            warnings_module.warn(warning)
        return warnings

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
