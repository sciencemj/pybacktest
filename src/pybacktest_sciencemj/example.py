from backtest import *
from models import Action
import matplotlib.pyplot as plt

# Example usage
if __name__ == "__main__":
    # Load stock data
    stock_a = Stock('360750.KS', start='2022-01-01', end='2025-01-01')
    stock_b = Stock('225030.KS', start='2022-01-01', end='2025-01-01')
    stocks = [stock_a, stock_b]

    # Define a simple moving average crossover strategy
    def sma_crossover_strategy(portfolio, stocks, date):
        '''
        return list of Action
        '''
        actions = []
        for stock in stocks:
            data = stock.data.loc[:date]
            if len(data) < 20:
                continue
            sma_short = data['Close'].rolling(window=5).mean().iloc[-1]
            sma_long = data['Close'].rolling(window=20).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]
            if sma_short > sma_long and portfolio.money >= current_price:
                quantity = int(portfolio.money // current_price)
                actions.append(Action(ticker=stock.ticker, type='buy', quantity=quantity, price=current_price))
            elif sma_short < sma_long and portfolio.tickers[stock.ticker] > 0:
                quantity = portfolio.tickers[stock.ticker]
                actions.append(Action(ticker=stock.ticker, type='sell', quantity=quantity, price=current_price))
        return actions
    sma_crossover = Strategy('SMA Crossover', sma_crossover_strategy)

    # Define just buy and hold strategy
    def buy_and_hold_strategy(portfolio, stocks, date):
        actions = []
        for stock in stocks:
            data = stock.data.loc[:date]
            if len(data) == 1 and portfolio.money >= data['Close'].iloc[-1]:
                quantity = int(portfolio.money // data['Close'].iloc[-1])
                actions.append(Action(ticker=stock.ticker, type='buy', quantity=quantity, price=data['Close'].iloc[-1]))
        return actions
    buy_and_hold = Strategy('Buy and Hold', buy_and_hold_strategy)

    strategies = [sma_crossover, buy_and_hold]
    
    # Initialize and run backtest
    backtest = Backtest(stocks, strategies, initial_capital=10000.0)
    backtest.run(end_date='2024-12-31')
    # Plot performance
    #plt.style.use('dark_background')
    
    for strategy in strategies:
        print(f"Strategy: {strategy.get_name()}")
        for trade in backtest.trades[strategy]:
            print(trade)
    backtest.plot_performance(figsize=(14, 10), show_trades=True)