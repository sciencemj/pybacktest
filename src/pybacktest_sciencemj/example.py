from backtest import *
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
        return action dictionary
        ex) action: {
            'AAPL': {'type': 'buy', 'quantity': 10, price: 150.0},
            'MSFT': {'type': 'sell', 'quantity': 5, price: 250.0}
        }
        '''
        action = {}
        for stock in stocks:
            data = stock.data.loc[:date]
            if len(data) < 20:
                continue
            sma_short = data['Close'].rolling(window=5).mean().iloc[-1]
            sma_long = data['Close'].rolling(window=20).mean().iloc[-1]
            current_price = data['Close'].iloc[-1]
            if sma_short > sma_long and portfolio['money'] >= current_price:
                quantity = int(portfolio['money'] // current_price)
                action[stock.ticker] = {'type': 'buy', 'quantity': quantity, 'price': current_price}
            elif sma_short < sma_long and portfolio[stock.ticker] > 0:
                quantity = portfolio[stock.ticker]
                action[stock.ticker] = {'type': 'sell', 'quantity': quantity, 'price': current_price}
        return action
    sma_crossover = Strategy('SMA Crossover', sma_crossover_strategy)

    # Define just buy and hold strategy
    def buy_and_hold_strategy(portfolio, stocks, date):
        action = {}
        for stock in stocks:
            data = stock.data.loc[:date]
            if len(data) == 1 and portfolio['money'] >= data['Close'].iloc[-1]:
                quantity = int(portfolio['money'] // data['Close'].iloc[-1])
                action[stock.ticker] = {'type': 'buy', 'quantity': quantity, 'price': data['Close'].iloc[-1]}
        return action
    buy_and_hold = Strategy('Buy and Hold', buy_and_hold_strategy)

    strategies = [sma_crossover, buy_and_hold]
    
    # Initialize and run backtest
    backtest = Backtest(stocks, strategies, initial_capital=10000.0)
    backtest.run(end_date='2024-12-31')
    # Plot performance
    #plt.style.use('dark_background')
    backtest.plot_performance(figsize=(14, 10), show_trades=True)
    for strategy in strategies:
        print(f"Strategy: {strategy.get_name()}")
        for trade in backtest.trades[strategy]:
            print(trade)