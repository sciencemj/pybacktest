import pytest
import pandas as pd
from src.pybacktest.backtest import Backtest
from src.pybacktest.models import Stock, Action, Portfolio
from src.pybacktest.strategy import Strategy, StrategyWrapper, StrategyManager
import json

def test_get_portfolio_value():
    stock_a = Stock('360750.KS', start='2022-01-01', end='2022-01-10', fetch=False)
    stock_b = Stock('225030.KS', start='2022-01-01', end='2022-01-10', fetch=False)
    
    dates = pd.date_range(start='2022-01-01', end='2022-01-10')
    data_a = pd.DataFrame({'Close': [100, 102, 101, 103, 104, 105, 106, 107, 108, 109]}, index=dates)
    data_b = pd.DataFrame({'Close': [200, 202, 201, 203, 204, 205, 206, 207, 208, 209]}, index=dates)
    
    stock_a.data = data_a
    stock_b.data = data_b
    
    backtest = Backtest([stock_a, stock_b], [], initial_capital=1000.0)
    backtest.portfolio.cash = 500.0
    backtest.portfolio.stock_count[stock_a.ticker] = 3
    backtest.portfolio.stock_count[stock_b.ticker] = 2
    
    expected_value = (500.0 + 
                        3 * stock_a.data.loc[pd.to_datetime('2022-01-10'), 'Close'] + 
                        2 * stock_b.data.loc[pd.to_datetime('2022-01-10'), 'Close'])
    
    actual_value = backtest.get_protfolio_value('2022-01-10')
    assert expected_value == actual_value

def test_execute_action():
    stock_a = Stock('360750.KS', start='2022-01-01', end='2022-01-10', fetch=False)
    
    dates = pd.date_range(start='2022-01-01', end='2022-01-10')
    data_a = pd.DataFrame({'Close': [100, 102, 101, 103, 104, 105, 106, 107, 108, 109]}, index=dates)
    stock_a.data = data_a
    
    backtest = Backtest([stock_a], [], initial_capital=1000.0)
    
    action = {'ticker': '360750.KS', 'type': 'buy', 'quantity': 5, 'price': 100.0}

    action = Action(**action)
    
    backtest.execute_action([action], pd.to_datetime('2022-01-02'), Strategy('Test Strategy', lambda p, s, d: {}))
    
    assert backtest.portfolio.stock_count['360750.KS'] == 5
    assert backtest.portfolio.cash == 1000.0 - (5 * 100.0)
    
    action_sell = {
        'ticker': '360750.KS', 'type': 'sell', 'quantity': 3, 'price': 102.0
    }
    action_sell = Action(**action_sell)
    
    backtest.execute_action([action_sell], pd.to_datetime('2022-01-03'), Strategy('Test Strategy', lambda p, s, d: {}))
    
    assert backtest.portfolio.stock_count['360750.KS'] == 2
    assert backtest.portfolio.cash == 1000.0 - (5 * 100.0) + (3 * 102.0)

def test_strategy_init():
    with open('strategy_test_format.json', 'r') as json_file:
        test_data = json.load(json_file)
    strategy = StrategyWrapper.model_validate(test_data)
    assert strategy['AAPL'].buy.by == ["current", "Close"]
    assert strategy['TQQQ'].buy.ticker == "AAPL"
    assert strategy['TQQQ'].sell.quantity == ["percent", 100]

def test_strategy():
    with open('strategy_test_format.json', 'r') as json_file:
        test_data = json.load(json_file)
    strategy = StrategyManager("AAPL and TQQQ", StrategyWrapper.model_validate(test_data))
    stock_a = Stock('AAPL', start='2022-01-01', end='2022-01-10', fetch=False)
    dates = pd.date_range(start='2022-01-01', end='2022-01-10')
    data_a = pd.DataFrame({'Close': [100, 102, 101, 103, 104, 105, 106, 107, 108, 500]}, index=dates)
    data_a["Change"] = data_a - data_a.shift(1)
    data_a["Change_Pct"] = data_a['Close'].pct_change()
    stock_a.data = data_a
    stock_b = Stock('TQQQ', start='2022-01-01', end='2022-01-10', fetch=False)
    data_b = pd.DataFrame({'Close': [100, 102, 101, 103, 104, 105, 106, 107, 108, 109]}, index=dates)
    data_b["Change"] = data_b - data_b.shift(1)
    data_b["Change_Pct"] = data_b['Close'].pct_change()
    stock_b.data = data_b
    stocks = [stock_a, stock_b]
    actions = strategy.apply(Portfolio(1000000, ["AAPL", "TQQQ"]), stocks, pd.to_datetime("2022-01-10"))
    print(actions[0].model_dump_json())
    assert actions[0].type == "buy"
    assert actions[0].price == 500