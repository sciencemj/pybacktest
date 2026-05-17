# Pybacktest: A Python Backtesting Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pybacktest.streamlit.app)

Pybacktest is a Python library for backtesting trading strategies. It allows you to define strategies using a simple JSON format, run simulations on historical stock data, and visualize the performance of your strategies.

## Features

- **Easy-to-use API**: Simple and intuitive API for running backtests.
- **Flexible Strategy Configuration**: Define complex trading strategies using a JSON format.
- **Historical Data**: Fetches historical stock data from Yahoo Finance.
- **Performance Visualization**: Plot the performance of your portfolio over time.
- **Streamlit UI**: An interactive web interface for creating, editing, and backtesting strategies.

## Installation

You can install Pybacktest and its core dependencies using pip:

1. Clone the repository:
   ```bash
   git clone https://github.com/sciencemj/pybacktest.git
   cd pybacktest
   ```
2. Install in editable mode (this will also install the core dependencies):
   ```bash
   pip install -e .
   ```

## How to Use

Here's a simple example of how to use Pybacktest to backtest a trading strategy:

```python
from pybacktest.backtest import Backtest
from pybacktest.strategy import StrategyWrapper, StrategyManager
from pybacktest.models import Stock

# 1. Define the backtest period and create Stock objects
start = '2023-01-01'
end = '2023-12-31'
apple = Stock("AAPL", start, end)
tqqq = Stock("TQQQ", start, end)

# 2. Define your trading strategy in JSON format
strategy_json = {
    "AAPL": {
        "buy": {
            "ticker": "AAPL",
            "indicator": ["current", "Change_Pct"],
            "window": False,
            "threshold": ["percent-change", 0.5],
            "quantity": ["count", 10]
        },
        "sell": {
            "ticker": "AAPL",
            "indicator": ["current", "Close"],
            "window": False,
            "threshold": ["profit-rate", 10],
            "quantity": ["percent", 100]
        }
    },
    "TQQQ": {
        "buy": {
            "ticker": "TQQQ",
            "indicator": ["average", "Close"],
            "window": 3,
            "threshold": ["point", 0.05],
            "quantity": ["value", 1000]
        },
        "sell": {
            "ticker": "AAPL",
            "indicator": ["current", "Change_Pct"],
            "window": False,
            "threshold": ["percent-change", -3],
            "quantity":["percent", 50]
        }
    }
}

# 3. Create a StrategyManager
strategy = StrategyManager("MyStrategy", StrategyWrapper(**strategy_json))

# 4. Create a Backtest instance and run the simulation
backtest = Backtest([apple, tqqq], [strategy], 100000)
backtest.run()

# 5. Plot the performance
backtest.plot_performance()
```

### Rebalancing, Execution Limits, and Projection

`Backtest` also accepts optional `rebalance`, `execution`, and `projection`
keyword arguments to enable periodic portfolio rebalancing, per-day liquidity
caps on executed orders, and Monte Carlo projection of future portfolio value:

```python
backtest = Backtest(
    [apple, tqqq],
    [strategy],
    100000,
    rebalance={"enabled": True, "frequency": "monthly", "day": 15, "mode": "full"},
    execution={"liquidity_limit": 0.05},
    projection={"enabled": True, "days": 30, "scenarios": 500},
)
results = backtest.run()
```

## Strategy Configuration

Trading strategies are defined in a JSON format. Each key in the JSON object represents a stock ticker. The value is an object with `buy` and `sell` rules.

### TradeAction

A `TradeAction` object has the following fields:

- `ticker`: The ticker of the stock to trade.
- `indicator`: A list specifying the value to use for comparison.
  - The first element can be `"average"` or `"current"`.
  - The second element is the field to use from the stock data (e.g., `"Close"`, `"Change_Pct"`).
- `window`: The period for the moving average if `indicator` is `"average"`. `false` if not used.
- `threshold`: The condition for the trade.
  - The first element is the type of threshold (e.g., `"percent-change"`, `"profit-rate"`, `"point"`).
  - The second element is the value for the threshold.
- `quantity`: The quantity to trade.
  - The first element is the type of quantity (e.g., `"count"`, `"percent"`, `"value"`).
  - The second element is the value for the quantity.
- `price_point`: The price to use for the trade (e.g., `"Close"`, `"Open"`).

## Streamlit UI

You can access the live Streamlit application here: [pybacktest.streamlit.app](https://pybacktest.streamlit.app)

Pybacktest comes with an interactive Streamlit web interface to help you create, edit, and backtest your trading strategies.

To run the Streamlit app locally, execute the following command:

```bash
streamlit run streamlit_page.py
```

The UI allows you to:
- Create and edit strategies using a user-friendly form.
- View the JSON representation of your strategies.
- Run backtests and see the results, including performance plots and trade history.
- Download your strategies as a JSON file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.