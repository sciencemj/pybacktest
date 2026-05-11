# Engine Core Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular Pybacktest engine with robust data normalization, full rebalancing by default, volume-aware signals and liquidity limits, structured results, portfolio projections, and fixed baseline tests/docs.

**Architecture:** Keep the existing public `Backtest`, `Stock`, `Portfolio`, `StrategyWrapper`, and JSON strategy shape usable while moving core behavior into focused modules: `data.py`, `signals.py`, `rebalancing.py`, `execution.py`, `projection.py`, and `results.py`. Each module gets direct tests before it is wired into `Backtest.run()`.

**Tech Stack:** Python 3.9+, pandas, numpy, pydantic v2, yfinance, pytest, uv.

---

## File Structure

- Modify: `pyproject.toml`
  - Add pytest `pythonpath = ["src"]` so `uv run pytest` works without environment variables.
- Modify: `README.md`
  - Replace old schema examples with current `indicator`, `window`, `threshold`, and `price_point` names.
- Modify: `src/pybacktest/models.py`
  - Make `Stock` delegate OHLCV normalization to `data.py`.
- Modify: `src/pybacktest/strategy.py`
  - Keep `StrategyManager` as public compatibility layer while delegating signal and rebalance generation.
- Modify: `src/pybacktest/backtest.py`
  - Wire structured config, execution, results, and projection into the run loop.
- Create: `src/pybacktest/data.py`
  - Normalize OHLCV data and expose data errors.
- Create: `src/pybacktest/results.py`
  - Define structured result objects.
- Create: `src/pybacktest/signals.py`
  - Evaluate price and volume strategy rules.
- Create: `src/pybacktest/rebalancing.py`
  - Generate full and sell-only rebalance actions.
- Create: `src/pybacktest/execution.py`
  - Execute actions sell-first, with cash scaling and optional liquidity limits.
- Create: `src/pybacktest/projection.py`
  - Generate portfolio projection bands from recent returns.
- Modify: `tests/test_backtest.py`
  - Change imports from `src.pybacktest...` to `pybacktest...`.
- Create: `tests/test_data.py`
- Create: `tests/test_results.py`
- Create: `tests/test_signals.py`
- Create: `tests/test_rebalancing.py`
- Create: `tests/test_execution.py`
- Create: `tests/test_projection.py`
- Create: `tests/test_readme_examples.py`

---

### Task 1: Fix Test Import Baseline

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing baseline expectation**

Run:

```bash
uv run pytest -q
```

Expected before changes:

```text
ModuleNotFoundError: No module named 'pybacktest'
```

- [ ] **Step 2: Add pytest pythonpath config**

Modify `pyproject.toml` by appending:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 3: Use package imports in legacy tests**

In `tests/test_backtest.py`, replace these imports:

```python
from src.pybacktest.backtest import Backtest
from src.pybacktest.models import Stock, Action, Portfolio
from src.pybacktest.strategy import Strategy, StrategyWrapper, StrategyManager
```

with:

```python
from pybacktest.backtest import Backtest
from pybacktest.models import Action, Portfolio, Stock
from pybacktest.strategy import Strategy, StrategyManager, StrategyWrapper
```

- [ ] **Step 4: Verify baseline passes**

Run:

```bash
uv run pytest -q
```

Expected:

```text
9 passed
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_backtest.py
git commit -m "test: fix package import baseline"
```

---

### Task 2: Add Robust Data Normalization

**Files:**
- Create: `src/pybacktest/data.py`
- Modify: `src/pybacktest/models.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing data normalization tests**

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest

from pybacktest.data import DataError, normalize_ohlcv


def test_normalize_standard_ohlcv_columns():
    raw = pd.DataFrame(
        {
            "Open": [99.0, 101.0],
            "High": [101.0, 103.0],
            "Low": [98.0, 100.0],
            "Close": [100.0, 102.0],
            "Volume": [1000, 1500],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    normalized = normalize_ohlcv(raw)

    assert list(normalized.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Change",
        "Change_Pct",
    ]
    assert normalized.loc[pd.Timestamp("2024-01-02"), "Change"] == 2.0
    assert normalized.loc[pd.Timestamp("2024-01-02"), "Change_Pct"] == 2.0


def test_normalize_yfinance_adjusted_close_shape():
    raw = pd.DataFrame(
        {
            "Open": [99.0, 101.0],
            "High": [101.0, 103.0],
            "Low": [98.0, 100.0],
            "Close": [100.0, 102.0],
            "Adj Close": [99.5, 101.5],
            "Volume": [1000, 1500],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    normalized = normalize_ohlcv(raw)

    assert "Adj Close" not in normalized.columns
    assert normalized["Close"].tolist() == [100.0, 102.0]


def test_normalize_download_order_without_headers():
    raw = pd.DataFrame(
        [
            [100.0, 103.0, 99.0, 101.0, 2000],
            [102.0, 104.0, 101.0, 103.0, 3000],
        ],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    normalized = normalize_ohlcv(raw)

    assert normalized.loc[pd.Timestamp("2024-01-01"), "Close"] == 100.0
    assert normalized.loc[pd.Timestamp("2024-01-01"), "Volume"] == 2000


def test_normalize_empty_data_raises_clear_error():
    with pytest.raises(DataError, match="No price data"):
        normalize_ohlcv(pd.DataFrame(), ticker="BAD")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_data.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'pybacktest.data'
```

- [ ] **Step 3: Implement `src/pybacktest/data.py`**

Create `src/pybacktest/data.py`:

```python
from __future__ import annotations

import pandas as pd


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
YFINANCE_DOWNLOAD_ORDER = ["Close", "High", "Low", "Open", "Volume"]


class DataError(ValueError):
    """Raised when market data cannot be normalized for backtesting."""


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [
            next(str(part) for part in column if str(part) in OHLCV_COLUMNS or str(part) == "Adj Close")
            for column in data.columns
        ]
    return data


def normalize_ohlcv(data: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    label = f" for {ticker}" if ticker else ""
    if data.empty:
        raise DataError(f"No price data{label}. Check ticker and date range.")

    normalized = _flatten_columns(data.copy())

    if set(OHLCV_COLUMNS).issubset(normalized.columns):
        normalized = normalized[OHLCV_COLUMNS]
    elif len(normalized.columns) == 5:
        normalized.columns = YFINANCE_DOWNLOAD_ORDER
        normalized = normalized[OHLCV_COLUMNS]
    elif set(OHLCV_COLUMNS + ["Adj Close"]).issubset(normalized.columns):
        normalized = normalized[OHLCV_COLUMNS]
    else:
        raise DataError(
            f"Price data{label} must contain Open, High, Low, Close, and Volume columns."
        )

    normalized = normalized.sort_index()
    normalized["Change"] = normalized["Close"] - normalized["Close"].shift(1)
    normalized["Change_Pct"] = normalized["Close"].pct_change() * 100
    return normalized
```

- [ ] **Step 4: Update `Stock.data_processing()` to delegate**

In `src/pybacktest/models.py`, add:

```python
from pybacktest.data import normalize_ohlcv
```

Replace `data_processing()` with:

```python
def data_processing(self, data: pd.DataFrame) -> pd.DataFrame:
    return normalize_ohlcv(data, ticker=self.ticker)
```

- [ ] **Step 5: Verify data tests pass**

Run:

```bash
uv run pytest tests/test_data.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Verify full suite still passes**

Run:

```bash
uv run pytest -q
```

Expected:

```text
13 passed
```

- [ ] **Step 7: Commit**

```bash
git add src/pybacktest/data.py src/pybacktest/models.py tests/test_data.py
git commit -m "feat: normalize market data robustly"
```

---

### Task 3: Add Structured Results and Isolated Snapshots

**Files:**
- Create: `src/pybacktest/results.py`
- Modify: `src/pybacktest/backtest.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write failing result isolation tests**

Create `tests/test_results.py`:

```python
import pandas as pd

from pybacktest.backtest import Backtest
from pybacktest.models import Action, Portfolio, Stock
from pybacktest.strategy import Strategy


def _stock(ticker: str) -> Stock:
    stock = Stock(ticker, "2024-01-01", "2024-01-03", fetch=False)
    stock.data = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 100.0, 100.0],
            "Low": [100.0, 100.0, 100.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1000, 1000, 1000],
            "Change": [None, 0.0, 0.0],
            "Change_Pct": [None, 0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    return stock


def test_run_returns_results_by_strategy_without_mixing_snapshots():
    first = Strategy("first", lambda p, s, d: [Action(ticker="A", type="buy", quantity=1, price=100.0)] if d.day == 1 else [])
    second = Strategy("second", lambda p, s, d: [])
    backtest = Backtest([_stock("A")], [first, second], initial_capital=1000.0)

    results = backtest.run()

    assert set(results.strategies.keys()) == {"first", "second"}
    assert len(results.strategies["first"].daily_snapshots) == 3
    assert len(results.strategies["second"].daily_snapshots) == 3
    assert results.strategies["first"].daily_snapshots[-1]["Stock_Amount_A"] == 1
    assert results.strategies["second"].daily_snapshots[-1]["Stock_Amount_A"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_results.py::test_run_returns_results_by_strategy_without_mixing_snapshots -q
```

Expected:

```text
AttributeError
```

- [ ] **Step 3: Implement result models**

Create `src/pybacktest/results.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyResult:
    name: str
    equity_curve: dict[pd.Timestamp, float] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    daily_snapshots: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_cash: float = 0.0
    final_holdings: dict[str, int] = field(default_factory=dict)
    projection: pd.DataFrame | None = None

    def monthly_snapshots(self) -> pd.DataFrame:
        if not self.daily_snapshots:
            return pd.DataFrame()
        df = pd.DataFrame(self.daily_snapshots)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").resample("ME").last()


@dataclass
class BacktestResult:
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Wire `Backtest.run()` to return result object**

In `src/pybacktest/backtest.py`, import:

```python
from pybacktest.results import BacktestResult, StrategyResult
```

Inside `Backtest.run()`, initialize and populate structured results:

```python
result = BacktestResult()
for strategy in self.strategies:
    strategy_result = StrategyResult(name=strategy.get_name())
    self.portfolio = Portfolio(self.initial_capital, [stock.ticker for stock in self.stocks])
    self.daily_snapshots = []
    self.value_over_time[strategy] = {}
    self.trades[strategy] = []
    for date in run_dates:
        stock_data = [stock.cut_data(stock.start, date) for stock in self.stocks]
        actions = strategy.apply(self.portfolio, stock_data, date)
        self.execute_action(actions, date, strategy)
        value = self.get_protfolio_value(date)
        self.value_over_time[strategy][date] = value
        strategy_result.equity_curve[date] = value
        snapshot = self.record_daily_snapshot(date)
        strategy_result.daily_snapshots.append(snapshot)
    strategy_result.trades = list(self.trades[strategy])
    strategy_result.final_cash = self.portfolio.cash
    strategy_result.final_holdings = dict(self.portfolio.stock_count)
    result.strategies[strategy.get_name()] = strategy_result
self.result = result
return result
```

Change `record_daily_snapshot()` to return the snapshot:

```python
self.daily_snapshots.append(snapshot)
return snapshot
```

- [ ] **Step 5: Verify result tests pass**

Run:

```bash
uv run pytest tests/test_results.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Verify existing monthly snapshot behavior still works**

Run:

```bash
uv run pytest tests/test_features.py::test_monthly_snapshots -q
```

Expected:

```text
1 passed
```

- [ ] **Step 7: Commit**

```bash
git add src/pybacktest/results.py src/pybacktest/backtest.py tests/test_results.py
git commit -m "feat: return structured backtest results"
```

---

### Task 4: Extract Signal Evaluation and Add Volume Indicators

**Files:**
- Create: `src/pybacktest/signals.py`
- Modify: `src/pybacktest/strategy.py`
- Create: `tests/test_signals.py`

- [ ] **Step 1: Write failing signal tests**

Create `tests/test_signals.py`:

```python
import pandas as pd

from pybacktest.models import Portfolio, Stock
from pybacktest.signals import evaluate_trade_action
from pybacktest.strategy import StrategyConfig, TradeAction


def _stock() -> Stock:
    stock = Stock("AAPL", "2024-01-01", "2024-01-03", fetch=False)
    stock.data = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000, 1100, 3000],
            "Change": [None, 1.0, 1.0],
            "Change_Pct": [None, 1.0, 0.990099],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    return stock


def test_volume_ratio_signal_triggers_when_current_volume_exceeds_average():
    action = TradeAction(
        ticker="AAPL",
        indicator=["average", "Volume"],
        window=2,
        threshold=["volume-ratio", 1.5],
        quantity=["count", 3],
        price_point="Close",
    )
    portfolio = Portfolio(10000.0, ["AAPL"])

    triggered = evaluate_trade_action(
        action=action,
        target_ticker="AAPL",
        order_type="buy",
        portfolio=portfolio,
        stocks=[_stock()],
    )

    assert triggered is not None
    assert triggered.ticker == "AAPL"
    assert triggered.type == "buy"
    assert triggered.quantity == 3
    assert triggered.price == 102.0


def test_volume_ratio_signal_does_not_trigger_when_ratio_is_low():
    action = TradeAction(
        ticker="AAPL",
        indicator=["average", "Volume"],
        window=3,
        threshold=["volume-ratio", 10.0],
        quantity=["count", 3],
        price_point="Close",
    )
    portfolio = Portfolio(10000.0, ["AAPL"])

    triggered = evaluate_trade_action(
        action=action,
        target_ticker="AAPL",
        order_type="buy",
        portfolio=portfolio,
        stocks=[_stock()],
    )

    assert triggered is None


def test_split_quantity_uses_portfolio_weight_for_buy_signal():
    action = TradeAction(
        ticker="AAPL",
        indicator=["current", "Close"],
        window=False,
        threshold=["point", 0],
        quantity=["split", 10],
        price_point="Close",
    )
    portfolio = Portfolio(10000.0, ["AAPL"])

    triggered = evaluate_trade_action(
        action=action,
        target_ticker="AAPL",
        order_type="buy",
        portfolio=portfolio,
        stocks=[_stock()],
        portfolio_weight=0.5,
    )

    assert triggered is not None
    assert triggered.quantity == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_signals.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'pybacktest.signals'
```

- [ ] **Step 3: Extend strategy model literals**

In `src/pybacktest/strategy.py`, allow `Volume` and `volume-ratio`:

```python
Literal["Close", "Open", "Low", "High", "Change", "Change_Pct", "Volume"]
Literal["point", "profit-rate", "percent-change", "volume-ratio"]
```

- [ ] **Step 4: Implement `src/pybacktest/signals.py`**

Create:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pybacktest.models import Action, Portfolio, Stock

if TYPE_CHECKING:
    from pybacktest.strategy import TradeAction


def _find_stock(ticker: str, stocks: list[Stock]) -> Stock:
    for stock in stocks:
        if stock.ticker == ticker:
            return stock
    raise KeyError(f"No Stock Data for {ticker}")


def _indicator_value(action: TradeAction, stock: Stock) -> float:
    method, field = action.indicator
    if method == "average":
        if isinstance(action.window, int):
            return float(stock.data[field].rolling(window=action.window, min_periods=1).mean().iloc[-1])
        return float(stock.data[field].mean())
    if method == "current":
        return float(stock.data[field].iloc[-1])
    raise ValueError(f"Unsupported indicator method: {method}")


def _volume_ratio(action: TradeAction, stock: Stock) -> float:
    if not isinstance(action.window, int):
        raise ValueError("volume-ratio threshold requires an integer window")
    average_volume = float(stock.data["Volume"].rolling(window=action.window, min_periods=1).mean().iloc[-1])
    if average_volume == 0:
        return 0.0
    return float(stock.data["Volume"].iloc[-1]) / average_volume


def _threshold_triggered(action: TradeAction, compare_value: float, portfolio: Portfolio, target_ticker: str, stock: Stock) -> bool:
    threshold_type, threshold_value = action.threshold
    if threshold_type == "volume-ratio":
        return _volume_ratio(action, stock) >= float(threshold_value)
    if threshold_type == "percent-change":
        threshold = float(threshold_value)
    elif threshold_type == "point":
        threshold = portfolio.buy_value[target_ticker] + float(threshold_value)
    elif threshold_type == "profit-rate":
        threshold = portfolio.buy_value[target_ticker] * (100 + float(threshold_value)) / 100
    else:
        raise ValueError(f"Unsupported threshold type: {threshold_type}")
    return compare_value <= threshold if float(threshold_value) <= 0 else compare_value >= threshold


def evaluate_trade_action(
    action: TradeAction,
    target_ticker: str,
    order_type: Literal["buy", "sell"],
    portfolio: Portfolio,
    stocks: list[Stock],
    portfolio_weight: float = 1.0,
) -> Action | None:
    target_stock = _find_stock(target_ticker, stocks)
    indicator_stock = _find_stock(action.ticker, stocks)
    price = float(target_stock.data[action.price_point].iloc[-1])
    compare_value = _indicator_value(action, indicator_stock)

    if action.threshold[0] != "percent-change" and portfolio.buy_value[target_ticker] == 0 and order_type == "buy":
        triggered = True
    else:
        triggered = _threshold_triggered(action, compare_value, portfolio, target_ticker, indicator_stock)

    if not triggered:
        return None

    from pybacktest.strategy import StrategyManager

    quantity_type = action.quantity[0]
    quantity_value = action.quantity[1]
    if order_type == "buy" and quantity_type == "split":
        quantity_type = "value"
        quantity_value = (portfolio.initial_capital / quantity_value) * portfolio_weight

    return StrategyManager.create_action(
        order_type,
        target_ticker,
        price,
        quantity_type,
        quantity_value,
        portfolio,
    )
```

- [ ] **Step 5: Delegate `StrategyManager.apply_strategy()` gradually**

In `src/pybacktest/strategy.py`, import:

```python
from pybacktest.signals import evaluate_trade_action
```

Replace the body of `apply_strategy()` with:

```python
actions = []
portfolio_weight = strategy.portfolio_weight or 1.0
buy_action = evaluate_trade_action(
    strategy.buy,
    ticker,
    "buy",
    portfolio,
    stocks,
    portfolio_weight=portfolio_weight,
)
if buy_action is not None:
    actions.append(buy_action)
sell_action = evaluate_trade_action(strategy.sell, ticker, "sell", portfolio, stocks)
if sell_action is not None:
    actions.append(sell_action)
return actions
```

- [ ] **Step 6: Verify signal and legacy strategy tests**

Run:

```bash
uv run pytest tests/test_signals.py tests/test_backtest.py::test_strategy tests/test_features.py::test_split_purchase -q
```

Expected:

```text
4 passed
```

- [ ] **Step 7: Commit**

```bash
git add src/pybacktest/signals.py src/pybacktest/strategy.py tests/test_signals.py
git commit -m "feat: add volume-aware signal evaluation"
```

---

### Task 5: Add Configurable Rebalancing With Full Default

**Files:**
- Create: `src/pybacktest/rebalancing.py`
- Modify: `src/pybacktest/strategy.py`
- Modify: `src/pybacktest/backtest.py`
- Create: `tests/test_rebalancing.py`

- [ ] **Step 1: Write failing rebalancing tests**

Create `tests/test_rebalancing.py`:

```python
import pandas as pd
import pytest

from pybacktest.models import Portfolio, Stock
from pybacktest.rebalancing import RebalanceConfig, generate_rebalance_actions, validate_weights


def _stock(ticker: str, close: float = 100.0) -> Stock:
    stock = Stock(ticker, "2024-01-15", "2024-01-15", fetch=False)
    stock.data = pd.DataFrame(
        {"Close": [close], "Volume": [10000]},
        index=pd.to_datetime(["2024-01-15"]),
    )
    return stock


def test_full_rebalance_is_default_and_generates_sell_and_buy_actions():
    portfolio = Portfolio(0.0, ["A", "B"])
    portfolio.stock_count["A"] = 100
    portfolio.stock_count["B"] = 0
    config = RebalanceConfig(enabled=True)

    actions, warnings = generate_rebalance_actions(
        portfolio=portfolio,
        stocks=[_stock("A"), _stock("B")],
        weights={"A": 0.5, "B": 0.5},
        date=pd.Timestamp("2024-01-15"),
        config=config,
    )

    assert warnings == []
    assert [(a.ticker, a.type, a.quantity) for a in actions] == [
        ("A", "sell", 50),
        ("B", "buy", 50),
    ]


def test_sell_only_rebalance_keeps_existing_trim_behavior():
    portfolio = Portfolio(0.0, ["A", "B"])
    portfolio.stock_count["A"] = 100
    portfolio.stock_count["B"] = 0
    config = RebalanceConfig(enabled=True, mode="sell_only")

    actions, warnings = generate_rebalance_actions(
        portfolio=portfolio,
        stocks=[_stock("A"), _stock("B")],
        weights={"A": 0.5, "B": 0.5},
        date=pd.Timestamp("2024-01-15"),
        config=config,
    )

    assert warnings == []
    assert [(a.ticker, a.type, a.quantity) for a in actions] == [("A", "sell", 50)]


def test_weights_over_one_raise_error():
    with pytest.raises(ValueError, match="Target weights sum to 1.20"):
        validate_weights({"A": 0.7, "B": 0.5})
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_rebalancing.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'pybacktest.rebalancing'
```

- [ ] **Step 3: Implement `src/pybacktest/rebalancing.py`**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from pybacktest.models import Action, Portfolio, Stock


@dataclass
class RebalanceConfig:
    enabled: bool = False
    frequency: Literal["monthly"] = "monthly"
    day: int = 15
    mode: Literal["full", "sell_only"] = "full"


def validate_weights(weights: dict[str, float]) -> list[str]:
    total = round(sum(weights.values()), 10)
    if total > 1.0:
        raise ValueError(f"Target weights sum to {total:.2f}; must be <= 1.00.")
    if total < 1.0 and total > 0:
        return [f"Target weights sum to {total:.2f}; remaining {1.0 - total:.2f} stays as cash."]
    return []


def should_rebalance(date: pd.Timestamp, config: RebalanceConfig) -> bool:
    return config.enabled and config.frequency == "monthly" and date.day == config.day


def _current_prices(stocks: list[Stock]) -> dict[str, float]:
    return {stock.ticker: float(stock.data["Close"].iloc[-1]) for stock in stocks if not stock.data.empty}


def generate_rebalance_actions(
    portfolio: Portfolio,
    stocks: list[Stock],
    weights: dict[str, float],
    date: pd.Timestamp,
    config: RebalanceConfig,
) -> tuple[list[Action], list[str]]:
    if not should_rebalance(date, config):
        return [], []

    warnings = validate_weights(weights)
    prices = _current_prices(stocks)
    total_value = portfolio.cash + sum(
        portfolio.stock_count.get(ticker, 0) * prices.get(ticker, 0.0)
        for ticker in portfolio.tickers
    )

    actions: list[Action] = []
    for ticker, weight in weights.items():
        price = prices.get(ticker, 0.0)
        if price <= 0:
            continue
        current_value = portfolio.stock_count.get(ticker, 0) * price
        target_value = total_value * weight
        diff = target_value - current_value
        quantity = int(abs(diff) // price)
        if quantity <= 0:
            continue
        if diff < 0:
            actions.append(Action(ticker=ticker, type="sell", quantity=quantity, price=price))
        elif config.mode == "full":
            actions.append(Action(ticker=ticker, type="buy", quantity=quantity, price=price))

    actions.sort(key=lambda action: 0 if action.type == "sell" else 1)
    return actions, warnings
```

- [ ] **Step 4: Wire compatibility into `StrategyManager`**

Add a signal-only method so `Backtest` can opt into the new rebalance module without double-counting legacy `StrategyManager.apply()` rebalances:

```python
def apply_signals(
    self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp
) -> List[Action]:
    actions = []
    for ticker, strategy in self.strategies.items():
        actions.extend(self.apply_strategy(ticker, strategy, portfolio, stocks, date))
    return actions
```

Update `apply()` to call `apply_signals()` and preserve legacy direct-method behavior:

```python
def apply(
    self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp
) -> List[Action]:
    actions = self.apply_signals(portfolio, stocks, date)
    if date.day == 15:
        actions.extend(self.rebalance(portfolio, stocks, date))
    return actions
```

Keep `StrategyManager.rebalance()` but delegate to the new module with `sell_only` so current direct calls keep their behavior:

```python
from pybacktest.rebalancing import RebalanceConfig, generate_rebalance_actions
```

```python
def rebalance(self, portfolio: Portfolio, stocks: List[Stock], date: pd.Timestamp) -> List[Action]:
    weights = {
        ticker: strategy.portfolio_weight
        for ticker, strategy in self.strategies.items()
        if strategy.portfolio_weight > 0
    }
    actions, _warnings = generate_rebalance_actions(
        portfolio,
        stocks,
        weights,
        date,
        RebalanceConfig(enabled=True, day=date.day, mode="sell_only"),
    )
    return actions
```

- [ ] **Step 5: Add optional `Backtest` rebalance config**

In `Backtest.__init__()`, add a parameter:

```python
rebalance: dict | None = None,
```

Store:

```python
self.rebalance_config = RebalanceConfig(**rebalance) if rebalance else None
```

In `Backtest.run()`, replace the action-generation line:

```python
actions = strategy.apply(self.portfolio, stock_data, date)
```

with:

```python
if self.rebalance_config is None:
    actions = strategy.apply(self.portfolio, stock_data, date)
else:
    actions = strategy.apply_signals(self.portfolio, stock_data, date)
```

Then, after signal actions and before execution, add:

```python
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
```

- [ ] **Step 6: Verify rebalancing tests**

Run:

```bash
uv run pytest tests/test_rebalancing.py tests/test_features.py::test_rebalancing tests/test_features.py::test_rebalancing_mid_month -q
```

Expected:

```text
5 passed
```

- [ ] **Step 7: Commit**

```bash
git add src/pybacktest/rebalancing.py src/pybacktest/strategy.py src/pybacktest/backtest.py tests/test_rebalancing.py
git commit -m "feat: add configurable portfolio rebalancing"
```

---

### Task 6: Add Execution Module and Liquidity Limits

**Files:**
- Create: `src/pybacktest/execution.py`
- Modify: `src/pybacktest/backtest.py`
- Create: `tests/test_execution.py`

- [ ] **Step 1: Write failing execution tests**

Create `tests/test_execution.py`:

```python
import pandas as pd

from pybacktest.execution import ExecutionConfig, execute_actions
from pybacktest.models import Action, Portfolio, Stock


def _stock(ticker: str, close: float = 100.0, volume: int = 1000) -> Stock:
    stock = Stock(ticker, "2024-01-01", "2024-01-01", fetch=False)
    stock.data = pd.DataFrame(
        {"Close": [close], "Volume": [volume]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    return stock


def test_liquidity_limit_scales_buy_quantity():
    portfolio = Portfolio(100000.0, ["A"])
    trades, warnings = execute_actions(
        portfolio=portfolio,
        actions=[Action(ticker="A", type="buy", quantity=100, price=100.0)],
        stocks=[_stock("A", volume=1000)],
        date=pd.Timestamp("2024-01-01"),
        config=ExecutionConfig(liquidity_limit=0.05),
    )

    assert portfolio.stock_count["A"] == 50
    assert trades[0]["quantity"] == 50
    assert warnings == ["Scaled buy A from 100 to 50 due to liquidity limit."]


def test_sell_executes_before_buy_after_liquidity_scaling():
    portfolio = Portfolio(0.0, ["A", "B"])
    portfolio.stock_count["A"] = 100
    trades, warnings = execute_actions(
        portfolio=portfolio,
        actions=[
            Action(ticker="B", type="buy", quantity=100, price=100.0),
            Action(ticker="A", type="sell", quantity=100, price=100.0),
        ],
        stocks=[_stock("A", volume=10000), _stock("B", volume=10000)],
        date=pd.Timestamp("2024-01-01"),
        config=ExecutionConfig(),
    )

    assert [trade["type"] for trade in trades] == ["sell", "buy"]
    assert portfolio.stock_count == {"A": 0, "B": 100}
    assert warnings == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_execution.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'pybacktest.execution'
```

- [ ] **Step 3: Implement `src/pybacktest/execution.py`**

Create:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from pybacktest.models import Action, Portfolio, Stock


@dataclass
class ExecutionConfig:
    liquidity_limit: float | None = None


def _volume_by_ticker(stocks: list[Stock]) -> dict[str, int]:
    volumes = {}
    for stock in stocks:
        if not stock.data.empty and "Volume" in stock.data:
            volumes[stock.ticker] = int(stock.data["Volume"].iloc[-1])
    return volumes


def _apply_liquidity_limit(action: Action, volumes: dict[str, int], limit: float | None) -> tuple[Action, str | None]:
    if limit is None:
        return action, None
    max_quantity = math.floor(volumes.get(action.ticker, 0) * limit)
    if action.quantity <= max_quantity:
        return action, None
    scaled = action.model_copy(update={"quantity": max_quantity})
    warning = f"Scaled {action.type} {action.ticker} from {action.quantity} to {max_quantity} due to liquidity limit."
    return scaled, warning


def execute_actions(
    portfolio: Portfolio,
    actions: list[Action],
    stocks: list[Stock],
    date: pd.Timestamp,
    config: ExecutionConfig,
) -> tuple[list[dict], list[str]]:
    volumes = _volume_by_ticker(stocks)
    warnings: list[str] = []
    trades: list[dict] = []
    buys: list[Action] = []
    sells: list[Action] = []

    for raw_action in actions:
        if raw_action.quantity <= 0:
            continue
        action, warning = _apply_liquidity_limit(raw_action, volumes, config.liquidity_limit)
        if warning:
            warnings.append(warning)
        if action.quantity <= 0:
            continue
        if action.type == "sell":
            sells.append(action)
        elif action.type == "buy":
            buys.append(action)

    for action in sells:
        if portfolio.stock_count[action.ticker] < action.quantity:
            raise ValueError(f"Not enough shares to sell {action.quantity} of {action.ticker} on {date}! Check your strategy.")
        portfolio.update(action.ticker, -action.quantity, action.price)
        trades.append({"date": date, "ticker": action.ticker, "type": "sell", "quantity": action.quantity, "price": action.price})

    total_buy_cost = sum(action.price * action.quantity for action in buys)
    ratio = 1.0
    if total_buy_cost > portfolio.cash and portfolio.cash > 0:
        ratio = portfolio.cash / total_buy_cost
        warnings.append(f"Insufficient cash on {date}. Scaling buy orders by ratio {ratio:.4f}.")
    elif total_buy_cost > portfolio.cash and portfolio.cash <= 0:
        warnings.append(f"No cash available on {date} to process buy orders.")
        return trades, warnings

    for action in buys:
        quantity = math.floor(action.quantity * ratio)
        cost = quantity * action.price
        if quantity > 0 and portfolio.cash >= cost:
            portfolio.update(action.ticker, quantity, action.price)
            trades.append({"date": date, "ticker": action.ticker, "type": "buy", "quantity": quantity, "price": action.price})

    return trades, warnings
```

- [ ] **Step 4: Delegate `Backtest.execute_action()`**

In `src/pybacktest/backtest.py`, import:

```python
from pybacktest.execution import ExecutionConfig, execute_actions
```

In `Backtest.__init__()`, add:

```python
execution: dict | None = None,
```

Store:

```python
self.execution_config = ExecutionConfig(**execution) if execution else ExecutionConfig()
```

Replace `execute_action()` body with:

```python
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
```

Rename the module import `import warnings` to:

```python
import warnings as warnings_module
```

Before `self.execute_action(actions, date, strategy)` in `run()`, set:

```python
self.current_stock_data = stock_data
```

Capture warnings:

```python
execution_warnings = self.execute_action(actions, date, strategy)
strategy_result.warnings.extend(execution_warnings)
```

- [ ] **Step 5: Verify execution and cash allocation tests**

Run:

```bash
uv run pytest tests/test_execution.py tests/test_features.py::test_fair_cash_allocation tests/test_backtest.py::test_execute_action -q
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/pybacktest/execution.py src/pybacktest/backtest.py tests/test_execution.py
git commit -m "feat: add liquidity-aware execution"
```

---

### Task 7: Add Portfolio Projection

**Files:**
- Create: `src/pybacktest/projection.py`
- Modify: `src/pybacktest/backtest.py`
- Create: `tests/test_projection.py`

- [ ] **Step 1: Write failing projection tests**

Create `tests/test_projection.py`:

```python
import pandas as pd

from pybacktest.models import Portfolio, Stock
from pybacktest.projection import ProjectionConfig, project_portfolio


def _stock(ticker: str) -> Stock:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    stock = Stock(ticker, "2024-01-01", "2024-01-08", fetch=False)
    stock.data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 99.0, 102.0, 103.0, 104.0, 103.0, 105.0],
            "Volume": [1000] * 8,
        },
        index=dates,
    )
    return stock


def test_projection_returns_value_bands_for_future_dates():
    portfolio = Portfolio(100.0, ["A"])
    portfolio.stock_count["A"] = 10

    projection, warnings = project_portfolio(
        portfolio=portfolio,
        stocks=[_stock("A")],
        config=ProjectionConfig(enabled=True, days=5, scenarios=25, lookback=5, random_seed=7),
    )

    assert warnings == []
    assert list(projection.columns) == ["low", "median", "high"]
    assert len(projection) == 5
    assert projection.index[0] > pd.Timestamp("2024-01-08")
    assert (projection["low"] <= projection["median"]).all()
    assert (projection["median"] <= projection["high"]).all()


def test_projection_skips_with_too_little_history():
    portfolio = Portfolio(100.0, ["A"])
    portfolio.stock_count["A"] = 10
    stock = _stock("A")
    stock.data = stock.data.iloc[:2]

    projection, warnings = project_portfolio(
        portfolio=portfolio,
        stocks=[stock],
        config=ProjectionConfig(enabled=True, days=5, scenarios=25, lookback=5),
    )

    assert projection is None
    assert warnings == ["Projection skipped: at least 3 return observations are required."]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/test_projection.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'pybacktest.projection'
```

- [ ] **Step 3: Implement `src/pybacktest/projection.py`**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pybacktest.models import Portfolio, Stock


@dataclass
class ProjectionConfig:
    enabled: bool = False
    days: int = 30
    scenarios: int = 500
    lookback: int = 60
    random_seed: int | None = None


def project_portfolio(
    portfolio: Portfolio,
    stocks: list[Stock],
    config: ProjectionConfig,
) -> tuple[pd.DataFrame | None, list[str]]:
    if not config.enabled:
        return None, []

    returns_by_ticker = {}
    latest_prices = {}
    for stock in stocks:
        if stock.ticker not in portfolio.stock_count or portfolio.stock_count[stock.ticker] <= 0:
            continue
        returns = stock.data["Close"].pct_change().dropna().tail(config.lookback)
        if len(returns) >= 3:
            returns_by_ticker[stock.ticker] = returns.to_numpy()
            latest_prices[stock.ticker] = float(stock.data["Close"].iloc[-1])

    if not returns_by_ticker:
        return None, ["Projection skipped: at least 3 return observations are required."]

    rng = np.random.default_rng(config.random_seed)
    paths = np.zeros((config.scenarios, config.days))
    tickers = list(returns_by_ticker)

    for scenario in range(config.scenarios):
        prices = latest_prices.copy()
        for day in range(config.days):
            value = portfolio.cash
            for ticker in tickers:
                sampled_return = rng.choice(returns_by_ticker[ticker])
                prices[ticker] = prices[ticker] * (1 + sampled_return)
                value += portfolio.stock_count[ticker] * prices[ticker]
            paths[scenario, day] = value

    start_date = max(stock.data.index.max() for stock in stocks if not stock.data.empty)
    index = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=config.days)
    projection = pd.DataFrame(
        {
            "low": np.percentile(paths, 10, axis=0),
            "median": np.percentile(paths, 50, axis=0),
            "high": np.percentile(paths, 90, axis=0),
        },
        index=index,
    )
    return projection, []
```

- [ ] **Step 4: Wire projection into `Backtest`**

In `src/pybacktest/backtest.py`, import:

```python
from pybacktest.projection import ProjectionConfig, project_portfolio
```

In `Backtest.__init__()`, add:

```python
projection: dict | None = None,
```

Store:

```python
self.projection_config = ProjectionConfig(**projection) if projection else ProjectionConfig()
```

After each strategy historical run, add:

```python
projection_df, projection_warnings = project_portfolio(
    self.portfolio,
    self.stocks,
    self.projection_config,
)
strategy_result.projection = projection_df
strategy_result.warnings.extend(projection_warnings)
```

- [ ] **Step 5: Verify projection tests**

Run:

```bash
uv run pytest tests/test_projection.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/pybacktest/projection.py src/pybacktest/backtest.py tests/test_projection.py
git commit -m "feat: add portfolio projection bands"
```

---

### Task 8: Update README Examples and Add Smoke Test

**Files:**
- Modify: `README.md`
- Create: `tests/test_readme_examples.py`

- [ ] **Step 1: Write failing README schema smoke test**

Create `tests/test_readme_examples.py`:

```python
from pybacktest.strategy import StrategyWrapper


def test_readme_strategy_schema_validates():
    strategy_json = {
        "AAPL": {
            "buy": {
                "ticker": "AAPL",
                "indicator": ["current", "Change_Pct"],
                "window": False,
                "threshold": ["percent-change", 0.5],
                "quantity": ["count", 10],
                "price_point": "Close",
            },
            "sell": {
                "ticker": "AAPL",
                "indicator": ["current", "Close"],
                "window": False,
                "threshold": ["profit-rate", 10],
                "quantity": ["percent", 100],
                "price_point": "Close",
            },
            "portfolio_weight": 0.5,
        }
    }

    strategy = StrategyWrapper.model_validate(strategy_json)

    assert strategy["AAPL"].buy.indicator == ["current", "Change_Pct"]
    assert strategy["AAPL"].buy.price_point == "Close"
    assert strategy["AAPL"].portfolio_weight == 0.5
```

- [ ] **Step 2: Run smoke test**

Run:

```bash
uv run pytest tests/test_readme_examples.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 3: Update README schema names**

In `README.md`, replace every strategy example field:

```text
by -> indicator
period -> window
criteria -> threshold
trade_as -> price_point
```

Add a rebalancing/projection example:

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

- [ ] **Step 4: Verify README smoke test and full suite**

Run:

```bash
uv run pytest -q
```

Expected:

```text
All tests pass
```

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme_examples.py
git commit -m "docs: update strategy schema examples"
```

---

### Task 9: Final Integration Verification

**Files:**
- Review all modified files.

- [ ] **Step 1: Run full tests**

Run:

```bash
uv run pytest -q
```

Expected:

```text
All tests pass
```

- [ ] **Step 2: Run focused import check**

Run:

```bash
uv run python -c "from pybacktest.backtest import Backtest; from pybacktest.strategy import StrategyWrapper; from pybacktest.models import Stock; print('ok')"
```

Expected:

```text
ok
```

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git diff --stat HEAD
```

Expected:

```text
Only engine, tests, and README files changed.
```

- [ ] **Step 4: Commit any final small fixes**

If Step 1 or Step 2 required a small corrective patch, commit it:

```bash
git add .
git commit -m "fix: stabilize engine core upgrade"
```

If no final patch was needed, do not create an empty commit.
