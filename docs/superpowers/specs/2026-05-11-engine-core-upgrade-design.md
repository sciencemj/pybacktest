# Engine Core Upgrade Design

## Purpose

Improve Pybacktest as a Python-first backtesting library before redesigning the Streamlit webview. The first implementation phase will strengthen the core engine, add configurable portfolio rebalancing, add trading-volume support, add portfolio projection, and fix existing correctness and usability issues discovered during the audit.

## Current Findings

The current test suite does not run cleanly with plain `uv run pytest` because the editable-install `.pth` file in `.venv` is hidden on macOS, and Python 3.13 skips hidden `.pth` files. Running with `PYTHONPATH=src uv run pytest -q` passes all current tests: 9 passed, 1 cash-scaling warning.

Several existing issues should be fixed as part of the engine foundation:

- README examples use old strategy field names: `by`, `period`, `criteria`, and `trade_as`. The current models use `indicator`, `window`, `threshold`, and `price_point`.
- Rebalancing is hard-coded to trigger on day 15 and currently performs sell-only trimming.
- `Backtest.run()` resets the portfolio per strategy but stores `daily_snapshots` on the shared `Backtest` object, which can mix snapshots across strategies.
- `Stock.data_processing()` assumes exactly five yfinance columns, so empty downloads or data that includes `Adj Close` can raise `ValueError`.
- Results are spread across mutable attributes, making the webview harder to improve and making warnings/errors less visible.

## Approved Approach

Use a modular core design. Keep the existing public objects and JSON strategy shape recognizable, but split responsibilities into small engine modules with clear interfaces.

The existing API should continue to work:

```python
backtest = Backtest(stocks, [strategy], initial_capital=100000)
results = backtest.run()
```

New capabilities will be opt-in through optional configuration:

```python
backtest = Backtest(
    stocks,
    [strategy],
    initial_capital=100000,
    rebalance={"enabled": True, "frequency": "monthly", "day": 15, "mode": "full"},
    execution={"liquidity_limit": 0.05},
    projection={"enabled": True, "days": 30, "scenarios": 500},
)
results = backtest.run()
```

## Core Modules

### `data.py`

Normalize yfinance and manual OHLCV data into a consistent schema:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Change`
- `Change_Pct`

The normalizer must handle empty downloads with a clear data error, tolerate common yfinance shapes including adjusted-close data, and calculate volume-derived fields needed by strategy rules, such as rolling volume averages and volume ratios.

### `signals.py`

Evaluate strategy rules against normalized stock data. Existing price indicators remain supported. Volume indicators become first-class strategy fields.

Example volume rule:

```json
{
  "ticker": "AAPL",
  "indicator": ["average", "Volume"],
  "window": 20,
  "threshold": ["volume-ratio", 1.5],
  "quantity": ["value", 1000]
}
```

The signal layer should return proposed actions without mutating portfolio state.

### `rebalancing.py`

Generate rebalance actions from target portfolio weights.

Default mode: `full`.

Supported modes:

- `full`: sell overweight assets and buy underweight assets toward target weights.
- `sell_only`: sell overweight assets only and leave released cash for later strategy buys.

Supported schedule for the first implementation:

- monthly frequency
- configurable day of month, defaulting to 15 for compatibility with current behavior
- disabled mode for users who do not want scheduled rebalancing

Weight validation:

- weights above `1.0` total should be an error
- weights below `1.0` total are allowed and leave the remainder as cash, with a warning

### `execution.py`

Convert proposed actions into filled trades.

Execution rules:

- process sells before buys
- preserve current fair cash allocation when buy demand exceeds available cash
- optionally apply a liquidity limit, expressed as max percentage of daily market volume
- record skipped or scaled trades as result warnings

Liquidity limits should apply to both buy and sell actions. If an action exceeds the allowed quantity for the day, it should be scaled down rather than silently ignored.

### `projection.py`

Add portfolio projection as scenario-based forward simulation, not exact price prediction.

Projection behavior:

- run after historical backtest
- use recent historical returns for the currently held assets
- simulate future portfolio value paths
- output value bands such as low, median, and high paths over future dates
- skip projection with a warning when there is too little history

The projection result should be clearly labeled as scenario projection, not a trading recommendation or exact forecast.

### `results.py`

Provide structured result objects for:

- equity curve per strategy
- trades per strategy
- daily snapshots per strategy
- monthly snapshots per strategy
- final holdings and cash
- warnings
- projection output

The existing mutable attributes can remain during transition, but `Backtest.run()` should return structured results so the webview can later render richer summaries without scraping internal state.

## Data Flow

1. Load stock data through the normalization layer.
2. For each historical date, evaluate strategy buy/sell rules in `signals.py`.
3. Add scheduled rebalance actions in `rebalancing.py` when enabled.
4. Execute actions in `execution.py`, applying sell-first ordering, cash constraints, and optional liquidity limits.
5. Store isolated strategy results in `results.py`.
6. Run optional projection from final holdings and recent returns.
7. Return structured results from `Backtest.run()`.

## Error Handling

Errors and warnings should be explicit:

- invalid ticker or empty yfinance data: clear data error
- invalid strategy schema: pydantic validation error with current field names
- target weights sum above `1.0`: error
- target weights sum below `1.0`: warning
- liquidity limit scales or blocks a trade: warning
- projection enabled with too little history: warning and no projection result

Warnings should be available in the returned result object and still be visible to Streamlit later.

## Testing Scope

Engine tests should cover:

- data normalization for empty downloads, standard OHLCV data, and adjusted-close data
- existing price strategy rules
- new volume indicators
- full rebalancing as the default mode
- explicit `sell_only` rebalancing mode
- cash scaling and sell-first execution
- liquidity limits based on daily volume
- strategy-isolated daily and monthly results
- projection output shape and too-little-history warnings
- README examples using the current schema

## Out Of Scope

This phase will not include:

- full Streamlit redesign
- machine-learning price prediction
- broker integration or live trading
- intraday data support
- short selling, margin, futures, options, or other derivatives

## Follow-Up Webview Direction

After the engine upgrade, the Streamlit webview should consume structured results rather than internal attributes. The likely next UI phase is a clearer workflow for strategy setup, rebalancing settings, volume rules, execution assumptions, result summaries, and projection charts.
