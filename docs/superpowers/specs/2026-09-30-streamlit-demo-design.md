# Streamlit Demo for Pybacktest 0.2 — Design Spec

**Date:** 2026-09-30
**Status:** Approved in conversation; awaiting written-spec review
**Scope:** A deployable Streamlit showcase that runs Pybacktest 0.2 on yfinance
data. No change to the core `pybacktest` package.

---

## 1. Intent

**Stated by the user**

- Purpose: a demo / showcase page for Pybacktest, deployed publicly.
- Deployment: the Streamlit Community Cloud app already connected to this
  repository. Its main file path is the V1 entrypoint, repository-root
  `streamlit_page.py` (deleted in `fa4282e`, so the deployed app is currently
  broken).
- Data: yfinance.
- Strategies: two or three demo strategies implemented in the UI package.
- Language: Korean / English toggle.
- Universe: multiple tickers per run (up to five).

**Assumed and accepted**

- The UI lives outside the core. It consumes only the public `pybacktest`
  API; Streamlit and yfinance never become core dependencies or extras.
- Every ticker in one run shares one quote currency. The core has no FX
  accounting, so mixed currencies are rejected before the run.
- Capital is split equally: each instrument targets weight `1/N`.

**Success criteria**

1. From the Cloud link, a visitor can enter tickers, pick a strategy, run it,
   and read the results without an unhandled error.
2. Data conversion, strategies, and result shaping are tested without network
   access.
3. Existing CI stays green, and the core suite still proves that
   `import pybacktest` loads neither Streamlit nor yfinance.

## 2. Non-goals

Saving or loading strategies, parameter optimization, intraday bars, short
selling, multi-currency FX, user-supplied code execution, custom theming, and
any change to `src/pybacktest`. `legacy/v1/ui` stays untouched.

## 3. Structure and deployment

```
streamlit_page.py          # Cloud entrypoint: puts src/ and repo root on sys.path, calls app.main()
streamlit_ui/
  __init__.py
  i18n.py                  # STRINGS = {"en": {...}, "ko": {...}}; t(key, lang)
  market_data.py           # fetch_history (network) + pure validation/conversion
  strategies.py            # BuyAndHold, MovingAverageCrossAll, RsiReversion
  runner.py                # RunConfig -> engine runs -> RunReport
  app.py                   # Streamlit widgets, charts, tables only
tests_ui/                  # UI tests; no network
```

**Dependencies.** `pyproject.toml` gains a `ui` dependency group
(`streamlit`, `yfinance`, `pandas`) and `[tool.uv] default-groups = ["dev", "ui"]`.
Community Cloud gives `uv.lock` top priority and installs it with `uv sync`,
which installs the project and the default groups but no extras. `uv.lock` is
regenerated in the same change.

**Core isolation.** `tests/` never imports `streamlit_ui`, so
`tests/test_package.py`'s in-process check that Streamlit and yfinance are
never imported stays meaningful.

**CI changes.**

- `test-core` switches from `--no-dev` to `--no-default-groups`, so it still
  runs without pandas, pyarrow, Streamlit, or yfinance.
- A new `ui` job runs `python -m pytest tests_ui -q` on Python 3.11 and 3.14.
- Ruff lint and format checks cover `streamlit_ui`, `tests_ui`, and
  `streamlit_page.py` in addition to the current paths.

## 4. Data flow: yfinance to `MarketDataSet`

1. **Input.** Up to five comma-separated tickers (for example `AAPL, MSFT` or
   `005930.KS, 000660.KS`), a start and end date (default: the last five
   years), daily bars only. Tickers are trimmed, upper-cased, and
   de-duplicated.
2. **Fetch** — `fetch_history(tickers, start, end)`, the only network
   function, wrapped in `st.cache_data(ttl=3600)` by the app. Per ticker it
   calls `yf.Ticker(t).history(start=..., end=..., auto_adjust=True)` and reads
   the currency from `fast_info["currency"]`. It returns plain
   `{ticker: (DataFrame, currency)}` data so the pure functions below can be
   tested with fabricated frames.
3. **Validate** (pure). Each failure raises `DemoInputError(code, detail)`,
   whose `code` maps to a localized message:
   - `no_tickers`, `too_many_tickers` (more than five);
   - `empty_history` naming the ticker(s) with no rows;
   - `mixed_currency` listing each ticker's currency;
   - `insufficient_history` when fewer bars remain than the chosen strategy's
     warm-up requires plus a margin of 5 bars.
4. **Convert** — `to_market_dataset(histories, currency)` (pure):
   - `InstrumentId.parse(f"YF:{ticker}")` (verified for `AAPL`, `005930.KS`,
     `BRK-B`, `^GSPC`);
   - `tick_size` is `1` for zero-decimal currencies (`KRW`, `JPY`) and `0.01`
     otherwise; OHLC are rounded to the tick with `ROUND_HALF_EVEN`, because
     the core risk policy rejects marks that are not tick-aligned;
   - `lot_size = 1`, instrument `timezone = UTC`;
   - rows with any NaN in OHLCV are dropped; the index is converted to UTC and
     then to naive `datetime64[ns]`;
   - one `BarSeries` per instrument, combined into a `MarketDataSet` with
     `Timeframe.days(1)`.
5. **Calendar.** `CalendarPolicy.union()`, so instruments with different
   holidays still run together.

## 5. Strategies

All strategies are frozen, stateless dataclasses that implement the public
`Strategy` protocol (`build_features` / `on_bar`). Each is constructed with the
tuple of instruments; each instrument's target weight is `Decimal(1) / N`,
quantized to 6 decimal places. Feature names are namespaced per instrument,
for example `fast:YF:AAPL`, so plans for several instruments never collide.

| Strategy | Rule | Parameters | Warm-up bars |
|---|---|---|---|
| `BuyAndHold` | Target `1/N` when the instrument has no position and no active order; otherwise emit nothing. | — | 1 |
| `MovingAverageCrossAll` | Per instrument: fast SMA crosses above slow → `1/N`; crosses below → `0`. Same crossover semantics as the core `MovingAverageCross` (current and one-bar-lagged SMAs). | `fast < slow` | `slow + 1` |
| `RsiReversion` | Per instrument: RSI below `lower` → `1/N`; RSI above `upper` → `0`. | `period` 2–30, `lower < upper` within 1–99 | `period + 1` |

`RsiReversion` computes a simple-average RSI (Cutler's RSI) in `on_bar` from
close lags `0..period`, declared with `FeatureBuilder.lag` so they follow each
instrument's own clock. With average gain `G` and average loss `L` over the
last `period` differences: `L == 0` gives RSI `100` (or `50` when `G == 0` as
well); otherwise `RSI = 100 - 100 / (1 + G / L)`. Any NaN input emits nothing.

Invalid parameters raise the core `ConfigurationError` at construction, so the
runner reports them like any other configuration error.

## 6. Running and results

`RunConfig` (frozen dataclass): tickers, start, end, strategy name and
parameters, initial cash amount, commission per share (`0` means
`NoCommission`).

`run_backtest(config, histories) -> RunReport` validates, converts, builds the
engine, and runs twice on the same dataset: the chosen strategy and a
`BuyAndHold` benchmark.

- Broker: `SimulatedBrokerFactory(fill_model=NextBarOpenFill(IntrabarPolicy.CONSERVATIVE), commission=NoCommission() | PerShareCommission(rate), slippage=NoSlippage(), liquidity=NoLiquidityLimit(), borrow_cost=NoBorrowCost())`.
- Risk: `LongShortRisk(max_leverage=Decimal("1"), max_position_weight=None, allow_short=False)`.
- Simulation: universe, period `[start, end + 1 day)` in UTC, daily timeframe,
  union calendar, `Money.of(initial_cash, currency)`, `seed=0`,
  `MetricsConfig(risk_free_rate=Decimal("0"), annualization_periods=252)`.

`RunReport` holds display-ready data built by pure functions:

- `metrics`: rows for total return, CAGR, volatility, Sharpe, Sortino, maximum
  drawdown, win rate, and turnover, with strategy and benchmark columns
  (`None` shown as "—");
- `equity`: a DataFrame indexed by timestamp with `strategy` and `benchmark`
  equity columns;
- `orders` and `fills` tables (timestamp, instrument, side, quantity, price or
  status);
- `warnings`: warning codes and messages from both runs;
- `info`: pybacktest version, currency, bar count per instrument, and the
  configuration.

**Errors.** `DemoInputError` and the core `PybacktestError` family are caught
and shown as a localized `st.error`. Any other exception shows a generic
localized message with the traceback in an expander.

## 7. Page layout

`app.main()` with `st.set_page_config(layout="wide")`.

- **Sidebar:** language radio (English / 한국어), tickers, date range,
  strategy select with parameters shown for that strategy, initial cash,
  commission per share, and a **Run** button inside an `st.form`, so edits do
  not trigger reruns.
- **Main area:** before the first run, a short introduction and usage steps.
  After a run: a metric row comparing strategy and benchmark
  (total return, CAGR, Sharpe, maximum drawdown), an equity-curve
  `st.line_chart`, and tabs for Orders, Fills, Warnings, and Run info.
- The last `RunReport` is kept in `st.session_state`, so switching language
  re-renders without re-running.

**i18n.** `STRINGS = {"en": {...}, "ko": {...}}` and `t(key, lang)`. A missing
key raises `KeyError` in tests instead of rendering silently.

## 8. Testing (`tests_ui/`, no network)

- `test_market_data.py`: fabricated yfinance-shaped frames → tick rounding for
  USD and KRW, NaN-row removal, UTC conversion; each `DemoInputError` code.
- `test_strategies.py`: synthetic price paths → each strategy's targets on the
  expected bars; RSI against a hand-computed value; namespaced feature names
  for two instruments; parameter validation.
- `test_runner.py`: an end-to-end run on fabricated histories for two
  instruments → report shape, benchmark present, equity index aligned, and a
  core configuration error surfaced as a handled error.
- `test_i18n.py`: `en` and `ko` have identical key sets.
- `test_app.py`: `streamlit.testing.v1.AppTest` renders the page without an
  exception, and with `fetch_history` patched to return fabricated data,
  submitting the form shows the metrics and chart.

## 9. Rollout

1. Merge to `main`; the connected Cloud app redeploys from
   `streamlit_page.py`.
2. Verify the live app with one US run (`AAPL, MSFT`) and one Korean run
   (`005930.KS`).
3. Update the README with a short "Streamlit demo" section: the Cloud link and
   `uv run streamlit run streamlit_page.py`.
