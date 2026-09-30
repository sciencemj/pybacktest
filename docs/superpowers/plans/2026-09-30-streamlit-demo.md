# Streamlit Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a bilingual (EN/KO) Streamlit page that backtests up to five same-currency yfinance tickers with three demo strategies on the Pybacktest 0.2 engine, deployed from the existing Community Cloud app at `streamlit_page.py`.

**Architecture:** A repository-local `streamlit_ui/` package that uses only the public `pybacktest` API. Pure modules (`market_data`, `strategies`, `runner`, `i18n`) hold all logic and are tested without network; `app.py` holds only Streamlit calls; the repository-root `streamlit_page.py` is a thin entrypoint. Streamlit and yfinance come from a `ui` dependency group that is a uv default group, so Community Cloud's `uv sync` installs them while the core package stays NumPy-only.

**Tech Stack:** Python 3.11+, Pybacktest 0.2 (this repo), Streamlit ≥ 1.50 (verified on 1.64.0), yfinance ≥ 1.0 (verified on 1.1.0), pandas ≥ 2.2, pytest, `streamlit.testing.v1.AppTest`, uv, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-30-streamlit-demo-design.md`

## Global Constraints

- No change to `src/pybacktest/` or to the `[project]` dependencies/extras in `pyproject.toml`.
- `tests/` must never import `streamlit_ui`; demo tests live only in `tests_ui/`.
- Entrypoint path is exactly `streamlit_page.py` at the repository root.
- At most five tickers per run (`MAX_TICKERS = 5`); every ticker must share one currency.
- `InstrumentId` for a ticker is `YF:<TICKER>`; tick size `1` for `KRW`/`JPY`, `0.01` otherwise; `lot_size = 1`.
- Each instrument targets weight `1/N`, quantized to `0.000001`.
- Engine setup: `NextBarOpenFill(IntrabarPolicy.CONSERVATIVE)`, `NoSlippage`, `NoLiquidityLimit`, `NoBorrowCost`, `NoCommission` or `PerShareCommission`, `LongShortRisk(max_leverage=1, max_position_weight=None, allow_short=False)`, `CalendarPolicy.union(max_staleness_bars=5)`, `seed=0`, `MetricsConfig(risk_free_rate=0, annualization_periods=252)`.
- Every user-visible string goes through `streamlit_ui.i18n.t(key, lang)`; `en` and `ko` have identical keys.
- Ruff (`E, F, I, B, UP, SIM, RUF`, line length 88) must pass on `streamlit_ui`, `tests_ui`, `streamlit_page.py`.
- Tests never touch the network: `market_data.fetch_history` is the only network function and is patched in app tests.

## Review Focus

Failure modes the spec implies but a straightforward implementation would miss. Each has a pinned test in the owning task.

1. **A held instrument misses a bar** (local holiday, or a bar yfinance dropped): the run must still succeed. `CalendarPolicy.union()`'s default `max_staleness_bars=0` fails the whole run — pinned by `test_holiday_in_one_series_does_not_break_the_union_calendar` (Task 3).
2. **Adjusted yfinance bars with open/close outside `[low, high]`** (observed on `005930.KS`, `000660.KS`): conversion must repair the range, or the core rejects the dataset — pinned by `test_clean_frame_widens_high_low_to_contain_open_and_close` (Task 1).
3. **A ticker listed mid-period**: it must be bought once its data begins, without failing the run — pinned by `test_ticker_listed_mid_period_is_bought_when_its_data_starts` (Task 3).
4. **yfinance returns bars but no currency**: must be reported as missing data naming the ticker, not as a mixed-currency error — pinned by `test_to_market_dataset_treats_unknown_currency_as_missing_data` (Task 1).
5. **Changing only the strategy or the language after a run**: must not re-download data, and switching language must keep the last report — pinned by `test_changing_only_the_strategy_reuses_cached_data` and `test_switching_language_keeps_the_last_report` (Task 4).

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` (modify) | `ui` dependency group; `[tool.uv] default-groups = ["dev", "ui"]` |
| `uv.lock` (regenerate) | Lock the new group |
| `streamlit_ui/__init__.py` | Package marker and docstring |
| `streamlit_ui/errors.py` | `DemoInputError(code, detail)` |
| `streamlit_ui/market_data.py` | Ticker parsing, `fetch_history` (network), cleaning, validation, `to_market_dataset` |
| `streamlit_ui/strategies.py` | `BuyAndHold`, `MovingAverageCrossAll`, `RsiReversion`, `rsi`, helpers |
| `streamlit_ui/runner.py` | `RunConfig`, `RunReport`, `run_backtest`, result tables |
| `streamlit_ui/i18n.py` | `LANGUAGES`, `STRINGS`, `t` |
| `streamlit_ui/app.py` | Streamlit page (`main`) and `format_metric` |
| `streamlit_page.py` | Cloud entrypoint |
| `tests_ui/__init__.py`, `tests_ui/support.py`, `tests_ui/conftest.py` | Fabricated data helpers and fixtures |
| `tests_ui/test_*.py` | One test module per `streamlit_ui` module |
| `.github/workflows/ci.yml` (modify) | Lint paths, core-only flag, `ui` job |
| `README.md` (modify) | "Streamlit demo" section |

Run all commands from the repository root. `pyproject.toml`'s pytest `pythonpath = [".", "src"]` makes `streamlit_ui` and `tests_ui` importable.

---

### Task 1: Dependencies, package skeleton, and market-data conversion

**Files:**
- Modify: `pyproject.toml`
- Regenerate: `uv.lock`
- Create: `streamlit_ui/__init__.py`, `streamlit_ui/errors.py`, `streamlit_ui/market_data.py`
- Create: `tests_ui/__init__.py`, `tests_ui/support.py`, `tests_ui/conftest.py`
- Test: `tests_ui/test_market_data.py`

**Interfaces:**
- Consumes: public `pybacktest` names `BarSeries`, `Instrument`, `InstrumentId`, `MarketDataSet`, `Timeframe`.
- Produces:
  - `DemoInputError(code: str, detail: str = "")` with `.code`, `.detail`.
  - `TickerHistory(frame: pd.DataFrame, currency: str)` (frozen dataclass).
  - `MAX_TICKERS = 5`, `WARMUP_MARGIN_BARS = 5`.
  - `parse_tickers(raw: str) -> tuple[str, ...]`
  - `instrument_id(ticker: str) -> InstrumentId`
  - `tick_size_for(currency: str) -> Decimal`
  - `fetch_history(tickers: Sequence[str], start: date, end: date) -> dict[str, TickerHistory]` (network)
  - `clean_frame(frame: pd.DataFrame, tick: Decimal) -> pd.DataFrame`
  - `validate_histories(histories: Mapping[str, TickerHistory]) -> str`
  - `to_market_dataset(histories, *, min_bars: int) -> tuple[MarketDataSet, str]`
  - Test helpers `make_frame(closes, *, start="2022-01-03", tz="America/New_York")`, `random_walk(seed, bars=400)`, `fake_fetch(tickers, start, end)`; fixture `two_usd_histories`.

- [ ] **Step 1: Add the `ui` dependency group**

In `pyproject.toml`, add a `ui` group after the `dev` group inside `[dependency-groups]`, and a new `[tool.uv]` table:

```toml
ui = [
    "pandas>=2.2",
    "streamlit>=1.50",
    "yfinance>=1.0",
]
```

```toml
[tool.uv]
default-groups = ["dev", "ui"]
```

Then lock and sync:

```bash
uv lock
uv sync
uv run python -c "import streamlit, yfinance; print(streamlit.__version__, yfinance.__version__)"
```

Expected: two version numbers print (for example `1.64.0 1.1.0`).

- [ ] **Step 2: Create the package skeletons**

`streamlit_ui/__init__.py`:

```python
"""Streamlit demo for Pybacktest 0.2; not part of the core package."""
```

`streamlit_ui/errors.py`:

```python
"""Typed input errors the demo reports with a localized message."""

from __future__ import annotations


class DemoInputError(ValueError):
    """A user input problem detected before the engine runs.

    ``code`` selects the localized message (``error.<code>`` in
    :mod:`streamlit_ui.i18n`); ``detail`` is interpolated into it.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail
```

`tests_ui/__init__.py`: an empty file.

`tests_ui/support.py`:

```python
"""Fabricated yfinance-shaped data shared by the UI tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from streamlit_ui.market_data import TickerHistory


def make_frame(
    closes: list[float] | np.ndarray,
    *,
    start: str = "2022-01-03",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Build a yfinance ``history()``-shaped frame from closing prices."""
    close = np.asarray(closes, dtype=np.float64)
    index = pd.bdate_range(start, periods=len(close), tz=tz, name="Date")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), 1_000_000.0),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=index,
    )


def random_walk(seed: int, bars: int = 400) -> np.ndarray:
    """Seeded log random walk starting near 100."""
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.02, bars)))


def fake_fetch(
    tickers: Sequence[str],
    start: date,
    end: date,
) -> dict[str, TickerHistory]:
    """Stand-in for ``market_data.fetch_history`` with USD random walks."""
    del start, end
    return {
        ticker: TickerHistory(make_frame(random_walk(seed)), "USD")
        for seed, ticker in enumerate(tickers)
    }
```

`tests_ui/conftest.py`:

```python
"""Fixtures for the UI tests."""

from __future__ import annotations

import pytest

from streamlit_ui.market_data import TickerHistory
from tests_ui.support import make_frame, random_walk


@pytest.fixture
def two_usd_histories() -> dict[str, TickerHistory]:
    return {
        "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
        "MSFT": TickerHistory(make_frame(random_walk(2)), "USD"),
    }
```

- [ ] **Step 3: Write the failing tests**

`tests_ui/test_market_data.py`:

```python
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from pybacktest import InstrumentId
from streamlit_ui.errors import DemoInputError
from streamlit_ui.market_data import (
    MAX_TICKERS,
    TickerHistory,
    clean_frame,
    parse_tickers,
    tick_size_for,
    to_market_dataset,
)
from tests_ui.support import make_frame, random_walk


def test_parse_tickers_trims_upper_cases_and_deduplicates():
    assert parse_tickers(" aapl, MSFT ,aapl,, 005930.ks ") == (
        "AAPL",
        "MSFT",
        "005930.KS",
    )


@pytest.mark.parametrize(
    ("raw", "code"),
    [("", "no_tickers"), (" , ,", "no_tickers"), ("A,B,C,D,E,F", "too_many_tickers")],
)
def test_parse_tickers_rejects_empty_and_too_many(raw, code):
    with pytest.raises(DemoInputError) as caught:
        parse_tickers(raw)
    assert caught.value.code == code


def test_max_tickers_is_five():
    assert MAX_TICKERS == 5


@pytest.mark.parametrize(
    ("currency", "tick"),
    [
        ("USD", Decimal("0.01")),
        ("EUR", Decimal("0.01")),
        ("KRW", Decimal("1")),
        ("JPY", Decimal("1")),
    ],
)
def test_tick_size_for_currency(currency, tick):
    assert tick_size_for(currency) == tick


def test_clean_frame_rounds_to_tick_and_converts_to_naive_utc():
    frame = make_frame([100.004, 100.006], tz="America/New_York")
    cleaned = clean_frame(frame, Decimal("0.01"))
    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert cleaned["Close"].tolist() == [100.0, 100.01]
    assert cleaned.index.tz is None
    # 2022-01-03 00:00 New York is 05:00 UTC.
    assert cleaned.index[0] == pd.Timestamp("2022-01-03 05:00")


def test_clean_frame_rounds_krw_to_whole_won():
    frame = make_frame([71234.6, 71234.4], tz="Asia/Seoul")
    cleaned = clean_frame(frame, Decimal("1"))
    assert cleaned["Close"].tolist() == [71235.0, 71234.0]


def test_clean_frame_drops_rows_with_missing_values():
    frame = make_frame([100.0, 101.0, 102.0])
    frame.iloc[1, frame.columns.get_loc("Close")] = np.nan
    assert len(clean_frame(frame, Decimal("0.01"))) == 2


def test_clean_frame_widens_high_low_to_contain_open_and_close():
    frame = make_frame([100.0])
    frame.loc[frame.index[0], ["Open", "High", "Low", "Close"]] = [
        101.0,
        100.5,
        100.2,
        99.0,
    ]
    cleaned = clean_frame(frame, Decimal("0.01"))
    assert cleaned.iloc[0]["High"] == 101.0
    assert cleaned.iloc[0]["Low"] == 99.0


def test_clean_frame_of_frame_without_ohlcv_is_empty():
    empty = pd.DataFrame(columns=["Open", "Close"])
    assert clean_frame(empty, Decimal("0.01")).empty


def test_to_market_dataset_builds_one_series_per_ticker(two_usd_histories):
    dataset, currency = to_market_dataset(two_usd_histories, min_bars=10)
    assert currency == "USD"
    assert set(dataset.series) == {
        InstrumentId.parse("YF:AAPL"),
        InstrumentId.parse("YF:MSFT"),
    }
    instrument = dataset.instruments[InstrumentId.parse("YF:AAPL")]
    assert instrument.tick_size == Decimal("0.01")
    assert instrument.quote_currency == "USD"


def test_to_market_dataset_rejects_empty_history():
    histories = {
        "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
        "NOPE": TickerHistory(pd.DataFrame(), ""),
    }
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=10)
    assert caught.value.code == "empty_history"
    assert "NOPE" in caught.value.detail


def test_to_market_dataset_rejects_mixed_currency():
    histories = {
        "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
        "005930.KS": TickerHistory(make_frame(random_walk(2)), "KRW"),
    }
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=10)
    assert caught.value.code == "mixed_currency"
    assert "AAPL=USD" in caught.value.detail
    assert "005930.KS=KRW" in caught.value.detail


def test_to_market_dataset_rejects_history_shorter_than_warmup():
    histories = {"AAPL": TickerHistory(make_frame(random_walk(1, bars=20)), "USD")}
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=61)
    assert caught.value.code == "insufficient_history"
    assert "AAPL=20" in caught.value.detail


def test_to_market_dataset_treats_unknown_currency_as_missing_data():
    histories = {"AAPL": TickerHistory(make_frame(random_walk(1)), "")}
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=10)
    assert caught.value.code == "empty_history"
    assert caught.value.detail == "AAPL"
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run python -m pytest tests_ui/test_market_data.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'streamlit_ui.market_data'`.

- [ ] **Step 5: Implement `streamlit_ui/market_data.py`**

```python
"""yfinance history fetching and conversion into a core ``MarketDataSet``.

Only :func:`fetch_history` touches the network. Everything else is pure so it
can be tested with fabricated yfinance-shaped frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pybacktest import (
    BarSeries,
    Instrument,
    InstrumentId,
    MarketDataSet,
    Timeframe,
)
from streamlit_ui.errors import DemoInputError

MAX_TICKERS = 5
WARMUP_MARGIN_BARS = 5
ZERO_DECIMAL_CURRENCIES = frozenset({"KRW", "JPY"})
OHLCV = ("Open", "High", "Low", "Close", "Volume")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True, slots=True)
class TickerHistory:
    """One ticker's raw daily bars and quote currency."""

    frame: pd.DataFrame
    currency: str


def parse_tickers(raw: str) -> tuple[str, ...]:
    """Split comma-separated input into trimmed, upper-cased unique tickers."""
    seen: dict[str, None] = {}
    for part in raw.split(","):
        ticker = part.strip().upper()
        if ticker:
            seen.setdefault(ticker, None)
    tickers = tuple(seen)
    if not tickers:
        raise DemoInputError("no_tickers")
    if len(tickers) > MAX_TICKERS:
        raise DemoInputError("too_many_tickers", str(MAX_TICKERS))
    return tickers


def instrument_id(ticker: str) -> InstrumentId:
    """Map a yfinance ticker to the demo's instrument identifier."""
    return InstrumentId.parse(f"YF:{ticker}")


def tick_size_for(currency: str) -> Decimal:
    """Return the price increment used for ``currency``."""
    return Decimal("1") if currency in ZERO_DECIMAL_CURRENCIES else Decimal("0.01")


def fetch_history(
    tickers: Sequence[str],
    start: date,
    end: date,
) -> dict[str, TickerHistory]:
    """Download adjusted daily bars for each ticker (network access)."""
    import yfinance as yf

    histories: dict[str, TickerHistory] = {}
    for ticker in tickers:
        handle = yf.Ticker(ticker)
        frame = handle.history(start=start, end=end, auto_adjust=True)
        try:
            currency = str(handle.fast_info["currency"] or "").upper()
        except Exception:  # yfinance raises assorted errors for unknown tickers
            currency = ""
        histories[ticker] = TickerHistory(frame=frame, currency=currency)
    return histories


def clean_frame(frame: pd.DataFrame, tick: Decimal) -> pd.DataFrame:
    """Keep OHLCV, drop incomplete rows, round prices, index by naive UTC."""
    missing = [column for column in OHLCV if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=list(OHLCV))
    cleaned = frame.loc[:, list(OHLCV)].astype("float64").dropna()
    index = pd.DatetimeIndex(cleaned.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    cleaned.index = index.tz_localize(None)
    prices = ["Open", "High", "Low", "Close"]
    for column in prices:
        cleaned[column] = [_round_to_tick(value, tick) for value in cleaned[column]]
    # Adjusted yfinance bars can leave open/close outside [low, high]; widen
    # the range so every bar satisfies the core OHLC invariant.
    cleaned["High"] = cleaned[prices].max(axis=1)
    cleaned["Low"] = cleaned[prices].min(axis=1)
    return cleaned[cleaned["Low"] > 0]


def _round_to_tick(value: float, tick: Decimal) -> float:
    steps = (Decimal(repr(value)) / tick).quantize(Decimal("1"), ROUND_HALF_EVEN)
    return float(steps * tick)


def validate_histories(histories: Mapping[str, TickerHistory]) -> str:
    """Return the single shared currency, or raise ``DemoInputError``."""
    empty = [ticker for ticker, item in histories.items() if item.frame.empty]
    if empty:
        raise DemoInputError("empty_history", ", ".join(empty))
    currencies = {ticker: item.currency for ticker, item in histories.items()}
    unknown = [ticker for ticker, currency in currencies.items() if not currency]
    if unknown:
        raise DemoInputError("empty_history", ", ".join(unknown))
    if len(set(currencies.values())) != 1:
        detail = ", ".join(f"{ticker}={code}" for ticker, code in currencies.items())
        raise DemoInputError("mixed_currency", detail)
    return next(iter(currencies.values()))


def to_market_dataset(
    histories: Mapping[str, TickerHistory],
    *,
    min_bars: int,
) -> tuple[MarketDataSet, str]:
    """Validate ``histories`` and convert them into one core dataset."""
    currency = validate_histories(histories)
    tick = tick_size_for(currency)
    series: dict[InstrumentId, BarSeries] = {}
    instruments: dict[InstrumentId, Instrument] = {}
    short: list[str] = []
    for ticker, item in histories.items():
        frame = clean_frame(item.frame, tick)
        if len(frame) < min_bars + WARMUP_MARGIN_BARS:
            short.append(f"{ticker}={len(frame)}")
            continue
        identifier = instrument_id(ticker)
        series[identifier] = BarSeries(
            timestamps=frame.index.to_numpy(dtype="datetime64[ns]"),
            open=frame["Open"].to_numpy(dtype=np.float64),
            high=frame["High"].to_numpy(dtype=np.float64),
            low=frame["Low"].to_numpy(dtype=np.float64),
            close=frame["Close"].to_numpy(dtype=np.float64),
            volume=frame["Volume"].to_numpy(dtype=np.float64),
        )
        instruments[identifier] = Instrument(
            id=identifier,
            quote_currency=currency,
            tick_size=tick,
            lot_size=Decimal("1"),
            timezone=UTC,
        )
    if short:
        raise DemoInputError(
            "insufficient_history",
            f"{', '.join(short)} < {min_bars + WARMUP_MARGIN_BARS}",
        )
    dataset = MarketDataSet(
        series=series,
        instruments=instruments,
        timeframe=Timeframe.days(1),
    )
    return dataset, currency
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run python -m pytest tests_ui/test_market_data.py -q`
Expected: all tests pass.

- [ ] **Step 7: Verify the core suite and core-only environment are unaffected**

```bash
uv run python -m pytest -q
uv run --no-default-groups --with pytest --with hypothesis python -m pytest -q
uv run ruff check streamlit_ui tests_ui && uv run ruff format --check streamlit_ui tests_ui
```

Expected: the default suite passes (on macOS 27 the 7 known `tests/risk/test_decimal_boundaries.py` failures are pre-existing and unrelated; deselect them with `--deselect tests/risk/test_decimal_boundaries.py`), the core-only run passes with skips, and Ruff reports no issues.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock streamlit_ui tests_ui
git commit -m "feat(ui): convert yfinance history into a MarketDataSet"
```

---

### Task 2: Demo strategies

**Files:**
- Create: `streamlit_ui/strategies.py`
- Test: `tests_ui/test_strategies.py`

**Interfaces:**
- Consumes: public `pybacktest` `ConfigurationError`, `InstrumentId`, `StrategyContext`; `pybacktest.data.features.FeatureBuilder`/`FeaturePlan`; `pybacktest.domain.market.MarketSlice`; `pybacktest.domain.orders.DecisionReason`/`TargetWeight`.
- Note: inside `on_bar`, `context.portfolio.positions` maps `InstrumentId` to `Quantity` (use `.value`), and `context.active_orders` is a tuple of `Order` with `.instrument`.
- Produces:
  - `BuyAndHold(instruments: tuple[InstrumentId, ...])`
  - `MovingAverageCrossAll(instruments, fast: int, slow: int)`
  - `RsiReversion(instruments, period: int, lower: int, upper: int)`
  - each with `.instruments` and `.warmup_bars: int`
  - `DemoStrategy = BuyAndHold | MovingAverageCrossAll | RsiReversion`
  - `equal_weight(count: int) -> Decimal`, `feature_name(feature: str, instrument: InstrumentId) -> str`, `rsi(closes: Sequence[float]) -> float`

- [ ] **Step 1: Write the failing tests**

`tests_ui/test_strategies.py`:

```python
from decimal import Decimal

import pytest

from pybacktest import ConfigurationError, InstrumentId
from pybacktest.data.features import FeatureBuilder
from streamlit_ui.strategies import (
    BuyAndHold,
    MovingAverageCrossAll,
    RsiReversion,
    equal_weight,
    feature_name,
    rsi,
)

AAPL = InstrumentId.parse("YF:AAPL")
MSFT = InstrumentId.parse("YF:MSFT")


def test_equal_weight_is_quantized_to_six_places():
    assert equal_weight(1) == Decimal("1.000000")
    assert equal_weight(3) == Decimal("0.333333")


def test_rsi_matches_hand_computed_value():
    # Changes: +1, -0.5, +2, -1 -> gains 3, losses 1.5 -> RS 2 -> RSI 66.67.
    assert rsi([10.0, 11.0, 10.5, 12.5, 11.5]) == pytest.approx(200 / 3)


def test_rsi_edge_cases():
    assert rsi([1.0, 2.0, 3.0]) == 100.0
    assert rsi([3.0, 3.0, 3.0]) == 50.0
    assert rsi([3.0, 2.0, 1.0]) == 0.0


def test_feature_names_are_namespaced_per_instrument():
    strategy = MovingAverageCrossAll(instruments=(AAPL, MSFT), fast=2, slow=3)
    plan = strategy.build_features(FeatureBuilder())
    names = {node.name for node in plan.nodes}
    assert feature_name("fast", AAPL) in names
    assert feature_name("fast", MSFT) in names
    assert feature_name("fast", AAPL) == "fast:YF:AAPL"


def test_rsi_declares_period_lags():
    strategy = RsiReversion(instruments=(AAPL,), period=3, lower=30, upper=70)
    plan = strategy.build_features(FeatureBuilder())
    names = {node.name for node in plan.nodes}
    assert {"close:YF:AAPL", "close_lag1:YF:AAPL", "close_lag3:YF:AAPL"} <= names


@pytest.mark.parametrize(
    ("strategy", "warmup"),
    [
        (BuyAndHold(instruments=(AAPL,)), 1),
        (MovingAverageCrossAll(instruments=(AAPL,), fast=5, slow=20), 21),
        (RsiReversion(instruments=(AAPL,), period=14, lower=30, upper=70), 15),
    ],
)
def test_warmup_bars(strategy, warmup):
    assert strategy.warmup_bars == warmup


@pytest.mark.parametrize(
    "build",
    [
        lambda: MovingAverageCrossAll(instruments=(AAPL,), fast=20, slow=20),
        lambda: MovingAverageCrossAll(instruments=(AAPL,), fast=0, slow=20),
        lambda: RsiReversion(instruments=(AAPL,), period=1, lower=30, upper=70),
        lambda: RsiReversion(instruments=(AAPL,), period=31, lower=30, upper=70),
        lambda: RsiReversion(instruments=(AAPL,), period=14, lower=70, upper=30),
        lambda: BuyAndHold(instruments=()),
        lambda: BuyAndHold(instruments=(AAPL, AAPL)),
    ],
)
def test_invalid_parameters_raise_configuration_error(build):
    with pytest.raises(ConfigurationError):
        build()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m pytest tests_ui/test_strategies.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'streamlit_ui.strategies'`.

- [ ] **Step 3: Implement `streamlit_ui/strategies.py`**

```python
"""Demo strategies built only on the public ``pybacktest`` strategy contract.

Every strategy is a frozen, stateless dataclass that trades each instrument in
``instruments`` independently and targets an equal ``1/N`` weight. Feature
names are namespaced as ``<feature>:<instrument>`` so the plans of several
instruments never collide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from pybacktest import ConfigurationError, InstrumentId, StrategyContext
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.orders import DecisionReason, TargetWeight

WEIGHT_QUANTUM = Decimal("0.000001")
RSI_PERIOD_RANGE = (2, 30)


def _instruments(value: Sequence[InstrumentId]) -> tuple[InstrumentId, ...]:
    instruments = tuple(value)
    if not instruments or not all(isinstance(i, InstrumentId) for i in instruments):
        raise ConfigurationError(
            "demo strategies need at least one InstrumentId.",
            code="invalid_demo_instruments",
        )
    if len(set(instruments)) != len(instruments):
        raise ConfigurationError(
            "demo strategy instruments must be unique.",
            code="invalid_demo_instruments",
        )
    return instruments


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{name} must be an integer.",
            code="invalid_demo_parameter",
        )
    return value


def equal_weight(count: int) -> Decimal:
    """Return ``1/count`` quantized to six decimal places."""
    return (Decimal(1) / Decimal(count)).quantize(WEIGHT_QUANTUM)


def feature_name(feature: str, instrument: InstrumentId) -> str:
    """Namespace ``feature`` for one instrument."""
    return f"{feature}:{instrument}"


def rsi(closes: Sequence[float]) -> float:
    """Simple-average (Cutler) RSI over consecutive closes, oldest first."""
    gains = 0.0
    losses = 0.0
    for previous, current in pairwise(closes):
        change = current - previous
        if change > 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 50.0 if gains == 0 else 100.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


def _holding(context: StrategyContext, instrument: InstrumentId) -> bool:
    quantity = context.portfolio.positions.get(instrument)
    return quantity is not None and quantity.value != 0


def _pending(context: StrategyContext) -> frozenset[InstrumentId]:
    return frozenset(order.instrument for order in context.active_orders)


@dataclass(frozen=True, slots=True)
class BuyAndHold:
    """Buy ``1/N`` of each instrument once and hold it."""

    instruments: tuple[InstrumentId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instruments(self.instruments))

    @property
    def warmup_bars(self) -> int:
        return 1

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        for instrument in self.instruments:
            builder.source(feature_name("close", instrument), instrument, "close")
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[TargetWeight, ...]:
        del market
        pending = _pending(context)
        weight = equal_weight(len(self.instruments))
        intents = []
        for instrument in self.instruments:
            close = context.features.at(feature_name("close", instrument))
            if close != close or instrument in pending or _holding(context, instrument):
                continue
            intents.append(
                TargetWeight(
                    instrument=instrument,
                    weight=weight,
                    reason=DecisionReason.of("buy_and_hold"),
                )
            )
        return tuple(intents)


@dataclass(frozen=True, slots=True)
class MovingAverageCrossAll:
    """Per instrument: target ``1/N`` on a fast/slow SMA cross up, ``0`` down."""

    instruments: tuple[InstrumentId, ...]
    fast: int
    slow: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instruments(self.instruments))
        fast = _int(self.fast, "fast")
        slow = _int(self.slow, "slow")
        if not 0 < fast < slow:
            raise ConfigurationError(
                "moving-average cross requires 0 < fast < slow.",
                code="invalid_demo_parameter",
            )

    @property
    def warmup_bars(self) -> int:
        return self.slow + 1

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        for instrument in self.instruments:
            close = builder.source(
                feature_name("close", instrument), instrument, "close"
            )
            fast = builder.sma(feature_name("fast", instrument), close, self.fast)
            slow = builder.sma(feature_name("slow", instrument), close, self.slow)
            builder.lag(feature_name("fast_prev", instrument), fast, 1)
            builder.lag(feature_name("slow_prev", instrument), slow, 1)
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[TargetWeight, ...]:
        del market
        view = context.features
        weight = equal_weight(len(self.instruments))
        intents = []
        for instrument in self.instruments:
            fast = view.at(feature_name("fast", instrument))
            slow = view.at(feature_name("slow", instrument))
            fast_prev = view.at(feature_name("fast_prev", instrument))
            slow_prev = view.at(feature_name("slow_prev", instrument))
            if any(value != value for value in (fast, slow, fast_prev, slow_prev)):
                continue
            if fast_prev <= slow_prev and fast > slow:
                target, outcome = weight, "cross_up"
            elif fast_prev >= slow_prev and fast < slow:
                target, outcome = Decimal(0), "cross_down"
            else:
                continue
            intents.append(
                TargetWeight(
                    instrument=instrument,
                    weight=target,
                    reason=DecisionReason.of("ma_cross", outcome=outcome),
                )
            )
        return tuple(intents)


@dataclass(frozen=True, slots=True)
class RsiReversion:
    """Per instrument: target ``1/N`` below ``lower`` RSI, ``0`` above ``upper``."""

    instruments: tuple[InstrumentId, ...]
    period: int
    lower: int
    upper: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instruments(self.instruments))
        period = _int(self.period, "period")
        lower = _int(self.lower, "lower")
        upper = _int(self.upper, "upper")
        low, high = RSI_PERIOD_RANGE
        if not low <= period <= high:
            raise ConfigurationError(
                f"RSI period must be between {low} and {high}.",
                code="invalid_demo_parameter",
            )
        if not 1 <= lower < upper <= 99:
            raise ConfigurationError(
                "RSI thresholds require 1 <= lower < upper <= 99.",
                code="invalid_demo_parameter",
            )

    @property
    def warmup_bars(self) -> int:
        return self.period + 1

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        for instrument in self.instruments:
            close = builder.source(
                feature_name("close", instrument), instrument, "close"
            )
            for lag in range(1, self.period + 1):
                builder.lag(feature_name(f"close_lag{lag}", instrument), close, lag)
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[TargetWeight, ...]:
        del market
        view = context.features
        pending = _pending(context)
        weight = equal_weight(len(self.instruments))
        intents = []
        for instrument in self.instruments:
            if instrument in pending:
                continue
            closes = [
                view.at(feature_name(f"close_lag{lag}", instrument))
                for lag in range(self.period, 0, -1)
            ]
            closes.append(view.at(feature_name("close", instrument)))
            if any(value != value for value in closes):
                continue
            value = rsi(closes)
            holding = _holding(context, instrument)
            if value < self.lower and not holding:
                target, outcome = weight, "oversold"
            elif value > self.upper and holding:
                target, outcome = Decimal(0), "overbought"
            else:
                continue
            intents.append(
                TargetWeight(
                    instrument=instrument,
                    weight=target,
                    reason=DecisionReason.of(
                        "rsi_reversion", rsi=round(value, 4), outcome=outcome
                    ),
                )
            )
        return tuple(intents)


DemoStrategy = BuyAndHold | MovingAverageCrossAll | RsiReversion
"""Any demo strategy; each exposes ``instruments`` and ``warmup_bars``."""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m pytest tests_ui/test_strategies.py -q`
Expected: all tests pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check streamlit_ui tests_ui && uv run ruff format --check streamlit_ui tests_ui
git add streamlit_ui/strategies.py tests_ui/test_strategies.py
git commit -m "feat(ui): add buy-and-hold, MA cross, and RSI demo strategies"
```

Behaviour on real price paths is exercised end to end in Task 3.

---

### Task 3: Runner and result tables

**Files:**
- Create: `streamlit_ui/runner.py`
- Test: `tests_ui/test_runner.py`

**Interfaces:**
- Consumes: Task 1 `TickerHistory`, `instrument_id`, `to_market_dataset`; Task 2 strategies and `DemoStrategy`.
- Produces:
  - `STRATEGY_NAMES = ("buy_and_hold", "ma_cross", "rsi")`, `MAX_STALENESS_BARS = 5`, `METRIC_ROWS: tuple[MetricName, ...]`
  - `RunConfig(tickers, start: date, end: date, strategy: str, params: Mapping[str, int] = {}, initial_cash: Decimal = Decimal("10000"), commission_per_share: Decimal = Decimal("0"))`
  - `RunReport(metrics, equity, orders, fills, warnings: pd.DataFrame, info: dict[str, object])`
    - `metrics`: index = metric names (`"total_return"`, …), columns `strategy`, `benchmark` (floats or `None`)
    - `equity`: index `timestamp`, columns `strategy`, `benchmark`
    - `orders`: `submitted_at, instrument, side, quantity, filled, status`
    - `fills`: `timestamp, instrument, side, quantity, price, fee`
    - `warnings`: `run, code, message`
    - `info`: `pybacktest, currency, bars, strategy, params, period`
  - `build_strategy(config: RunConfig) -> DemoStrategy`
  - `run_backtest(config: RunConfig, histories: Mapping[str, TickerHistory]) -> RunReport` — raises `DemoInputError` or `PybacktestError`.

- [ ] **Step 1: Write the failing tests**

`tests_ui/test_runner.py`:

```python
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from pybacktest import PybacktestError
from streamlit_ui.market_data import TickerHistory
from streamlit_ui.runner import METRIC_ROWS, RunConfig, run_backtest
from tests_ui.support import make_frame

START = date(2022, 1, 1)
END = date(2023, 12, 31)


def _config(strategy: str, params: dict[str, int], tickers=("AAPL", "MSFT")):
    return RunConfig(
        tickers=tickers,
        start=START,
        end=END,
        strategy=strategy,
        params=params,
        initial_cash=Decimal("10000"),
        commission_per_share=Decimal("0"),
    )


def _v_shape(bars: int = 120) -> np.ndarray:
    """Fall from 100 to 50, then rise back to 150."""
    half = bars // 2
    return np.concatenate(
        [np.linspace(100, 50, half), np.linspace(50, 150, bars - half)]
    )


def test_report_shape_and_benchmark(two_usd_histories):
    report = run_backtest(
        _config("ma_cross", {"fast": 5, "slow": 20}), two_usd_histories
    )
    assert list(report.metrics.index) == [name.value for name in METRIC_ROWS]
    assert list(report.metrics.columns) == ["strategy", "benchmark"]
    assert list(report.equity.columns) == ["strategy", "benchmark"]
    assert report.equity.index.is_monotonic_increasing
    assert report.equity.notna().all().all()
    assert report.info["currency"] == "USD"
    assert set(report.info["bars"]) == {"YF:AAPL", "YF:MSFT"}
    assert list(report.fills.columns) == [
        "timestamp",
        "instrument",
        "side",
        "quantity",
        "price",
        "fee",
    ]


def test_buy_and_hold_buys_each_instrument_once(two_usd_histories):
    report = run_backtest(_config("buy_and_hold", {}), two_usd_histories)
    assert sorted(report.fills["instrument"]) == ["YF:AAPL", "YF:MSFT"]
    assert set(report.fills["side"]) == {"buy"}
    # The chosen strategy is the benchmark, so both curves match exactly.
    assert report.equity["strategy"].equals(report.equity["benchmark"])


def test_ma_cross_buys_after_the_bottom_on_a_v_shape():
    histories = {"AAPL": TickerHistory(make_frame(_v_shape()), "USD")}
    report = run_backtest(
        _config("ma_cross", {"fast": 3, "slow": 10}, tickers=("AAPL",)), histories
    )
    buys = report.fills[report.fills["side"] == "buy"]
    assert len(buys) == 1
    assert buys.iloc[0]["price"] > 50.0


def test_rsi_buys_the_dip_and_sells_the_rally():
    histories = {"AAPL": TickerHistory(make_frame(_v_shape()), "USD")}
    report = run_backtest(
        _config("rsi", {"period": 5, "lower": 30, "upper": 70}, tickers=("AAPL",)),
        histories,
    )
    # A straight decline pins RSI at 0 and a straight rally pins it at 100, so
    # the buy lands during the fall and the sell right after the bottom.
    bottom = report.equity.index[len(report.equity) // 2]
    assert report.fills["side"].tolist() == ["buy", "sell"]
    assert report.fills.iloc[0]["timestamp"] < bottom
    assert report.fills.iloc[1]["timestamp"] >= bottom


def test_commission_is_charged_per_share(two_usd_histories):
    config = RunConfig(
        tickers=("AAPL", "MSFT"),
        start=START,
        end=END,
        strategy="buy_and_hold",
        initial_cash=Decimal("10000"),
        commission_per_share=Decimal("0.01"),
    )
    report = run_backtest(config, two_usd_histories)
    expected = (report.fills["quantity"] * 0.01).round(6).tolist()
    assert report.fills["fee"].round(6).tolist() == expected


def test_invalid_strategy_parameters_surface_as_core_errors(two_usd_histories):
    with pytest.raises(PybacktestError):
        run_backtest(_config("ma_cross", {"fast": 30, "slow": 10}), two_usd_histories)


def test_ticker_listed_mid_period_is_bought_when_its_data_starts():
    histories = {
        "AAPL": TickerHistory(make_frame(np.linspace(100, 120, 200)), "USD"),
        "NEWCO": TickerHistory(
            make_frame(np.linspace(20, 30, 100), start="2022-05-23"), "USD"
        ),
    }
    report = run_backtest(
        _config("buy_and_hold", {}, tickers=("AAPL", "NEWCO")), histories
    )
    newco = report.fills[report.fills["instrument"] == "YF:NEWCO"]
    assert len(newco) == 1
    assert newco.iloc[0]["timestamp"].date() > date(2022, 5, 23)


def test_holiday_in_one_series_does_not_break_the_union_calendar():
    gappy = make_frame(np.linspace(100, 120, 200))
    gappy = gappy.drop(gappy.index[50])
    histories = {
        "AAPL": TickerHistory(gappy, "USD"),
        "MSFT": TickerHistory(make_frame(np.linspace(50, 70, 200)), "USD"),
    }
    report = run_backtest(_config("ma_cross", {"fast": 3, "slow": 10}), histories)
    assert report.info["bars"] == {"YF:AAPL": 199, "YF:MSFT": 200}
    assert len(report.equity) == 200
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m pytest tests_ui/test_runner.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'streamlit_ui.runner'`.

- [ ] **Step 3: Implement `streamlit_ui/runner.py`**

```python
"""Run a demo configuration through the core engine and shape the results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd

import pybacktest
from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    BacktestResult,
    CalendarPolicy,
    DateRange,
    IntrabarPolicy,
    LongShortRisk,
    MarketDataSet,
    MetricName,
    MetricsConfig,
    Money,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    PerShareCommission,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
)
from streamlit_ui.market_data import TickerHistory, instrument_id, to_market_dataset
from streamlit_ui.strategies import (
    BuyAndHold,
    DemoStrategy,
    MovingAverageCrossAll,
    RsiReversion,
)

STRATEGY_NAMES = ("buy_and_hold", "ma_cross", "rsi")
# A held instrument may miss up to this many union bars (a local holiday or a
# bar yfinance dropped) before the core rejects its stale mark.
MAX_STALENESS_BARS = 5
METRIC_ROWS = (
    MetricName.TOTAL_RETURN,
    MetricName.CAGR,
    MetricName.VOLATILITY,
    MetricName.SHARPE,
    MetricName.SORTINO,
    MetricName.MAXIMUM_DRAWDOWN,
    MetricName.WIN_RATE,
    MetricName.TURNOVER,
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything the sidebar collects for one run."""

    tickers: tuple[str, ...]
    start: date
    end: date
    strategy: str
    params: Mapping[str, int] = field(default_factory=dict)
    initial_cash: Decimal = Decimal("10000")
    commission_per_share: Decimal = Decimal("0")


@dataclass(frozen=True, slots=True)
class RunReport:
    """Display-ready results for the chosen strategy and its benchmark."""

    metrics: pd.DataFrame
    equity: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    warnings: pd.DataFrame
    info: dict[str, object]


class _InMemoryData:
    """A data source that owns one already-validated dataset."""

    def __init__(self, dataset: MarketDataSet) -> None:
        self._dataset = dataset

    def load(
        self,
        universe: object,
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        del universe, period, timeframe
        return self._dataset


def build_strategy(config: RunConfig) -> DemoStrategy:
    """Construct the configured demo strategy."""
    instruments = tuple(instrument_id(ticker) for ticker in config.tickers)
    params = dict(config.params)
    if config.strategy == "buy_and_hold":
        return BuyAndHold(instruments=instruments)
    if config.strategy == "ma_cross":
        return MovingAverageCrossAll(
            instruments=instruments,
            fast=params["fast"],
            slow=params["slow"],
        )
    if config.strategy == "rsi":
        return RsiReversion(
            instruments=instruments,
            period=params["period"],
            lower=params["lower"],
            upper=params["upper"],
        )
    raise ValueError(f"unknown strategy: {config.strategy!r}")


def _engine(dataset: MarketDataSet, config: RunConfig) -> BacktestEngine:
    commission = (
        NoCommission()
        if config.commission_per_share == 0
        else PerShareCommission(rate_per_share=config.commission_per_share)
    )
    return BacktestEngine(
        data_source=_InMemoryData(dataset),
        broker_factory=SimulatedBrokerFactory(
            fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
            commission=commission,
            slippage=NoSlippage(),
            liquidity=NoLiquidityLimit(),
            borrow_cost=NoBorrowCost(),
        ),
        risk_policy=LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=False,
        ),
    )


def _simulation(config: RunConfig, currency: str) -> SimulationRequest:
    start = datetime(
        config.start.year, config.start.month, config.start.day, tzinfo=UTC
    )
    end = datetime(config.end.year, config.end.month, config.end.day, tzinfo=UTC)
    return SimulationRequest(
        universe=tuple(instrument_id(ticker) for ticker in config.tickers),
        period=DateRange(start, end + timedelta(days=1)),
        timeframe=Timeframe.days(1),
        calendar=CalendarPolicy.union(max_staleness_bars=MAX_STALENESS_BARS),
        initial_cash=Money.of(config.initial_cash, currency),
        seed=0,
        metrics=MetricsConfig(
            risk_free_rate=Decimal("0"),
            annualization_periods=252,
        ),
    )


def run_backtest(
    config: RunConfig,
    histories: Mapping[str, TickerHistory],
) -> RunReport:
    """Run ``config`` and a buy-and-hold benchmark on the same dataset."""
    strategy = build_strategy(config)
    benchmark = BuyAndHold(instruments=strategy.instruments)
    dataset, currency = to_market_dataset(histories, min_bars=strategy.warmup_bars)
    engine = _engine(dataset, config)
    simulation = _simulation(config, currency)
    result = engine.run(BacktestRequest(strategy=strategy, simulation=simulation))
    base = engine.run(BacktestRequest(strategy=benchmark, simulation=simulation))
    return RunReport(
        metrics=metrics_table(result, base),
        equity=equity_frame(result, base),
        orders=orders_table(result),
        fills=fills_table(result),
        warnings=warnings_table(result, base),
        info={
            "pybacktest": pybacktest.__version__,
            "currency": currency,
            "bars": {str(i): len(s.timestamps) for i, s in dataset.series.items()},
            "strategy": config.strategy,
            "params": dict(config.params),
            "period": f"{config.start.isoformat()} to {config.end.isoformat()}",
        },
    )


def _metric(result: BacktestResult, name: MetricName) -> float | None:
    value = result.summary.result_for(name).value
    return None if value is None else float(value)


def metrics_table(result: BacktestResult, benchmark: BacktestResult) -> pd.DataFrame:
    """One row per metric with ``strategy`` and ``benchmark`` columns."""
    return pd.DataFrame(
        {
            "strategy": [_metric(result, name) for name in METRIC_ROWS],
            "benchmark": [_metric(benchmark, name) for name in METRIC_ROWS],
        },
        index=[name.value for name in METRIC_ROWS],
    )


def _equity(result: BacktestResult) -> pd.Series:
    points = {
        snapshot.timestamp: float(snapshot.equity.amount)
        for snapshot in result.snapshots
        if snapshot.timestamp is not None
    }
    series = pd.Series(points, dtype="float64")
    series.index = pd.DatetimeIndex(series.index)
    return series


def equity_frame(result: BacktestResult, benchmark: BacktestResult) -> pd.DataFrame:
    """Equity per timestamp for the strategy and the benchmark."""
    frame = pd.DataFrame(
        {"strategy": _equity(result), "benchmark": _equity(benchmark)}
    ).sort_index()
    frame.index.name = "timestamp"
    return frame


def orders_table(result: BacktestResult) -> pd.DataFrame:
    """Orders with their final status."""
    return pd.DataFrame(
        [
            {
                "submitted_at": order.submitted_at,
                "instrument": str(order.instrument),
                "side": order.side.value,
                "quantity": float(order.quantity.value),
                "filled": float(order.filled_quantity.value),
                "status": order.status.value,
            }
            for order in result.orders
        ],
        columns=["submitted_at", "instrument", "side", "quantity", "filled", "status"],
    )


def fills_table(result: BacktestResult) -> pd.DataFrame:
    """Executions with price and fee."""
    return pd.DataFrame(
        [
            {
                "timestamp": fill.timestamp,
                "instrument": str(fill.instrument),
                "side": fill.side.value,
                "quantity": float(fill.quantity.value),
                "price": float(fill.price.amount),
                "fee": float(fill.fee.amount),
            }
            for fill in result.fills
        ],
        columns=["timestamp", "instrument", "side", "quantity", "price", "fee"],
    )


def warnings_table(result: BacktestResult, benchmark: BacktestResult) -> pd.DataFrame:
    """Warnings from both runs, tagged with their source."""
    rows = [
        {"run": run, "code": warning.code.value, "message": warning.message}
        for run, source in (("strategy", result), ("benchmark", benchmark))
        for warning in source.warnings
    ]
    return pd.DataFrame(rows, columns=["run", "code", "message"])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m pytest tests_ui/test_runner.py -q`
Expected: all tests pass. If `test_holiday_in_one_series_does_not_break_the_union_calendar` fails with `held position mark exceeded the configured calendar staleness policy`, the calendar is missing `max_staleness_bars=MAX_STALENESS_BARS`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check streamlit_ui tests_ui && uv run ruff format --check streamlit_ui tests_ui
git add streamlit_ui/runner.py tests_ui/test_runner.py
git commit -m "feat(ui): run demo strategies with a buy-and-hold benchmark"
```

---

### Task 4: Strings, page, and entrypoint

**Files:**
- Create: `streamlit_ui/i18n.py`, `streamlit_ui/app.py`, `streamlit_page.py`
- Test: `tests_ui/test_i18n.py`, `tests_ui/test_app.py`

**Interfaces:**
- Consumes: Task 1 `market_data` (module attribute `fetch_history`, `parse_tickers`, `TickerHistory`), `DemoInputError`; Task 3 `STRATEGY_NAMES`, `METRIC_ROWS`, `RunConfig`, `RunReport`, `run_backtest`; public `pybacktest.PybacktestError`.
- Produces:
  - `LANGUAGES = {"en": "English", "ko": "한국어"}`, `STRINGS`, `t(key: str, lang: str, **values) -> str` (missing key → `KeyError`)
  - `format_metric(name: str, value: float | None) -> str`
  - `main() -> None`
  - Widget keys used by tests: `lang` (radio), `strategy` (selectbox), `tickers` (text input), `start`, `end` (date inputs), `fast`, `slow`, `period`, `lower`, `upper`, `cash`, `commission` (number inputs); the form submit button is `app.sidebar.button[0]`.
- `app.py` must call `market_data.fetch_history` through the module (`market_data.fetch_history(...)`) so tests can patch `streamlit_ui.market_data.fetch_history`.

- [ ] **Step 1: Write the failing tests**

`tests_ui/test_i18n.py`:

```python
import pytest

from streamlit_ui.i18n import LANGUAGES, STRINGS, t
from streamlit_ui.runner import METRIC_ROWS, STRATEGY_NAMES

ERROR_CODES = (
    "no_tickers",
    "too_many_tickers",
    "empty_history",
    "mixed_currency",
    "insufficient_history",
    "invalid_period",
)


def test_languages_have_identical_keys():
    assert set(STRINGS) == set(LANGUAGES)
    assert set(STRINGS["en"]) == set(STRINGS["ko"])


@pytest.mark.parametrize("lang", list(LANGUAGES))
def test_every_referenced_key_exists(lang):
    for name in STRATEGY_NAMES:
        t(f"strategy.{name}", lang)
    for metric in METRIC_ROWS:
        t(f"metric.{metric.value}", lang)
    for code in ERROR_CODES:
        t(f"error.{code}", lang, detail="x")


def test_missing_key_raises():
    with pytest.raises(KeyError):
        t("does.not.exist", "en")


def test_interpolation():
    assert t("error.too_many_tickers", "en", detail="5") == "Enter at most 5 tickers."
```

`tests_ui/test_app.py`:

```python
from datetime import date
from pathlib import Path
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from streamlit_ui.app import format_metric
from streamlit_ui.market_data import TickerHistory
from tests_ui.support import fake_fetch, make_frame, random_walk

ENTRYPOINT = str(Path(__file__).resolve().parents[1] / "streamlit_page.py")
FETCH = "streamlit_ui.market_data.fetch_history"


@pytest.fixture(autouse=True)
def _clear_cache():
    st.cache_data.clear()


def _app() -> AppTest:
    return AppTest.from_file(ENTRYPOINT, default_timeout=30).run()


def test_page_renders_intro_without_running():
    app = _app()
    assert not app.exception
    assert app.title[0].value == "Pybacktest demo"
    assert not app.metric


def test_submitting_the_form_shows_results():
    with mock.patch(FETCH, side_effect=fake_fetch):
        app = _app()
        app.sidebar.selectbox(key="strategy").set_value("ma_cross").run()
        app.sidebar.button[0].click().run()
    assert not app.exception
    assert not app.error, [element.value for element in app.error]
    assert len(app.metric) == 4
    assert len(app.tabs) == 4


def test_switching_language_keeps_the_last_report():
    with mock.patch(FETCH, side_effect=fake_fetch) as fetch:
        app = _app()
        app.sidebar.button[0].click().run()
        app.sidebar.radio(key="lang").set_value("ko").run()
    assert fetch.call_count == 1
    assert app.title[0].value == "Pybacktest 데모"
    assert len(app.metric) == 4


def test_mixed_currency_shows_localized_error():
    def mixed(tickers, start, end):
        del tickers, start, end
        return {
            "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
            "005930.KS": TickerHistory(make_frame(random_walk(2)), "KRW"),
        }

    with mock.patch(FETCH, side_effect=mixed):
        app = _app()
        app.sidebar.radio(key="lang").set_value("ko").run()
        app.sidebar.text_input(key="tickers").set_value("AAPL, 005930.KS")
        app.sidebar.button[0].click().run()
    assert "통화" in app.error[0].value


def test_start_after_end_is_rejected_before_fetching():
    with mock.patch(FETCH, side_effect=fake_fetch) as fetch:
        app = _app()
        app.sidebar.date_input(key="start").set_value(date(2024, 1, 2))
        app.sidebar.date_input(key="end").set_value(date(2024, 1, 1))
        app.sidebar.button[0].click().run()
    assert fetch.call_count == 0
    assert "start date" in app.error[0].value


def test_unexpected_failure_shows_generic_error_with_details():
    with mock.patch(FETCH, side_effect=RuntimeError("boom")):
        app = _app()
        app.sidebar.button[0].click().run()
    assert not app.exception
    assert "unexpected" in app.error[0].value
    assert "boom" in app.expander[0].code[0].value


def test_changing_only_the_strategy_reuses_cached_data():
    with mock.patch(FETCH, side_effect=fake_fetch) as fetch:
        app = _app()
        app.sidebar.button[0].click().run()
        app.sidebar.selectbox(key="strategy").set_value("rsi").run()
        app.sidebar.button[0].click().run()
    assert not app.error
    assert fetch.call_count == 1


@pytest.mark.parametrize(
    ("name", "value", "text"),
    [
        ("total_return", 0.1234, "12.34%"),
        ("maximum_drawdown", -0.5, "-50.00%"),
        ("sharpe", 1.234, "1.23"),
        ("win_rate", None, "—"),
    ],
)
def test_format_metric(name, value, text):
    assert format_metric(name, value) == text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run python -m pytest tests_ui/test_i18n.py tests_ui/test_app.py -q`
Expected: collection errors for `streamlit_ui.i18n` and `streamlit_ui.app`.

- [ ] **Step 3: Implement `streamlit_ui/i18n.py`**

```python
"""Korean and English UI strings."""

from __future__ import annotations

LANGUAGES = {"en": "English", "ko": "한국어"}

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "page.title": "Pybacktest demo",
        "page.intro": (
            "Backtest a simple strategy on Yahoo Finance daily data with the "
            "deterministic Pybacktest 0.2 engine."
        ),
        "page.steps": (
            "1. Enter up to five tickers that share one currency.\n"
            "2. Pick a period and a strategy.\n"
            "3. Press **Run backtest**."
        ),
        "sidebar.language": "Language",
        "sidebar.tickers": "Tickers (comma-separated)",
        "sidebar.tickers_help": "Examples: AAPL, MSFT or 005930.KS, 000660.KS",
        "sidebar.start": "Start date",
        "sidebar.end": "End date",
        "sidebar.strategy": "Strategy",
        "sidebar.initial_cash": "Initial cash",
        "sidebar.commission": "Commission per share",
        "sidebar.run": "Run backtest",
        "strategy.buy_and_hold": "Buy & Hold",
        "strategy.ma_cross": "Moving-average cross",
        "strategy.rsi": "RSI mean reversion",
        "param.fast": "Fast SMA window",
        "param.slow": "Slow SMA window",
        "param.period": "RSI period",
        "param.lower": "Buy below RSI",
        "param.upper": "Sell above RSI",
        "result.spinner": "Downloading data and running the engine…",
        "result.metrics": "Strategy vs. buy & hold",
        "result.equity": "Equity curve",
        "result.orders": "Orders",
        "result.fills": "Fills",
        "result.warnings": "Warnings",
        "result.info": "Run info",
        "result.none": "None",
        "column.strategy": "Strategy",
        "column.benchmark": "Buy & Hold",
        "metric.total_return": "Total return",
        "metric.cagr": "CAGR",
        "metric.volatility": "Volatility",
        "metric.sharpe": "Sharpe",
        "metric.sortino": "Sortino",
        "metric.maximum_drawdown": "Max drawdown",
        "metric.win_rate": "Win rate",
        "metric.turnover": "Turnover",
        "error.no_tickers": "Enter at least one ticker.",
        "error.too_many_tickers": "Enter at most {detail} tickers.",
        "error.empty_history": "No data was found for: {detail}.",
        "error.mixed_currency": (
            "All tickers must share one currency (found {detail})."
        ),
        "error.insufficient_history": (
            "Not enough bars for this strategy ({detail}). Choose a longer period."
        ),
        "error.invalid_period": "The start date must be before the end date.",
        "error.engine": "The engine rejected this configuration: {detail}",
        "error.unexpected": "Something unexpected went wrong.",
        "error.details": "Details",
    },
    "ko": {
        "page.title": "Pybacktest 데모",
        "page.intro": (
            "야후 파이낸스 일봉 데이터로 간단한 전략을 결정적(deterministic) "
            "Pybacktest 0.2 엔진에서 백테스트해 보세요."
        ),
        "page.steps": (
            "1. 같은 통화의 티커를 최대 5개까지 입력하세요.\n"
            "2. 기간과 전략을 고르세요.\n"
            "3. **백테스트 실행**을 누르세요."
        ),
        "sidebar.language": "언어",
        "sidebar.tickers": "티커 (쉼표로 구분)",
        "sidebar.tickers_help": "예: AAPL, MSFT 또는 005930.KS, 000660.KS",
        "sidebar.start": "시작일",
        "sidebar.end": "종료일",
        "sidebar.strategy": "전략",
        "sidebar.initial_cash": "초기 자금",
        "sidebar.commission": "주당 수수료",
        "sidebar.run": "백테스트 실행",
        "strategy.buy_and_hold": "매수 후 보유",
        "strategy.ma_cross": "이동평균 크로스",
        "strategy.rsi": "RSI 역추세",
        "param.fast": "단기 이동평균 기간",
        "param.slow": "장기 이동평균 기간",
        "param.period": "RSI 기간",
        "param.lower": "RSI 이하에서 매수",
        "param.upper": "RSI 이상에서 매도",
        "result.spinner": "데이터를 받고 엔진을 실행하는 중…",
        "result.metrics": "전략 vs 매수 후 보유",
        "result.equity": "자산 곡선",
        "result.orders": "주문",
        "result.fills": "체결",
        "result.warnings": "경고",
        "result.info": "실행 정보",
        "result.none": "없음",
        "column.strategy": "전략",
        "column.benchmark": "매수 후 보유",
        "metric.total_return": "총수익률",
        "metric.cagr": "연평균 수익률",
        "metric.volatility": "변동성",
        "metric.sharpe": "샤프 지수",
        "metric.sortino": "소르티노 지수",
        "metric.maximum_drawdown": "최대 낙폭",
        "metric.win_rate": "승률",
        "metric.turnover": "회전율",
        "error.no_tickers": "티커를 하나 이상 입력하세요.",
        "error.too_many_tickers": "티커는 최대 {detail}개까지 입력할 수 있어요.",
        "error.empty_history": "다음 티커의 데이터를 찾지 못했어요: {detail}.",
        "error.mixed_currency": ("모든 티커의 통화가 같아야 해요 (현재 {detail})."),
        "error.insufficient_history": (
            "이 전략에 필요한 봉 수가 부족해요 ({detail}). 기간을 늘려주세요."
        ),
        "error.invalid_period": "시작일은 종료일보다 앞서야 해요.",
        "error.engine": "엔진이 이 설정을 거부했어요: {detail}",
        "error.unexpected": "예상치 못한 오류가 발생했어요.",
        "error.details": "상세 정보",
    },
}


def t(key: str, lang: str, **values: object) -> str:
    """Return the ``lang`` string for ``key`` with ``values`` interpolated."""
    text = STRINGS[lang][key]
    return text.format(**values) if values else text
```

- [ ] **Step 4: Implement `streamlit_ui/app.py`**

```python
"""Streamlit page: widgets, charts, and tables only."""

from __future__ import annotations

import traceback
from datetime import date, timedelta
from decimal import Decimal

import streamlit as st

from pybacktest import PybacktestError
from streamlit_ui import market_data
from streamlit_ui.errors import DemoInputError
from streamlit_ui.i18n import LANGUAGES, t
from streamlit_ui.runner import STRATEGY_NAMES, RunConfig, RunReport, run_backtest

PERCENT_METRICS = frozenset(
    {"total_return", "cagr", "volatility", "maximum_drawdown", "win_rate"}
)
HEADLINE_METRICS = ("total_return", "cagr", "sharpe", "maximum_drawdown")
DEFAULT_TICKERS = "AAPL, MSFT"
DEFAULT_YEARS = 5


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_history(
    tickers: tuple[str, ...],
    start: date,
    end: date,
) -> dict[str, market_data.TickerHistory]:
    # Looked up on the module at call time so tests can patch it.
    return market_data.fetch_history(tickers, start, end)


def format_metric(name: str, value: float | None) -> str:
    """Render one metric value for display."""
    if value is None:
        return "—"
    if name in PERCENT_METRICS:
        return f"{value:.2%}"
    return f"{value:.2f}"


def _strategy_params(strategy: str, lang: str) -> dict[str, int]:
    if strategy == "ma_cross":
        fast = st.number_input(t("param.fast", lang), 2, 200, 20, key="fast")
        slow = st.number_input(t("param.slow", lang), 3, 400, 60, key="slow")
        return {"fast": int(fast), "slow": int(slow)}
    if strategy == "rsi":
        period = st.number_input(t("param.period", lang), 2, 30, 14, key="period")
        lower = st.number_input(t("param.lower", lang), 1, 98, 30, key="lower")
        upper = st.number_input(t("param.upper", lang), 2, 99, 70, key="upper")
        return {"period": int(period), "lower": int(lower), "upper": int(upper)}
    return {}


def _language() -> str:
    with st.sidebar:
        return st.radio(
            "Language / 언어",
            list(LANGUAGES),
            format_func=LANGUAGES.__getitem__,
            horizontal=True,
            key="lang",
        )


def _sidebar(lang: str) -> RunConfig | None:
    """Collect the run configuration; ``None`` until the form is submitted."""
    with st.sidebar:
        strategy = st.selectbox(
            t("sidebar.strategy", lang),
            STRATEGY_NAMES,
            format_func=lambda name: t(f"strategy.{name}", lang),
            key="strategy",
        )
        with st.form("run"):
            raw_tickers = st.text_input(
                t("sidebar.tickers", lang),
                DEFAULT_TICKERS,
                help=t("sidebar.tickers_help", lang),
                key="tickers",
            )
            today = date.today()
            start = st.date_input(
                t("sidebar.start", lang),
                today - timedelta(days=365 * DEFAULT_YEARS),
                key="start",
            )
            end = st.date_input(t("sidebar.end", lang), today, key="end")
            params = _strategy_params(strategy, lang)
            cash = st.number_input(
                t("sidebar.initial_cash", lang), 100.0, 1e12, 10000.0, key="cash"
            )
            commission = st.number_input(
                t("sidebar.commission", lang), 0.0, 100.0, 0.0, key="commission"
            )
            submitted = st.form_submit_button(t("sidebar.run", lang), type="primary")
    if not submitted:
        return None
    if start >= end:
        raise DemoInputError("invalid_period")
    return RunConfig(
        tickers=market_data.parse_tickers(raw_tickers),
        start=start,
        end=end,
        strategy=strategy,
        params=params,
        initial_cash=Decimal(str(cash)),
        commission_per_share=Decimal(str(commission)),
    )


def _render_report(report: RunReport, lang: str) -> None:
    st.subheader(t("result.metrics", lang))
    columns = st.columns(len(HEADLINE_METRICS))
    for column, name in zip(columns, HEADLINE_METRICS, strict=True):
        strategy_value = report.metrics.loc[name, "strategy"]
        benchmark_value = report.metrics.loc[name, "benchmark"]
        column.metric(
            t(f"metric.{name}", lang),
            format_metric(name, _optional(strategy_value)),
            f"{t('column.benchmark', lang)} "
            f"{format_metric(name, _optional(benchmark_value))}",
            delta_color="off",
        )
    st.subheader(t("result.equity", lang))
    st.line_chart(
        report.equity.rename(
            columns={
                "strategy": t("column.strategy", lang),
                "benchmark": t("column.benchmark", lang),
            }
        )
    )
    st.table(
        {
            t("column.strategy", lang): _formatted_column(report, "strategy", lang),
            t("column.benchmark", lang): _formatted_column(report, "benchmark", lang),
        }
    )
    orders, fills, warnings, info = st.tabs(
        [
            t("result.orders", lang),
            t("result.fills", lang),
            t("result.warnings", lang),
            t("result.info", lang),
        ]
    )
    with orders:
        st.dataframe(report.orders, hide_index=True, width="stretch")
    with fills:
        st.dataframe(report.fills, hide_index=True, width="stretch")
    with warnings:
        if report.warnings.empty:
            st.write(t("result.none", lang))
        else:
            st.dataframe(report.warnings, hide_index=True, width="stretch")
    with info:
        st.json(report.info)


def _formatted_column(report: RunReport, column: str, lang: str) -> dict[str, str]:
    return {
        t(f"metric.{name}", lang): format_metric(
            name, _optional(report.metrics.loc[name, column])
        )
        for name in report.metrics.index
    }


def _optional(value: object) -> float | None:
    if value is None or value != value:
        return None
    return float(value)


def main() -> None:
    """Render the demo page."""
    st.set_page_config(page_title="Pybacktest demo", layout="wide")
    lang = _language()
    st.title(t("page.title", lang))
    try:
        config = _sidebar(lang)
        if config is not None:
            with st.spinner(t("result.spinner", lang)):
                histories = _cached_history(config.tickers, config.start, config.end)
                st.session_state["report"] = run_backtest(config, histories)
    except DemoInputError as error:
        st.error(t(f"error.{error.code}", lang, detail=error.detail))
        return
    except PybacktestError as error:
        st.error(t("error.engine", lang, detail=str(error)))
        return
    except Exception:
        st.error(t("error.unexpected", lang))
        with st.expander(t("error.details", lang)):
            st.code(traceback.format_exc())
        return
    report = st.session_state.get("report")
    if report is None:
        st.write(t("page.intro", lang))
        st.markdown(t("page.steps", lang))
        return
    _render_report(report, lang)
```

- [ ] **Step 5: Create the entrypoint `streamlit_page.py`**

```python
"""Streamlit Community Cloud entrypoint for the Pybacktest demo."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from streamlit_ui.app import main  # noqa: E402

main()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run python -m pytest tests_ui -q`
Expected: every `tests_ui` test passes (58 in the verified prototype).

- [ ] **Step 7: Smoke-test the page against live yfinance**

```bash
uv run streamlit run streamlit_page.py
```

In the browser: run `AAPL, MSFT` with the MA cross strategy, then switch to 한국어 and run `005930.KS, 000660.KS` with RSI and initial cash `10000000`. Expected: metric row, equity chart, and four tabs render with no error in either language. Stop the server with Ctrl+C.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check streamlit_ui tests_ui streamlit_page.py
uv run ruff format --check streamlit_ui tests_ui streamlit_page.py
git add streamlit_ui/i18n.py streamlit_ui/app.py streamlit_page.py tests_ui/test_i18n.py tests_ui/test_app.py
git commit -m "feat(ui): add bilingual Streamlit demo page at streamlit_page.py"
```

---

### Task 5: CI and README

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `tests_ui/` from Tasks 1–4; `streamlit_page.py` from Task 4.
- Produces: CI job `ui`; `test-core` using `--no-default-groups`; Ruff gates covering the demo.

- [ ] **Step 1: Replace `.github/workflows/ci.yml`**

The changes are: Ruff lint and format now also cover `streamlit_ui tests_ui streamlit_page.py`; `test-core` uses `--no-default-groups` (otherwise the new default `ui` group would install pandas, Streamlit, and yfinance into the core-only job); a new `ui` job runs `tests_ui` on 3.11 and 3.14.

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Lint and type check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10.2.0
        with:
          python-version: "3.11"
          enable-cache: true
      - name: Ruff lint
        run: uv run --locked ruff check src tests benchmarks streamlit_ui tests_ui streamlit_page.py
      - name: Ruff format
        run: uv run --locked ruff format --check src tests benchmarks streamlit_ui tests_ui streamlit_page.py
      - name: ty
        run: uv run --locked ty check src

  test:
    name: Tests (Python ${{ matrix.python-version }}, optional deps)
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10.2.0
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      # The default suite includes tests/integration/test_readme.py, which
      # executes every README example, including the pandas/pyarrow blocks.
      - name: Test suite and README examples
        run: uv run --locked python -m pytest -q

  test-core:
    name: Tests (Python ${{ matrix.python-version }}, core only)
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.14"]
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10.2.0
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - name: Test suite without optional dependencies
        run: >-
          uv run --locked --no-default-groups --with pytest --with hypothesis
          python -m pytest -q

  ui:
    name: Streamlit demo (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.14"]
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10.2.0
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - name: Demo tests (no network)
        run: uv run --locked python -m pytest tests_ui -q

  build:
    name: Package build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10.2.0
        with:
          python-version: "3.11"
          enable-cache: true
      - name: Build sdist and wheel
        run: uv build
      - name: Import installed wheel
        run: >-
          uv run --isolated --no-project --with dist/*.whl
          python -c "import pybacktest; print(pybacktest.__version__)"
      - name: Import installed sdist
        run: >-
          uv run --isolated --no-project --with dist/*.tar.gz
          python -c "import pybacktest; print(pybacktest.__version__)"
      - uses: actions/upload-artifact@v7
        with:
          name: dist
          path: dist/
```

- [ ] **Step 2: Run every changed CI step locally**

```bash
uv run --locked ruff check src tests benchmarks streamlit_ui tests_ui streamlit_page.py
uv run --locked ruff format --check src tests benchmarks streamlit_ui tests_ui streamlit_page.py
uv run --locked --no-default-groups --with pytest --with hypothesis python -m pytest -q --deselect tests/risk/test_decimal_boundaries.py
uv run --locked python -m pytest tests_ui -q
uv run --locked python -m pytest -q --deselect tests/risk/test_decimal_boundaries.py
```

Expected: all pass. In the core-only run, confirm Streamlit is absent: `uv run --no-default-groups python -c "import streamlit"` must fail with `ModuleNotFoundError`.

- [ ] **Step 3: Add a README section**

In `README.md`, insert this section immediately before `## Roadmap`:

````markdown
## Streamlit demo

A bilingual (English / 한국어) demo page backtests up to five same-currency
Yahoo Finance tickers with a buy-and-hold, moving-average cross, or RSI
strategy, and compares the result with buy-and-hold. It lives in
`streamlit_ui/`, outside the core package, and uses only the public API.

```bash
uv run streamlit run streamlit_page.py
```

`uv sync` installs Streamlit and yfinance through the default `ui` dependency
group; the published `pybacktest` package still depends only on NumPy.
````

Also extend the Development section's gate list so it matches CI:

```bash
uv run ruff check src tests benchmarks streamlit_ui tests_ui streamlit_page.py
uv run ruff format --check src tests benchmarks streamlit_ui tests_ui streamlit_page.py
uv run ty check src
uv run python -m pytest -q  # includes the executable README examples
uv run python -m pytest tests_ui -q  # Streamlit demo, no network
uv build
```

(Replace the existing fenced block under "## Development" that starts with `uv run ruff check src tests benchmarks`.)

- [ ] **Step 4: Verify README examples still pass**

Run: `uv run python -m pytest tests/integration/test_readme.py -q`
Expected: all pass (the new blocks are `bash`/`markdown`, not Python).

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "ci: test the Streamlit demo and keep the core-only job extra-free"
```

---

### Task 6: Publish and verify the live deployment

**Files:** none.

- [ ] **Step 1: Push and open a PR**

```bash
git push -u origin feature/streamlit-demo
gh pr create --base main --title "Add bilingual Streamlit demo" --body "<summary, test plan>"
```

- [ ] **Step 2: Wait for CI**

Run: `gh pr checks --watch`
Expected: `lint`, `test` (4), `test-core` (2), `ui` (2), and `build` all pass.

- [ ] **Step 3: After the user merges, verify Community Cloud**

The Cloud app redeploys from `main`. Open it and repeat Task 4 Step 7's two runs. If the Cloud log shows `ModuleNotFoundError: No module named 'streamlit'` or `'yfinance'`, Cloud did not install the default `ui` group from `uv.lock`. Stop and report the log to the user; do not change dependency files without their approval.
