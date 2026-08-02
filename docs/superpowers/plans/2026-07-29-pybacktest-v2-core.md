# Pybacktest V2 Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V1 package with a deterministic, explicit, Python-first hybrid backtesting core that supports multi-instrument daily/minute bars, long/short portfolios, market/limit orders, costs, partial fills, causal features, replayable results, and a public step-able simulation session.

**Architecture:** Vectorized causal feature plans run over immutable NumPy-backed market data, while orders, fills, risk decisions, and ledger changes run through a deterministic timestamp event loop. Ports define data, broker, strategy, risk, and artifact contracts; concrete adapters depend inward on those contracts. `BacktestEngine.run()` and `SimulationSession.advance()` share exactly the same execution path.

**Tech Stack:** Python 3.11+, stdlib dataclasses/Decimal/zoneinfo, NumPy, optional Pandas/PyArrow adapters, pytest, Hypothesis, pytest-benchmark, Ruff, ty, uv.

## Global Constraints

- This is a clean break targeting package version `0.2.0`; do not add a V1 compatibility shim.
- Core modules must not import yfinance, matplotlib, Streamlit, or MCP.
- Constructors perform no network or filesystem I/O.
- A run has exactly one base currency; cross-currency conversion is outside this plan.
- OHLCV/features use finite `float64`; booked money, fees, and quantities use quantized `Decimal`.
- A close-observed order cannot fill on the same bar; default market fills occur at the next tradable bar open.
- Every ledger mutation must be attributable to a `Fill` or explicit cash event.
- Public configuration uses typed objects and enums, never unvalidated dictionaries or behavior-selecting free-form strings.
- All random behavior receives the run RNG seeded by `BacktestRequest.seed`.
- Engine/service objects are reentrant after construction; all mutable broker,
  ledger, recorder, RNG, clock, and ID state belongs to one
  `SimulationSession`.
- Every shell command is prefixed with `rtk` per the repository instructions.
- Run focused tests after each red/green cycle; run the complete suite, Ruff, and ty before the final commit.

## Plan Boundaries

This is plan 1 of 3:

1. **This plan:** domain, data, features, broker simulation, risk, ledger, engine, results, artifacts, public Python API.
2. `2026-07-29-pybacktest-v2-strategy-spec.md`: safe `StrategySpec`, compiler, component catalog, experiment lineage.
3. `2026-07-29-pybacktest-v2-mcp.md`: local stdio MCP v2 adapter, bounded tools/resources, quotas, security tests.

The next plans consume only interfaces explicitly produced here.

## File Map

The final core package is organized as follows:

```text
src/pybacktest/
├── __init__.py                 # Curated public API and __version__
├── domain/
│   ├── errors.py               # Typed fatal/configuration errors
│   ├── identifiers.py          # RunId, OrderId, FillId
│   ├── instruments.py          # InstrumentId and Instrument
│   ├── money.py                # Money, Quantity, quantization
│   ├── time.py                 # DateRange and Timeframe
│   ├── market.py               # BarView and MarketSlice
│   ├── orders.py               # Intents, Order, Fill, status transitions
│   ├── portfolio.py            # Position and immutable snapshots
│   └── events.py               # Structured engine/broker/risk events
├── ports/
│   ├── data.py                 # MarketDataSource protocol
│   ├── broker.py               # Broker and cost model protocols
│   ├── strategy.py             # Strategy protocol and StrategyContext
│   ├── risk.py                 # OrderSizer/RiskPolicy protocols
│   └── artifacts.py            # ArtifactStore protocol
├── data/
│   ├── dataset.py              # NumPy-backed BarSeries/MarketDataSet
│   ├── validation.py           # Dataset contract validation
│   ├── calendar.py             # UNION/INTERSECTION clock construction
│   ├── features.py             # FeatureBuilder/Plan/Executor/Set
│   └── fingerprint.py          # Stable dataset hashing
├── strategy/
│   ├── intents.py              # Public intent re-exports/helpers
│   └── components.py           # MovingAverageCross reference strategy
├── risk/
│   ├── sizing.py               # Target intent conversion
│   └── policies.py             # LongShortRisk
├── adapters/
│   ├── data/pandas.py          # Explicit DataFrame adapter
│   ├── data/parquet.py         # Optional Parquet source
│   ├── broker/models.py        # Fill/fee/slippage/borrow models
│   ├── broker/simulated.py     # SimulatedBroker
│   └── artifacts/local.py      # Atomic local artifact storage
├── engine/
│   ├── accounting.py           # PortfolioLedger
│   ├── recorder.py             # Ordered trace/result accumulation
│   ├── session.py              # SimulationSession
│   └── engine.py               # BacktestEngine orchestration
├── application/
│   ├── requests.py             # SimulationRequest/BacktestRequest
│   └── service.py              # BacktestService for Python callers
└── results/
    ├── models.py               # Immutable result DTOs
    ├── metrics.py              # Metric definitions/calculation
    ├── explain.py              # Causal trace explanation
    └── serialization.py        # JSON/Parquet artifact conversion
```

## Task 1: Clean-Break Package and Tooling Baseline

**Files:**
- Create: `legacy/v1/README.md`
- Move: `src/pybacktest/` → `legacy/v1/src/pybacktest/`
- Move: `tests/` → `legacy/v1/tests/`
- Move: `streamlit_page.py`, `streamlit_page_en.py`, `streamlit_page_ko.py`, `streamlit_ui/` → `legacy/v1/ui/`
- Move: `backtest_test.ipynb`, `strategy_test.json` → `legacy/v1/examples/`
- Delete: `src/pybacktest.egg-info/`
- Create: `src/pybacktest/__init__.py`
- Create then move: `tests/test_v2_package.py` → `tests/test_package.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: none.
- Produces: `pybacktest.__version__ == "0.2.0"` and a clean Python 3.11+ package/test/tooling baseline.

- [ ] **Step 1: Write the failing package-boundary test**

Add `tests/test_v2_package.py` before moving the old test directory:

```python
import sys


def test_v2_package_has_version_without_optional_imports():
    import pybacktest

    assert pybacktest.__version__ == "0.2.0"
    assert "yfinance" not in sys.modules
    assert "matplotlib" not in sys.modules
    assert "streamlit" not in sys.modules
    assert "mcp" not in sys.modules
```

- [ ] **Step 2: Run the test and verify the V1 package fails the new contract**

Run:

```bash
rtk proxy uv run --no-sync python -m pytest tests/test_v2_package.py -q
```

Expected: FAIL because `pybacktest.__version__` does not exist.

- [ ] **Step 3: Archive V1 and create the new package root**

Run these commands individually:

```bash
rtk proxy mkdir -p legacy/v1/src legacy/v1/ui legacy/v1/examples
rtk git add tests/test_v2_package.py
rtk git mv src/pybacktest legacy/v1/src/pybacktest
rtk git mv tests legacy/v1/tests
rtk proxy mkdir -p src/pybacktest tests
rtk git mv legacy/v1/tests/test_v2_package.py tests/test_package.py
rtk git mv streamlit_page.py legacy/v1/ui/streamlit_page.py
rtk git mv streamlit_page_en.py legacy/v1/ui/streamlit_page_en.py
rtk git mv streamlit_page_ko.py legacy/v1/ui/streamlit_page_ko.py
rtk git mv streamlit_ui legacy/v1/ui/streamlit_ui
rtk git mv backtest_test.ipynb legacy/v1/examples/backtest_test.ipynb
rtk git mv strategy_test.json legacy/v1/examples/strategy_test.json
rtk git rm -r src/pybacktest.egg-info
```

`tests/fixtures/strategy_test_format.json` moves with the V1 test directory;
there is no package-root `strategy_test_format.json`.

Create `src/pybacktest/__init__.py`:

```python
"""Deterministic, explicit backtesting primitives and engine."""

__version__ = "0.2.0"
```

Create `legacy/v1/README.md`:

```markdown
# Pybacktest V1 Archive

This directory preserves the pre-0.2 source, tests, Streamlit application,
notebook, and examples for history only. It is not installed, tested, or
supported by Pybacktest 0.2. No V1 compatibility shim is provided.
```

- [ ] **Step 4: Replace project metadata with the V2 dependency boundary**

Set the relevant `pyproject.toml` sections to:

```toml
[project]
name = "pybacktest"
version = "0.2.0"
description = "A deterministic, extensible Python backtesting engine"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "sciencemj", email = "sciencemj.park@gmail.com"}]
dependencies = [
    "numpy>=2.0",
]

[project.optional-dependencies]
data = ["pandas>=2.2"]
parquet = ["pyarrow>=17"]
yfinance = ["yfinance>=0.2"]
plot = ["matplotlib>=3.9"]

[dependency-groups]
dev = [
    "hypothesis",
    "pandas>=2.2",
    "pyarrow>=17",
    "pytest>=8",
    "pytest-benchmark",
    "ruff",
    "ty",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
```

Replace `README.md` with a short V2 notice that links to the approved design
and states that API documentation will be filled in by Task 11. Do not retain
V1 usage examples at the package root.

Regenerate the lock:

```bash
rtk proxy uv lock
```

- [ ] **Step 5: Verify the baseline**

Run:

```bash
rtk proxy uv run python -m pytest tests/test_package.py -q
rtk proxy uv run ruff check src tests
```

Expected: one passing test and no Ruff errors.

- [ ] **Step 6: Commit**

```bash
rtk git add pyproject.toml uv.lock README.md src tests legacy
rtk git commit -m "refactor: establish pybacktest v2 package"
```

## Task 2: Domain Value Objects, Time, and Instrument Identity

**Files:**
- Create: `src/pybacktest/domain/__init__.py`
- Create: `src/pybacktest/domain/errors.py`
- Create: `src/pybacktest/domain/identifiers.py`
- Create: `src/pybacktest/domain/instruments.py`
- Create: `src/pybacktest/domain/money.py`
- Create: `src/pybacktest/domain/time.py`
- Create: `tests/domain/test_values.py`

**Interfaces:**
- Consumes: Python 3.11 stdlib only.
- Produces:
  - `InstrumentId.parse(value: str) -> InstrumentId`
  - `Instrument(id, quote_currency, tick_size, lot_size, timezone)`
  - `Money.of(amount, currency)`, `Money.usd(amount)`
  - `Quantity.of(value).quantized(lot_size) -> Quantity`
  - `Timeframe.minutes(count)`, `Timeframe.days(count)`
  - `DateRange(start, end)` with `[start, end)` semantics
  - `RunId.new()`, `OrderId.new()`, `FillId.new()`
  - `RunId.parse()`, `OrderId.parse()`, `FillId.parse()`

- [ ] **Step 1: Write failing value-object tests**

Create `tests/domain/test_values.py`:

```python
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.time import DateRange, Timeframe


def test_instrument_id_round_trips_canonical_form():
    instrument_id = InstrumentId.parse("XNAS:AAPL")
    assert instrument_id.venue == "XNAS"
    assert instrument_id.symbol == "AAPL"
    assert str(instrument_id) == "XNAS:AAPL"


def test_instrument_rejects_non_positive_tick_and_lot():
    with pytest.raises(ConfigurationError, match="tick_size"):
        Instrument(
            id=InstrumentId.parse("XNAS:AAPL"),
            quote_currency="USD",
            tick_size=Decimal("0"),
            lot_size=Decimal("1"),
            timezone=ZoneInfo("America/New_York"),
        )


def test_money_requires_matching_currency_for_addition():
    with pytest.raises(ConfigurationError, match="currency"):
        Money.usd("1") + Money.of("1", "KRW")


def test_quantity_quantizes_down_to_lot_size():
    quantity = Quantity.of("1.234").quantized(Decimal("0.01"))
    assert quantity.value == Decimal("1.23")


def test_date_range_is_start_inclusive_end_exclusive():
    period = DateRange(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
    )
    assert period.contains(datetime(2024, 1, 1, tzinfo=UTC))
    assert not period.contains(datetime(2024, 2, 1, tzinfo=UTC))


def test_timeframe_rejects_zero_count():
    with pytest.raises(ConfigurationError, match="count"):
        Timeframe.minutes(0)
```

- [ ] **Step 2: Run the tests and verify missing modules**

```bash
rtk proxy uv run python -m pytest tests/domain/test_values.py -q
```

Expected: collection FAIL because `pybacktest.domain` does not exist.

- [ ] **Step 3: Implement typed errors and immutable value objects**

Use frozen, slotted dataclasses. The core validation pattern is:

```python
# src/pybacktest/domain/errors.py
class PybacktestError(Exception):
    """Base class for stable Pybacktest failures."""


class ConfigurationError(PybacktestError, ValueError):
    """Raised when typed configuration violates a declared invariant."""


class DataValidationError(PybacktestError, ValueError):
    """Raised when market data violates the source contract."""


class LookaheadViolation(PybacktestError):
    """Raised when code requests data after the current engine timestamp."""


class AccountingInvariantError(PybacktestError):
    """Raised when a ledger transition cannot be reconciled."""


class ClockRegressionError(PybacktestError):
    """Raised when simulation time moves backwards."""


class AdapterContractError(PybacktestError):
    """Raised when a port implementation violates its contract."""
```

Implement `InstrumentId` with an exact two-part parser:

```python
@dataclass(frozen=True, slots=True, order=True)
class InstrumentId:
    venue: str
    symbol: str

    def __post_init__(self) -> None:
        if not self.venue or not self.symbol:
            raise ConfigurationError("InstrumentId requires venue and symbol.")
        if ":" in self.venue or ":" in self.symbol:
            raise ConfigurationError("InstrumentId parts cannot contain ':'.")

    @classmethod
    def parse(cls, value: str) -> "InstrumentId":
        parts = value.split(":")
        if len(parts) != 2:
            raise ConfigurationError(
                "InstrumentId must use the canonical 'VENUE:SYMBOL' form."
            )
        return cls(venue=parts[0].upper(), symbol=parts[1].upper())

    def __str__(self) -> str:
        return f"{self.venue}:{self.symbol}"
```

Implement `Money` and `Quantity` with `Decimal(str(value))`, finite checks,
currency normalization, currency-safe arithmetic, and `ROUND_DOWN` lot
quantization. Implement `DateRange` with timezone-aware datetime validation and
`start < end`. Implement `Timeframe` as a frozen value with `unit` restricted
to `"minute"` or `"day"` and positive `count`.

Implement IDs as frozen wrappers around canonical lowercase identifiers. Each
type requires its own prefix followed by exactly 32 lowercase hex characters;
this rejects path separators and cross-type confusion. `new()` uses that type
prefix plus UUID hex:

```python
@dataclass(frozen=True, slots=True)
class OrderId:
    value: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"order_[0-9a-f]{32}", self.value) is None:
            raise ConfigurationError("OrderId is not in canonical form.")

    @classmethod
    def new(cls) -> "OrderId":
        return cls(f"order_{uuid4().hex}")

    @classmethod
    def parse(cls, value: str) -> "OrderId":
        return cls(value)

    def __str__(self) -> str:
        return self.value
```

Repeat the same shape and type-specific regex for `FillId` (`fill_...`) and
`RunId` (`run_...`). Tests that need stable IDs use 32 repeated hex
characters after the prefix, never relaxed production validation.

- [ ] **Step 4: Run domain tests**

```bash
rtk proxy uv run python -m pytest tests/domain/test_values.py -q
rtk proxy uv run ruff check src/pybacktest/domain tests/domain
rtk proxy uv run ty check src/pybacktest/domain
```

Expected: all tests pass; Ruff and ty report no errors.

- [ ] **Step 5: Commit**

```bash
rtk git add src/pybacktest/domain tests/domain
rtk git commit -m "feat: add immutable domain value objects"
```

## Task 3: Orders, Fills, and Structured Events

**Files:**
- Create: `src/pybacktest/domain/orders.py`
- Create: `src/pybacktest/domain/events.py`
- Create: `tests/domain/test_orders.py`

**Interfaces:**
- Consumes: `InstrumentId`, `Money`, `Quantity`, `OrderId`, `FillId`.
- Produces:
  - Enums `OrderSide`, `OrderType`, `TimeInForce`, `OrderStatus`
  - Intents `TargetWeight`, `TargetQuantity`, `MarketOrderIntent`,
    `LimitOrderIntent`, `CancelOrderIntent`
  - immutable `DecisionReason(code, details)`
  - `Order.accept()`, `Order.apply_fill()`, `Order.cancel()`, `Order.reject()`
  - `Fill`
  - Events `OrderAccepted`, `OrderAdjusted`, `OrderRejected`, `OrderExpired`,
    `PartialFill`, `DataUnavailable`, `DecisionTraceEntry`

- [ ] **Step 1: Write failing order-state tests**

```python
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


NOW = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
AAPL = InstrumentId.parse("XNAS:AAPL")


def make_order() -> Order:
    return Order.pending(
        id=OrderId.new(),
        instrument=AAPL,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Quantity.of("10"),
        limit_price=Money.usd("100"),
        time_in_force=TimeInForce.DAY,
        submitted_at=NOW,
        active_from=NOW,
        reason=DecisionReason.of(
            "ma_cross",
            signal="fast_crossed_above_slow",
        ),
    )


def test_order_state_machine_accepts_partial_then_full_fill():
    accepted = make_order().accept()
    first = Fill(
        id=FillId.new(),
        order_id=accepted.id,
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("4"),
        price=Money.usd("99"),
        fee=Money.usd("1"),
        timestamp=NOW,
    )
    partial = accepted.apply_fill(first)
    assert partial.status is OrderStatus.PARTIALLY_FILLED
    assert partial.remaining_quantity.value == Decimal("6")

    second = Fill(
        id=FillId.new(),
        order_id=accepted.id,
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("6"),
        price=Money.usd("100"),
        fee=Money.usd("1"),
        timestamp=NOW,
    )
    filled = partial.apply_fill(second)
    assert filled.status is OrderStatus.FILLED
    assert filled.remaining_quantity.value == Decimal("0")


def test_filled_order_cannot_be_cancelled():
    order = make_order().accept()
    fill = Fill(
        id=FillId.new(),
        order_id=order.id,
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("10"),
        price=Money.usd("100"),
        fee=Money.usd("0"),
        timestamp=NOW,
    )
    with pytest.raises(ConfigurationError, match="terminal"):
        order.apply_fill(fill).cancel("too late")
```

- [ ] **Step 2: Verify the tests fail**

```bash
rtk proxy uv run python -m pytest tests/domain/test_orders.py -q
```

Expected: collection FAIL because order types do not exist.

- [ ] **Step 3: Implement intent DTOs and the order transition table**

Use frozen dataclasses. `DecisionReason.details` accepts only JSON scalar
values (`str | int | float | bool | None`), copies the input, and exposes an
immutable `MappingProxyType`. Its convenience constructor is explicit:

```python
@classmethod
def of(cls, code: str, **details: JSONScalar) -> "DecisionReason":
    return cls(code=code, details=MappingProxyType(dict(details)))
```

Every intent and order requires a `DecisionReason`; plain strings and mutable
dictionaries are rejected. `Order.apply_fill()` must reject an instrument, side,
currency, order ID, or quantity mismatch. The only legal transitions are:

```text
PENDING -> ACCEPTED | REJECTED | CANCELLED
ACCEPTED -> PARTIALLY_FILLED | FILLED | CANCELLED | REJECTED
PARTIALLY_FILLED -> PARTIALLY_FILLED | FILLED | CANCELLED
FILLED | CANCELLED | REJECTED -> no transition
```

Implement transitions by returning a replaced immutable order:

```python
def apply_fill(self, fill: Fill) -> "Order":
    if self.status not in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}:
        raise ConfigurationError("Cannot fill an order outside an active state.")
    if fill.order_id != self.id or fill.instrument != self.instrument:
        raise ConfigurationError("Fill identity does not match the order.")
    next_filled = self.filled_quantity.value + fill.quantity.value
    if next_filled > self.quantity.value:
        raise ConfigurationError("Fill quantity exceeds remaining order quantity.")
    next_status = (
        OrderStatus.FILLED
        if next_filled == self.quantity.value
        else OrderStatus.PARTIALLY_FILLED
    )
    return replace(
        self,
        status=next_status,
        filled_quantity=Quantity.of(next_filled),
    )
```

Give every structured event `code`, `timestamp`, and typed identity fields.
Messages are display text and are never parsed for control flow.

- [ ] **Step 4: Verify order tests and static checks**

```bash
rtk proxy uv run python -m pytest tests/domain/test_orders.py -q
rtk proxy uv run ruff check src/pybacktest/domain tests/domain
rtk proxy uv run ty check src/pybacktest/domain
```

- [ ] **Step 5: Commit**

```bash
rtk git add src/pybacktest/domain/orders.py src/pybacktest/domain/events.py tests/domain/test_orders.py
rtk git commit -m "feat: model order lifecycle and events"
```

## Task 4: Validated Columnar Market Data and Calendar

**Files:**
- Create: `src/pybacktest/ports/__init__.py`
- Create: `src/pybacktest/ports/data.py`
- Create: `src/pybacktest/data/__init__.py`
- Create: `src/pybacktest/data/dataset.py`
- Create: `src/pybacktest/data/validation.py`
- Create: `src/pybacktest/data/calendar.py`
- Create: `src/pybacktest/data/fingerprint.py`
- Create: `src/pybacktest/domain/market.py`
- Create: `src/pybacktest/adapters/__init__.py`
- Create: `src/pybacktest/adapters/data/__init__.py`
- Create: `src/pybacktest/adapters/data/pandas.py`
- Create: `src/pybacktest/adapters/data/parquet.py`
- Create: `tests/data/test_dataset.py`
- Create: `tests/data/test_adapters.py`

**Interfaces:**
- Consumes: `Instrument`, `InstrumentId`, `Timeframe`, `DateRange`.
- Produces:
  - `BarSeries` with read-only NumPy `datetime64[ns]` and `float64` arrays
  - `MarketDataSet(series, instruments, timeframe, fingerprint)`
  - `MarketSlice(timestamp, bars)` with immutable `BarView`
  - `CalendarPolicy.union(max_staleness_bars)` and `.intersection()`
  - `MarketDataSource.load(universe, period, timeframe) -> MarketDataSet`
  - `PandasDataSource` and optional `ParquetDataSource`

- [ ] **Step 1: Write failing validation and calendar tests**

```python
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from pybacktest.adapters.data.pandas import PandasDataSource
from pybacktest.data.calendar import CalendarPolicy
from pybacktest.domain.errors import DataValidationError
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.time import DateRange, Timeframe


AAPL = Instrument(
    id=InstrumentId.parse("XNAS:AAPL"),
    quote_currency="USD",
    tick_size=Decimal("0.01"),
    lot_size=Decimal("1"),
    timezone=ZoneInfo("America/New_York"),
)


def frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument": [str(AAPL.id)] * len(index),
            "open": np.full(len(index), 100.0),
            "high": np.full(len(index), 101.0),
            "low": np.full(len(index), 99.0),
            "close": np.full(len(index), 100.5),
            "volume": np.full(len(index), 1_000.0),
        },
        index=index,
    )


def test_pandas_source_normalizes_to_read_only_float64_arrays():
    index = pd.date_range("2024-01-02", periods=2, tz="UTC", freq="D")
    source = PandasDataSource(frame(index), instruments={AAPL.id: AAPL})
    dataset = source.load(
        [AAPL.id],
        DateRange(index[0].to_pydatetime(), (index[-1] + pd.Timedelta(days=1)).to_pydatetime()),
        Timeframe.days(1),
    )
    series = dataset.series[AAPL.id]
    assert series.close.dtype == np.float64
    assert not series.close.flags.writeable


def test_duplicate_timestamp_is_rejected():
    index = pd.DatetimeIndex(
        [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)]
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        PandasDataSource(frame(index), instruments={AAPL.id: AAPL}).load(
            [AAPL.id],
            DateRange(index[0], datetime(2024, 1, 3, tzinfo=UTC)),
            Timeframe.days(1),
        )


def test_high_low_invariant_is_rejected():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    invalid = frame(index)
    invalid.loc[index[0], "high"] = 90.0
    with pytest.raises(DataValidationError, match="high"):
        PandasDataSource(invalid, instruments={AAPL.id: AAPL}).load(
            [AAPL.id],
            DateRange(index[0].to_pydatetime(), datetime(2024, 1, 3, tzinfo=UTC)),
            Timeframe.days(1),
        )


def test_union_calendar_keeps_dates_seen_by_only_one_instrument():
    timestamps = {
        InstrumentId.parse("XNAS:AAPL"): np.array(
            ["2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
        ),
        InstrumentId.parse("XNAS:MSFT"): np.array(
            ["2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
        ),
    }
    calendar = CalendarPolicy.union(max_staleness_bars=1).build(timestamps)
    assert calendar.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]


def test_source_never_loads_at_or_after_exclusive_end():
    index = pd.date_range("2024-01-01", periods=5, tz="UTC", freq="D")
    source = PandasDataSource(frame(index), instruments={AAPL.id: AAPL})
    dataset = source.load(
        [AAPL.id],
        DateRange(index[0].to_pydatetime(), index[3].to_pydatetime()),
        Timeframe.days(1),
    )
    np.testing.assert_array_equal(
        dataset.series[AAPL.id].timestamps,
        index[:3].to_numpy(dtype="datetime64[ns]"),
    )
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run python -m pytest tests/data/test_dataset.py tests/data/test_adapters.py -q
```

Expected: collection FAIL because data modules do not exist.

- [ ] **Step 3: Implement the data port and immutable array store**

Define:

```python
class MarketDataSource(Protocol):
    def load(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        """Load a fixed, validated dataset without refreshing external data."""
        raise NotImplementedError
```

`BarSeries.__post_init__()` must:

1. Coerce timestamps to `datetime64[ns]`.
2. Coerce OHLCV to `float64`.
3. Reject unequal lengths, non-finite values, non-positive prices, negative
   volume, duplicate/non-increasing timestamps, and OHLC invariant violations.
4. Copy arrays and set `write=False`.

`MarketDataSet` stores a mapping of instrument to `BarSeries`, a matching
instrument metadata mapping, a `Timeframe`, and a SHA-256 fingerprint over
canonical metadata plus raw array bytes. It copies both mappings into
`MappingProxyType`; callers cannot replace a series or instrument after the
fingerprint is calculated.

`MarketSlice` must expose only the current timestamp's `BarView`; stale
mark-to-market values belong in a separate lookup and never make an
instrument tradable.

- [ ] **Step 4: Implement Pandas and Parquet adapters**

`PandasDataSource` accepts a DataFrame with UTC-aware index and exact lowercase
columns `instrument, open, high, low, close, volume`. It groups by canonical
instrument ID, slices `[period.start, period.end)`, constructs `BarSeries`, and
never relabels unknown columns by position.

`ParquetDataSource(path, instruments)` performs no I/O in `__init__`; `load()`
uses `pyarrow.dataset` to filter period/instrument columns and delegates the
result to `PandasDataSource`. Raise `AdapterContractError` with code
`optional_dependency_missing` when PyArrow is unavailable.

- [ ] **Step 5: Verify data contracts**

```bash
rtk proxy uv run python -m pytest tests/data -q
rtk proxy uv run ruff check src/pybacktest/data src/pybacktest/adapters/data tests/data
rtk proxy uv run ty check src/pybacktest/data src/pybacktest/adapters/data
```

- [ ] **Step 6: Commit**

```bash
rtk git add src/pybacktest/domain/market.py src/pybacktest/ports src/pybacktest/data src/pybacktest/adapters/__init__.py src/pybacktest/adapters/data tests/data
rtk git commit -m "feat: add validated columnar market data"
```

## Task 5: Causal Vectorized Feature Plans and Strategy Protocol

**Files:**
- Create: `src/pybacktest/data/features.py`
- Create: `src/pybacktest/ports/strategy.py`
- Create: `src/pybacktest/strategy/__init__.py`
- Create: `src/pybacktest/strategy/intents.py`
- Create: `tests/features/test_features.py`
- Create: `tests/strategy/test_protocol.py`

**Interfaces:**
- Consumes: `MarketDataSet`, `MarketSlice`, order intent DTOs, immutable portfolio snapshot interface.
- Produces:
  - `FeatureBuilder.source(name, instrument, field) -> FeatureNode`
  - `FeatureBuilder.lag(name, node, periods) -> FeatureNode`
  - `FeatureBuilder.sma(name, node, window) -> FeatureNode`
  - `FeatureBuilder.ema(name, node, span) -> FeatureNode`
  - `FeatureBuilder.plan() -> FeaturePlan`
  - `FeatureExecutor.execute(plan, dataset) -> FeatureSet`
  - `FeatureSet.view(timestamp) -> FeatureView`
  - `Strategy.build_features(builder) -> FeaturePlan`
  - `Strategy.on_bar(context, market) -> Sequence[OrderIntent]`

- [ ] **Step 1: Write failing causal-feature tests**

```python
import numpy as np
import pytest

from pybacktest.data.features import FeatureBuilder, FeatureExecutor
from pybacktest.domain.errors import ConfigurationError, LookaheadViolation
from tests.factories import one_instrument_dataset


def test_sma_is_vectorized_and_prefix_invariant():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3, 4, 5])
    builder = FeatureBuilder()
    close = builder.source("close", instrument, "close")
    builder.sma("sma3", close, window=3)
    plan = builder.plan()

    full = FeatureExecutor().execute(plan, dataset)
    prefix = FeatureExecutor().execute(plan, dataset.prefix(3))

    np.testing.assert_allclose(full.column("sma3")[:3], prefix.column("sma3"))
    np.testing.assert_allclose(full.column("sma3"), [np.nan, np.nan, 2, 3, 4])


def test_negative_lag_and_duplicate_feature_names_are_rejected():
    builder = FeatureBuilder()
    dataset, instrument = one_instrument_dataset(closes=[1, 2])
    close = builder.source("close", instrument, "close")
    with pytest.raises(ConfigurationError, match="positive"):
        builder.lag("future", close, periods=-1)
    with pytest.raises(ConfigurationError, match="duplicate"):
        builder.source("close", instrument, "open")


def test_feature_view_rejects_future_timestamp():
    dataset, instrument = one_instrument_dataset(closes=[1, 2, 3])
    builder = FeatureBuilder()
    builder.source("close", instrument, "close")
    features = FeatureExecutor().execute(builder.plan(), dataset)
    view = features.view(dataset.timestamps[1])
    with pytest.raises(LookaheadViolation):
        view.at("close", offset=1)
```

Create `tests/factories.py` in this task with deterministic builders for
`Instrument`, `MarketDataSet`, `Order`, and timestamps; later tests must reuse
these factories instead of copying fixtures.

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run python -m pytest tests/features/test_features.py -q
```

Expected: collection FAIL because `FeatureBuilder` does not exist.

- [ ] **Step 3: Implement a closed causal operator graph**

Represent every node as:

```python
@dataclass(frozen=True, slots=True)
class FeatureNode:
    name: str
    operator: Literal["source", "lag", "sma", "ema"]
    inputs: tuple[str, ...]
    instrument: InstrumentId | None
    field: Literal["open", "high", "low", "close", "volume"] | None
    parameter: int | None
    lookback: int
```

`FeatureBuilder` must reject duplicate names, non-positive windows/spans,
negative/zero lag, unknown source fields, and references from a different
builder. Topologically order the immutable nodes in `FeaturePlan`.

Implement SMA with cumulative sums and an output initialized to NaN; emit a
value only after `window` observations. Implement EMA with an explicit
adjust-false recurrence seeded after the first finite input. Do not expose a
generic callable operator in V2 core.

`FeatureView.at(name, offset=0)` accepts only `offset <= 0`; positive offsets
raise `LookaheadViolation`.

- [ ] **Step 4: Define the strategy boundary**

Create immutable `StrategyContext(timestamp, portfolio, active_orders,
features, run_id)` and the protocol:

```python
class Strategy(Protocol):
    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        """Declare a causal feature plan without reading market values."""
        raise NotImplementedError

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> Sequence[OrderIntent]:
        """Return intents without mutating portfolio or market state."""
        raise NotImplementedError
```

Add a runtime helper `validate_strategy_output()` that rejects non-intents,
duplicate cancel requests, and intents for instruments outside the universe.

- [ ] **Step 5: Verify features and protocol**

```bash
rtk proxy uv run python -m pytest tests/features tests/strategy -q
rtk proxy uv run ruff check src/pybacktest/data/features.py src/pybacktest/ports/strategy.py tests/features tests/strategy
rtk proxy uv run ty check src/pybacktest/data/features.py src/pybacktest/ports/strategy.py
```

- [ ] **Step 6: Commit**

```bash
rtk git add src/pybacktest/data/features.py src/pybacktest/ports/strategy.py src/pybacktest/strategy tests/features tests/strategy tests/factories.py
rtk git commit -m "feat: add causal feature and strategy contracts"
```

## Task 6: Portfolio Ledger and Accounting Invariants

**Files:**
- Create: `src/pybacktest/domain/portfolio.py`
- Create: `src/pybacktest/engine/__init__.py`
- Create: `src/pybacktest/engine/accounting.py`
- Create: `tests/engine/test_accounting.py`
- Create: `tests/engine/test_accounting_properties.py`

**Interfaces:**
- Consumes: `Fill`, `Money`, `Quantity`, `Instrument`, `MarketSlice`.
- Produces:
  - `Position(instrument, quantity: Quantity, average_price: Money | None, realized_pnl: Money)`
  - immutable `PortfolioSnapshot`
  - `PortfolioLedger.apply_fill(fill) -> PortfolioSnapshot`
  - `PortfolioLedger.mark_to_market(timestamp, prices) -> PortfolioSnapshot`
  - `PortfolioLedger.apply_cash_event(event) -> PortfolioSnapshot`

- [ ] **Step 1: Write exact long/short accounting tests**

```python
from decimal import Decimal

from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import Fill, OrderSide
from pybacktest.engine.accounting import PortfolioLedger
from tests.factories import aapl, fill


def test_long_entry_partial_exit_and_fees_reconcile():
    ledger = PortfolioLedger(base_currency="USD", initial_cash=Money.usd("10000"))
    ledger.apply_fill(
        fill(aapl().id, OrderSide.BUY, quantity="10", price="100", fee="1")
    )
    snapshot = ledger.apply_fill(
        fill(aapl().id, OrderSide.SELL, quantity="4", price="110", fee="1")
    )

    assert snapshot.cash.amount == Decimal("9438")
    assert snapshot.positions[aapl().id].quantity.value == Decimal("6")
    assert snapshot.positions[aapl().id].average_price.amount == Decimal("100")
    assert snapshot.realized_pnl.amount == Decimal("40")
    assert snapshot.total_fees.amount == Decimal("2")


def test_sell_crossing_zero_opens_short_at_fill_price():
    ledger = PortfolioLedger(base_currency="USD", initial_cash=Money.usd("10000"))
    ledger.apply_fill(
        fill(aapl().id, OrderSide.BUY, quantity="10", price="100", fee="0")
    )
    snapshot = ledger.apply_fill(
        fill(aapl().id, OrderSide.SELL, quantity="15", price="110", fee="0")
    )
    position = snapshot.positions[aapl().id]
    assert position.quantity.value == Decimal("-5")
    assert position.average_price.amount == Decimal("110")
    assert snapshot.realized_pnl.amount == Decimal("100")


def test_mark_to_market_does_not_mutate_previous_snapshot():
    ledger = PortfolioLedger(base_currency="USD", initial_cash=Money.usd("10000"))
    before = ledger.snapshot()
    after = ledger.mark_to_market(
        prices={aapl().id: Money.usd("105")},
        timestamp=fill.timestamp,
    )
    assert before is not after
    assert before.timestamp is None
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run python -m pytest tests/engine/test_accounting.py -q
```

Expected: collection FAIL because `PortfolioLedger` does not exist.

- [ ] **Step 3: Implement fill accounting as one explicit transition**

Implement `PortfolioLedger` with private cash/position state and immutable
snapshots. For a fill:

1. Quantize quantity to instrument lot size and price/fee to currency units.
2. Reject currency mismatch and non-finite/negative values.
3. Split a fill crossing zero into a closing leg and opening leg.
4. Calculate realized P&L only on the closing leg:
   - long close: `(sell_price - average_price) * closed_quantity`
   - short close: `(average_price - buy_price) * closed_quantity`
5. Change cash by signed notional minus fee.
6. Preserve average price on partial close; reset it on flat; use fill price on
   a new or crossed position.
7. Reconcile cash and position deltas before publishing a snapshot.

Raise `AccountingInvariantError` when reconciliation differs after Decimal
quantization.

- [ ] **Step 4: Add property tests**

Use Hypothesis to generate sequences of buy/sell fills with positive prices
and quantities. Assert:

```python
assert final_position == sum(
    fill.quantity.value if fill.side is OrderSide.BUY else -fill.quantity.value
    for fill in fills
)
assert snapshot.total_fees.amount == sum(fill.fee.amount for fill in fills)
assert all(event.timestamp >= previous for previous, event in adjacent_events)
```

Generate only USD fills and explicitly include zero-crossing sequences.

- [ ] **Step 5: Verify accounting**

```bash
rtk proxy uv run python -m pytest tests/engine/test_accounting.py tests/engine/test_accounting_properties.py -q
rtk proxy uv run ruff check src/pybacktest/engine/accounting.py src/pybacktest/domain/portfolio.py tests/engine
rtk proxy uv run ty check src/pybacktest/engine/accounting.py src/pybacktest/domain/portfolio.py
```

- [ ] **Step 6: Commit**

```bash
rtk git add src/pybacktest/domain/portfolio.py src/pybacktest/engine/accounting.py tests/engine
rtk git commit -m "feat: add reconciled long short ledger"
```

## Task 7: Target Sizing and Long/Short Risk Decisions

**Files:**
- Create: `src/pybacktest/ports/risk.py`
- Create: `src/pybacktest/risk/__init__.py`
- Create: `src/pybacktest/risk/sizing.py`
- Create: `src/pybacktest/risk/policies.py`
- Create: `tests/risk/test_sizing.py`
- Create: `tests/risk/test_policies.py`

**Interfaces:**
- Consumes: order intents, `PortfolioSnapshot`, current prices/instruments.
- Produces:
  - `OrderSizer.size(intent, context) -> Order | OrderRejected`
  - `RiskPolicy.evaluate(order, context) -> RiskDecision`
  - `RiskDecision(status, original_quantity, final_quantity, codes, message)`
  - `LongShortRisk(max_leverage, max_position_weight, allow_short)`

- [ ] **Step 1: Write failing sizing and risk tests**

```python
from decimal import Decimal

from pybacktest.domain.orders import DecisionReason, OrderSide, TargetWeight
from pybacktest.risk.policies import LongShortRisk, RiskStatus
from pybacktest.risk.sizing import DefaultOrderSizer
from tests.factories import aapl, portfolio_snapshot, risk_context


def test_target_weight_sizes_from_current_equity_and_position():
    snapshot = portfolio_snapshot(cash="10000", positions={})
    intent = TargetWeight(
        instrument=aapl().id,
        weight=Decimal("0.50"),
        reason=DecisionReason.of("allocate_half"),
    )
    order = DefaultOrderSizer().size(
        intent,
        risk_context(snapshot=snapshot, prices={aapl().id: "100"}),
    )
    assert order.side is OrderSide.BUY
    assert order.quantity.value == Decimal("50")


def test_risk_adjustment_is_visible_not_silent():
    snapshot = portfolio_snapshot(cash="10000", positions={})
    order = DefaultOrderSizer().size(
        TargetWeight(
            instrument=aapl().id,
            weight=Decimal("1"),
            reason=DecisionReason.of("full_allocation"),
        ),
        risk_context(snapshot=snapshot, prices={aapl().id: "100"}),
    )
    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("0.25"),
        allow_short=True,
    ).evaluate(order, risk_context(snapshot=snapshot, prices={aapl().id: "100"}))
    assert decision.status is RiskStatus.ADJUSTED
    assert decision.original_quantity.value == Decimal("100")
    assert decision.final_quantity.value == Decimal("25")
    assert decision.codes == ("max_position_weight",)


def test_short_order_is_rejected_when_disabled():
    snapshot = portfolio_snapshot(cash="10000", positions={})
    order = DefaultOrderSizer().size(
        TargetWeight(
            instrument=aapl().id,
            weight=Decimal("-0.5"),
            reason=DecisionReason.of("short_signal"),
        ),
        risk_context(snapshot=snapshot, prices={aapl().id: "100"}),
    )
    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(order, risk_context(snapshot=snapshot, prices={aapl().id: "100"}))
    assert decision.status is RiskStatus.REJECTED
    assert decision.codes == ("short_not_allowed",)


def test_zero_target_flattens_position_without_creating_one_share():
    snapshot = portfolio_snapshot(cash="9500", positions={aapl().id: "5"})
    order = DefaultOrderSizer().size(
        TargetWeight(
            instrument=aapl().id,
            weight=Decimal("0"),
            reason=DecisionReason.of("flatten"),
        ),
        risk_context(snapshot=snapshot, prices={aapl().id: "100"}),
    )
    assert order.side is OrderSide.SELL
    assert order.quantity.value == Decimal("5")
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run python -m pytest tests/risk -q
```

- [ ] **Step 3: Implement sizing before risk, without reading broker cash**

`DefaultOrderSizer` converts target intents using marked portfolio equity,
current price, existing signed position, and instrument lot size. Explicit
market/limit intents keep their requested quantity. `CancelOrderIntent` bypasses
sizing.

Return a proposed immutable `Order`; do not cap it to cash in the sizer. This
preserves proceeds from fills processed before the order reaches risk/broker.

- [ ] **Step 4: Implement risk decision composition**

`LongShortRisk.evaluate()` computes projected signed position, gross exposure,
leverage, and position weight. Apply constraints in stable order:

1. instrument/tradability
2. short permission
3. max position weight
4. max gross leverage
5. available cash/margin

Each constraint returns an immutable `RiskDecision`; composed adjustments use
the smallest absolute allowed quantity and accumulate reason codes. A zero
adjusted quantity becomes `REJECTED`, never a zero order.

- [ ] **Step 5: Verify risk behavior**

```bash
rtk proxy uv run python -m pytest tests/risk -q
rtk proxy uv run ruff check src/pybacktest/risk src/pybacktest/ports/risk.py tests/risk
rtk proxy uv run ty check src/pybacktest/risk src/pybacktest/ports/risk.py
```

- [ ] **Step 6: Commit**

```bash
rtk git add src/pybacktest/ports/risk.py src/pybacktest/risk tests/risk
rtk git commit -m "feat: add explicit sizing and risk decisions"
```

## Task 8: Simulated Broker, Costs, Limit Orders, and Partial Fills

**Files:**
- Create: `src/pybacktest/ports/broker.py`
- Create: `src/pybacktest/adapters/broker/__init__.py`
- Create: `src/pybacktest/adapters/broker/models.py`
- Create: `src/pybacktest/adapters/broker/simulated.py`
- Create: `tests/broker/test_market_fills.py`
- Create: `tests/broker/test_limit_fills.py`
- Create: `tests/broker/test_partial_fills.py`
- Create: `tests/broker/test_order_expiry.py`

**Interfaces:**
- Consumes: `Order`, `Fill`, `MarketSlice`, instrument metadata, engine RNG.
- Produces:
  - `Broker.submit(order) -> Sequence[BrokerEvent]`
  - `Broker.cancel(order_id, timestamp) -> Sequence[BrokerEvent]`
  - `Broker.process(market, rng) -> Sequence[BrokerEvent]`
  - `SimulatedBroker`
  - immutable `SimulatedBrokerFactory.create(run_context) -> SimulatedBroker`
  - `NextBarOpenFill`, `IntrabarPolicy.CONSERVATIVE`
  - `PerShareCommission`, `NoCommission`
  - `VolumeShareSlippage`, `NoSlippage`
  - `VolumeParticipationLimit`, `NoLiquidityLimit`
  - `NoBorrowCost`

- [ ] **Step 1: Write failing next-bar and limit-price tests**

```python
import pytest

from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoSlippage,
    SimulatedBroker,
    VolumeParticipationLimit,
)
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.orders import OrderStatus
from tests.factories import accepted_order, market_slice, rng


def broker() -> SimulatedBroker:
    return SimulatedBroker(
        fill_model=NextBarOpenFill(
            intrabar_policy=IntrabarPolicy.CONSERVATIVE
        ),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(max_volume_ratio=Decimal("0.10")),
        borrow_cost=NoBorrowCost(),
    )


def test_market_order_does_not_fill_on_submission_bar():
    order = accepted_order(
        order_type="market",
        quantity="10",
        submitted_at="2024-01-02T21:00:00Z",
        active_from="2024-01-03T14:30:00Z",
    )
    sim = broker()
    sim.submit(order)
    assert sim.process(market_slice("2024-01-02T21:00:00Z", open="100"), rng()) == []
    events = sim.process(
        market_slice("2024-01-03T14:30:00Z", open="101"), rng()
    )
    assert events[0].fill.price.amount == Decimal("101")


def test_buy_limit_opening_below_limit_gets_open_price_improvement():
    order = accepted_order(
        order_type="limit",
        quantity="10",
        limit_price="100",
        active_from="2024-01-03T14:30:00Z",
    )
    sim = broker()
    sim.submit(order)
    event = sim.process(
        market_slice(
            "2024-01-03T14:30:00Z",
            open="98",
            high="102",
            low="97",
            close="101",
        ),
        rng(),
    )[0]
    assert event.fill.price.amount == Decimal("98")


def test_volume_limit_creates_partial_fill_and_keeps_order_active():
    order = accepted_order(
        order_type="market",
        quantity="200",
        active_from="2024-01-03T14:30:00Z",
    )
    sim = broker()
    sim.submit(order)
    events = sim.process(
        market_slice("2024-01-03T14:30:00Z", open="100", volume="1000"),
        rng(),
    )
    assert events[0].fill.quantity.value == Decimal("100")
    assert sim.active_orders[order.id].status is OrderStatus.PARTIALLY_FILLED


def test_negative_volume_ratio_is_rejected():
    with pytest.raises(ConfigurationError, match="max_volume_ratio"):
        VolumeParticipationLimit(
            max_volume_ratio=Decimal("-0.01"),
        )
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run python -m pytest tests/broker -q
```

- [ ] **Step 3: Implement explicit model protocols**

Define protocols:

```python
class CommissionModel(Protocol):
    def calculate(self, order: Order, quantity: Quantity, price: Money) -> Money:
        raise NotImplementedError


class SlippageModel(Protocol):
    def apply(
        self,
        order: Order,
        quantity: Quantity,
        reference_price: Money,
        market: BarView,
        rng: np.random.Generator,
    ) -> Money:
        raise NotImplementedError


class LiquidityModel(Protocol):
    def available_quantity(
        self,
        order: Order,
        market: BarView,
    ) -> Quantity:
        raise NotImplementedError
```

`PerShareCommission` multiplies filled quantity by rate and enforces an optional
minimum fee. `VolumeParticipationLimit` caps fill quantity from current bar
volume and `max_volume_ratio`. `VolumeShareSlippage` applies signed basis-point
price impact proportional to the actual participation returned by the
liquidity model. Each model validates finite, non-negative parameters in its
constructor.

- [ ] **Step 4: Implement deterministic broker processing**

Store active orders in submission sequence order. Reject duplicate IDs.
`process()` ignores orders whose `active_from > market.timestamp` or whose
instrument lacks a tradable bar.

Market orders use open. Limit fills follow the approved rules:

- buy: open <= limit → open; otherwise low <= limit → limit
- sell: open >= limit → open; otherwise high >= limit → limit

Apply available volume, lot rounding, slippage, and commission in that order.
Emit `PartialFill` or a terminal fill event and replace the immutable active
order using `Order.apply_fill()`.

Expire DAY orders only when the instrument session boundary supplied by the
calendar passes; GTC orders persist. Cancellation and expiry emit events and
never touch the ledger.

`BacktestEngine` receives `SimulatedBrokerFactory`, not a live broker. The
factory stores only immutable model configuration and creates a fresh broker
for each session, sharing only immutable cost/fill model values.

```python
broker_factory = SimulatedBrokerFactory(
    fill_model=NextBarOpenFill(
        intrabar_policy=IntrabarPolicy.CONSERVATIVE,
    ),
    commission=PerShareCommission(rate_per_share=Decimal("0.005")),
    slippage=VolumeShareSlippage(impact_bps=Decimal("5")),
    liquidity=VolumeParticipationLimit(
        max_volume_ratio=Decimal("0.05"),
    ),
    borrow_cost=NoBorrowCost(),
)
```

This exact construction appears in README; no model is selected by a string
or hidden default.

- [ ] **Step 5: Verify broker contracts**

```bash
rtk proxy uv run python -m pytest tests/broker -q
rtk proxy uv run ruff check src/pybacktest/adapters/broker src/pybacktest/ports/broker.py tests/broker
rtk proxy uv run ty check src/pybacktest/adapters/broker src/pybacktest/ports/broker.py
```

- [ ] **Step 6: Commit**

```bash
rtk git add src/pybacktest/ports/broker.py src/pybacktest/adapters/broker tests/broker
rtk git commit -m "feat: add deterministic simulated broker"
```

## Task 9: Results, Metrics, Causal Traces, and Local Artifacts

**Files:**
- Create: `src/pybacktest/results/__init__.py`
- Create: `src/pybacktest/results/models.py`
- Create: `src/pybacktest/results/metrics.py`
- Create: `src/pybacktest/results/explain.py`
- Create: `src/pybacktest/results/serialization.py`
- Create: `src/pybacktest/ports/artifacts.py`
- Create: `src/pybacktest/adapters/artifacts/__init__.py`
- Create: `src/pybacktest/adapters/artifacts/local.py`
- Create: `src/pybacktest/engine/recorder.py`
- Create: `tests/results/test_metrics.py`
- Create: `tests/results/test_explain.py`
- Create: `tests/results/test_artifacts.py`

**Interfaces:**
- Consumes: ordered engine events, snapshots, orders, fills, run metadata.
- Produces:
  - `MetricsConfig(risk_free_rate, annualization_periods)`
  - immutable `RunManifest`, `ArtifactManifest`, `SummaryMetrics`,
    `BacktestResult`
  - `BacktestResult.replay_fingerprint`
  - `calculate_replay_fingerprint(result) -> str`
  - `BacktestResult.explain_trade(order_id) -> TradeExplanation`
  - `build_trade_explanation(events, order_id) -> TradeExplanation`
  - `ArtifactStore.write(result) -> ArtifactRef`
  - atomic `LocalArtifactStore(root, max_bytes=None)`

- [ ] **Step 1: Write failing metric and trace tests**

```python
import pytest

from pybacktest.results.metrics import MetricsConfig, calculate_metrics
from tests.factories import result_with_trace


def test_max_drawdown_and_total_return_use_declared_series():
    metrics = calculate_metrics(
        equity=[100.0, 120.0, 90.0, 110.0],
        config=MetricsConfig(
            risk_free_rate=0.0,
            annualization_periods=252,
        ),
    )
    assert metrics.total_return == pytest.approx(0.10)
    assert metrics.maximum_drawdown == pytest.approx(-0.25)


def test_explanation_preserves_recorded_causal_order():
    result, order_id = result_with_trace(
        stages=[
            ("feature", {"fast": 101, "slow": 100}),
            ("rule", {"crossed_above": True}),
            ("intent", {"target_weight": "0.5"}),
            ("risk", {"status": "approved"}),
            ("fill", {"price": "102"}),
        ]
    )
    explanation = result.explain_trade(order_id)
    assert [entry.stage for entry in explanation.entries] == [
        "feature",
        "rule",
        "intent",
        "risk",
        "fill",
    ]
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run python -m pytest tests/results -q
```

- [ ] **Step 3: Implement immutable result DTOs and metrics**

Store ordered tuples, not mutable lists, in results. Define metric metadata
alongside each value: formula ID, annualization periods, risk-free rate, and
NaN policy.

Implement total return, CAGR, volatility, Sharpe, Sortino, maximum drawdown,
turnover, win rate, and average gross/net exposure. Return `None` plus a typed
warning when a metric has insufficient observations; never emit infinity.

Use these versioned formulas:

```text
simple_return[t] = equity[t] / equity[t-1] - 1
total_return = final_equity / initial_equity - 1
CAGR = (final_equity / initial_equity) ** (annualization_periods / intervals) - 1
volatility = sample_std(simple_returns, ddof=1) * sqrt(annualization_periods)
sharpe = mean(simple_returns - risk_free_rate / annualization_periods)
         / sample_std(simple_returns, ddof=1)
         * sqrt(annualization_periods)
sortino = mean(simple_returns - risk_free_rate / annualization_periods)
          / sqrt(mean(min(excess_return, 0) ** 2))
          * sqrt(annualization_periods)
maximum_drawdown = min(equity / cumulative_max(equity) - 1)
turnover = sum(abs(fill_quantity * fill_price)) / mean(equity)
win_rate = profitable_closing_legs / all_nonzero_closing_legs
gross_exposure = mean(sum(abs(position_market_value)) / equity)
net_exposure = mean(sum(position_market_value) / equity)
```

Formula IDs are `total_return.v1`, `cagr.v1`, `volatility.v1`,
`sharpe.v1`, `sortino.v1`, `maximum_drawdown.v1`, `turnover.v1`,
`win_rate.v1`, `gross_exposure.v1`, and `net_exposure.v1`. CAGR is `None`
for non-positive endpoint equity; dispersion ratios are `None` for an
insufficient/zero denominator; exposure observations with non-positive equity
are excluded and reported in warnings.

Define the immutable result boundary explicitly:

```python
@dataclass(frozen=True, slots=True)
class BacktestResult:
    manifest: RunManifest
    summary: SummaryMetrics
    market_timestamps: tuple[datetime, ...]
    snapshots: tuple[PortfolioSnapshot, ...]
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    events: tuple[EngineEvent, ...]
    warnings: tuple[RunWarning, ...]

    @property
    def run_id(self) -> RunId:
        return self.manifest.run_id

    def explain_trade(self, order_id: OrderId) -> TradeExplanation:
        return build_trade_explanation(self.events, order_id)

    def replay_fingerprint(self) -> str:
        return calculate_replay_fingerprint(self)
```

`RunManifest` includes run ID, library/schema versions, canonical request,
strategy/spec/compiler identities, dataset fingerprint, seed, adapter/model
versions, and run start/end timestamps. `SummaryMetrics` contains named values
plus their formula metadata and warnings.

Persistence wraps that execution metadata in `ArtifactManifest`, which adds
artifact schema version, artifact ID, sizes, and checksums. Avoid a checksum
cycle: `manifest.json` records checksums for every other artifact but not
itself. The returned `ArtifactRef` carries the finalized `ArtifactManifest`
and the manifest file's own checksum. Persist that checksum as the one-line
`manifest.sha256` sidecar so a later process can verify the manifest before
trusting its table checksums. The immutable engine result is never mutated
after artifact writing.

`replay_fingerprint()` hashes the canonical behavioral payload after replacing
run-scoped run/order/fill IDs with stable sequence ordinals and excluding
wall-clock run metadata and artifact locations. Prices, quantities, event
ordering, decisions, snapshots, and metric metadata remain in the hash.

- [ ] **Step 4: Implement recorder and causal explanation**

`RunRecorder` appends typed events in monotonic `(timestamp, sequence)` order.
It indexes trace entries by `OrderId`. Reject time regression. Build
`BacktestResult` only once at run completion.

`explain_trade()` returns the recorded chain and raises a typed key error for
an unknown order. It must not synthesize a narrative or infer missing stages.
Implement it as a thin call to the pure `build_trade_explanation()` function
so an artifact reader can apply the identical logic to persisted events
without reconstructing an engine.

- [ ] **Step 5: Implement atomic artifact serialization**

Write JSON files using canonical sorted keys and Decimal-as-string encoding.
Write tabular data as Parquet. `LocalArtifactStore.write()` must write to a
temporary sibling directory, compute non-manifest checksums, write the final
manifest and its checksum sidecar last, fsync files and the directory, then
atomically rename it to `<root>/<run_id>`. Return checksums for:

```text
manifest.json
manifest.sha256
config.json
summary.json
equity.parquet
positions.parquet
orders.parquet
fills.parquet
events.parquet
```

Reject paths outside the configured root and duplicate completed `run_id`.
When `max_bytes` is set, count temporary artifact bytes as each file closes
and abort before rename if the total exceeds the limit; remove only the
verified run-specific temporary sibling.

- [ ] **Step 6: Verify results and artifacts**

```bash
rtk proxy uv run python -m pytest tests/results -q
rtk proxy uv run ruff check src/pybacktest/results src/pybacktest/engine/recorder.py src/pybacktest/adapters/artifacts tests/results
rtk proxy uv run ty check src/pybacktest/results src/pybacktest/engine/recorder.py
```

- [ ] **Step 7: Commit**

```bash
rtk git add src/pybacktest/results src/pybacktest/ports/artifacts.py src/pybacktest/adapters/artifacts src/pybacktest/engine/recorder.py tests/results
rtk git commit -m "feat: add replayable results and artifacts"
```

## Task 10: SimulationSession and BacktestEngine

**Files:**
- Create: `src/pybacktest/application/__init__.py`
- Create: `src/pybacktest/application/requests.py`
- Create: `src/pybacktest/application/service.py`
- Modify: `src/pybacktest/engine/__init__.py`
- Create: `src/pybacktest/engine/session.py`
- Create: `src/pybacktest/engine/engine.py`
- Create: `tests/engine/test_session.py`
- Create: `tests/engine/test_engine.py`
- Create: `tests/engine/test_determinism.py`
- Create: `tests/engine/test_lookahead.py`

**Interfaces:**
- Consumes: all core ports/models produced in Tasks 2-9.
- Produces:
  - `SimulationRequest(universe, period, timeframe, calendar, initial_cash, seed, metrics)`
  - `BacktestRequest(strategy, simulation, run_id=None)`
  - `Observation(timestamp, market, portfolio, active_orders, features)`
  - `StepResult(observation, events, done)`
  - `BacktestEngine.create_session(simulation, *, feature_plan, run_id=None, order_sizer=None, risk_policy=None) -> SimulationSession`
  - `SimulationSession.reset() -> Observation`
  - `SimulationSession.advance(intents) -> StepResult`
  - `BacktestEngine.run(request, *, order_sizer=None, risk_policy=None) -> BacktestResult`
  - `BacktestService.run_python(request) -> BacktestResult`

- [ ] **Step 1: Write the failing execution-order test**

```python
from decimal import Decimal

from pybacktest import BacktestEngine
from pybacktest.domain.identifiers import RunId
from tests.factories import (
    always_buy_strategy,
    backtest_request,
    core_engine,
    two_bar_source,
)


def test_close_observed_order_fills_at_next_bar_open():
    engine = core_engine(
        source=two_bar_source(
            first_close="100",
            second_open="105",
            second_close="106",
        )
    )
    result = engine.run(
        backtest_request(strategy=always_buy_strategy(quantity="1"))
    )
    assert len(result.fills) == 1
    assert result.fills[0].price.amount == Decimal("105")
    assert result.fills[0].timestamp == result.market_timestamps[1]


def test_run_and_manual_session_produce_identical_event_stream():
    request = backtest_request(
        strategy=always_buy_strategy(quantity="1"),
        run_id=RunId.parse(f"run_{'1' * 32}"),
    )
    engine = core_engine(source=two_bar_source())
    run_result = engine.run(request)

    builder = FeatureBuilder()
    feature_plan = request.strategy.build_features(builder)
    session = engine.create_session(
        request.simulation,
        feature_plan=feature_plan,
        run_id=request.run_id,
        order_sizer=None,
        risk_policy=None,
    )
    observation = session.reset()
    while not session.done:
        intents = request.strategy.on_bar(
            session.strategy_context(observation),
            observation.market,
        )
        step = session.advance(intents)
        if step.observation is not None:
            observation = step.observation
    manual_result = session.result()

    assert manual_result.events == run_result.events
    assert manual_result.snapshots == run_result.snapshots
```

- [ ] **Step 2: Verify tests fail**

```bash
rtk proxy uv run python -m pytest tests/engine/test_session.py tests/engine/test_engine.py -q
```

- [ ] **Step 3: Implement request validation and session lifecycle**

`SimulationRequest` validates a non-empty unique universe, one quote currency,
period/timeframe compatibility, positive initial cash, and a non-negative
63-bit seed integer.

Add a focused regression test that constructing `SimulationRequest` with
`universe=()` raises `ConfigurationError(code="empty_universe")` rather than
reaching the calendar and failing with an index error.

`BacktestEngine.create_session()` has one public signature:

```python
def create_session(
    self,
    simulation: SimulationRequest,
    *,
    feature_plan: FeaturePlan,
    run_id: RunId | None = None,
    order_sizer: OrderSizer | None = None,
    risk_policy: RiskPolicy | None = None,
) -> SimulationSession: ...
```

There is no overload that accepts a full `BacktestRequest`; callers supplying
external actions, including a future RL wrapper, declare or use an empty
`FeaturePlan` explicitly. `None` creates a new `RunId`; application layers that
must return an ID before asynchronous execution may preallocate one and pass it
through `BacktestRequest`. A `None` order sizer or risk policy uses the
engine's configured default; supplied values are explicit per-run overrides
used by `run_spec()` for the compiled sizer and composed engine/spec policy.

`reset()`:

1. loads and validates the fixed dataset;
2. builds the calendar;
3. initializes RNG, a fresh broker from the factory, ledger, recorder, clock,
   and feature set;
4. processes no strategy intent;
5. returns the first observation.

`advance(intents)`:

1. validates intents against current observation;
2. sizes and risk-checks in sequence;
3. records adjustment/rejection or submits accepted orders with
   `active_from=next tradable timestamp`;
4. advances the clock;
5. processes active broker orders against the new `MarketSlice`;
6. applies fills to ledger;
7. marks positions and records snapshot/events;
8. returns the next observation or `done=True`.

Reject `advance()` before `reset()`, after completion, or with a stale
observation token.

- [ ] **Step 4: Implement `BacktestEngine.run()` only as a session driver**

The run loop must contain no second execution implementation:

```python
def run(
    self,
    request: BacktestRequest,
    *,
    order_sizer: OrderSizer | None = None,
    risk_policy: RiskPolicy | None = None,
) -> BacktestResult:
    builder = FeatureBuilder()
    plan = request.strategy.build_features(builder)
    session = self.create_session(
        request.simulation,
        feature_plan=plan,
        run_id=request.run_id,
        order_sizer=order_sizer,
        risk_policy=risk_policy,
    )
    observation = session.reset()
    while not session.done:
        context = session.strategy_context(observation)
        intents = request.strategy.on_bar(context, observation.market)
        step = session.advance(intents)
        if step.observation is not None:
            observation = step.observation
    return session.result(strategy_identity=type(request.strategy).__qualname__)
```

`BacktestService.run_python()` delegates directly to `BacktestEngine.run()`.

At reset, create one run-scoped ID sequence. It derives order and fill IDs
from `(run_id, kind, monotonic counter)` with UUID5 and never reads wall-clock
time or global randomness. IDs are collision-free across runs; replay
comparison uses the stable ordinal normalization defined in Task 9.

- [ ] **Step 5: Add determinism and look-ahead regression tests**

Run the same request twice and assert equal `replay_fingerprint`, normalized
orders/fills, snapshots, metrics, strategy fingerprint, and dataset
fingerprint. Assert the two collision-free `RunId` values differ. Add a
strategy that requests `features.at("close", offset=1)` and assert
`LookaheadViolation`. Add a duplicate strategy-name test proving `RunId`, not
name, keys results.

Run two sessions concurrently against the same engine and assert neither
session observes the other's active orders, fills, RNG draws, recorder events,
or ledger state.

- [ ] **Step 6: Verify end-to-end engine**

```bash
rtk proxy uv run python -m pytest tests/engine tests/domain tests/data tests/features tests/risk tests/broker tests/results -q
rtk proxy uv run ruff check src tests
rtk proxy uv run ty check src/pybacktest
```

- [ ] **Step 7: Commit**

```bash
rtk git add src/pybacktest/application src/pybacktest/engine tests/engine
rtk git commit -m "feat: add stepable deterministic backtest engine"
```

## Task 11: Public API, Reference Strategy, Documentation, and Performance Gate

**Files:**
- Modify: `src/pybacktest/__init__.py`
- Create: `src/pybacktest/strategy/components.py`
- Create: `tests/integration/test_public_api.py`
- Create: `tests/integration/test_golden_scenario.py`
- Create: `benchmarks/test_engine_benchmark.py`
- Create: `benchmarks/README.md`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: all core interfaces.
- Produces: the curated imports and explicit example approved in the design,
  plus a measured one-million-observation performance gate.

- [ ] **Step 1: Write a failing public API example test**

```python
from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    CalendarPolicy,
    DateRange,
    InstrumentId,
    IntrabarPolicy,
    MetricsConfig,
    Money,
    MovingAverageCross,
    SimulationRequest,
    Timeframe,
)


def test_curated_public_api_is_importable():
    assert InstrumentId.parse("XNAS:AAPL").symbol == "AAPL"
    assert MovingAverageCross(fast=20, slow=60).fast == 20
    assert IntrabarPolicy.CONSERVATIVE.value == "conservative"
```

- [ ] **Step 2: Verify the curated API test fails**

```bash
rtk proxy uv run python -m pytest tests/integration/test_public_api.py -q
```

Expected: FAIL because public re-exports/reference strategy are absent.

- [ ] **Step 3: Add the reference strategy and curated exports**

`MovingAverageCross` must:

1. validate `0 < fast < slow`;
2. declare close, fast SMA, and slow SMA through `FeatureBuilder`;
3. emit `TargetWeight` only on a true cross, not on every bar above/below;
4. record fast/slow values and rule outcome in the reason payload;
5. use explicit `long_weight` and `flat_weight`.

Export only approved value objects, requests, engine/session/result types,
standard adapters/models, risk policy, intents, and this reference strategy
from `pybacktest.__init__`.

- [ ] **Step 4: Add a hand-calculated golden integration scenario**

Use five bars and fixed fills to assert the complete sequence:

```text
bar close crossover
→ target intent
→ sized order
→ approved risk decision
→ next-open partial fill
→ remaining fill
→ ledger cash/position
→ final metric and causal explanation
```

Store expected values directly in the test, including exact Decimal cash,
fees, order statuses, timestamps, and event codes. This is the behavioral
reference for future adapters.

- [ ] **Step 5: Add the performance benchmark and documented runner metadata**

Generate 100 instruments × 10,000 observations with two rolling features and
a sparse market-order strategy. Benchmark only `engine.run()` after dataset
construction. Record peak RSS with `resource.getrusage`; normalize macOS bytes
and Linux KiB to bytes in one tested helper before applying the limit.

The assertion is:

```python
assert benchmark_result_seconds <= 30.0
assert peak_rss_bytes <= 1_500_000_000
```

`benchmarks/README.md` must document CPU, memory, OS, Python, NumPy, dataset
shape, excluded ingestion/serialization time, and the 20% regression rule.
Mark the benchmark `@pytest.mark.performance` so the default unit suite stays
fast.

- [ ] **Step 6: Rewrite README as executable documentation**

Include:

- installation with core and optional extras;
- explicit Parquet ingestion/run example from the design;
- event timing diagram;
- market/limit fill rules;
- ledger/risk/cost explanation;
- `SimulationSession` example;
- artifact and `explain_trade()` example;
- link to the design document;
- statement that V1 and Streamlit are archived and unsupported.

Every README Python block must be exercised by
`tests/integration/test_public_api.py` or a dedicated doctest-style test.

- [ ] **Step 7: Run the final core verification**

```bash
rtk proxy uv run python -m pytest -q
rtk proxy uv run python -m pytest -m performance benchmarks/test_engine_benchmark.py -q
rtk proxy uv run ruff check src tests benchmarks
rtk proxy uv run ruff format --check src tests benchmarks
rtk proxy uv run ty check src/pybacktest
rtk git diff --check
```

Expected: all tests/checks pass; benchmark is at most 30 seconds and 1.5 GB.

- [ ] **Step 8: Commit**

```bash
rtk git add src/pybacktest/__init__.py src/pybacktest/strategy/components.py tests/integration benchmarks README.md pyproject.toml uv.lock
rtk git commit -m "feat: complete pybacktest v2 core"
```

## Core Plan Completion Check

Before starting plan 2, verify:

```bash
rtk git status --short
rtk proxy uv run python -m pytest -q
rtk proxy uv run ruff check src tests benchmarks
rtk proxy uv run ty check src/pybacktest
```

The worktree must be clean. The public interfaces consumed by plan 2 are:

```python
FeatureBuilder
FeaturePlan
Strategy
StrategyContext
OrderIntent
BacktestRequest
BacktestResult
BacktestService.run_python
LocalArtifactStore
RunId
```
