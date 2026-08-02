# Pybacktest 0.2

Pybacktest 0.2 is a deterministic, extensible Python backtesting engine. Broker
execution assumptions — fill model, commission, slippage, liquidity, and
borrow cost — are selected explicitly with typed objects. When omitted, the
engine supplies the documented typed defaults `DefaultOrderSizer()` and an
unrestricted risk policy. Nothing is chosen by a behavior-selecting string,
and the same request always produces the same result.

Its approved design is documented in the
[Pybacktest V2 architecture design](docs/superpowers/specs/2026-07-29-pybacktest-v2-architecture-design.md).

**Pybacktest V1 (0.1.x) and its Streamlit UI are archived and unsupported.**
They are not importable from this package, they receive no fixes, and none of
their usage patterns carry over. The 0.2 surface below is the whole supported
API.

Every Python block in this file is executed or compiled by
`tests/integration/test_readme.py`, so the documentation cannot drift away from
the code.

## Installation

The core install depends only on NumPy.

```bash
uv add pybacktest
```

Optional extras are additive and never required by the core engine.

```bash
uv add "pybacktest[data]"      # PandasDataSource
uv add "pybacktest[parquet]"   # ParquetDataSource (pandas + pyarrow)
uv add "pybacktest[yfinance]"  # third-party yfinance dependency convenience
uv add "pybacktest[plot]"      # third-party matplotlib dependency convenience
```

The `yfinance` and `plot` extras only install those third-party dependencies;
this core provides no bundled yfinance ingestion adapter or plotting helper.

Importing `pybacktest` never imports pandas, pyarrow, yfinance, matplotlib,
Streamlit, an MCP package, or a reinforcement-learning package. The fixed data
adapters therefore live behind their own explicit import path:

```bash
python -c "from pybacktest.adapters.data import ParquetDataSource"
```

`PandasDataSource` and `ParquetDataSource` are deliberately **not** top-level
exports; importing them is what pulls their extra into your process.

## An offline run, end to end

This example needs nothing but NumPy. The dataset, the instrument, the broker
factory, the risk policy, and the request split are all written out.

```python
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    BarSeries,
    CalendarPolicy,
    DateRange,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LongShortRisk,
    MarketDataSet,
    MetricsConfig,
    Money,
    MovingAverageCross,
    NextBarOpenFill,
    NoBorrowCost,
    NoSlippage,
    OrderStatus,
    PerShareCommission,
    Quantity,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
    VolumeParticipationLimit,
)

AAPL = InstrumentId.parse("XNAS:AAPL")

dataset = MarketDataSet(
    series={
        AAPL: BarSeries(
            timestamps=np.asarray(
                [
                    "2024-01-02T14:30:00",
                    "2024-01-03T14:30:00",
                    "2024-01-04T14:30:00",
                    "2024-01-05T14:30:00",
                    "2024-01-08T14:30:00",
                ],
                dtype="datetime64[ns]",
            ),
            open=np.asarray([100.0, 100.0, 98.0, 104.0, 106.0]),
            high=np.asarray([101.0, 101.0, 103.0, 105.0, 107.0]),
            low=np.asarray([99.0, 97.0, 97.0, 103.0, 105.0]),
            close=np.asarray([100.0, 98.0, 102.0, 104.0, 106.0]),
            volume=np.asarray([1000.0, 1000.0, 1000.0, 400.0, 2000.0]),
        )
    },
    instruments={
        AAPL: Instrument(
            id=AAPL,
            quote_currency="USD",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("1"),
            timezone=UTC,
        )
    },
    timeframe=Timeframe.days(1),
)


class InMemoryData:
    """A market data source that owns one already-validated dataset."""

    def __init__(self, dataset):
        self._dataset = dataset

    def load(self, universe, period, timeframe):
        return self._dataset


engine = BacktestEngine(
    data_source=InMemoryData(dataset),
    broker_factory=SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
        commission=PerShareCommission(rate_per_share=Decimal("0.005")),
        slippage=NoSlippage(),
        liquidity=VolumeParticipationLimit(max_volume_ratio=Decimal("0.05")),
        borrow_cost=NoBorrowCost(),
    ),
    risk_policy=LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    ),
)

simulation = SimulationRequest(
    universe=(AAPL,),
    period=DateRange(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 9, tzinfo=UTC),
    ),
    timeframe=Timeframe.days(1),
    calendar=CalendarPolicy.union(),
    initial_cash=Money.usd("10000"),
    seed=7,
    metrics=MetricsConfig(
        risk_free_rate=Decimal("0"),
        annualization_periods=252,
    ),
)

result = engine.run(
    BacktestRequest(
        strategy=MovingAverageCross(
            fast=1,
            slow=2,
            long_weight=Decimal("0.5"),
            flat_weight=Decimal("0"),
            instrument=AAPL,
        ),
        simulation=simulation,
    )
)

assert len(result.orders) == 1
assert result.orders[0].quantity == Quantity.of("49")
# The crossover closes at 102.00 on 2024-01-04, but nothing executes on that
# bar: the first fill is the 2024-01-05 open, capped at 5% of its 400 volume.
assert result.fills[0].timestamp == datetime(2024, 1, 5, 14, 30, tzinfo=UTC)
assert result.fills[0].quantity == Quantity.of("20")
assert result.fills[0].price == Money.usd("104.00")
# `DefaultOrderSizer` sizes a target as a DAY order, so the 29-share remainder
# expires at the next session boundary instead of resting. See "Orders, fills,
# and time in force" below for how to ask for GOOD_TIL_CANCELLED instead.
assert result.orders[0].status is OrderStatus.CANCELLED
assert result.snapshots[-1].cash == Money.usd("7919.900")
assert result.snapshots[-1].positions[AAPL].quantity == Quantity.of("20")
```

A request is deliberately split in two. `SimulationRequest` is the keyword-only
market configuration — universe, period, timeframe, calendar, initial cash,
seed, metrics. `BacktestRequest(strategy, simulation, run_id=None,
provenance=None)` binds a Python strategy to it. `BacktestEngine` takes a
`broker_factory`, never a broker instance: each session must own a broker no
other session has used, and a reused instance is rejected.

Constructors perform no network or filesystem I/O; data loading begins at `reset()` / `run()`.
`ParquetDataSource.__init__` only stores a path, and
`LocalArtifactStore.__init__` only normalizes one, so building objects is free
of side effects and every read happens inside a session you started.

## Optional Parquet ingestion

`ParquetDataSource` reads only the requested rows and delegates canonical
validation to the shared Pandas path. Its import is explicit because it is what
pulls the `parquet` extra into your process.

```python title="requires: pandas,pyarrow"
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    CalendarPolicy,
    DateRange,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LongShortRisk,
    MetricsConfig,
    Money,
    MovingAverageCross,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    Quantity,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
)
from pybacktest.adapters.data import ParquetDataSource

AAPL = InstrumentId.parse("XNAS:AAPL")

frame = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(
            [
                "2024-01-02T14:30:00",
                "2024-01-03T14:30:00",
                "2024-01-04T14:30:00",
                "2024-01-05T14:30:00",
                "2024-01-08T14:30:00",
            ],
            utc=True,
        ),
        "instrument": ["XNAS:AAPL"] * 5,
        "open": [100.0, 100.0, 98.0, 104.0, 106.0],
        "high": [101.0, 101.0, 103.0, 105.0, 107.0],
        "low": [99.0, 97.0, 97.0, 103.0, 105.0],
        "close": [100.0, 98.0, 102.0, 104.0, 106.0],
        "volume": [1000.0] * 5,
    }
)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "bars.parquet"
    frame.to_parquet(path, index=False)

    source = ParquetDataSource(
        path,
        instruments={
            AAPL: Instrument(
                id=AAPL,
                quote_currency="USD",
                tick_size=Decimal("0.01"),
                lot_size=Decimal("1"),
                timezone=UTC,
            )
        },
    )
    broker_factory = SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
        slippage=NoSlippage(),
        liquidity=NoLiquidityLimit(),
        borrow_cost=NoBorrowCost(),
    )
    risk_policy = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    )
    engine = BacktestEngine(
        data_source=source,
        broker_factory=broker_factory,
        risk_policy=risk_policy,
    )
    request = BacktestRequest(
        strategy=MovingAverageCross(
            fast=1,
            slow=2,
            long_weight=Decimal("0.5"),
            flat_weight=Decimal("0"),
            instrument=AAPL,
        ),
        simulation=SimulationRequest(
            universe=(AAPL,),
            period=DateRange(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 9, tzinfo=UTC),
            ),
            timeframe=Timeframe.days(1),
            calendar=CalendarPolicy.union(),
            initial_cash=Money.usd("10000"),
            seed=7,
            metrics=MetricsConfig(
                risk_free_rate=Decimal("0"),
                annualization_periods=252,
            ),
        ),
    )
    result = engine.run(request)

assert len(result.market_timestamps) == 5
assert len(result.orders) == 1
assert result.fills[0].price == Money.usd("104.00")
assert result.fills[0].quantity == Quantity.of("49")
assert result.snapshots[-1].cash == Money.usd("4904.00")
assert result.snapshots[-1].positions[AAPL].quantity == Quantity.of("49")
```

## Event timing

A strategy decides on the bar it can see, and the engine executes that decision
on the next bar. There is no same-bar execution and no lookahead.

```text
bar t                                   bar t+1
─────────────────────────────────────   ─────────────────────────────────
observation (OHLCV + pinned features)
        │
        ├─ strategy.on_bar()  ──────►  intent
        ├─ order_sizer.size() ──────►  proposed order
        ├─ risk_policy.evaluate() ──►  passed / adjusted / rejected
        └─ scheduled: active_from = t+1
                                        broker fills at the t+1 open
                                        ledger applies cash, position, fees
                                        recorder appends the causal events
```

The feature view handed to `on_bar` is pinned to bar `t`: a positive offset
raises `LookaheadViolation`, and rolling windows are NaN until enough history
exists. The five-bar run above is the reference: the crossover happens on
2024-01-04, the order becomes active on 2024-01-05, and the first fill is at
the 2024-01-05 open of 104.00 — never at the 102.00 close that produced it.

## Orders, fills, and time in force

- **Market orders** fill at the next eligible bar open.
- **Limit orders** fill at the open when it is already through the limit,
  otherwise at the limit price when the bar range touches it, under
  `IntrabarPolicy.CONSERVATIVE`.
- **`TimeInForce.DAY`** orders expire at the next session boundary. A partially
  filled DAY order is cancelled with its remainder unfilled.
- **`TimeInForce.GOOD_TIL_CANCELLED`** orders keep resting across boundaries
  until they fill or you send a `CancelOrderIntent`.

`DefaultOrderSizer` gives `TargetWeight` and `TargetQuantity` targets a DAY time
in force, and preserves the explicit time in force you set on
`MarketOrderIntent` and `LimitOrderIntent`. Pass your own sizer through
`BacktestEngine(order_sizer=...)` or `engine.run(request, order_sizer=...)` when
you want different execution instructions.

## Execution models

Every broker execution model is an explicit object with no hidden default.
Sizing and risk have typed engine defaults, and both remain replaceable.

| Concern | Models |
| --- | --- |
| Fill | `NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE)` |
| Commission | `NoCommission()`, `PerShareCommission(rate_per_share=..., minimum_fee=...)` |
| Slippage | `NoSlippage()`, `VolumeShareSlippage(impact_bps=...)` |
| Liquidity | `NoLiquidityLimit()`, `VolumeParticipationLimit(max_volume_ratio=...)` |
| Borrow | `NoBorrowCost()` |
| Sizing | `DefaultOrderSizer()` (engine default) |
| Risk | unrestricted policy (engine default), `LongShortRisk(max_leverage=..., max_position_weight=..., allow_short=...)` |

`VolumeParticipationLimit` caps each bar's fill to a ratio of that bar's volume,
which is what turns one 49-share order into a 20-share fill on a 400-share bar
and a 29-share fill on the next 2000-share bar. The ledger applies every fill in
exact `Decimal` arithmetic and reconciles cash, positions, realized and
unrealized P&L, and fees on every bar; a transition that cannot be reconciled
raises `AccountingInvariantError` instead of drifting.

## Driving a session yourself

`BacktestEngine.run()` is a thin loop over the public session API, so you can
step the simulation yourself — one bar per `advance()` call. `observation` is
keyword-only and identity-checked against the observation the session handed
you, which is what makes an external agent unable to act on stale state.

```python
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from pybacktest import (
    BacktestEngine,
    BarSeries,
    CalendarPolicy,
    DateRange,
    FeatureBuilder,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LongShortRisk,
    MarketDataSet,
    MetricsConfig,
    Money,
    MovingAverageCross,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
)

MSFT = InstrumentId.parse("XNAS:MSFT")
closes = np.asarray([100.0, 98.0, 102.0, 104.0, 106.0])

dataset = MarketDataSet(
    series={
        MSFT: BarSeries(
            timestamps=np.asarray(
                [
                    "2024-01-02T14:30:00",
                    "2024-01-03T14:30:00",
                    "2024-01-04T14:30:00",
                    "2024-01-05T14:30:00",
                    "2024-01-08T14:30:00",
                ],
                dtype="datetime64[ns]",
            ),
            open=closes,
            high=closes,
            low=closes,
            close=closes,
            volume=np.full(5, 1_000_000.0),
        )
    },
    instruments={
        MSFT: Instrument(
            id=MSFT,
            quote_currency="USD",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("1"),
            timezone=UTC,
        )
    },
    timeframe=Timeframe.days(1),
)


class InMemoryData:
    """A market data source that owns one already-validated dataset."""

    def __init__(self, dataset):
        self._dataset = dataset

    def load(self, universe, period, timeframe):
        return self._dataset


engine = BacktestEngine(
    data_source=InMemoryData(dataset),
    broker_factory=SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
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

strategy = MovingAverageCross(
    fast=1,
    slow=2,
    long_weight=Decimal("1"),
    flat_weight=Decimal("0"),
    instrument=MSFT,
)
feature_plan = strategy.build_features(FeatureBuilder())

session = engine.create_session(
    SimulationRequest(
        universe=(MSFT,),
        period=DateRange(
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 9, tzinfo=UTC),
        ),
        timeframe=Timeframe.days(1),
        calendar=CalendarPolicy.union(),
        initial_cash=Money.usd("10000"),
        seed=7,
        metrics=MetricsConfig(
            risk_free_rate=Decimal("0"),
            annualization_periods=252,
        ),
    ),
    feature_plan=feature_plan,
)

observation = session.reset()
while not session.done:
    context = session.strategy_context(observation)
    intents = strategy.on_bar(context, observation.market)
    step = session.advance(intents, observation=observation)
    if step.observation is not None:
        observation = step.observation

result = session.result()
assert len(result.market_timestamps) == 5
assert result.manifest.spec_identity == "external.actions"
```

A session created this way records **external-action** provenance, because the
engine did not compile the decisions it consumed. Pass
`create_session(..., provenance=...)` when you can state what produced them.

These call shapes are rejected on purpose:

```python title="not-executed: every statement below fails by design"
# The engine takes a factory; a reusable broker instance is rejected.
BacktestEngine(data_source=source, broker_factory=broker_instance)
# `observation` is keyword-only and identity-checked against the session.
session.advance(intents)
session.advance(intents, observation=an_older_observation)
# A strategy must be bound to an instrument before it can declare features.
MovingAverageCross(fast=20, slow=60).build_features(FeatureBuilder())
```

## Artifacts and causal explanations

`result.explain_trade(order_id)` returns the exact recorded events for one
order, in recorder order, so you can answer "why did this trade happen" without
re-running anything. `LocalArtifactStore.write()` publishes a result atomically
with a checksum manifest. Artifact publication writes Parquet payloads and
therefore requires PyArrow; install it with `uv add "pybacktest[parquet]"`.

```python title="requires: pyarrow"
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    BarSeries,
    CalendarPolicy,
    DateRange,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LocalArtifactStore,
    LongShortRisk,
    MarketDataSet,
    MetricsConfig,
    Money,
    MovingAverageCross,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
)

NVDA = InstrumentId.parse("XNAS:NVDA")
closes = np.asarray([100.0, 98.0, 102.0, 104.0, 106.0])

dataset = MarketDataSet(
    series={
        NVDA: BarSeries(
            timestamps=np.asarray(
                [
                    "2024-01-02T14:30:00",
                    "2024-01-03T14:30:00",
                    "2024-01-04T14:30:00",
                    "2024-01-05T14:30:00",
                    "2024-01-08T14:30:00",
                ],
                dtype="datetime64[ns]",
            ),
            open=closes,
            high=closes,
            low=closes,
            close=closes,
            volume=np.full(5, 1_000_000.0),
        )
    },
    instruments={
        NVDA: Instrument(
            id=NVDA,
            quote_currency="USD",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("1"),
            timezone=UTC,
        )
    },
    timeframe=Timeframe.days(1),
)


class InMemoryData:
    """A market data source that owns one already-validated dataset."""

    def __init__(self, dataset):
        self._dataset = dataset

    def load(self, universe, period, timeframe):
        return self._dataset


engine = BacktestEngine(
    data_source=InMemoryData(dataset),
    broker_factory=SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
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

result = engine.run(
    BacktestRequest(
        strategy=MovingAverageCross(
            fast=1,
            slow=2,
            long_weight=Decimal("1"),
            flat_weight=Decimal("0"),
            instrument=NVDA,
        ),
        simulation=SimulationRequest(
            universe=(NVDA,),
            period=DateRange(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 9, tzinfo=UTC),
            ),
            timeframe=Timeframe.days(1),
            calendar=CalendarPolicy.union(),
            initial_cash=Money.usd("10000"),
            seed=7,
            metrics=MetricsConfig(
                risk_free_rate=Decimal("0"),
                annualization_periods=252,
            ),
        ),
    )
)

explanation = result.explain_trade(result.orders[0].id)
codes = [entry.code.value for entry in explanation.entries]
assert codes[:5] == [
    "intent.received",
    "order.sized",
    "risk.passed",
    "order.scheduled",
    "order.accepted",
]

with tempfile.TemporaryDirectory() as directory:
    # The store refuses a root reached through a symlink, and macOS temporary
    # directories live under one, so resolve the path before handing it over.
    reference = LocalArtifactStore(Path(directory).resolve()).write(result)
    assert reference.manifest.replay_fingerprint == result.replay_fingerprint()
    assert [item.name for item in reference.files]
```

## Determinism, replay, and provenance

`result.replay_fingerprint()` hashes all observable behavior — manifest,
summary, timestamps, snapshots, orders, fills, events, warnings — after
normalizing run-scoped identifiers, so two identical runs agree even though
their `RunId`s differ. `result.manifest` records the library and schema
versions, the canonical request, the dataset fingerprint, the seed, and the
adapter and model identities that produced the run.

The engine derives Python-strategy provenance by hashing the strategy class
source, its distributing package, and its configuration. When it cannot read a
trustworthy implementation — a generated class, a compiled `StrategySpec`, an
MCP-produced strategy — it fails closed instead of guessing, and you supply the
descriptor yourself. That explicit escape hatch is public API:

```python
import hashlib
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from pybacktest import (
    BacktestEngine,
    BacktestRequest,
    BarSeries,
    CalendarPolicy,
    DateRange,
    DecisionReason,
    FeatureBuilder,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LongShortRisk,
    MarketDataSet,
    MarketOrderIntent,
    MetricsConfig,
    Money,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    OrderSide,
    ProvenanceDescriptor,
    Quantity,
    SimulatedBrokerFactory,
    SimulationRequest,
    Timeframe,
    TimeInForce,
)

AMD = InstrumentId.parse("XNAS:AMD")
closes = np.asarray([100.0, 101.0, 102.0])


class BuyOnceThenHold:
    """A generated strategy standing in for a compiled StrategySpec."""

    def build_features(self, builder: FeatureBuilder):
        builder.source("close", AMD, "close")
        return builder.plan()

    def on_bar(self, context, market):
        if context.portfolio.positions or context.active_orders:
            return ()
        return (
            MarketOrderIntent(
                instrument=AMD,
                side=OrderSide.BUY,
                quantity=Quantity.of("1"),
                time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
                reason=DecisionReason.of("buy_once"),
            ),
        )


dataset = MarketDataSet(
    series={
        AMD: BarSeries(
            timestamps=np.asarray(
                [
                    "2024-01-02T14:30:00",
                    "2024-01-03T14:30:00",
                    "2024-01-04T14:30:00",
                ],
                dtype="datetime64[ns]",
            ),
            open=closes,
            high=closes,
            low=closes,
            close=closes,
            volume=np.full(3, 1_000_000.0),
        )
    },
    instruments={
        AMD: Instrument(
            id=AMD,
            quote_currency="USD",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("1"),
            timezone=UTC,
        )
    },
    timeframe=Timeframe.days(1),
)


class InMemoryData:
    """A market data source that owns one already-validated dataset."""

    def __init__(self, dataset):
        self._dataset = dataset

    def load(self, universe, period, timeframe):
        return self._dataset


engine = BacktestEngine(
    data_source=InMemoryData(dataset),
    broker_factory=SimulatedBrokerFactory(
        fill_model=NextBarOpenFill(intrabar_policy=IntrabarPolicy.CONSERVATIVE),
        commission=NoCommission(),
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

result = engine.run(
    BacktestRequest(
        strategy=BuyOnceThenHold(),
        simulation=SimulationRequest(
            universe=(AMD,),
            period=DateRange(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 5, tzinfo=UTC),
            ),
            timeframe=Timeframe.days(1),
            calendar=CalendarPolicy.union(),
            initial_cash=Money.usd("10000"),
            seed=7,
            metrics=MetricsConfig(
                risk_free_rate=Decimal("0"),
                annualization_periods=252,
            ),
        ),
        provenance=ProvenanceDescriptor(
            strategy_identity="docs.readme.BuyOnceThenHold",
            strategy_fingerprint=hashlib.sha256(
                b"docs.readme.BuyOnceThenHold@v1"
            ).hexdigest(),
            spec_identity="docs.readme.spec.v1",
            compiler_identity="docs.readme.compiler.v1",
        ),
    )
)

assert result.manifest.strategy_identity == "docs.readme.BuyOnceThenHold"
assert result.manifest.compiler_identity == "docs.readme.compiler.v1"
```

## Roadmap

Pybacktest 0.2 core is complete. This roadmap is directional: it orders the
work that is currently justified, but it does not promise release dates. Every
extension remains a separate package that consumes the public typed API; no
MCP, model, training, UI, or adapter dependency enters the core.

### Next — Core readiness

- [ ] Automate the supported Python matrix, core and optional-dependency tests,
  Ruff, ty, executable README examples, and package-build checks in CI. Keep the
  reference performance gate on a controlled manual or scheduled runner.
- [ ] Remove inherited repository-wide formatting debt in a formatting-only
  change, then enforce `ruff format --check` in CI.
- [ ] Restore the intended dependency direction by moving engine-consumed
  request and provenance contracts to a neutral layer with compatibility
  re-exports.
- [ ] Add reusable contract suites for data, broker, and artifact adapters.

This stage is complete when every quality gate runs automatically and a new
adapter can prove compatibility without copying implementation-specific tests.

### Next project — StrategySpec and MCP

- [ ] Revise the existing [StrategySpec plan](docs/superpowers/plans/2026-07-29-pybacktest-v2-strategy-spec.md)
  and [MCP plan](docs/superpowers/plans/2026-07-29-pybacktest-v2-mcp.md) so they
  produce separate consumer packages instead of modules inside the core.
- [ ] Build StrategySpec first: versioned strict schemas, an allowlisted
  component registry, semantic validation, an `eval`-free compiler, parity with
  Python strategies, and risk-policy intersection.
- [ ] After StrategySpec is stable, build a bounded local-stdio MCP server with
  allowlisted datasets and components, request/run/artifact quotas, sanitized
  failures, and no arbitrary code, URL, or filesystem-path execution.

This stage is complete when both packages are independently installable, depend
only on the public Pybacktest contract, and pass parity, quota, and security
tests.

### Later — Ecosystem and operations

- [ ] Add standalone market-data adapters, including a real yfinance adapter.
- [ ] Add standalone plotting/reporting and artifact-storage adapters; the
  current `yfinance` and `plot` extras install dependencies but do not provide
  these integrations.
- [ ] Automate package publication, release notes, and compatibility checks for
  the core and extension packages.

Each item is complete only with a reusable contract test and an end-to-end
example. Paper/live brokers, remote multi-user MCP hosting, and distributed
execution remain unscheduled rather than implied commitments.

### Research — AI training and reinforcement learning

- [ ] Record reproducible experiment lineage across StrategySpec, validation
  feedback, dataset/configuration fingerprints, metrics, artifacts, and
  parent-child runs; export training trajectories without hidden model
  reasoning.
- [ ] Wrap `SimulationSession` in a separate Gym-style package with explicit
  observation, action, and reward contracts and parity with engine results.
- [ ] Explore walk-forward, parameter-sweep, and Monte Carlo workflows after the
  preceding contracts are stable.

Automatic FX accounting, derivatives, tick/order-book simulation, and other
explicitly excluded domains stay outside the committed roadmap until they have
separate approved designs.

## Performance

`benchmarks/` holds a marked gate for 100 instruments × 10,000 bars. It is
outside the default `testpaths`, so run it explicitly:

```bash
uv run python -m pytest -m performance benchmarks/test_engine_benchmark.py -q
```

See [`benchmarks/README.md`](benchmarks/README.md) for the measured runtime,
peak RSS, hardware, and the regression policy.
