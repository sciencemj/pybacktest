"""The curated top-level namespace and its reference strategy."""

import sys
import threading
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

import pybacktest
from pybacktest import (
    AccountingInvariantError,
    AdapterContractError,
    ArtifactDurabilityError,
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    BacktestEngine,
    BacktestRequest,
    BacktestResult,
    BacktestService,
    BarSeries,
    CalendarMode,
    CalendarPolicy,
    CancelOrderIntent,
    CausalStage,
    ClockRegressionError,
    ComponentDescriptor,
    ConfigurationError,
    DataValidationError,
    DateRange,
    DecisionReason,
    DefaultOrderSizer,
    DeterministicComponent,
    EngineEvent,
    EngineEventCode,
    FeatureBuilder,
    FeaturePlan,
    FeatureView,
    FillId,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LimitOrderIntent,
    LocalArtifactStore,
    LongShortRisk,
    LookaheadViolation,
    MarketDataSet,
    MarketOrderIntent,
    MetricMetadata,
    MetricName,
    MetricResult,
    MetricsConfig,
    MissingPolicy,
    Money,
    MovingAverageCross,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    Observation,
    OrderId,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    PerShareCommission,
    ProvenanceDescriptor,
    PybacktestError,
    Quantity,
    ResultValidationError,
    RiskStatus,
    RunId,
    RunManifest,
    RunWarning,
    SerializationError,
    SimulatedBroker,
    SimulatedBrokerFactory,
    SimulationRequest,
    SimulationSession,
    StepResult,
    Strategy,
    StrategyContext,
    SummaryMetrics,
    TargetQuantity,
    TargetWeight,
    Timeframe,
    TimeframeUnit,
    TimeInForce,
    TradeExplanation,
    UnknownTradeError,
    VolumeParticipationLimit,
    VolumeShareSlippage,
    WarningCode,
    external_action_provenance,
    python_strategy_provenance,
)
from pybacktest.data.features import FeatureSet
from pybacktest.ports.strategy import PortfolioSnapshot
from tests.factories import market_slice

AAPL = InstrumentId.parse("XNAS:AAPL")
MSFT = InstrumentId.parse("XNAS:MSFT")

EXPECTED_PUBLIC_API = (
    "AccountingInvariantError",
    "AdapterContractError",
    "ArtifactDurabilityError",
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactRef",
    "BacktestEngine",
    "BacktestRequest",
    "BacktestResult",
    "BacktestService",
    "BarSeries",
    "CalendarMode",
    "CalendarPolicy",
    "CancelOrderIntent",
    "CausalStage",
    "ClockRegressionError",
    "ComponentDescriptor",
    "ConfigurationError",
    "DataValidationError",
    "DateRange",
    "DecisionReason",
    "DefaultOrderSizer",
    "DeterministicComponent",
    "EngineEvent",
    "EngineEventCode",
    "FeatureBuilder",
    "FeaturePlan",
    "FeatureView",
    "FillId",
    "Instrument",
    "InstrumentId",
    "IntrabarPolicy",
    "LimitOrderIntent",
    "LocalArtifactStore",
    "LongShortRisk",
    "LookaheadViolation",
    "MarketDataSet",
    "MarketOrderIntent",
    "MetricMetadata",
    "MetricName",
    "MetricResult",
    "MetricsConfig",
    "MissingPolicy",
    "Money",
    "MovingAverageCross",
    "NextBarOpenFill",
    "NoBorrowCost",
    "NoCommission",
    "NoLiquidityLimit",
    "NoSlippage",
    "Observation",
    "OrderId",
    "OrderIntent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PerShareCommission",
    "ProvenanceDescriptor",
    "PybacktestError",
    "Quantity",
    "ResultValidationError",
    "RiskStatus",
    "RunId",
    "RunManifest",
    "RunWarning",
    "SerializationError",
    "SimulatedBroker",
    "SimulatedBrokerFactory",
    "SimulationRequest",
    "SimulationSession",
    "StepResult",
    "Strategy",
    "StrategyContext",
    "SummaryMetrics",
    "TargetQuantity",
    "TargetWeight",
    "TimeInForce",
    "Timeframe",
    "TimeframeUnit",
    "TradeExplanation",
    "UnknownTradeError",
    "VolumeParticipationLimit",
    "VolumeShareSlippage",
    "WarningCode",
    "external_action_provenance",
    "python_strategy_provenance",
)

_IMPORTED_NAMES = (
    AccountingInvariantError,
    AdapterContractError,
    ArtifactDurabilityError,
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    BacktestEngine,
    BacktestRequest,
    BacktestResult,
    BacktestService,
    BarSeries,
    CalendarMode,
    CalendarPolicy,
    CancelOrderIntent,
    CausalStage,
    ClockRegressionError,
    ComponentDescriptor,
    ConfigurationError,
    DataValidationError,
    DateRange,
    DecisionReason,
    DefaultOrderSizer,
    DeterministicComponent,
    EngineEvent,
    EngineEventCode,
    FeatureBuilder,
    FeaturePlan,
    FeatureView,
    FillId,
    Instrument,
    InstrumentId,
    IntrabarPolicy,
    LimitOrderIntent,
    LocalArtifactStore,
    LongShortRisk,
    LookaheadViolation,
    MarketDataSet,
    MarketOrderIntent,
    MetricMetadata,
    MetricName,
    MetricResult,
    MetricsConfig,
    MissingPolicy,
    Money,
    MovingAverageCross,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    Observation,
    OrderId,
    OrderIntent,
    OrderSide,
    OrderStatus,
    OrderType,
    PerShareCommission,
    ProvenanceDescriptor,
    PybacktestError,
    Quantity,
    ResultValidationError,
    RiskStatus,
    RunId,
    RunManifest,
    RunWarning,
    SerializationError,
    SimulatedBroker,
    SimulatedBrokerFactory,
    SimulationRequest,
    SimulationSession,
    StepResult,
    Strategy,
    StrategyContext,
    SummaryMetrics,
    TargetQuantity,
    TargetWeight,
    TimeInForce,
    Timeframe,
    TimeframeUnit,
    TradeExplanation,
    UnknownTradeError,
    VolumeParticipationLimit,
    VolumeShareSlippage,
    WarningCode,
    external_action_provenance,
    python_strategy_provenance,
)


def test_curated_public_api_is_importable() -> None:
    assert InstrumentId.parse("XNAS:AAPL").symbol == "AAPL"
    assert MovingAverageCross(fast=20, slow=60).fast == 20
    assert IntrabarPolicy.CONSERVATIVE.value == "conservative"


def test_public_all_is_explicit_sorted_and_complete() -> None:
    assert pybacktest.__all__ == list(EXPECTED_PUBLIC_API)
    assert pybacktest.__all__ == sorted(pybacktest.__all__)
    assert len(set(pybacktest.__all__)) == len(pybacktest.__all__)
    assert len(_IMPORTED_NAMES) == len(EXPECTED_PUBLIC_API)


def test_every_exported_name_is_bound_on_the_package() -> None:
    for name in pybacktest.__all__:
        assert hasattr(pybacktest, name), name


def test_data_adapters_are_not_top_level_exports() -> None:
    # The fresh-subprocess proof that importing the package loads no optional
    # dependency lives in tests/test_package.py; this pins the export surface.
    assert "pybacktest" in sys.modules
    assert "PandasDataSource" not in pybacktest.__all__
    assert "ParquetDataSource" not in pybacktest.__all__
    assert not hasattr(pybacktest, "PandasDataSource")
    assert not hasattr(pybacktest, "ParquetDataSource")


# --- MovingAverageCross: configuration and declared causal plan -------------


@pytest.mark.parametrize(
    "fast,slow",
    [
        (0, 2),
        (-1, 2),
        (2, 2),
        (3, 2),
        (True, 2),
        (1, False),
        (1.0, 2),
        (1, 2.0),
        ("1", 2),
    ],
)
def test_windows_must_be_ordered_positive_non_bool_integers(
    fast: object,
    slow: object,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        MovingAverageCross(fast=fast, slow=slow, instrument=AAPL)

    assert raised.value.code == "invalid_moving_average_windows"


@pytest.mark.parametrize(
    "long_weight,flat_weight",
    [
        (float("nan"), 0),
        (float("inf"), 0),
        (1, float("-inf")),
        (Decimal("NaN"), 0),
        ("wide", 0),
        (1, None),
    ],
)
def test_weights_must_be_finite_and_representable(
    long_weight: object,
    flat_weight: object,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        MovingAverageCross(
            fast=1,
            slow=2,
            long_weight=long_weight,
            flat_weight=flat_weight,
            instrument=AAPL,
        )

    assert raised.value.code == "invalid_moving_average_weight"


def test_weights_are_normalized_to_exact_decimals() -> None:
    strategy = MovingAverageCross(
        fast=1,
        slow=2,
        long_weight=0.5,
        flat_weight="0.00",
        instrument=AAPL,
    )

    assert strategy.long_weight == Decimal("0.5")
    assert strategy.flat_weight == Decimal("0.00")
    assert isinstance(strategy.long_weight, Decimal)
    assert isinstance(strategy.flat_weight, Decimal)


def test_documented_defaults_keep_the_import_only_constructor_valid() -> None:
    strategy = MovingAverageCross(fast=20, slow=60)

    assert strategy.long_weight == Decimal("1")
    assert strategy.flat_weight == Decimal("0")
    assert strategy.instrument is None


def test_instrument_binding_must_be_an_instrument_id() -> None:
    with pytest.raises(ConfigurationError) as raised:
        MovingAverageCross(fast=1, slow=2, instrument="XNAS:AAPL")

    assert raised.value.code == "invalid_strategy_instrument"


def test_build_features_fails_closed_without_a_bound_instrument() -> None:
    with pytest.raises(ConfigurationError) as raised:
        MovingAverageCross(fast=20, slow=60).build_features(FeatureBuilder())

    assert raised.value.code == "unbound_strategy_instrument"


def test_build_features_declares_the_exact_causal_plan() -> None:
    plan = MovingAverageCross(fast=2, slow=5, instrument=AAPL).build_features(
        FeatureBuilder()
    )

    assert isinstance(plan, FeaturePlan)
    assert tuple(
        (
            node.name,
            node.operator,
            node.inputs,
            node.instrument,
            node.field,
            node.parameter,
            node.lookback,
        )
        for node in plan.nodes
    ) == (
        ("close", "source", (), AAPL, "close", None, 0),
        ("fast", "sma", ("close",), None, None, 2, 1),
        ("slow", "sma", ("close",), None, None, 5, 4),
        ("fast_prev", "lag", ("fast",), None, None, 1, 2),
        ("slow_prev", "lag", ("slow",), None, None, 1, 5),
    )


def test_the_strategy_object_is_frozen_and_slotted() -> None:
    strategy = MovingAverageCross(fast=1, slow=2, instrument=AAPL)

    assert not hasattr(strategy, "__dict__")
    with pytest.raises(FrozenInstanceError):
        strategy.fast = 3  # type: ignore[misc]


# --- MovingAverageCross: signal semantics ----------------------------------


def _pinned_context(
    columns: dict[str, list[float]],
    index: int,
) -> StrategyContext:
    """Pin a strategy context onto an explicit lagged feature history."""
    length = len(next(iter(columns.values())))
    timestamps = np.asarray(
        [
            np.datetime64("2024-01-02T14:30:00", "ns") + np.timedelta64(step, "D")
            for step in range(length)
        ],
        dtype="datetime64[ns]",
    )
    features = FeatureSet(
        timestamps=timestamps,
        columns={
            name: np.asarray(values, dtype=np.float64)
            for name, values in columns.items()
        },
    )
    return StrategyContext(
        timestamps[index],
        PortfolioSnapshot(positions={}),
        (),
        features.view(timestamps[index]),
        RunId.parse("run_" + "1" * 32),
    )


def _decide(
    strategy: MovingAverageCross,
    columns: dict[str, list[float]],
    index: int,
) -> tuple[OrderIntent, ...]:
    return tuple(
        strategy.on_bar(
            _pinned_context(columns, index),
            market_slice("2024-01-02T14:30:00Z"),
        )
    )


_NAN = float("nan")
_EQUALITY_HISTORY = {
    "fast": [98.0, 100.0, 102.0, 100.0, 98.0],
    "slow": [99.0, 100.0, 100.0, 100.0, 100.0],
    "fast_prev": [_NAN, 98.0, 100.0, 102.0, 100.0],
    "slow_prev": [_NAN, 99.0, 100.0, 100.0, 100.0],
}
_TRENDING_HISTORY = {
    "fast": [102.0, 104.0, 106.0],
    "slow": [100.0, 100.0, 100.0],
    "fast_prev": [98.0, 102.0, 104.0],
    "slow_prev": [99.0, 100.0, 100.0],
}


@pytest.mark.parametrize("missing", ["fast", "slow", "fast_prev", "slow_prev"])
def test_nan_warm_up_emits_no_signal(missing: str) -> None:
    columns = {
        "fast": [102.0],
        "slow": [100.0],
        "fast_prev": [98.0],
        "slow_prev": [99.0],
    }
    columns[missing] = [_NAN]

    assert _decide(_strategy(), columns, 0) == ()


def _strategy() -> MovingAverageCross:
    return MovingAverageCross(
        fast=1,
        slow=2,
        long_weight=Decimal("0.75"),
        flat_weight=Decimal("0"),
        instrument=AAPL,
    )


def test_true_cross_up_emits_the_long_weight_and_a_finite_reason() -> None:
    (intent,) = _decide(_strategy(), _TRENDING_HISTORY, 0)

    assert isinstance(intent, TargetWeight)
    assert intent.instrument == AAPL
    assert intent.weight == Decimal("0.75")
    assert intent.reason.code == "ma_cross"
    assert dict(intent.reason.details) == {
        "fast": 102.0,
        "slow": 100.0,
        "fast_prev": 98.0,
        "slow_prev": 99.0,
        "outcome": "cross_up",
    }
    assert all(
        type(value) is float
        for key, value in intent.reason.details.items()
        if key != "outcome"
    )


def test_repeated_bars_on_one_side_emit_nothing() -> None:
    strategy = _strategy()

    assert len(_decide(strategy, _TRENDING_HISTORY, 0)) == 1
    assert _decide(strategy, _TRENDING_HISTORY, 1) == ()
    assert _decide(strategy, _TRENDING_HISTORY, 2) == ()


def test_current_equality_emits_nothing() -> None:
    assert _decide(_strategy(), _EQUALITY_HISTORY, 1) == ()
    assert _decide(_strategy(), _EQUALITY_HISTORY, 3) == ()


def test_previous_equality_is_a_neutral_pivot_in_both_directions() -> None:
    (up,) = _decide(_strategy(), _EQUALITY_HISTORY, 2)
    (down,) = _decide(_strategy(), _EQUALITY_HISTORY, 4)

    assert isinstance(up, TargetWeight)
    assert up.weight == Decimal("0.75")
    assert up.reason.details["outcome"] == "cross_up"
    assert isinstance(down, TargetWeight)
    assert down.weight == Decimal("0")
    assert down.reason.details["outcome"] == "cross_down"


def test_on_bar_fails_closed_without_a_bound_instrument() -> None:
    unbound = MovingAverageCross(fast=1, slow=2)

    with pytest.raises(ConfigurationError) as raised:
        _decide(unbound, _TRENDING_HISTORY, 0)

    assert raised.value.code == "unbound_strategy_instrument"


# --- MovingAverageCross: statelessness under reuse -------------------------


class _FixedSource:
    """Serve one already-constructed dataset without any I/O."""

    def __init__(self, dataset: MarketDataSet) -> None:
        self._dataset = dataset

    def load(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        del universe, period, timeframe
        return self._dataset


def _instrument(instrument_id: InstrumentId) -> Instrument:
    return Instrument(
        id=instrument_id,
        quote_currency="USD",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("1"),
        timezone=UTC,
    )


def _flat_dataset(
    instrument_id: InstrumentId,
    closes: list[float],
) -> MarketDataSet:
    prices = np.asarray(closes, dtype=np.float64)
    timestamps = np.asarray(
        [
            np.datetime64("2024-01-02T14:30:00", "ns") + np.timedelta64(step, "D")
            for step in range(len(closes))
        ],
        dtype="datetime64[ns]",
    )
    return MarketDataSet(
        series={
            instrument_id: BarSeries(
                timestamps=timestamps,
                open=prices,
                high=prices,
                low=prices,
                close=prices,
                volume=np.full(len(closes), 1_000_000.0),
            )
        },
        instruments={instrument_id: _instrument(instrument_id)},
        timeframe=Timeframe.days(1),
    )


def _run(strategy: MovingAverageCross, dataset: MarketDataSet) -> BacktestResult:
    engine = BacktestEngine(
        data_source=_FixedSource(dataset),
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
    return engine.run(
        BacktestRequest(
            strategy=strategy,
            simulation=SimulationRequest(
                universe=tuple(dataset.instruments),
                period=DateRange(
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 3, 1, tzinfo=UTC),
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


_FIRST_CLOSES = [100.0, 98.0, 102.0, 104.0, 101.0, 99.0]
_SECOND_CLOSES = [100.0, 101.0, 99.0, 97.0, 103.0, 105.0]


def test_a_reused_strategy_object_leaks_no_crossover_state() -> None:
    shared = MovingAverageCross(
        fast=1,
        slow=2,
        long_weight=Decimal("1"),
        flat_weight=Decimal("0"),
        instrument=AAPL,
    )
    first = _flat_dataset(AAPL, _FIRST_CLOSES)
    second = _flat_dataset(AAPL, _SECOND_CLOSES)

    _run(shared, first)
    reused = _run(shared, second)
    fresh = _run(
        MovingAverageCross(
            fast=1,
            slow=2,
            long_weight=Decimal("1"),
            flat_weight=Decimal("0"),
            instrument=AAPL,
        ),
        second,
    )

    assert len(reused.fills) > 0
    assert reused.replay_fingerprint() == fresh.replay_fingerprint()


def test_concurrent_runs_can_share_one_strategy_object() -> None:
    shared = MovingAverageCross(
        fast=1,
        slow=2,
        long_weight=Decimal("1"),
        flat_weight=Decimal("0"),
        instrument=MSFT,
    )
    dataset = _flat_dataset(MSFT, _FIRST_CLOSES)
    expected = _run(shared, dataset).replay_fingerprint()
    observed: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        fingerprint = _run(shared, _flat_dataset(MSFT, _FIRST_CLOSES))
        with lock:
            observed.append(fingerprint.replay_fingerprint())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert observed == [expected] * 4
