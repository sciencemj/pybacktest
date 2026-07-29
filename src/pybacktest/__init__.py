"""Deterministic, explicit backtesting primitives and engine.

Every name below is reachable with the core dependencies alone. Importing
this package never imports pandas, pyarrow, yfinance, matplotlib, Streamlit,
an MCP package, or a reinforcement-learning package. The fixed data adapters
therefore stay behind their own explicit import path::

    from pybacktest.adapters.data import ParquetDataSource  # needs [parquet]
    from pybacktest.adapters.data import PandasDataSource   # needs [data]
"""

from pybacktest.adapters.artifacts import LocalArtifactStore
from pybacktest.adapters.broker import (
    IntrabarPolicy,
    NextBarOpenFill,
    NoBorrowCost,
    NoCommission,
    NoLiquidityLimit,
    NoSlippage,
    PerShareCommission,
    SimulatedBroker,
    SimulatedBrokerFactory,
    VolumeParticipationLimit,
    VolumeShareSlippage,
)
from pybacktest.application.provenance import (
    ProvenanceDescriptor,
    external_action_provenance,
    python_strategy_provenance,
)
from pybacktest.application.requests import BacktestRequest, SimulationRequest
from pybacktest.application.service import BacktestService
from pybacktest.data.calendar import CalendarMode, CalendarPolicy
from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.data.features import FeatureBuilder, FeaturePlan, FeatureView
from pybacktest.domain.errors import (
    AccountingInvariantError,
    AdapterContractError,
    ClockRegressionError,
    ConfigurationError,
    DataValidationError,
    LookaheadViolation,
    PybacktestError,
)
from pybacktest.domain.identifiers import FillId, OrderId, RunId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from pybacktest.domain.time import DateRange, Timeframe, TimeframeUnit
from pybacktest.engine.engine import BacktestEngine
from pybacktest.engine.session import Observation, SimulationSession, StepResult
from pybacktest.ports.artifacts import ArtifactDurabilityError
from pybacktest.ports.components import (
    ComponentDescriptor,
    DeterministicComponent,
)
from pybacktest.ports.risk import RiskStatus
from pybacktest.ports.strategy import Strategy, StrategyContext
from pybacktest.results.metrics import MetricsConfig
from pybacktest.results.models import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    BacktestResult,
    CausalStage,
    EngineEvent,
    EngineEventCode,
    MetricMetadata,
    MetricName,
    MetricResult,
    MissingPolicy,
    ResultValidationError,
    RunManifest,
    RunWarning,
    SummaryMetrics,
    TradeExplanation,
    UnknownTradeError,
    WarningCode,
)
from pybacktest.results.serialization import SerializationError
from pybacktest.risk.policies import LongShortRisk
from pybacktest.risk.sizing import DefaultOrderSizer
from pybacktest.strategy.components import MovingAverageCross
from pybacktest.strategy.intents import (
    CancelOrderIntent,
    DecisionReason,
    LimitOrderIntent,
    MarketOrderIntent,
    OrderIntent,
    TargetQuantity,
    TargetWeight,
)

__version__ = "0.2.0"

__all__ = [
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
]
