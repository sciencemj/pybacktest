"""Validated immutable requests for deterministic simulations."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from pybacktest.application.provenance import ProvenanceDescriptor
from pybacktest.data.calendar import CalendarPolicy
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.money import Money
from pybacktest.domain.time import DateRange, Timeframe, TimeframeUnit
from pybacktest.ports.strategy import Strategy
from pybacktest.results.metrics import MetricsConfig

_MAX_SEED = 2**63 - 1


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """Complete typed configuration for one fixed-data simulation."""

    universe: tuple[InstrumentId, ...]
    period: DateRange
    timeframe: Timeframe
    calendar: CalendarPolicy
    initial_cash: Money
    seed: int
    metrics: MetricsConfig

    def __init__(
        self,
        *,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
        calendar: CalendarPolicy,
        initial_cash: Money,
        seed: int,
        metrics: MetricsConfig,
    ) -> None:
        if isinstance(universe, (str, bytes, bytearray)) or not isinstance(
            universe, Sequence
        ):
            raise ConfigurationError(
                "universe must be a sequence of InstrumentId values."
            )
        copied_universe = tuple(universe)
        if not copied_universe:
            raise ConfigurationError(
                "universe must contain at least one instrument.",
                code="empty_universe",
            )
        if not all(
            isinstance(instrument_id, InstrumentId)
            for instrument_id in copied_universe
        ):
            raise ConfigurationError(
                "universe must contain only InstrumentId values."
            )
        if len(set(copied_universe)) != len(copied_universe):
            raise ConfigurationError("universe instruments must be unique.")
        if not isinstance(period, DateRange):
            raise ConfigurationError("period must be a DateRange.")
        if not isinstance(timeframe, Timeframe):
            raise ConfigurationError("timeframe must be a Timeframe.")
        minimum_span = timedelta(
            minutes=timeframe.count
            if timeframe.unit is TimeframeUnit.MINUTE
            else 0,
            days=timeframe.count
            if timeframe.unit is TimeframeUnit.DAY
            else 0,
        )
        if period.end - period.start < minimum_span:
            raise ConfigurationError(
                "period must span at least one requested timeframe."
            )
        if not isinstance(calendar, CalendarPolicy):
            raise ConfigurationError("calendar must be a CalendarPolicy.")
        if not isinstance(initial_cash, Money) or initial_cash.amount <= 0:
            raise ConfigurationError(
                "initial_cash must be positive Money."
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= _MAX_SEED
        ):
            raise ConfigurationError(
                "seed must be a nonnegative signed 63-bit integer."
            )
        if not isinstance(metrics, MetricsConfig):
            raise ConfigurationError("metrics must be MetricsConfig.")
        object.__setattr__(self, "universe", copied_universe)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "initial_cash", initial_cash)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    """Bind a Python strategy and optional identity to a simulation.

    ``provenance`` is the public binding path for callers that compile
    strategies themselves — ``StrategySpec`` front ends, MCP servers, and
    other generators — and can therefore state spec, compiler, and schema
    fingerprints the engine cannot derive. When it is ``None`` the engine
    derives Python provenance and fails closed if it cannot be trusted.
    """

    strategy: Strategy
    simulation: SimulationRequest
    run_id: RunId | None = None
    provenance: ProvenanceDescriptor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, Strategy):
            raise ConfigurationError(
                "strategy must implement the Strategy protocol."
            )
        if not isinstance(self.simulation, SimulationRequest):
            raise ConfigurationError(
                "simulation must be a SimulationRequest."
            )
        if self.run_id is not None and not isinstance(self.run_id, RunId):
            raise ConfigurationError("run_id must be a RunId when provided.")
        if self.provenance is not None and not isinstance(
            self.provenance,
            ProvenanceDescriptor,
        ):
            raise ConfigurationError(
                "provenance must be a ProvenanceDescriptor when provided.",
                code="invalid_provenance_descriptor",
            )
