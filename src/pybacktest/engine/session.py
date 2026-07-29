"""One stepable, deterministic simulation state machine."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol
from uuid import UUID, uuid5

import numpy as np

from pybacktest.application.requests import SimulationRequest
from pybacktest.data.calendar import CalendarMode
from pybacktest.data.dataset import MarketDataSet
from pybacktest.data.features import (
    FeatureExecutor,
    FeaturePlan,
    FeatureSet,
    FeatureView,
)
from pybacktest.domain.errors import (
    AdapterContractError,
    ConfigurationError,
    DataValidationError,
    PybacktestError,
)
from pybacktest.domain.events import OrderRejected
from pybacktest.domain.identifiers import FillId, OrderId, RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.market import BarView, MarketSlice
from pybacktest.domain.money import Money
from pybacktest.domain.orders import CancelOrderIntent, Order, OrderStatus
from pybacktest.domain.portfolio import PortfolioSnapshot
from pybacktest.domain.time import TimeframeUnit
from pybacktest.engine.accounting import PortfolioLedger
from pybacktest.engine.recorder import RunRecorder
from pybacktest.ports.broker import (
    Broker,
    BrokerEvent,
    BrokerRunContext,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
)
from pybacktest.ports.data import MarketDataSource
from pybacktest.ports.risk import (
    OrderSizer,
    RiskContext,
    RiskDecision,
    RiskPolicy,
    RiskStatus,
)
from pybacktest.ports.strategy import (
    PortfolioSnapshot as StrategyPortfolioSnapshot,
)
from pybacktest.ports.strategy import StrategyContext, validate_strategy_output
from pybacktest.results.models import (
    BacktestResult,
    CausalStage,
    EngineEvent,
    EngineEventCode,
    FrozenMapping,
    RunManifest,
)
from pybacktest.results.serialization import canonical_json_bytes
from pybacktest.strategy.intents import OrderIntent

_LIBRARY_VERSION = "0.2.0"
_SCHEMA_VERSION = "results.v1"


class SessionStateError(PybacktestError, RuntimeError):
    """Raised when a simulation session lifecycle is used out of order."""


class BrokerFactory(Protocol):
    """Create a fresh broker for one immutable run context."""

    def create(self, run_context: BrokerRunContext) -> Broker:
        """Return a broker with no state shared with another run."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Observation:
    """Immutable current-only state exposed at one market timestamp."""

    timestamp: np.datetime64
    market: MarketSlice
    portfolio: PortfolioSnapshot
    active_orders: tuple[Order, ...]
    features: FeatureView

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, np.datetime64):
            raise ConfigurationError(
                "observation timestamp must be np.datetime64."
            )
        normalized = self.timestamp.astype("datetime64[ns]")
        if np.isnat(normalized):
            raise ConfigurationError("observation timestamp cannot be NaT.")
        if (
            not isinstance(self.market, MarketSlice)
            or self.market.timestamp != normalized
        ):
            raise ConfigurationError(
                "observation market must match its timestamp."
            )
        if not isinstance(self.portfolio, PortfolioSnapshot):
            raise ConfigurationError(
                "observation portfolio must be a PortfolioSnapshot."
            )
        if self.portfolio.timestamp != _as_datetime(normalized):
            raise ConfigurationError(
                "observation portfolio must match its timestamp."
            )
        active_orders = tuple(self.active_orders)
        if not all(isinstance(order, Order) for order in active_orders):
            raise ConfigurationError(
                "observation active_orders must contain Order values."
            )
        if (
            not isinstance(self.features, FeatureView)
            or self.features.timestamp != normalized
        ):
            raise ConfigurationError(
                "observation features must match its timestamp."
            )
        object.__setattr__(self, "timestamp", normalized)
        object.__setattr__(self, "active_orders", active_orders)


@dataclass(frozen=True, slots=True)
class StepResult:
    """The events and optional next observation from consuming one step."""

    observation: Observation | None
    events: tuple[EngineEvent, ...]
    done: bool

    def __post_init__(self) -> None:
        if self.observation is not None and not isinstance(
            self.observation, Observation
        ):
            raise ConfigurationError(
                "step observation must be Observation or None."
            )
        events = tuple(self.events)
        if not all(isinstance(event, EngineEvent) for event in events):
            raise ConfigurationError(
                "step events must contain EngineEvent values."
            )
        if type(self.done) is not bool:
            raise ConfigurationError("step done must be a bool.")
        if self.done != (self.observation is None):
            raise ConfigurationError(
                "done steps cannot expose a next observation."
            )
        object.__setattr__(self, "events", events)


class _RunIdSequence:
    """Run-namespaced UUID5 identities with independent kind counters."""

    def __init__(self, run_id: RunId) -> None:
        self._namespace = UUID(run_id.value.removeprefix("run_"))
        self._order_counter = 0

    def next_order_id(self) -> OrderId:
        value = uuid5(
            self._namespace,
            f"order:{self._order_counter}",
        )
        self._order_counter += 1
        return OrderId.parse(f"order_{value.hex}")

    def fill_id(self, sequence: int) -> FillId:
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ConfigurationError(
                "fill sequence must be a nonnegative integer."
            )
        value = uuid5(self._namespace, f"fill:{sequence}")
        return FillId.parse(f"fill_{value.hex}")


class _FixedDatasetSessionBoundary:
    """Explicit DAY expiry derived from the run's fixed market clock.

    Daily bars each define one session. Intraday bars share a session while
    their instrument-local calendar date is unchanged. The provider never
    reads wall-clock time and is injected only by the engine; direct brokers
    retain their documented no-inference behavior.
    """

    def __init__(
        self,
        dataset: MarketDataSet,
        calendar: np.ndarray,
    ) -> None:
        self._daily = dataset.timeframe.unit is TimeframeUnit.DAY
        self._instruments = dataset.instruments
        self._calendar = tuple(
            _as_datetime(timestamp) for timestamp in calendar
        )
        self._indices = {
            timestamp: index
            for index, timestamp in enumerate(self._calendar)
        }

    def day_order_expired(
        self,
        order: Order,
        timestamp: datetime,
    ) -> bool:
        if order.instrument not in self._instruments:
            raise AdapterContractError(
                "DAY order instrument is outside the fixed dataset.",
                code="invalid_session_boundary_instrument",
            )
        current_index = self._indices.get(timestamp)
        active_index = self._indices.get(order.active_from)
        if current_index is None or active_index is None:
            raise AdapterContractError(
                "DAY expiry timestamp is outside the fixed run calendar.",
                code="invalid_session_boundary_timestamp",
            )
        if current_index <= active_index:
            return False
        if self._daily:
            return True
        timezone = self._instruments[order.instrument].timezone
        return (
            timestamp.astimezone(timezone).date()
            > order.active_from.astimezone(timezone).date()
        )


@dataclass(frozen=True, slots=True)
class _Provenance:
    strategy_identity: str
    strategy_fingerprint: str
    spec_identity: str
    compiler_identity: str


class SimulationSession:
    """A single-use session that consumes each current observation once.

    Intents observed on a bar become active only at the next timestamp in the
    fixed calendar. Intents on the final bar are sized and risk-checked, then
    deterministically rejected with ``terminal.no_next_bar``; the session
    never fabricates a future timestamp.
    """

    def __init__(
        self,
        *,
        simulation: SimulationRequest,
        feature_plan: FeaturePlan,
        run_id: RunId,
        data_source: MarketDataSource,
        broker_factory: BrokerFactory,
        order_sizer: OrderSizer,
        risk_policy: RiskPolicy,
    ) -> None:
        if not isinstance(simulation, SimulationRequest):
            raise ConfigurationError(
                "simulation must be a SimulationRequest."
            )
        if not isinstance(feature_plan, FeaturePlan):
            raise ConfigurationError("feature_plan must be a FeaturePlan.")
        if not isinstance(run_id, RunId):
            raise ConfigurationError("run_id must be a RunId.")
        self._simulation = simulation
        self._feature_plan = feature_plan
        self._run_id = run_id
        self._data_source = data_source
        self._broker_factory = broker_factory
        self._order_sizer = order_sizer
        self._risk_policy = risk_policy
        self._provenance = _external_provenance(feature_plan)

        self._reset = False
        self._done = False
        self._failed = False
        self._context_created = False
        self._dataset: MarketDataSet | None = None
        self._calendar: np.ndarray | None = None
        self._feature_set: FeatureSet | None = None
        self._id_sequence: _RunIdSequence | None = None
        self._rng: np.random.Generator | None = None
        self._broker: Broker | None = None
        self._ledger: PortfolioLedger | None = None
        self._recorder: RunRecorder | None = None
        self._index = -1
        self._observation: Observation | None = None
        self._last_marks: dict[InstrumentId, tuple[int, Money]] = {}
        self._result: BacktestResult | None = None

    @property
    def run_id(self) -> RunId:
        """Return the identity allocated when this session was created."""
        return self._run_id

    @property
    def done(self) -> bool:
        """Return whether the final current observation has been consumed."""
        return self._done

    def _bind_python_strategy(
        self,
        *,
        strategy: object,
        feature_plan: FeaturePlan,
    ) -> None:
        """Bind deterministic Python provenance before any dataset I/O."""
        if self._reset:
            raise SessionStateError(
                "strategy provenance must be bound before reset."
            )
        if feature_plan != self._feature_plan:
            raise ConfigurationError(
                "bound feature plan must equal the session feature plan."
            )
        self._provenance = _python_provenance(strategy, feature_plan)

    def reset(self) -> Observation:
        """Load fixed data and return the first current-only observation."""
        if self._reset:
            raise SessionStateError("session reset may be called only once.")
        dataset = _load_dataset(self._data_source, self._simulation)
        calendar = self._simulation.calendar.build(
            {
                instrument_id: dataset.series[instrument_id].timestamps
                for instrument_id in self._simulation.universe
            }
        )
        if len(calendar) == 0:
            raise DataValidationError(
                "calendar contains no tradable timestamps."
            )

        feature_set = FeatureExecutor().execute(
            self._feature_plan,
            dataset,
        )
        id_sequence = _RunIdSequence(self._run_id)
        broker = self._broker_factory.create(
            BrokerRunContext(
                instruments=dataset.instruments,
                session_boundary=_FixedDatasetSessionBoundary(
                    dataset,
                    calendar,
                ),
                fill_ids=id_sequence,
            )
        )
        if not isinstance(broker, Broker):
            raise AdapterContractError(
                "broker factory must return a Broker.",
                code="invalid_broker",
            )
        ledger = PortfolioLedger(
            base_currency=self._simulation.initial_cash.currency,
            initial_cash=self._simulation.initial_cash,
            instruments=dataset.instruments,
        )
        manifest = _manifest(
            simulation=self._simulation,
            feature_plan=self._feature_plan,
            dataset=dataset,
            run_id=self._run_id,
            provenance=self._provenance,
            data_source=self._data_source,
            broker_factory=self._broker_factory,
            order_sizer=self._order_sizer,
            risk_policy=self._risk_policy,
            calendar=calendar,
        )
        recorder = RunRecorder(
            manifest=manifest,
            metrics_config=self._simulation.metrics,
        )

        self._dataset = dataset
        self._calendar = calendar
        self._feature_set = feature_set
        self._id_sequence = id_sequence
        self._rng = np.random.default_rng(self._simulation.seed)
        self._broker = broker
        self._ledger = ledger
        self._recorder = recorder
        self._index = 0
        self._reset = True

        market = self._market_slice(0)
        timestamp = _as_datetime(market.timestamp)
        recorder.record_market_timestamp(timestamp)
        initial_marks, initial_mark_state = (
            self._candidate_valuation_marks(market, 0)
        )
        self._require_held_position_marks(initial_marks)
        snapshot = ledger.mark_to_market(
            timestamp=timestamp,
            prices=initial_marks,
        )
        recorder.record_snapshot(snapshot)
        self._last_marks = initial_mark_state
        observation = self._build_observation(market, snapshot)
        self._observation = observation
        return observation

    def strategy_context(self, observation: Observation) -> StrategyContext:
        """Create one strategy capability for the current observation only."""
        self._require_active()
        if observation is not self._observation:
            raise SessionStateError(
                "observation is stale or belongs to another session."
            )
        if self._context_created:
            raise SessionStateError(
                "strategy context for this observation was already created."
            )
        self._context_created = True
        return StrategyContext(
            timestamp=observation.timestamp,
            portfolio=StrategyPortfolioSnapshot(
                positions={
                    instrument_id: position.quantity
                    for instrument_id, position in (
                        observation.portfolio.positions.items()
                    )
                }
            ),
            active_orders=observation.active_orders,
            features=observation.features,
            run_id=self._run_id,
        )

    def advance(self, intents: Sequence[OrderIntent]) -> StepResult:
        """Consume decisions for the current bar and move at most one bar."""
        self._require_active()
        try:
            validated = validate_strategy_output(
                intents,
                universe=self._simulation.universe,
            )
            current_events: list[EngineEvent] = []
            next_index = self._index + 1
            has_next = next_index < len(self._required_calendar())
            next_market: MarketSlice | None = None
            next_marks: Mapping[InstrumentId, Money] | None = None
            next_mark_state: dict[
                InstrumentId, tuple[int, Money]
            ] | None = None
            if has_next:
                next_market = self._market_slice(next_index)
                next_marks, next_mark_state = (
                    self._candidate_valuation_marks(
                        next_market,
                        next_index,
                    )
                )
                self._require_held_position_marks(next_marks)
            for intent in validated:
                current_events.extend(
                    self._apply_intent(intent, has_next=has_next)
                )

            self._observation = None
            self._context_created = False
            if not has_next:
                self._done = True
                return StepResult(
                    observation=None,
                    events=tuple(current_events),
                    done=True,
                )

            if (
                next_market is None
                or next_marks is None
                or next_mark_state is None
            ):
                raise SessionStateError(
                    "next market preparation is incomplete."
                )
            current_timestamp = _as_datetime(next_market.timestamp)
            recorder = self._required_recorder()
            recorder.record_market_timestamp(current_timestamp)
            broker_events = self._required_broker().process(
                next_market,
                self._required_rng(),
            )
            current_events.extend(
                self._apply_broker_events(broker_events)
            )
            self._require_held_position_marks(next_marks)
            snapshot = self._required_ledger().mark_to_market(
                timestamp=current_timestamp,
                prices=next_marks,
            )
            recorder.record_snapshot(snapshot)
            observation = self._build_observation(
                next_market,
                snapshot,
            )
            self._index = next_index
            self._last_marks = next_mark_state
            self._observation = observation
            return StepResult(
                observation=observation,
                events=tuple(current_events),
                done=False,
            )
        except Exception:
            self._failed = True
            raise

    def result(self) -> BacktestResult:
        """Finalize and return the immutable result after completion."""
        if not self._reset:
            raise SessionStateError("session must be reset before result().")
        if self._failed:
            raise SessionStateError(
                "failed session cannot produce a result."
            )
        if not self._done:
            raise SessionStateError(
                "session must be complete before result()."
            )
        if self._result is None:
            self._result = self._required_recorder().finalize()
        return self._result

    def _require_active(self) -> None:
        if not self._reset:
            raise SessionStateError(
                "session must be reset before it can advance."
            )
        if self._failed:
            raise SessionStateError(
                "failed session cannot be advanced."
            )
        if self._done:
            raise SessionStateError(
                "completed session cannot advance."
            )
        if self._observation is None:
            raise SessionStateError(
                "current observation was already consumed."
            )

    def _apply_intent(
        self,
        intent: OrderIntent,
        *,
        has_next: bool,
    ) -> tuple[EngineEvent, ...]:
        timestamp = _as_datetime(
            self._required_calendar()[self._index]
        )
        order_id = (
            intent.order_id
            if isinstance(intent, CancelOrderIntent)
            else self._required_ids().next_order_id()
        )
        active_from = (
            _as_datetime(self._required_calendar()[self._index + 1])
            if has_next
            else timestamp
        )
        context = RiskContext(
            snapshot=self._current_snapshot(),
            prices=self._current_marks(),
            instruments=self._required_dataset().instruments,
            tradable=frozenset(self._current_market().bars),
            order_id=order_id,
            submitted_at=timestamp,
            active_from=active_from,
        )
        sized = self._order_sizer.size(intent, context)
        if not isinstance(
            sized,
            (Order, OrderRejected, CancelOrderIntent),
        ):
            raise AdapterContractError(
                "order sizer must return Order, OrderRejected, "
                "or CancelOrderIntent.",
                code="invalid_sizer_result",
            )
        if isinstance(intent, CancelOrderIntent):
            if (
                not isinstance(sized, CancelOrderIntent)
                or sized.order_id != intent.order_id
            ):
                raise AdapterContractError(
                    "order sizer must preserve cancellation identity.",
                    code="invalid_sizer_result",
                )
            return self._apply_broker_events(
                self._required_broker().cancel(
                    sized.order_id,
                    timestamp,
                )
            )
        if isinstance(sized, CancelOrderIntent):
            raise AdapterContractError(
                "order sizer cannot turn a new order into a cancellation.",
                code="invalid_sizer_result",
            )
        if isinstance(sized, OrderRejected):
            if (
                sized.order_id != order_id
                or sized.instrument != intent.instrument
                or sized.timestamp != timestamp
            ):
                raise AdapterContractError(
                    "sizer rejection must preserve the run context.",
                    code="invalid_sizer_result",
                )
            event = self._required_recorder().emit_event(
                timestamp=timestamp,
                stage=CausalStage.of("sizing"),
                code=EngineEventCode.of("order.rejected"),
                details={
                    "reason": sized.reason.code,
                    "instrument": str(sized.instrument),
                },
                message=sized.message,
            )
            return (event,)
        if (
            sized.id != order_id
            or sized.instrument != intent.instrument
            or sized.submitted_at != timestamp
            or sized.active_from != active_from
            or sized.status is not OrderStatus.PENDING
        ):
            raise AdapterContractError(
                "sized order must preserve the run context and be PENDING.",
                code="invalid_sizer_result",
            )

        decision = self._risk_policy.evaluate(sized, context)
        if not isinstance(decision, RiskDecision):
            raise AdapterContractError(
                "risk policy must return RiskDecision.",
                code="invalid_risk_decision",
            )
        instrument = self._required_dataset().instruments[
            sized.instrument
        ]
        if (
            decision.original_quantity != sized.quantity
            or not _is_lot_aligned(
                decision.final_quantity.value,
                instrument.lot_size,
            )
        ):
            raise AdapterContractError(
                "risk decision quantities must match the sized order "
                "and instrument lot.",
                code="invalid_risk_decision",
            )
        if decision.status is RiskStatus.REJECTED:
            rejected = sized.reject()
            self._required_recorder().record_order(rejected)
            event = self._required_recorder().emit_event(
                timestamp=timestamp,
                stage=CausalStage.of("risk"),
                code=EngineEventCode.of("order.rejected"),
                order_id=rejected.id,
                details={"codes": decision.codes},
                message=decision.message,
            )
            return (event,)

        events: list[EngineEvent] = []
        accepted_source = sized
        if decision.status is RiskStatus.ADJUSTED:
            accepted_source = replace(
                sized,
                quantity=decision.final_quantity,
            )
            self._required_recorder().record_order(accepted_source)
            events.append(
                self._required_recorder().emit_event(
                    timestamp=timestamp,
                    stage=CausalStage.of("risk"),
                    code=EngineEventCode.of("order.adjusted"),
                    order_id=accepted_source.id,
                    details={
                        "requested_quantity": str(
                            decision.original_quantity.value
                        ),
                        "adjusted_quantity": str(
                            decision.final_quantity.value
                        ),
                        "codes": decision.codes,
                    },
                    message=decision.message,
                )
            )

        if not has_next:
            terminal = accepted_source.reject()
            self._required_recorder().record_order(terminal)
            events.append(
                self._required_recorder().emit_event(
                    timestamp=timestamp,
                    stage=CausalStage.of("scheduling"),
                    code=EngineEventCode.of("terminal.no_next_bar"),
                    order_id=terminal.id,
                    details={
                        "policy": "reject_after_sizing_and_risk",
                    },
                    message=(
                        "Order rejected because the fixed calendar has "
                        "no next tradable timestamp."
                    ),
                )
            )
            return tuple(events)

        accepted = accepted_source.accept()
        self._required_recorder().record_order(accepted)
        events.append(
            self._required_recorder().emit_event(
                timestamp=timestamp,
                stage=CausalStage.of("broker"),
                code=EngineEventCode.of("order.accepted"),
                order_id=accepted.id,
                details={
                    "active_from": accepted.active_from.isoformat(),
                },
                message="Order accepted for next-bar execution.",
            )
        )
        submit_events = self._required_broker().submit(accepted)
        events.extend(self._apply_broker_events(submit_events))
        return tuple(events)

    def _apply_broker_events(
        self,
        broker_events: Sequence[BrokerEvent],
    ) -> tuple[EngineEvent, ...]:
        if isinstance(broker_events, (str, bytes, bytearray)) or not isinstance(
            broker_events, Sequence
        ):
            raise AdapterContractError(
                "broker events must be a sequence.",
                code="invalid_broker_events",
            )
        emitted: list[EngineEvent] = []
        recorder = self._required_recorder()
        for broker_event in broker_events:
            if isinstance(
                broker_event,
                (OrderFilledEvent, OrderPartiallyFilledEvent),
            ):
                recorder.record_order(broker_event.order)
                recorder.record_fill(broker_event.fill)
                self._required_ledger().apply_fill(broker_event.fill)
                code = (
                    "order.filled"
                    if isinstance(broker_event, OrderFilledEvent)
                    else "order.partial_fill"
                )
                emitted.append(
                    recorder.emit_event(
                        timestamp=broker_event.fill.timestamp,
                        stage=CausalStage.of("broker"),
                        code=EngineEventCode.of(code),
                        order_id=broker_event.order.id,
                        details={
                            "fill_id": str(broker_event.fill.id),
                            "quantity": str(
                                broker_event.fill.quantity.value
                            ),
                            "price": str(broker_event.fill.price.amount),
                        },
                        message="Broker execution applied to the ledger.",
                    )
                )
            elif isinstance(
                broker_event,
                (OrderCancelledEvent, OrderExpiredEvent),
            ):
                recorder.record_order(broker_event.order)
                code = (
                    "order.cancelled"
                    if isinstance(broker_event, OrderCancelledEvent)
                    else "order.expired"
                )
                emitted.append(
                    recorder.emit_event(
                        timestamp=broker_event.timestamp,
                        stage=CausalStage.of("broker"),
                        code=EngineEventCode.of(code),
                        order_id=broker_event.order.id,
                        message=broker_event.message,
                    )
                )
            else:
                raise AdapterContractError(
                    "broker returned an unsupported event value.",
                    code="invalid_broker_event",
                )
        return tuple(emitted)

    def _market_slice(self, index: int) -> MarketSlice:
        dataset = self._required_dataset()
        timestamp = self._required_calendar()[index]
        bars: dict[InstrumentId, BarView] = {}
        for instrument_id in self._simulation.universe:
            series = dataset.series[instrument_id]
            position = int(
                np.searchsorted(series.timestamps, timestamp)
            )
            if (
                position >= len(series.timestamps)
                or series.timestamps[position] != timestamp
            ):
                continue
            bars[instrument_id] = BarView(
                timestamp=timestamp,
                open=float(series.open[position]),
                high=float(series.high[position]),
                low=float(series.low[position]),
                close=float(series.close[position]),
                volume=float(series.volume[position]),
            )
        return MarketSlice(timestamp=timestamp, bars=bars)

    def _candidate_valuation_marks(
        self,
        market: MarketSlice,
        index: int,
    ) -> tuple[
        Mapping[InstrumentId, Money],
        dict[InstrumentId, tuple[int, Money]],
    ]:
        dataset = self._required_dataset()
        candidate_state = dict(self._last_marks)
        for instrument_id, bar in market.bars.items():
            candidate_state[instrument_id] = (
                index,
                Money.of(
                    str(bar.close),
                    dataset.instruments[
                        instrument_id
                    ].quote_currency,
                ),
            )
        max_staleness = (
            0
            if self._simulation.calendar.mode is CalendarMode.INTERSECTION
            else self._simulation.calendar.max_staleness_bars
        )
        marks = MappingProxyType(
            {
                instrument_id: price
                for instrument_id, (observed_index, price) in (
                    candidate_state.items()
                )
                if index - observed_index <= max_staleness
            }
        )
        return marks, candidate_state

    def _require_held_position_marks(
        self,
        marks: Mapping[InstrumentId, Money],
    ) -> None:
        missing = tuple(
            instrument_id
            for instrument_id, position in (
                self._required_ledger().snapshot().positions.items()
            )
            if position.quantity.value != 0 and instrument_id not in marks
        )
        if missing:
            instruments = ", ".join(str(item) for item in missing)
            raise DataValidationError(
                "held position mark exceeded the configured calendar "
                f"staleness policy: {instruments}."
            )

    def _current_marks(self) -> Mapping[InstrumentId, Money]:
        return self._current_snapshot().valuation_prices

    def _current_market(self) -> MarketSlice:
        observation = self._observation
        if observation is None:
            raise SessionStateError("current observation is unavailable.")
        return observation.market

    def _current_snapshot(self) -> PortfolioSnapshot:
        observation = self._observation
        if observation is None:
            raise SessionStateError("current observation is unavailable.")
        return observation.portfolio

    def _build_observation(
        self,
        market: MarketSlice,
        snapshot: PortfolioSnapshot,
    ) -> Observation:
        feature_set = self._feature_set
        if feature_set is None:
            raise SessionStateError("feature set is unavailable.")
        return Observation(
            timestamp=market.timestamp,
            market=market,
            portfolio=snapshot,
            active_orders=self._active_orders(),
            features=feature_set.view(market.timestamp),
        )

    def _active_orders(self) -> tuple[Order, ...]:
        active_orders = self._required_broker().active_orders
        if not isinstance(active_orders, Mapping):
            raise AdapterContractError(
                "broker active_orders must be a mapping.",
                code="invalid_active_orders",
            )
        copied = dict(active_orders)
        if not all(
            isinstance(order_id, OrderId)
            and isinstance(order, Order)
            and order.id == order_id
            and order.status
            in {
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
            }
            for order_id, order in copied.items()
        ):
            raise AdapterContractError(
                "broker active_orders must map matching IDs to active orders.",
                code="invalid_active_orders",
            )
        return tuple(copied.values())

    def _required_dataset(self) -> MarketDataSet:
        if self._dataset is None:
            raise SessionStateError("dataset is unavailable before reset.")
        return self._dataset

    def _required_calendar(self) -> np.ndarray:
        if self._calendar is None:
            raise SessionStateError("calendar is unavailable before reset.")
        return self._calendar

    def _required_ids(self) -> _RunIdSequence:
        if self._id_sequence is None:
            raise SessionStateError("ID sequence is unavailable before reset.")
        return self._id_sequence

    def _required_rng(self) -> np.random.Generator:
        if self._rng is None:
            raise SessionStateError("RNG is unavailable before reset.")
        return self._rng

    def _required_broker(self) -> Broker:
        if self._broker is None:
            raise SessionStateError("broker is unavailable before reset.")
        return self._broker

    def _required_ledger(self) -> PortfolioLedger:
        if self._ledger is None:
            raise SessionStateError("ledger is unavailable before reset.")
        return self._ledger

    def _required_recorder(self) -> RunRecorder:
        if self._recorder is None:
            raise SessionStateError("recorder is unavailable before reset.")
        return self._recorder


def _load_dataset(
    data_source: MarketDataSource,
    simulation: SimulationRequest,
) -> MarketDataSet:
    try:
        load = data_source.load
    except AttributeError as exc:
        raise AdapterContractError(
            "data source must expose load().",
            code="invalid_data_source",
        ) from exc
    dataset = load(
        simulation.universe,
        simulation.period,
        simulation.timeframe,
    )
    if not isinstance(dataset, MarketDataSet):
        raise AdapterContractError(
            "data source must return MarketDataSet.",
            code="invalid_dataset_type",
        )
    if dataset.timeframe != simulation.timeframe:
        raise DataValidationError(
            "dataset timeframe does not match the request."
        )
    if set(dataset.instruments) != set(simulation.universe):
        raise DataValidationError(
            "dataset universe does not match the request."
        )
    currencies = {
        instrument.quote_currency
        for instrument in dataset.instruments.values()
    }
    if currencies != {simulation.initial_cash.currency}:
        raise DataValidationError(
            "dataset instruments must share the initial-cash currency."
        )
    start = _as_np_datetime(simulation.period.start)
    end = _as_np_datetime(simulation.period.end)
    for series in dataset.series.values():
        if (
            len(series.timestamps) == 0
            or series.timestamps[0] < start
            or series.timestamps[-1] >= end
        ):
            raise DataValidationError(
                "dataset timestamps must fall inside the requested period."
            )
    return dataset


def _manifest(
    *,
    simulation: SimulationRequest,
    feature_plan: FeaturePlan,
    dataset: MarketDataSet,
    run_id: RunId,
    provenance: _Provenance,
    data_source: object,
    broker_factory: object,
    order_sizer: object,
    risk_policy: object,
    calendar: np.ndarray,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        library_version=_LIBRARY_VERSION,
        schema_version=_SCHEMA_VERSION,
        canonical_request=FrozenMapping.from_mapping({
            "universe": [str(item) for item in simulation.universe],
            "period": {
                "start": simulation.period.start.astimezone(UTC).isoformat(),
                "end": simulation.period.end.astimezone(UTC).isoformat(),
            },
            "timeframe": {
                "unit": simulation.timeframe.unit.value,
                "count": simulation.timeframe.count,
            },
            "calendar": {
                "mode": simulation.calendar.mode.value,
                "max_staleness_bars": (
                    simulation.calendar.max_staleness_bars
                ),
            },
            "initial_cash": {
                "amount": str(simulation.initial_cash.amount),
                "currency": simulation.initial_cash.currency,
            },
            "seed": simulation.seed,
            "metrics": {
                "risk_free_rate": str(
                    simulation.metrics.risk_free_rate
                ),
                "annualization_periods": (
                    simulation.metrics.annualization_periods
                ),
            },
            "feature_plan_fingerprint": _fingerprint(feature_plan),
            "execution_fingerprint": _component_fingerprint(
                broker_factory,
                order_sizer,
                risk_policy,
            ),
        }),
        strategy_identity=provenance.strategy_identity,
        strategy_fingerprint=provenance.strategy_fingerprint,
        spec_identity=provenance.spec_identity,
        compiler_identity=provenance.compiler_identity,
        dataset_fingerprint=dataset.fingerprint,
        seed=simulation.seed,
        adapter_versions=FrozenMapping.from_mapping({
            "data": _version_identity(data_source),
            "broker": _version_identity(broker_factory),
        }),
        model_versions=FrozenMapping.from_mapping({
            "order_sizer": _version_identity(order_sizer),
            "risk_policy": _version_identity(risk_policy),
        }),
        started_at=_as_datetime(calendar[0]),
        ended_at=_as_datetime(calendar[-1]),
    )


def _external_provenance(feature_plan: FeaturePlan) -> _Provenance:
    return _Provenance(
        strategy_identity="external.actions",
        strategy_fingerprint=_fingerprint(
            {
                "identity": "external.actions",
                "feature_plan": feature_plan,
            }
        ),
        spec_identity="external.actions",
        compiler_identity="pybacktest.session.external.v1",
    )


def _python_provenance(
    strategy: object,
    feature_plan: FeaturePlan,
) -> _Provenance:
    identity = _type_identity(strategy)
    if hasattr(strategy, "__dict__"):
        state: object = dict(vars(strategy))
    elif hasattr(type(strategy), "__dataclass_fields__"):
        state = strategy
    else:
        slots = getattr(type(strategy), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        state = {
            slot: getattr(strategy, slot)
            for slot in slots
            if hasattr(strategy, slot)
        }
    try:
        fingerprint = _fingerprint(
            {
                "identity": identity,
                "configuration": state,
                "feature_plan": feature_plan,
            }
        )
    except Exception as exc:
        raise ConfigurationError(
            "strategy configuration contains unsupported deterministic state.",
            code="unsupported_strategy_state",
        ) from exc
    return _Provenance(
        strategy_identity=identity,
        strategy_fingerprint=fingerprint,
        spec_identity="python.strategy",
        compiler_identity="pybacktest.session.python.v1",
    )


def _fingerprint(value: object) -> str:
    try:
        encoded = canonical_json_bytes(value)
    except Exception as exc:
        raise ConfigurationError(
            "value contains unsupported deterministic state.",
            code="unsupported_deterministic_state",
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _component_fingerprint(*components: object) -> str:
    try:
        descriptors = tuple(
            {
                "identity": _type_identity(component),
                "configuration": _deterministic_object_state(component),
            }
            for component in components
        )
        return _fingerprint(descriptors)
    except Exception as exc:
        if (
            isinstance(exc, ConfigurationError)
            and exc.code == "unsupported_component_state"
        ):
            raise
        raise ConfigurationError(
            "engine component contains unsupported deterministic state.",
            code="unsupported_component_state",
        ) from exc


def _deterministic_object_state(value: object) -> object:
    if hasattr(type(value), "__dataclass_fields__"):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    return {
        slot: getattr(value, slot)
        for slot in slots
        if hasattr(value, slot)
    }


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _version_identity(value: object) -> str:
    identity = _type_identity(value).replace("<locals>", "locals")
    sanitized = re.sub(r"[^A-Za-z0-9._+-]", "_", identity)
    return (
        sanitized
        if sanitized and sanitized[0].isalnum()
        else f"type.{sanitized}"
    )


def _as_datetime(timestamp: np.datetime64) -> datetime:
    normalized = timestamp.astype("datetime64[us]")
    value = normalized.item()
    if not isinstance(value, datetime):
        raise DataValidationError(
            "market timestamp cannot be converted to datetime."
        )
    return value.replace(tzinfo=UTC)


def _as_np_datetime(value: datetime) -> np.datetime64:
    normalized = value.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(normalized, "ns")


def _is_lot_aligned(value: Decimal, lot_size: Decimal) -> bool:
    try:
        value_numerator, value_denominator = value.as_integer_ratio()
        lot_numerator, lot_denominator = lot_size.as_integer_ratio()
    except (AttributeError, OverflowError, ValueError):
        return False
    if lot_numerator <= 0:
        return False
    return (
        value_numerator * lot_denominator
    ) % (
        value_denominator * lot_numerator
    ) == 0


__all__ = [
    "BrokerFactory",
    "Observation",
    "SessionStateError",
    "SimulationSession",
    "StepResult",
]
