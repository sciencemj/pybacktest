"""One stepable, deterministic simulation state machine."""

from __future__ import annotations

import hashlib
import re
import threading
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol
from uuid import UUID, uuid5
from weakref import ref

import numpy as np

from pybacktest._introspection import deterministic_instance_state
from pybacktest.application.provenance import (
    ProvenanceDescriptor,
    external_action_provenance,
)
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
from pybacktest.domain.orders import (
    CancelOrderIntent,
    Order,
    OrderStatus,
    TimeInForce,
)
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
from pybacktest.ports.components import (
    ComponentDescriptor,
    DeterministicComponent,
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
from pybacktest.results._decimal import ExactDecimalError, exact_add
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

_BROKER_REGISTRY: dict[int, ref[Broker]] = {}
_BROKER_REGISTRY_LOCK = threading.Lock()

class _BrokerCallOrigin(StrEnum):
    """Which engine call produced a broker event batch.

    The legal shape of a batch depends entirely on what the engine asked
    for, so the origin travels with the batch and is validated explicitly
    rather than inferred from the events themselves.
    """

    SUBMIT = "submit"
    CANCEL = "cancel"
    PROCESS = "process"


_STABLE_ORDER_FIELDS = (
    "instrument",
    "side",
    "type",
    "quantity",
    "quote_currency",
    "limit_price",
    "time_in_force",
    "submitted_at",
    "active_from",
    "reason",
)

_ALLOWED_BROKER_TRANSITIONS = {
    OrderStatus.ACCEPTED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
}


class SessionStateError(PybacktestError, RuntimeError):
    """Raised when a simulation session lifecycle is used out of order."""


class BrokerFactory(Protocol):
    """Create a fresh broker for one immutable run context.

    Every call must return an instance no previous session has used. The
    engine enforces this by holding a weak reference to each broker it
    claims, so a custom broker must support weak references: a class using
    ``__slots__`` needs ``weakref_slot=True`` (or a ``__weakref__`` slot),
    otherwise ``reset()`` fails closed with ``unsupported_broker_instance``.
    """

    def create(self, run_context: BrokerRunContext) -> Broker:
        """Return a weak-referenceable broker sharing no state with a run."""
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


class _FillIdFacade:
    """Expose only ``fill_id`` from the run's identity sequence.

    Handing the broker the whole :class:`_RunIdSequence` would also hand it
    ``next_order_id()``, letting an adapter advance the engine's own order
    counter. The broker gets this narrowed capability instead.
    """

    def __init__(self, sequence: _RunIdSequence) -> None:
        self._sequence = sequence

    def fill_id(self, sequence: int) -> FillId:
        """Return the run-scoped identity for one committed fill ordinal."""
        return self._sequence.fill_id(sequence)


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
        # Membership is decided on exact datetime64[ns] values *before* the
        # microsecond conversion. Truncating first would let a series
        # timestamp the calendar excludes collapse onto a real calendar
        # entry and become a phantom DAY anchor.
        calendar_ns = frozenset(
            calendar.astype("datetime64[ns]").astype(np.int64).tolist()
        )
        self._instrument_calendars = {
            instrument_id: tuple(
                _as_datetime(value)
                for value, exact in zip(
                    dataset.series[instrument_id].timestamps,
                    dataset.series[instrument_id]
                    .timestamps.astype("datetime64[ns]")
                    .astype(np.int64)
                    .tolist(),
                    strict=True,
                )
                if exact in calendar_ns
            )
            for instrument_id in dataset.instruments
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
        if current_index is None or order.active_from not in self._indices:
            raise AdapterContractError(
                "DAY expiry timestamp is outside the fixed run calendar.",
                code="invalid_session_boundary_timestamp",
            )
        instrument_calendar = self._instrument_calendars[
            order.instrument
        ]
        anchor_index = bisect_left(
            instrument_calendar,
            order.active_from,
        )
        if anchor_index == len(instrument_calendar):
            return False
        anchor = instrument_calendar[anchor_index]
        if timestamp <= anchor:
            return False
        if self._daily:
            return True
        timezone = self._instruments[order.instrument].timezone
        return (
            timestamp.astimezone(timezone).date()
            > anchor.astimezone(timezone).date()
        )


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
        provenance: ProvenanceDescriptor | None = None,
    ) -> None:
        if not isinstance(simulation, SimulationRequest):
            raise ConfigurationError(
                "simulation must be a SimulationRequest."
            )
        if not isinstance(feature_plan, FeaturePlan):
            raise ConfigurationError("feature_plan must be a FeaturePlan.")
        if not isinstance(run_id, RunId):
            raise ConfigurationError("run_id must be a RunId.")
        if provenance is not None and not isinstance(
            provenance,
            ProvenanceDescriptor,
        ):
            raise ConfigurationError(
                "provenance must be a ProvenanceDescriptor when supplied.",
                code="invalid_provenance_descriptor",
            )
        self._simulation = simulation
        self._feature_plan = feature_plan
        self._run_id = run_id
        self._data_source = data_source
        self._broker_factory = broker_factory
        self._order_sizer = order_sizer
        self._risk_policy = risk_policy
        self._provenance = (
            external_action_provenance(feature_plan)
            if provenance is None
            else provenance
        )

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
        self._submitted_orders: dict[OrderId, Order] = {}
        self._fill_ordinal = 0
        self._result: BacktestResult | None = None

    @property
    def run_id(self) -> RunId:
        """Return the identity allocated when this session was created."""
        return self._run_id

    @property
    def done(self) -> bool:
        """Return whether the final current observation has been consumed."""
        return self._done

    @property
    def provenance(self) -> ProvenanceDescriptor:
        """Return the immutable provenance bound when this session was made."""
        return self._provenance

    def reset(self) -> Observation:
        """Load fixed data and return the first current-only observation."""
        if self._failed:
            raise SessionStateError("failed session cannot be reset.")
        if self._reset:
            raise SessionStateError("session reset may be called only once.")
        try:
            return self._initialize()
        except Exception:
            self._discard_runtime_state()
            self._failed = True
            raise

    def _initialize(self) -> Observation:
        """Build runtime state atomically for :meth:`reset`."""
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
        _validate_calendar_precision(calendar)

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
                fill_ids=_FillIdFacade(id_sequence),
            )
        )
        if not isinstance(broker, Broker):
            raise AdapterContractError(
                "broker factory must return a Broker.",
                code="invalid_broker",
            )
        _claim_fresh_broker(broker)
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
        self._submitted_orders = {}
        self._fill_ordinal = 0
        self._index = 0
        self._reset = True
        self._require_broker_state_agreement({})

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

    def _discard_runtime_state(self) -> None:
        """Remove any partially installed state after reset failure."""
        self._reset = False
        self._done = False
        self._context_created = False
        self._dataset = None
        self._calendar = None
        self._feature_set = None
        self._id_sequence = None
        self._rng = None
        self._broker = None
        self._ledger = None
        self._recorder = None
        self._index = -1
        self._observation = None
        self._last_marks = {}
        self._submitted_orders = {}
        self._fill_ordinal = 0
        self._result = None

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

    def advance(
        self,
        intents: Sequence[OrderIntent],
        *,
        observation: Observation,
    ) -> StepResult:
        """Consume decisions for the current bar and move at most one bar."""
        self._require_active()
        if observation is not self._observation:
            raise SessionStateError(
                "observation is stale or belongs to another session."
            )
        validated = validate_strategy_output(
            intents,
            universe=self._simulation.universe,
        )
        self._prevalidate_cancellations(validated, observation)
        try:
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
                self._apply_broker_events(
                    broker_events,
                    at=current_timestamp,
                    origin=_BrokerCallOrigin.PROCESS,
                )
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

    def _prevalidate_cancellations(
        self,
        intents: Sequence[OrderIntent],
        observation: Observation,
    ) -> None:
        """Reject every invalid cancellation before any batch item applies."""
        active_order_ids = {
            order.id for order in observation.active_orders
        }
        for intent in intents:
            if (
                isinstance(intent, CancelOrderIntent)
                and intent.order_id not in active_order_ids
            ):
                raise AdapterContractError(
                    "cancel intent must identify a current active order.",
                    code="unknown_cancel_order",
                )

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
            active_orders=self._active_orders(),
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
                ),
                at=timestamp,
                origin=_BrokerCallOrigin.CANCEL,
                cancelled_order_id=sized.order_id,
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
        recorder = self._required_recorder()
        recorded = _decision_order(sized, decision, has_next=has_next)
        recorder.record_order(recorded)
        events: list[EngineEvent] = [
            recorder.emit_event(
                timestamp=timestamp,
                stage=CausalStage.of("intent"),
                code=EngineEventCode.of("intent.received"),
                order_id=recorded.id,
                details=_intent_details(intent),
                message="Strategy intent accepted for sizing.",
            ),
            recorder.emit_event(
                timestamp=timestamp,
                stage=CausalStage.of("sizing"),
                code=EngineEventCode.of("order.sized"),
                order_id=recorded.id,
                details=_sizing_details(sized),
                message="Intent sized into one proposed order.",
            ),
            self._emit_risk_event(recorded, decision, timestamp),
        ]
        if decision.status is RiskStatus.REJECTED:
            return tuple(events)

        if not has_next:
            events.append(
                recorder.emit_event(
                    timestamp=timestamp,
                    stage=CausalStage.of("scheduling"),
                    code=EngineEventCode.of("terminal.no_next_bar"),
                    order_id=recorded.id,
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

        events.append(
            recorder.emit_event(
                timestamp=timestamp,
                stage=CausalStage.of("scheduling"),
                code=EngineEventCode.of("order.scheduled"),
                order_id=recorded.id,
                details={
                    "active_from": recorded.active_from.isoformat(),
                    "time_in_force": recorded.time_in_force.value,
                },
                message="Order scheduled for the next tradable timestamp.",
            )
        )
        events.append(
            recorder.emit_event(
                timestamp=timestamp,
                stage=CausalStage.of("broker"),
                code=EngineEventCode.of("order.accepted"),
                order_id=recorded.id,
                details={
                    "active_from": recorded.active_from.isoformat(),
                },
                message="Order accepted for next-bar execution.",
            )
        )
        self._submitted_orders[recorded.id] = recorded
        submit_events = self._required_broker().submit(recorded)
        events.extend(
            self._apply_broker_events(
                submit_events,
                at=timestamp,
                origin=_BrokerCallOrigin.SUBMIT,
            )
        )
        return tuple(events)

    def _emit_risk_event(
        self,
        order: Order,
        decision: RiskDecision,
        timestamp: datetime,
    ) -> EngineEvent:
        """Record every risk outcome, including an unchanged pass."""
        if decision.status is RiskStatus.PASSED:
            code = "risk.passed"
        elif decision.status is RiskStatus.ADJUSTED:
            code = "order.adjusted"
        else:
            code = "order.rejected"
        return self._required_recorder().emit_event(
            timestamp=timestamp,
            stage=CausalStage.of("risk"),
            code=EngineEventCode.of(code),
            order_id=order.id,
            details={
                "requested_quantity": str(
                    decision.original_quantity.value
                ),
                "adjusted_quantity": str(decision.final_quantity.value),
                "codes": decision.codes,
            },
            message=decision.message,
        )

    def _apply_broker_events(
        self,
        broker_events: Sequence[BrokerEvent],
        *,
        at: datetime,
        origin: _BrokerCallOrigin,
        cancelled_order_id: OrderId | None = None,
    ) -> tuple[EngineEvent, ...]:
        if isinstance(broker_events, (str, bytes, bytearray)) or not isinstance(
            broker_events, Sequence
        ):
            raise AdapterContractError(
                "broker events must be a sequence.",
                code="invalid_broker_events",
            )
        predicted = self._validate_broker_events(
            broker_events,
            at=at,
            origin=origin,
            cancelled_order_id=cancelled_order_id,
        )
        self._require_broker_state_agreement(predicted)
        return self._record_broker_events(broker_events)

    def _validate_broker_events(
        self,
        broker_events: Sequence[BrokerEvent],
        *,
        at: datetime,
        origin: _BrokerCallOrigin,
        cancelled_order_id: OrderId | None,
    ) -> dict[OrderId, Order]:
        """Predict engine order state, rejecting before anything mutates.

        Returns the complete engine-owned order state this batch would
        produce, so the caller can compare the broker's post-call
        ``active_orders`` against it before any recorder, ledger, fill
        ordinal, or order-registry mutation happens.
        """
        _require_broker_call_origin(
            broker_events,
            origin=origin,
            cancelled_order_id=cancelled_order_id,
        )
        staged = dict(self._submitted_orders)
        staged_ordinal = self._fill_ordinal
        ids = self._required_ids()
        for broker_event in broker_events:
            if not isinstance(
                broker_event,
                (
                    OrderFilledEvent,
                    OrderPartiallyFilledEvent,
                    OrderCancelledEvent,
                    OrderExpiredEvent,
                ),
            ):
                raise AdapterContractError(
                    "broker returned an unsupported event value.",
                    code="invalid_broker_event",
                )
            order = broker_event.order
            previous = staged.get(order.id)
            if previous is None:
                raise AdapterContractError(
                    "broker event references an order the engine never "
                    "submitted.",
                    code="unknown_broker_order",
                )
            _require_stable_order_identity(previous, order)
            if order.status not in _ALLOWED_BROKER_TRANSITIONS.get(
                previous.status,
                frozenset(),
            ):
                raise AdapterContractError(
                    "broker event moved an order through an invalid "
                    "lifecycle transition.",
                    code="invalid_broker_order_state",
                )
            if isinstance(
                broker_event,
                (OrderFilledEvent, OrderPartiallyFilledEvent),
            ):
                fill = broker_event.fill
                if (
                    fill.order_id != order.id
                    or fill.instrument != order.instrument
                    or fill.side is not order.side
                    or fill.price.currency != order.quote_currency
                    or fill.fee.currency != order.quote_currency
                ):
                    raise AdapterContractError(
                        "broker fill disagrees with the engine-owned order "
                        "on identity, side, or quote currency.",
                        code="invalid_broker_order_state",
                    )
                if fill.quantity.value > previous.remaining_quantity.value:
                    raise AdapterContractError(
                        "broker fill quantity exceeds the order's remaining "
                        "quantity.",
                        code="invalid_broker_order_state",
                    )
                if fill.timestamp != at or fill.timestamp < order.active_from:
                    raise AdapterContractError(
                        "broker fill timestamp does not match the timestamp "
                        "the broker was asked to process.",
                        code="invalid_broker_event_timestamp",
                    )
                if fill.id != ids.fill_id(staged_ordinal):
                    raise AdapterContractError(
                        "broker fill identity is not the next run-scoped "
                        "UUID5 fill ordinal.",
                        code="invalid_fill_identity",
                    )
                try:
                    expected_filled = exact_add(
                        previous.filled_quantity.value,
                        fill.quantity.value,
                    )
                except ExactDecimalError as error:
                    raise AdapterContractError(
                        "broker fill quantity exceeds the exact numeric "
                        "range.",
                        code="invalid_broker_order_state",
                    ) from error
                if order.filled_quantity.value != expected_filled:
                    raise AdapterContractError(
                        "broker order filled quantity does not equal the "
                        "prior quantity plus this fill.",
                        code="invalid_broker_order_state",
                    )
                staged_ordinal += 1
            elif broker_event.timestamp != at:
                raise AdapterContractError(
                    "broker lifecycle timestamp does not match the "
                    "timestamp the broker was asked to process.",
                    code="invalid_broker_event_timestamp",
                )
            elif order.filled_quantity != previous.filled_quantity:
                raise AdapterContractError(
                    "broker terminated an order while changing its filled "
                    "quantity.",
                    code="invalid_broker_order_state",
                )
            staged[order.id] = order
        return staged

    def _require_broker_state_agreement(
        self,
        predicted: Mapping[OrderId, Order],
    ) -> None:
        """Require the broker's active orders to equal the engine's view."""
        expected = {
            order_id: order
            for order_id, order in predicted.items()
            if order.status
            in {
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
            }
        }
        if self._active_orders_mapping() != expected:
            raise AdapterContractError(
                "broker active_orders disagree with the engine-owned order "
                "lifecycle after the call.",
                code="broker_state_disagreement",
            )

    def _record_broker_events(
        self,
        broker_events: Sequence[BrokerEvent],
    ) -> tuple[EngineEvent, ...]:
        emitted: list[EngineEvent] = []
        recorder = self._required_recorder()
        for broker_event in broker_events:
            self._submitted_orders[broker_event.order.id] = (
                broker_event.order
            )
            if isinstance(
                broker_event,
                (OrderFilledEvent, OrderPartiallyFilledEvent),
            ):
                self._fill_ordinal += 1
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
                emitted.append(self._emit_accounting_event(broker_event))
            else:
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
        return tuple(emitted)

    def _emit_accounting_event(
        self,
        broker_event: OrderFilledEvent | OrderPartiallyFilledEvent,
    ) -> EngineEvent:
        """Record the ledger effect that this fill produced."""
        snapshot = self._required_ledger().snapshot()
        position = snapshot.positions.get(broker_event.order.instrument)
        return self._required_recorder().emit_event(
            timestamp=broker_event.fill.timestamp,
            stage=CausalStage.of("accounting"),
            code=EngineEventCode.of("ledger.applied"),
            order_id=broker_event.order.id,
            details={
                "fill_id": str(broker_event.fill.id),
                "cash": str(snapshot.cash.amount),
                "position_quantity": str(
                    Decimal("0")
                    if position is None
                    else position.quantity.value
                ),
                "realized_pnl": str(snapshot.realized_pnl.amount),
                "total_fees": str(snapshot.total_fees.amount),
            },
            message="Fill applied to the portfolio ledger.",
        )

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
        return tuple(self._active_orders_mapping().values())

    def _active_orders_mapping(self) -> dict[OrderId, Order]:
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
        instruments = self._required_dataset().instruments
        for order in copied.values():
            instrument = instruments.get(order.instrument)
            if instrument is None:
                raise AdapterContractError(
                    "broker active_orders contain an instrument outside the "
                    "run catalog.",
                    code="invalid_active_orders",
                )
            if order.quote_currency != instrument.quote_currency:
                raise AdapterContractError(
                    "broker active_orders contain an order whose quote "
                    "currency disagrees with its instrument.",
                    code="invalid_active_orders",
                )
        return copied

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
    provenance: ProvenanceDescriptor,
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
            "provenance": provenance.canonical_details(),
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


def _claim_fresh_broker(broker: Broker) -> None:
    """Require every session to own a broker no other session has used.

    A factory that returns a cached instance would leak order, fill, and
    clock state between runs, so the reused instance is rejected instead of
    silently producing a contaminated result.
    """
    key = id(broker)
    with _BROKER_REGISTRY_LOCK:
        existing = _BROKER_REGISTRY.get(key)
        if existing is not None and existing() is broker:
            raise AdapterContractError(
                "broker factory returned an instance already used by "
                "another session.",
                code="reused_broker_instance",
            )
        def _discard(discarded: ref[Broker], key: int = key) -> None:
            with _BROKER_REGISTRY_LOCK:
                if _BROKER_REGISTRY.get(key) is discarded:
                    del _BROKER_REGISTRY[key]

        try:
            _BROKER_REGISTRY[key] = ref(broker, _discard)
        except TypeError as exc:
            raise AdapterContractError(
                "broker instance must support weak references so the "
                "engine can prove it is fresh.",
                code="unsupported_broker_instance",
            ) from exc


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
            _deterministic_component_state(component)
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


def _deterministic_component_state(value: object) -> dict[str, object]:
    """Fingerprint declared immutable configuration, never live telemetry.

    A component declares identity explicitly through
    :class:`~pybacktest.ports.components.DeterministicComponent`. Frozen
    dataclasses are immutable by construction, and a component with no
    instance state has nothing that could drift. Anything else carries
    mutable runtime state the engine refuses to hash, so it fails closed.
    """
    identity = _type_identity(value)
    if isinstance(value, DeterministicComponent):
        descriptor = value.component_descriptor()
        if not isinstance(descriptor, ComponentDescriptor):
            raise ConfigurationError(
                "component_descriptor() must return a ComponentDescriptor.",
                code="unsupported_component_state",
            )
        return {
            "identity": descriptor.identity,
            "version": descriptor.version,
            "configuration": dict(descriptor.configuration),
        }
    if _is_frozen_dataclass(value):
        return {"identity": identity, "configuration": value}
    if not _instance_state(value):
        return {"identity": identity, "configuration": {}}
    raise ConfigurationError(
        f"engine component {identity} carries mutable runtime state; "
        "implement DeterministicComponent to declare its immutable "
        "configuration.",
        code="unsupported_component_state",
    )


def _is_frozen_dataclass(value: object) -> bool:
    parameters = getattr(type(value), "__dataclass_params__", None)
    return bool(getattr(parameters, "frozen", False))


def _instance_state(value: object) -> dict[str, object]:
    return deterministic_instance_state(value)


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


def _decision_order(
    sized: Order,
    decision: RiskDecision,
    *,
    has_next: bool,
) -> Order:
    """Return the single order this decision commits to the recorder."""
    if decision.status is RiskStatus.REJECTED:
        return sized.reject()
    source = (
        sized
        if decision.status is RiskStatus.PASSED
        else replace(sized, quantity=decision.final_quantity)
    )
    return source.reject() if not has_next else source.accept()


def _intent_details(intent: OrderIntent) -> dict[str, object]:
    """Preserve the strategy's own decision reason and structured details."""
    details: dict[str, object] = {
        "intent": type(intent).__name__,
        "instrument": str(getattr(intent, "instrument", "")),
        "reason": intent.reason.code,
    }
    for key, value in intent.reason.details.items():
        details[f"reason.{key}"] = value
    return details


def _sizing_details(sized: Order) -> dict[str, object]:
    return {
        "side": sized.side.value,
        "type": sized.type.value,
        "quantity": str(sized.quantity.value),
        "time_in_force": sized.time_in_force.value,
        "limit_price": (
            ""
            if sized.limit_price is None
            else str(sized.limit_price.amount)
        ),
    }


def _require_broker_call_origin(
    broker_events: Sequence[BrokerEvent],
    *,
    origin: _BrokerCallOrigin,
    cancelled_order_id: OrderId | None,
) -> None:
    """Reject events the engine's specific broker call cannot produce.

    Every order the engine submits becomes active only at a later
    timestamp, so ``submit()`` has nothing to report. ``cancel()`` answers
    for exactly the order it was given. ``process()`` may execute and may
    end a DAY session, but it may not cancel, and it may not expire an
    order that carries no DAY instruction.
    """
    if origin is _BrokerCallOrigin.SUBMIT:
        if broker_events:
            raise AdapterContractError(
                "broker submit() reported an event for an order that is "
                "not active yet.",
                code="invalid_broker_event_origin",
            )
        return

    if origin is _BrokerCallOrigin.CANCEL:
        if len(broker_events) != 1:
            raise AdapterContractError(
                "broker cancel() must report exactly one cancellation.",
                code="invalid_broker_event_origin",
            )
        event = broker_events[0]
        if (
            not isinstance(event, OrderCancelledEvent)
            or cancelled_order_id is None
            or event.order.id != cancelled_order_id
        ):
            raise AdapterContractError(
                "broker cancel() must report a cancellation for exactly "
                "the requested order.",
                code="invalid_broker_event_origin",
            )
        return

    for event in broker_events:
        if isinstance(event, OrderCancelledEvent):
            raise AdapterContractError(
                "broker process() cannot cancel an order; only the engine "
                "requests cancellation.",
                code="invalid_broker_event_origin",
            )
        if (
            isinstance(event, OrderExpiredEvent)
            and event.order.time_in_force is not TimeInForce.DAY
        ):
            raise AdapterContractError(
                "broker process() expired an order that carries no DAY "
                "time-in-force instruction.",
                code="invalid_broker_event_origin",
            )


def _require_stable_order_identity(previous: Order, current: Order) -> None:
    """Reject a broker order whose immutable identity fields changed."""
    if any(
        getattr(previous, field_name) != getattr(current, field_name)
        for field_name in _STABLE_ORDER_FIELDS
    ):
        raise AdapterContractError(
            "broker event changed an immutable order identity field.",
            code="invalid_broker_order_state",
        )


def _validate_calendar_precision(calendar: np.ndarray) -> None:
    """Reject fixed clocks that Python ``datetime`` cannot represent."""
    microseconds = calendar.astype("datetime64[us]")
    round_tripped = microseconds.astype("datetime64[ns]")
    if not np.array_equal(calendar, round_tripped):
        raise DataValidationError(
            "calendar contains nanosecond timestamps that cannot be "
            "represented without precision loss.",
            code="timestamp_precision_loss",
        )


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
