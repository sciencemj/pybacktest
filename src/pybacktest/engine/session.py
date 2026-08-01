"""One stepable, deterministic simulation state machine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol

import numpy as np

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
from pybacktest.domain.identifiers import OrderId, RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.market import BarView, MarketSlice
from pybacktest.domain.money import Money
from pybacktest.domain.orders import (
    CancelOrderIntent,
    Order,
    OrderStatus,
)
from pybacktest.domain.portfolio import PortfolioSnapshot
from pybacktest.engine._boundary import _FixedDatasetSessionBoundary
from pybacktest.engine._broker_guard import (
    _BrokerCallOrigin,
    _claim_fresh_broker,
    _engine_active_orders,
    _require_broker_state_agreement,
    _snapshot_broker_events,
    _validate_active_orders,
    _validate_broker_events,
)
from pybacktest.engine._identity import _FillIdFacade, _RunIdSequence
from pybacktest.engine._loading import _load_dataset
from pybacktest.engine._manifest import _manifest
from pybacktest.engine._time import (
    _as_datetime,
    _validate_calendar_precision,
)
from pybacktest.engine.accounting import PortfolioLedger
from pybacktest.engine.recorder import RunRecorder
from pybacktest.ports.broker import (
    Broker,
    BrokerEvent,
    BrokerRunContext,
    OrderCancelledEvent,
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
)
from pybacktest.strategy.intents import OrderIntent


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
            raise ConfigurationError("observation timestamp must be np.datetime64.")
        normalized = self.timestamp.astype("datetime64[ns]")
        if np.isnat(normalized):
            raise ConfigurationError("observation timestamp cannot be NaT.")
        if (
            not isinstance(self.market, MarketSlice)
            or self.market.timestamp != normalized
        ):
            raise ConfigurationError("observation market must match its timestamp.")
        if not isinstance(self.portfolio, PortfolioSnapshot):
            raise ConfigurationError(
                "observation portfolio must be a PortfolioSnapshot."
            )
        if self.portfolio.timestamp != _as_datetime(normalized):
            raise ConfigurationError("observation portfolio must match its timestamp.")
        active_orders = tuple(self.active_orders)
        if not all(isinstance(order, Order) for order in active_orders):
            raise ConfigurationError(
                "observation active_orders must contain Order values."
            )
        if (
            not isinstance(self.features, FeatureView)
            or self.features.timestamp != normalized
        ):
            raise ConfigurationError("observation features must match its timestamp.")
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
            raise ConfigurationError("step observation must be Observation or None.")
        events = tuple(self.events)
        if not all(isinstance(event, EngineEvent) for event in events):
            raise ConfigurationError("step events must contain EngineEvent values.")
        if type(self.done) is not bool:
            raise ConfigurationError("step done must be a bool.")
        if self.done != (self.observation is None):
            raise ConfigurationError("done steps cannot expose a next observation.")
        object.__setattr__(self, "events", events)


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
            raise ConfigurationError("simulation must be a SimulationRequest.")
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
        self._session_boundary: _FixedDatasetSessionBoundary | None = None
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
            raise DataValidationError("calendar contains no tradable timestamps.")
        _validate_calendar_precision(calendar)

        feature_set = FeatureExecutor().execute(
            self._feature_plan,
            dataset,
        )
        id_sequence = _RunIdSequence(self._run_id)
        session_boundary = _FixedDatasetSessionBoundary(dataset, calendar)
        broker = self._broker_factory.create(
            BrokerRunContext(
                instruments=dataset.instruments,
                session_boundary=session_boundary,
                fill_ids=_FillIdFacade(self._run_id),
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
        self._session_boundary = session_boundary
        self._ledger = ledger
        self._recorder = recorder
        self._submitted_orders = {}
        self._fill_ordinal = 0
        self._index = 0
        self._reset = True
        active_orders = _validate_active_orders(
            broker.active_orders,
            instruments=dataset.instruments,
        )
        _require_broker_state_agreement({}, active_orders)

        market = self._market_slice(0)
        timestamp = _as_datetime(market.timestamp)
        recorder.record_market_timestamp(timestamp)
        initial_marks, initial_mark_state = self._candidate_valuation_marks(market, 0)
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
        self._session_boundary = None
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
            next_mark_state: dict[InstrumentId, tuple[int, Money]] | None = None
            if has_next:
                next_market = self._market_slice(next_index)
                next_marks, next_mark_state = self._candidate_valuation_marks(
                    next_market,
                    next_index,
                )
                self._require_held_position_marks(next_marks)
            for intent in validated:
                current_events.extend(self._apply_intent(intent, has_next=has_next))

            self._observation = None
            self._context_created = False
            if not has_next:
                self._done = True
                return StepResult(
                    observation=None,
                    events=tuple(current_events),
                    done=True,
                )

            if next_market is None or next_marks is None or next_mark_state is None:
                raise SessionStateError("next market preparation is incomplete.")
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
        active_order_ids = {order.id for order in observation.active_orders}
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
            raise SessionStateError("failed session cannot produce a result.")
        if not self._done:
            raise SessionStateError("session must be complete before result().")
        if self._result is None:
            self._result = self._required_recorder().finalize()
        return self._result

    def _require_active(self) -> None:
        if not self._reset:
            raise SessionStateError("session must be reset before it can advance.")
        if self._failed:
            raise SessionStateError("failed session cannot be advanced.")
        if self._done:
            raise SessionStateError("completed session cannot advance.")
        if self._observation is None:
            raise SessionStateError("current observation was already consumed.")

    def _apply_intent(
        self,
        intent: OrderIntent,
        *,
        has_next: bool,
    ) -> tuple[EngineEvent, ...]:
        timestamp = _as_datetime(self._required_calendar()[self._index])
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
            active_orders=_engine_active_orders(self._submitted_orders),
        )
        sized = self._order_sizer.size(intent, context)
        if not isinstance(
            sized,
            (Order, OrderRejected, CancelOrderIntent),
        ):
            raise AdapterContractError(
                "order sizer must return Order, OrderRejected, or CancelOrderIntent.",
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
        instrument = self._required_dataset().instruments[sized.instrument]
        if decision.original_quantity != sized.quantity or not _is_lot_aligned(
            decision.final_quantity.value,
            instrument.lot_size,
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
                "requested_quantity": str(decision.original_quantity.value),
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
        # Validation order is the trust boundary: nothing below records or
        # mutates engine-owned state until every adapter view agrees.
        events = _snapshot_broker_events(broker_events)
        predicted = _validate_broker_events(
            events,
            submitted_orders=self._submitted_orders,
            fill_ordinal=self._fill_ordinal,
            fill_ids=self._required_ids(),
            at=at,
            origin=origin,
            cancelled_order_id=cancelled_order_id,
            session_boundary=self._session_boundary,
        )
        active_orders = _validate_active_orders(
            self._required_broker().active_orders,
            instruments=self._required_dataset().instruments,
        )
        _require_broker_state_agreement(predicted, active_orders)
        return self._record_broker_events(events)

    def _record_broker_events(
        self,
        broker_events: Sequence[BrokerEvent],
    ) -> tuple[EngineEvent, ...]:
        emitted: list[EngineEvent] = []
        recorder = self._required_recorder()
        for broker_event in broker_events:
            self._submitted_orders[broker_event.order.id] = broker_event.order
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
                            "quantity": str(broker_event.fill.quantity.value),
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
                    Decimal("0") if position is None else position.quantity.value
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
            position = int(np.searchsorted(series.timestamps, timestamp))
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
                    dataset.instruments[instrument_id].quote_currency,
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
                for instrument_id, (observed_index, price) in (candidate_state.items())
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
            active_orders=_engine_active_orders(self._submitted_orders),
            features=feature_set.view(market.timestamp),
        )

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
            "" if sized.limit_price is None else str(sized.limit_price.amount)
        ),
    }


def _is_lot_aligned(value: Decimal, lot_size: Decimal) -> bool:
    try:
        value_numerator, value_denominator = value.as_integer_ratio()
        lot_numerator, lot_denominator = lot_size.as_integer_ratio()
    except (AttributeError, OverflowError, ValueError):
        return False
    if lot_numerator <= 0:
        return False
    return (value_numerator * lot_denominator) % (
        value_denominator * lot_numerator
    ) == 0


__all__ = [
    "BrokerFactory",
    "Observation",
    "SessionStateError",
    "SimulationSession",
    "StepResult",
]
