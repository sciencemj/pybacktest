"""Run-scoped chronological recording for Task 10 and artifact results."""

from __future__ import annotations

from datetime import datetime

from pybacktest.domain.errors import PybacktestError
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.orders import Fill, Order, OrderStatus
from pybacktest.domain.portfolio import PortfolioSnapshot
from pybacktest.results._decimal import ExactDecimalError, exact_add, exact_sum
from pybacktest.results.metrics import MetricsConfig, calculate_metrics
from pybacktest.results.models import (
    BacktestResult,
    CausalStage,
    EngineEvent,
    EngineEventCode,
    FrozenMapping,
    ResultValidationError,
    RunManifest,
    RunWarning,
    SummaryMetrics,
)


class RecorderStateError(PybacktestError, RuntimeError):
    """Raised for mutation or finalization after a recorder is finalized."""


_ALLOWED_ORDER_TRANSITIONS = {
    OrderStatus.PENDING: {
        OrderStatus.PENDING,
        OrderStatus.ACCEPTED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.ACCEPTED: {
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.FILLED: {OrderStatus.FILLED},
    OrderStatus.CANCELLED: {OrderStatus.CANCELLED},
    OrderStatus.REJECTED: {OrderStatus.REJECTED},
}


class RunRecorder:
    """The sole mutable owner of one ordered result stream."""

    def __init__(
        self,
        *,
        manifest: RunManifest,
        metrics_config: MetricsConfig,
    ) -> None:
        if not isinstance(manifest, RunManifest):
            raise ResultValidationError("manifest must be a RunManifest.")
        if not isinstance(metrics_config, MetricsConfig):
            raise ResultValidationError(
                "metrics_config must be a MetricsConfig."
            )
        self._manifest = manifest
        self._metrics_config = metrics_config
        self._market_timestamps: list[datetime] = []
        self._snapshots: list[PortfolioSnapshot] = []
        self._snapshot_timestamps: set[datetime] = set()
        self._orders: dict[OrderId, Order] = {}
        self._fills: list[Fill] = []
        self._fill_ids: set[FillId] = set()
        self._events: list[EngineEvent] = []
        self._traces: dict[OrderId, list[EngineEvent]] = {}
        self._warnings: list[RunWarning] = []
        self._warning_set: set[RunWarning] = set()
        self._finalized = False

    def _require_open(self) -> None:
        if self._finalized:
            raise RecorderStateError("run recorder is already finalized.")

    def record_market_timestamp(self, timestamp: datetime) -> None:
        """Append one strictly increasing aware market timestamp."""
        self._require_open()
        if (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ResultValidationError(
                "market timestamp must be timezone-aware."
            )
        if (
            self._market_timestamps
            and timestamp <= self._market_timestamps[-1]
        ):
            raise ResultValidationError(
                "market timestamp cannot repeat or regress."
            )
        if self._market_timestamps and (
            self._market_timestamps[-1] not in self._snapshot_timestamps
        ):
            raise ResultValidationError(
                "record the current market snapshot before advancing."
            )
        self._market_timestamps.append(timestamp)

    def record_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Record exactly one complete snapshot for the current market."""
        self._require_open()
        if not isinstance(snapshot, PortfolioSnapshot):
            raise ResultValidationError(
                "snapshot must be a PortfolioSnapshot."
            )
        if not self._market_timestamps:
            raise ResultValidationError(
                "a market timestamp must precede its snapshot."
            )
        current = self._market_timestamps[-1]
        if snapshot.timestamp != current:
            raise ResultValidationError(
                "snapshot timestamp must match the current market."
            )
        if current in self._snapshot_timestamps:
            raise ResultValidationError(
                "current market already has a snapshot."
            )
        self._snapshots.append(snapshot)
        self._snapshot_timestamps.add(current)

    def record_order(self, order: Order) -> None:
        """Insert or replace one final-order slot through a valid transition."""
        self._require_open()
        if not isinstance(order, Order):
            raise ResultValidationError("order must be an Order.")
        if not self._market_timestamps:
            raise ResultValidationError(
                "a market timestamp must precede an order."
            )
        if order.submitted_at > self._market_timestamps[-1]:
            raise ResultValidationError(
                "order submission cannot follow the current market."
            )
        previous = self._orders.get(order.id)
        if previous is not None:
            stable_fields = (
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
            if any(
                getattr(previous, field_name) != getattr(order, field_name)
                for field_name in stable_fields
            ):
                raise ResultValidationError(
                    "order replacement changed immutable identity fields."
                )
            if (
                order.status not in _ALLOWED_ORDER_TRANSITIONS[previous.status]
                or order.filled_quantity.value
                < previous.filled_quantity.value
            ):
                raise ResultValidationError(
                    "order replacement regressed its lifecycle."
                )
        self._orders[order.id] = order

    def record_fill(self, fill: Fill) -> None:
        """Append one fill before the snapshot for its current market."""
        self._require_open()
        if not isinstance(fill, Fill):
            raise ResultValidationError("fill must be a Fill.")
        if fill.id in self._fill_ids:
            raise ResultValidationError("fill ID is already recorded.")
        if not self._market_timestamps:
            raise ResultValidationError(
                "a market timestamp must precede a fill."
            )
        current = self._market_timestamps[-1]
        if fill.timestamp != current:
            raise ResultValidationError(
                "fill timestamp must match the current market."
            )
        if current in self._snapshot_timestamps:
            raise ResultValidationError(
                "fills cannot follow the current market snapshot."
            )
        if self._fills and fill.timestamp < self._fills[-1].timestamp:
            raise ResultValidationError("fill timestamp cannot regress.")
        order = self._orders.get(fill.order_id)
        if (
            order is None
            or fill.instrument != order.instrument
            or fill.side is not order.side
            or fill.price.currency != order.quote_currency
            or fill.fee.currency != order.quote_currency
            or fill.timestamp < order.active_from
        ):
            raise ResultValidationError(
                "fill identity and order relationship is inconsistent."
            )
        try:
            recorded_quantity = exact_sum(
                item.quantity.value
                for item in self._fills
                if item.order_id == fill.order_id
            )
            total_quantity = exact_add(
                recorded_quantity,
                fill.quantity.value,
            )
        except ExactDecimalError as error:
            raise ResultValidationError(
                "fill aggregation exceeds the supported numeric range."
            ) from error
        if total_quantity > order.quantity.value:
            raise ResultValidationError(
                "fill quantity exceeds the recorded order quantity."
            )
        self._fills.append(fill)
        self._fill_ids.add(fill.id)

    def record_event(self, event: EngineEvent) -> None:
        """Append a prebuilt event only when it owns the next sequence."""
        self._require_open()
        if not isinstance(event, EngineEvent):
            raise ResultValidationError("event must be an EngineEvent.")
        if event.sequence != len(self._events):
            raise ResultValidationError(
                "event must carry the next deterministic sequence."
            )
        if self._events and event.timestamp < self._events[-1].timestamp:
            raise ResultValidationError("event timestamp cannot regress.")
        if event.order_id is not None and event.order_id not in self._orders:
            raise ResultValidationError(
                "event order_id must reference a recorded order."
            )
        self._events.append(event)
        if event.order_id is not None:
            self._traces.setdefault(event.order_id, []).append(event)

    def emit_event(
        self,
        *,
        timestamp: datetime,
        stage: CausalStage,
        code: EngineEventCode,
        order_id: OrderId | None = None,
        details: FrozenMapping | dict[str, object] | None = None,
        message: str | None = None,
    ) -> EngineEvent:
        """Construct and append the next event through the same validation path."""
        self._require_open()
        event = EngineEvent(
            timestamp=timestamp,
            sequence=len(self._events),
            stage=stage,
            code=code,
            order_id=order_id,
            details=FrozenMapping.from_mapping(details or {}),
            message=message,
        )
        self.record_event(event)
        return event

    def record_warning(self, warning: RunWarning) -> None:
        """Append one typed warning with deterministic first-seen deduplication."""
        self._require_open()
        if not isinstance(warning, RunWarning):
            raise ResultValidationError("warning must be a RunWarning.")
        if warning not in self._warning_set:
            self._warning_set.add(warning)
            self._warnings.append(warning)

    def events_for(self, order_id: OrderId) -> tuple[EngineEvent, ...]:
        """Return the recorder's current immutable trace index entry."""
        if not isinstance(order_id, OrderId):
            raise ResultValidationError("order_id must be an OrderId.")
        return tuple(self._traces.get(order_id, ()))

    def finalize(
        self,
        summary: SummaryMetrics | None = None,
    ) -> BacktestResult:
        """Build the immutable result once, optionally using supplied metrics."""
        self._require_open()
        if self._market_timestamps and (
            self._market_timestamps[-1] not in self._snapshot_timestamps
        ):
            raise ResultValidationError(
                "the final market requires a snapshot before finalization."
            )
        if summary is None:
            summary = calculate_metrics(
                equity=tuple(
                    snapshot.equity.amount
                    for snapshot in self._snapshots
                ),
                fills=tuple(self._fills),
                snapshots=tuple(self._snapshots),
                config=self._metrics_config,
            )
        elif not isinstance(summary, SummaryMetrics):
            raise ResultValidationError(
                "summary must be SummaryMetrics when provided."
            )
        combined_warnings = list(summary.warnings)
        seen = set(combined_warnings)
        for warning in self._warnings:
            if warning not in seen:
                seen.add(warning)
                combined_warnings.append(warning)
        if tuple(combined_warnings) != summary.warnings:
            summary = SummaryMetrics(
                results=summary.results,
                warnings=tuple(combined_warnings),
            )
        result = BacktestResult(
            manifest=self._manifest,
            summary=summary,
            market_timestamps=tuple(self._market_timestamps),
            snapshots=tuple(self._snapshots),
            orders=tuple(self._orders.values()),
            fills=tuple(self._fills),
            events=tuple(self._events),
            warnings=summary.warnings,
        )
        self._finalized = True
        return result


__all__ = ["RecorderStateError", "RunRecorder"]
