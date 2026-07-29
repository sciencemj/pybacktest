"""Immutable result values shared by execution, metrics, and persistence."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from itertools import pairwise
from math import isfinite
from typing import TypeAlias, cast

from pybacktest.domain.errors import PybacktestError
from pybacktest.domain.identifiers import OrderId, RunId
from pybacktest.domain.orders import Fill, Order
from pybacktest.domain.portfolio import PortfolioSnapshot

JSONScalar: TypeAlias = str | int | float | bool | None
FrozenJSON: TypeAlias = (
    JSONScalar | tuple["FrozenJSON", ...] | Mapping[str, "FrozenJSON"]
)

_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_MAX_SEED = 2**63 - 1


class ResultValidationError(PybacktestError, ValueError):
    """Raised when an immutable result value violates its contract."""


class UnknownTradeError(PybacktestError, KeyError):
    """Raised when no recorded event belongs to the requested order."""


@dataclass(frozen=True, slots=True)
class FrozenMapping(Mapping[str, FrozenJSON]):
    """A recursively immutable, defensive JSON mapping."""

    _items: tuple[tuple[str, FrozenJSON], ...] = field(repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FrozenMapping:
        if not isinstance(value, Mapping):
            raise ResultValidationError("value must be a mapping.")
        copied: list[tuple[str, FrozenJSON]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResultValidationError("frozen mapping keys must be strings.")
            copied.append((key, freeze_json(item)))
        return cls(tuple(copied))

    def __getitem__(self, key: str) -> FrozenJSON:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def freeze_json(value: object) -> FrozenJSON:
    """Copy JSON-compatible data into recursively immutable values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ResultValidationError("JSON float values must be finite.")
        return value
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping.from_mapping(
            cast("Mapping[str, object]", value)
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(freeze_json(item) for item in value)
    raise ResultValidationError("value must contain only JSON-compatible data.")


def _token(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _TOKEN_PATTERN.fullmatch(value) is None:
        raise ResultValidationError(
            f"{field_name} must be a stable lowercase token."
        )
    return value


def _decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ResultValidationError(f"{field_name} must be a finite Decimal.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ResultValidationError(
            f"{field_name} must be a finite Decimal."
        ) from error
    if not result.is_finite():
        raise ResultValidationError(f"{field_name} must be a finite Decimal.")
    return result


def _aware(value: object, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ResultValidationError(f"{field_name} must be timezone-aware.")
    return value


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(
        ord(character) < 32 for character in value
    ):
        raise ResultValidationError(f"{field_name} must be non-empty text.")
    return value


def _version(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _VERSION_PATTERN.fullmatch(value) is None:
        raise ResultValidationError(f"{field_name} must be a stable version.")
    return value


def _hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ResultValidationError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )
    return value


def _version_mapping(
    value: object,
    field_name: str,
) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{field_name} must be a mapping.")
    copied: dict[str, str] = {}
    for key, version in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ResultValidationError(
                f"{field_name} keys must be non-empty strings."
            )
        copied[key] = _version(version, f"{field_name}[{key!r}]")
    return FrozenMapping.from_mapping(copied)


@dataclass(frozen=True, slots=True)
class CausalStage:
    """Stable typed stage in an engine decision/execution trace."""

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _token(self.value, "stage"))

    @classmethod
    def of(cls, value: str) -> CausalStage:
        return cls(value)


@dataclass(frozen=True, slots=True)
class EngineEventCode:
    """Stable typed event code independent of display prose."""

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _token(self.value, "event code"))

    @classmethod
    def of(cls, value: str) -> EngineEventCode:
        return cls(value)


@dataclass(frozen=True, slots=True)
class EngineEvent:
    """One ordered, immutable, Parquet-friendly engine event."""

    timestamp: datetime
    sequence: int
    stage: CausalStage
    code: EngineEventCode
    order_id: OrderId | None = None
    details: FrozenMapping = field(
        default_factory=lambda: FrozenMapping.from_mapping({})
    )
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _aware(self.timestamp, "timestamp"))
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ResultValidationError(
                "event sequence must be a nonnegative integer."
            )
        if not isinstance(self.stage, CausalStage):
            raise ResultValidationError("stage must be a CausalStage.")
        if not isinstance(self.code, EngineEventCode):
            raise ResultValidationError("code must be an EngineEventCode.")
        if self.order_id is not None and not isinstance(self.order_id, OrderId):
            raise ResultValidationError("order_id must be an OrderId when provided.")
        if not isinstance(self.details, FrozenMapping):
            object.__setattr__(
                self,
                "details",
                FrozenMapping.from_mapping(self.details),
            )
        if self.message is not None and (
            not isinstance(self.message, str) or not self.message.strip()
        ):
            raise ResultValidationError(
                "message must be a non-empty string when provided."
            )


class MetricName(StrEnum):
    """Names with versioned formulas in a backtest summary."""

    TOTAL_RETURN = "total_return"
    CAGR = "cagr"
    VOLATILITY = "volatility"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    TURNOVER = "turnover"
    WIN_RATE = "win_rate"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"


class MissingPolicy(StrEnum):
    """How unavailable or non-finite metric values are represented."""

    NONE_WITH_WARNING = "none_with_warning"


@dataclass(frozen=True, slots=True)
class WarningCode:
    """Stable typed warning code."""

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _token(self.value, "warning code"))

    @classmethod
    def of(cls, value: str) -> WarningCode:
        return cls(value)


@dataclass(frozen=True, slots=True)
class RunWarning:
    """Deterministic typed warning attached to a run or metric."""

    code: WarningCode
    message: str
    metric: MetricName | None = None
    details: FrozenMapping = field(
        default_factory=lambda: FrozenMapping.from_mapping({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.code, WarningCode):
            raise ResultValidationError("warning code must be a WarningCode.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ResultValidationError("warning message must be non-empty.")
        if self.metric is not None and not isinstance(self.metric, MetricName):
            raise ResultValidationError("warning metric must be a MetricName.")
        if not isinstance(self.details, FrozenMapping):
            object.__setattr__(
                self,
                "details",
                FrozenMapping.from_mapping(self.details),
            )


@dataclass(frozen=True, slots=True)
class MetricMetadata:
    """Versioned formula metadata carried beside every metric value."""

    formula_id: str
    annualization_periods: int
    risk_free_rate: Decimal
    missing_policy: MissingPolicy = MissingPolicy.NONE_WITH_WARNING
    parameters: FrozenMapping = field(
        default_factory=lambda: FrozenMapping.from_mapping({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "formula_id", _token(self.formula_id, "formula_id"))
        if (
            isinstance(self.annualization_periods, bool)
            or not isinstance(self.annualization_periods, int)
            or self.annualization_periods <= 0
        ):
            raise ResultValidationError(
                "annualization_periods must be a positive integer."
            )
        object.__setattr__(
            self,
            "risk_free_rate",
            _decimal(self.risk_free_rate, "risk_free_rate"),
        )
        if not isinstance(self.missing_policy, MissingPolicy):
            raise ResultValidationError(
                "missing_policy must be a MissingPolicy."
            )
        if not isinstance(self.parameters, FrozenMapping):
            object.__setattr__(
                self,
                "parameters",
                FrozenMapping.from_mapping(self.parameters),
            )


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One named optional value and its complete formula metadata."""

    name: MetricName
    value: Decimal | None
    metadata: MetricMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.name, MetricName):
            raise ResultValidationError("metric name must be a MetricName.")
        if self.value is not None:
            object.__setattr__(
                self,
                "value",
                _decimal(self.value, f"{self.name.value} value"),
            )
        if not isinstance(self.metadata, MetricMetadata):
            raise ResultValidationError("metric metadata must be MetricMetadata.")


@dataclass(frozen=True, slots=True)
class SummaryMetrics:
    """Immutable ordered metric results with numeric convenience properties."""

    results: tuple[MetricResult, ...]
    warnings: tuple[RunWarning, ...] = ()

    def __post_init__(self) -> None:
        results = tuple(self.results)
        warnings = tuple(self.warnings)
        if not all(isinstance(item, MetricResult) for item in results):
            raise ResultValidationError("results must contain MetricResult values.")
        if len({item.name for item in results}) != len(results):
            raise ResultValidationError("metric names must be unique.")
        if {item.name for item in results} != set(MetricName):
            raise ResultValidationError("summary must contain every declared metric.")
        if not all(isinstance(item, RunWarning) for item in warnings):
            raise ResultValidationError("warnings must contain RunWarning values.")
        if len(set(warnings)) != len(warnings):
            raise ResultValidationError("warnings must be deduplicated.")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "warnings", warnings)

    def result_for(self, name: MetricName | str) -> MetricResult:
        resolved = name if isinstance(name, MetricName) else MetricName(name)
        for item in self.results:
            if item.name is resolved:
                return item
        raise KeyError(resolved.value)

    def metadata_for(self, name: MetricName | str) -> MetricMetadata:
        return self.result_for(name).metadata

    @property
    def total_return(self) -> Decimal | None:
        return self.result_for(MetricName.TOTAL_RETURN).value

    @property
    def cagr(self) -> Decimal | None:
        return self.result_for(MetricName.CAGR).value

    @property
    def volatility(self) -> Decimal | None:
        return self.result_for(MetricName.VOLATILITY).value

    @property
    def sharpe(self) -> Decimal | None:
        return self.result_for(MetricName.SHARPE).value

    @property
    def sortino(self) -> Decimal | None:
        return self.result_for(MetricName.SORTINO).value

    @property
    def maximum_drawdown(self) -> Decimal | None:
        return self.result_for(MetricName.MAXIMUM_DRAWDOWN).value

    @property
    def turnover(self) -> Decimal | None:
        return self.result_for(MetricName.TURNOVER).value

    @property
    def win_rate(self) -> Decimal | None:
        return self.result_for(MetricName.WIN_RATE).value

    @property
    def gross_exposure(self) -> Decimal | None:
        return self.result_for(MetricName.GROSS_EXPOSURE).value

    @property
    def net_exposure(self) -> Decimal | None:
        return self.result_for(MetricName.NET_EXPOSURE).value


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Caller-supplied run identity and complete execution provenance."""

    run_id: RunId
    library_version: str
    schema_version: str
    canonical_request: FrozenMapping
    strategy_identity: str
    strategy_fingerprint: str
    spec_identity: str
    compiler_identity: str
    dataset_fingerprint: str
    seed: int
    adapter_versions: FrozenMapping
    model_versions: FrozenMapping
    started_at: datetime
    ended_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, RunId):
            raise ResultValidationError("run_id must be a RunId.")
        object.__setattr__(
            self,
            "library_version",
            _version(self.library_version, "library_version"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _version(self.schema_version, "schema_version"),
        )
        if not isinstance(self.canonical_request, Mapping):
            raise ResultValidationError("canonical_request must be a mapping.")
        object.__setattr__(
            self,
            "canonical_request",
            FrozenMapping.from_mapping(self.canonical_request),
        )
        for field_name in (
            "strategy_identity",
            "spec_identity",
            "compiler_identity",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "strategy_fingerprint",
            _hash(self.strategy_fingerprint, "strategy_fingerprint"),
        )
        object.__setattr__(
            self,
            "dataset_fingerprint",
            _hash(self.dataset_fingerprint, "dataset_fingerprint"),
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= _MAX_SEED
        ):
            raise ResultValidationError(
                "seed must be a nonnegative signed 63-bit integer."
            )
        object.__setattr__(
            self,
            "adapter_versions",
            _version_mapping(self.adapter_versions, "adapter_versions"),
        )
        object.__setattr__(
            self,
            "model_versions",
            _version_mapping(self.model_versions, "model_versions"),
        )
        started_at = _aware(self.started_at, "started_at")
        ended_at = _aware(self.ended_at, "ended_at")
        if ended_at < started_at:
            raise ResultValidationError("ended_at cannot precede started_at.")


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """Final size and SHA-256 digest for one artifact file."""

    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or "/" in self.name
            or "\\" in self.name
            or self.name in {".", ".."}
        ):
            raise ResultValidationError("artifact file name must be a safe basename.")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ResultValidationError(
                "artifact file size must be a nonnegative integer."
            )
        object.__setattr__(self, "sha256", _hash(self.sha256, "sha256"))


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Noncyclic manifest containing checksums for payload files only."""

    run_manifest: RunManifest
    artifact_schema_version: str
    artifact_id: str
    replay_fingerprint: str
    files: tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_manifest, RunManifest):
            raise ResultValidationError("run_manifest must be a RunManifest.")
        object.__setattr__(
            self,
            "artifact_schema_version",
            _version(
                self.artifact_schema_version,
                "artifact_schema_version",
            ),
        )
        if (
            not isinstance(self.artifact_id, str)
            or re.fullmatch(r"artifact_[0-9a-f]{64}", self.artifact_id) is None
        ):
            raise ResultValidationError(
                "artifact_id must use canonical deterministic form."
            )
        object.__setattr__(
            self,
            "replay_fingerprint",
            _hash(self.replay_fingerprint, "replay_fingerprint"),
        )
        files = tuple(self.files)
        if not all(isinstance(item, ArtifactFile) for item in files):
            raise ResultValidationError("files must contain ArtifactFile values.")
        names = tuple(item.name for item in files)
        if len(set(names)) != len(names):
            raise ResultValidationError("artifact file names must be unique.")
        if {"manifest.json", "manifest.sha256"} & set(names):
            raise ResultValidationError(
                "embedded manifest files must exclude manifest trust files."
            )
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Final artifact location and immutable all-file integrity view."""

    path: str
    manifest: ArtifactManifest
    manifest_checksum: str
    files: tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ResultValidationError("artifact path must be a non-empty string.")
        if not isinstance(self.manifest, ArtifactManifest):
            raise ResultValidationError("manifest must be an ArtifactManifest.")
        object.__setattr__(
            self,
            "manifest_checksum",
            _hash(self.manifest_checksum, "manifest_checksum"),
        )
        files = tuple(self.files)
        if not all(isinstance(item, ArtifactFile) for item in files):
            raise ResultValidationError("files must contain ArtifactFile values.")
        names = tuple(item.name for item in files)
        if len(set(names)) != len(names):
            raise ResultValidationError("artifact file names must be unique.")
        if set(names) != {
            "manifest.json",
            "manifest.sha256",
            *(item.name for item in self.manifest.files),
        }:
            raise ResultValidationError(
                "ArtifactRef files must include payload and trust files exactly."
            )
        manifest_file = next(
            item for item in files if item.name == "manifest.json"
        )
        if manifest_file.sha256 != self.manifest_checksum:
            raise ResultValidationError(
                "manifest_checksum must match the manifest file."
            )
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Fully validated immutable boundary for one completed engine run."""

    manifest: RunManifest
    summary: SummaryMetrics
    market_timestamps: tuple[datetime, ...]
    snapshots: tuple[PortfolioSnapshot, ...]
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    events: tuple[EngineEvent, ...]
    warnings: tuple[RunWarning, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RunManifest):
            raise ResultValidationError("manifest must be a RunManifest.")
        if not isinstance(self.summary, SummaryMetrics):
            raise ResultValidationError("summary must be SummaryMetrics.")
        market_timestamps = tuple(self.market_timestamps)
        snapshots = tuple(self.snapshots)
        orders = tuple(self.orders)
        fills = tuple(self.fills)
        events = tuple(self.events)
        warnings = tuple(self.warnings)
        if not all(isinstance(item, datetime) for item in market_timestamps):
            raise ResultValidationError(
                "market_timestamps must contain datetime values."
            )
        for timestamp in market_timestamps:
            _aware(timestamp, "market timestamp")
        if any(
            current <= previous
            for previous, current in pairwise(market_timestamps)
        ):
            raise ResultValidationError(
                "market timestamps must be strictly increasing."
            )
        if not all(isinstance(item, PortfolioSnapshot) for item in snapshots):
            raise ResultValidationError(
                "snapshots must contain PortfolioSnapshot values."
            )
        if len(snapshots) != len(market_timestamps) or any(
            snapshot.timestamp != timestamp
            for snapshot, timestamp in zip(
                snapshots,
                market_timestamps,
                strict=True,
            )
        ):
            raise ResultValidationError(
                "snapshots must correspond exactly to market timestamps."
            )
        if not all(isinstance(item, Order) for item in orders):
            raise ResultValidationError("orders must contain Order values.")
        order_by_id = {item.id: item for item in orders}
        if len(order_by_id) != len(orders):
            raise ResultValidationError("final order IDs must be unique.")
        if not all(isinstance(item, Fill) for item in fills):
            raise ResultValidationError("fills must contain Fill values.")
        if len({item.id for item in fills}) != len(fills):
            raise ResultValidationError("fill IDs must be unique.")
        if any(
            current.timestamp < previous.timestamp
            for previous, current in pairwise(fills)
        ):
            raise ResultValidationError("fills must be timestamp-monotonic.")
        filled_by_order: dict[OrderId, Decimal] = {}
        for fill in fills:
            order = order_by_id.get(fill.order_id)
            if (
                order is None
                or fill.timestamp not in market_timestamps
                or fill.instrument != order.instrument
                or fill.side is not order.side
                or fill.price.currency != order.quote_currency
                or fill.fee.currency != order.quote_currency
                or fill.timestamp < order.active_from
            ):
                raise ResultValidationError(
                    "fill identity and order relationship is inconsistent."
                )
            filled_by_order[order.id] = (
                filled_by_order.get(order.id, Decimal("0"))
                + fill.quantity.value
            )
        for order in orders:
            if filled_by_order.get(order.id, Decimal("0")) != (
                order.filled_quantity.value
            ):
                raise ResultValidationError(
                    "final order filled quantity is inconsistent with fills."
                )
        if not all(isinstance(item, EngineEvent) for item in events):
            raise ResultValidationError("events must contain EngineEvent values.")
        for expected_sequence, event in enumerate(events):
            if event.sequence != expected_sequence:
                raise ResultValidationError(
                    "event sequences must be contiguous from zero."
                )
            if (
                expected_sequence
                and event.timestamp < events[expected_sequence - 1].timestamp
            ):
                raise ResultValidationError(
                    "events must be timestamp-monotonic."
                )
            if event.order_id is not None and event.order_id not in order_by_id:
                raise ResultValidationError(
                    "event order_id must reference a final order."
                )
        if not all(isinstance(item, RunWarning) for item in warnings):
            raise ResultValidationError(
                "warnings must contain RunWarning values."
            )
        if warnings != self.summary.warnings:
            raise ResultValidationError(
                "result warnings must match summary warnings exactly."
            )
        object.__setattr__(self, "market_timestamps", market_timestamps)
        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "fills", fills)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "warnings", warnings)

    @property
    def run_id(self) -> RunId:
        return self.manifest.run_id

    def explain_trade(self, order_id: OrderId) -> TradeExplanation:
        from .explain import build_trade_explanation

        return build_trade_explanation(self.events, order_id)

    def replay_fingerprint(self) -> str:
        from .serialization import calculate_replay_fingerprint

        return calculate_replay_fingerprint(self)


@dataclass(frozen=True, slots=True)
class TradeExplanation:
    """The exact recorded causal events for one order."""

    order_id: OrderId
    entries: tuple[EngineEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, OrderId):
            raise ResultValidationError("order_id must be an OrderId.")
        entries = tuple(self.entries)
        if not entries or not all(
            isinstance(item, EngineEvent) and item.order_id == self.order_id
            for item in entries
        ):
            raise ResultValidationError(
                "trade explanation entries must match the requested order."
            )
        object.__setattr__(self, "entries", entries)


__all__ = [
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactRef",
    "BacktestResult",
    "CausalStage",
    "EngineEvent",
    "EngineEventCode",
    "FrozenMapping",
    "MetricMetadata",
    "MetricName",
    "MetricResult",
    "MissingPolicy",
    "ResultValidationError",
    "RunManifest",
    "RunWarning",
    "SummaryMetrics",
    "TradeExplanation",
    "UnknownTradeError",
    "WarningCode",
    "freeze_json",
]
