"""Closed, causal feature declarations and vectorized execution."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from pybacktest.data.dataset import MarketDataSet
from pybacktest.data.validation import OHLCV_FIELDS, freeze_array
from pybacktest.domain.errors import ConfigurationError, LookaheadViolation
from pybacktest.domain.instruments import InstrumentId

FeatureOperator = Literal["source", "lag", "sma", "ema"]
FeatureField = Literal["open", "high", "low", "close", "volume"]
MAX_FEATURE_PARAMETER = 1_000_000
"""Largest supported lag, SMA window, or EMA span."""

_FEATURE_VIEW_TOKEN = object()


def _require_feature_parameter(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_FEATURE_PARAMETER
    ):
        raise ConfigurationError(
            f"{label} parameter must be a positive integer "
            f"at most {MAX_FEATURE_PARAMETER}."
        )
    return value


@dataclass(frozen=True, slots=True)
class FeatureNode:
    """One immutable node in a closed feature operator graph."""

    name: str
    operator: FeatureOperator
    inputs: tuple[str, ...]
    instrument: InstrumentId | None
    field: FeatureField | None
    parameter: int | None
    lookback: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError(
                "feature name must be a non-empty string."
            )
        if self.operator not in ("source", "lag", "sma", "ema"):
            raise ConfigurationError(
                "feature operator must belong to the closed graph."
            )
        try:
            inputs = tuple(self.inputs)
        except TypeError as exc:
            raise ConfigurationError(
                "feature inputs must be an immutable name sequence."
            ) from exc
        if not all(
            isinstance(name, str) and name.strip()
            for name in inputs
        ):
            raise ConfigurationError(
                "feature inputs must contain feature names."
            )
        if (
            isinstance(self.lookback, bool)
            or not isinstance(self.lookback, int)
            or self.lookback < 0
        ):
            raise ConfigurationError(
                "feature lookback must be a nonnegative integer."
            )
        if self.operator == "source":
            if (
                inputs
                or not isinstance(self.instrument, InstrumentId)
                or self.field not in OHLCV_FIELDS
                or self.parameter is not None
                or self.lookback != 0
            ):
                raise ConfigurationError(
                    "source feature metadata is invalid."
                )
        else:
            if (
                len(inputs) != 1
                or self.instrument is not None
                or self.field is not None
            ):
                raise ConfigurationError(
                    "derived feature metadata is invalid."
                )
            _require_feature_parameter(
                self.parameter,
                "feature",
            )
        object.__setattr__(self, "inputs", inputs)


@dataclass(frozen=True, slots=True)
class FeaturePlan:
    """An immutable, topologically ordered feature graph."""

    nodes: tuple[FeatureNode, ...]

    def __post_init__(self) -> None:
        try:
            nodes = tuple(self.nodes)
        except TypeError as exc:
            raise ConfigurationError(
                "feature plan nodes must be a sequence."
            ) from exc
        seen: dict[str, FeatureNode] = {}
        for node in nodes:
            if not isinstance(node, FeatureNode):
                raise ConfigurationError(
                    "feature plan must contain FeatureNode values."
                )
            if node.name in seen:
                raise ConfigurationError(
                    f"duplicate feature name in plan: {node.name!r}."
                )
            if any(name not in seen for name in node.inputs):
                raise ConfigurationError(
                    "feature plan nodes must be topologically ordered."
                )
            if node.operator != "source":
                input_node = seen[node.inputs[0]]
                if node.operator == "lag":
                    expected_lookback = (
                        input_node.lookback + cast(int, node.parameter)
                    )
                elif node.operator == "sma":
                    expected_lookback = (
                        input_node.lookback
                        + cast(int, node.parameter)
                        - 1
                    )
                else:
                    expected_lookback = input_node.lookback
                if node.lookback != expected_lookback:
                    raise ConfigurationError(
                        "feature lookback recurrence is invalid."
                    )
            seen[node.name] = node
        object.__setattr__(self, "nodes", nodes)


class FeatureBuilder:
    """Declare a closed feature graph without reading market values."""

    __slots__ = ("_nodes",)

    def __init__(self) -> None:
        self._nodes: dict[str, FeatureNode] = {}

    def source(
        self,
        name: str,
        instrument: InstrumentId,
        field: str,
    ) -> FeatureNode:
        """Declare an OHLCV source column."""
        self._require_available_name(name)
        if not isinstance(instrument, InstrumentId):
            raise ConfigurationError(
                "feature instrument must be an InstrumentId."
            )
        if field not in OHLCV_FIELDS:
            raise ConfigurationError(f"unknown source field: {field!r}.")
        node = FeatureNode(
            name=name,
            operator="source",
            inputs=(),
            instrument=instrument,
            field=field,
            parameter=None,
            lookback=0,
        )
        self._nodes[name] = node
        return node

    def lag(
        self,
        name: str,
        node: FeatureNode,
        periods: int,
    ) -> FeatureNode:
        """Declare a strictly backward-looking lag."""
        self._require_available_name(name)
        source = self._require_owned_node(node)
        self._require_positive_parameter(periods, "lag periods")
        result = FeatureNode(
            name=name,
            operator="lag",
            inputs=(source.name,),
            instrument=None,
            field=None,
            parameter=periods,
            lookback=source.lookback + periods,
        )
        self._nodes[name] = result
        return result

    def sma(
        self,
        name: str,
        node: FeatureNode,
        window: int,
    ) -> FeatureNode:
        """Declare a causal simple moving average."""
        self._require_available_name(name)
        source = self._require_owned_node(node)
        self._require_positive_parameter(window, "SMA window")
        result = FeatureNode(
            name=name,
            operator="sma",
            inputs=(source.name,),
            instrument=None,
            field=None,
            parameter=window,
            lookback=source.lookback + window - 1,
        )
        self._nodes[name] = result
        return result

    def ema(
        self,
        name: str,
        node: FeatureNode,
        span: int,
    ) -> FeatureNode:
        """Declare an adjust-false exponential moving average."""
        self._require_available_name(name)
        source = self._require_owned_node(node)
        self._require_positive_parameter(span, "EMA span")
        result = FeatureNode(
            name=name,
            operator="ema",
            inputs=(source.name,),
            instrument=None,
            field=None,
            parameter=span,
            lookback=source.lookback,
        )
        self._nodes[name] = result
        return result

    def plan(self) -> FeaturePlan:
        """Publish the graph in declaration and topological order."""
        return FeaturePlan(nodes=tuple(self._nodes.values()))

    def _require_available_name(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError(
                "feature name must be a non-empty string."
            )
        if name in self._nodes:
            raise ConfigurationError(f"duplicate feature name: {name!r}.")

    def _require_owned_node(self, node: FeatureNode) -> FeatureNode:
        if (
            not isinstance(node, FeatureNode)
            or self._nodes.get(node.name) is not node
        ):
            raise ConfigurationError(
                "feature node belongs to a different builder."
            )
        return node

    @staticmethod
    def _require_positive_parameter(value: int, label: str) -> None:
        _require_feature_parameter(value, label)


@dataclass(frozen=True, slots=True)
class _ComputedFeature:
    instrument: InstrumentId
    timestamps: NDArray[np.datetime64]
    values: NDArray[np.float64]


class FeatureExecutor:
    """Execute a feature plan causally on instrument observation clocks."""

    def execute(
        self,
        plan: FeaturePlan,
        dataset: MarketDataSet,
    ) -> "FeatureSet":
        """Compute every node, then align published columns to the union clock."""
        if not isinstance(plan, FeaturePlan):
            raise ConfigurationError("plan must be a FeaturePlan.")
        if not isinstance(dataset, MarketDataSet):
            raise ConfigurationError("dataset must be a MarketDataSet.")

        computed: dict[str, _ComputedFeature] = {}
        columns: dict[str, NDArray[np.float64]] = {}
        union_timestamps = dataset.timestamps
        for node in plan.nodes:
            result = self._compute_node(node, computed, dataset)
            computed[node.name] = result
            aligned = np.full(len(union_timestamps), np.nan, dtype=np.float64)
            positions = np.searchsorted(union_timestamps, result.timestamps)
            aligned[positions] = result.values
            columns[node.name] = freeze_array(aligned)
        return FeatureSet(
            timestamps=union_timestamps,
            columns=columns,
        )

    def _compute_node(
        self,
        node: FeatureNode,
        computed: Mapping[str, _ComputedFeature],
        dataset: MarketDataSet,
    ) -> _ComputedFeature:
        if node.operator == "source":
            if node.instrument is None or node.field is None:
                raise ConfigurationError("source feature metadata is incomplete.")
            try:
                series = dataset.series[node.instrument]
            except KeyError as exc:
                raise ConfigurationError(
                    f"feature instrument {node.instrument} is outside the dataset."
                ) from exc
            values = cast(
                NDArray[np.float64],
                getattr(series, node.field),
            )
            return _ComputedFeature(
                instrument=node.instrument,
                timestamps=series.timestamps,
                values=values,
            )

        if len(node.inputs) != 1 or node.parameter is None:
            raise ConfigurationError("derived feature metadata is incomplete.")
        try:
            source = computed[node.inputs[0]]
        except KeyError as exc:
            raise ConfigurationError(
                "feature plan is not topologically ordered."
            ) from exc
        if node.operator == "lag":
            values = _lag(source.values, node.parameter)
        elif node.operator == "sma":
            values = _sma(source.values, node.parameter)
        elif node.operator == "ema":
            values = _ema(source.values, node.parameter)
        else:
            raise ConfigurationError(
                f"unsupported feature operator: {node.operator!r}."
            )
        return _ComputedFeature(
            instrument=source.instrument,
            timestamps=source.timestamps,
            values=values,
        )


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Immutable aligned columns computed for one union market clock."""

    timestamps: NDArray[np.datetime64]
    columns: Mapping[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        timestamps = freeze_array(
            np.array(
                self.timestamps,
                dtype="datetime64[ns]",
                order="C",
                copy=True,
            )
        )
        copied: dict[str, NDArray[np.float64]] = {}
        for name, values in self.columns.items():
            array = np.array(values, dtype=np.float64, order="C", copy=True)
            if array.ndim != 1 or len(array) != len(timestamps):
                raise ConfigurationError(
                    "feature columns must match the feature clock."
                )
            if np.isinf(array).any():
                raise ConfigurationError(
                    "feature columns cannot contain infinite values."
                )
            copied[name] = freeze_array(array)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "columns", MappingProxyType(copied))

    def column(self, name: str) -> NDArray[np.float64]:
        """Return an immutable aligned feature column."""
        try:
            return self.columns[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown feature column: {name!r}."
            ) from exc

    def view(self, timestamp: np.datetime64) -> "FeatureView":
        """Pin a feature view to one timestamp on this feature clock."""
        if not isinstance(timestamp, np.datetime64):
            raise ConfigurationError(
                "feature view timestamp must be a scalar np.datetime64."
            )
        try:
            normalized = timestamp.astype("datetime64[ns]")
        except (AttributeError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                "feature view timestamp must be valid."
            ) from exc
        index = int(np.searchsorted(self.timestamps, normalized))
        if (
            index >= len(self.timestamps)
            or self.timestamps[index] != normalized
        ):
            raise ConfigurationError(
                "feature view timestamp is outside the feature clock."
            )
        return FeatureView._create(
            columns=self.columns,
            index=index,
            timestamp=normalized,
        )


@dataclass(
    frozen=True,
    slots=True,
    init=False,
    repr=False,
    match_args=False,
)
class FeatureView:
    """Opaque causal access capability pinned to one union-clock position."""

    __columns: Mapping[str, NDArray[np.float64]]
    __index: int
    __timestamp: np.datetime64

    def __init__(
        self,
        *,
        _token: object = None,
        _columns: Mapping[str, NDArray[np.float64]] | None = None,
        _index: int = -1,
        _timestamp: np.datetime64 | None = None,
    ) -> None:
        if (
            _token is not _FEATURE_VIEW_TOKEN
            or _columns is None
            or _index < 0
            or _timestamp is None
        ):
            raise TypeError(
                "FeatureView instances are created by FeatureSet.view()."
            )
        object.__setattr__(self, "_FeatureView__columns", _columns)
        object.__setattr__(self, "_FeatureView__index", _index)
        object.__setattr__(self, "_FeatureView__timestamp", _timestamp)

    @classmethod
    def _create(
        cls,
        *,
        columns: Mapping[str, NDArray[np.float64]],
        index: int,
        timestamp: np.datetime64,
    ) -> "FeatureView":
        return cls(
            _token=_FEATURE_VIEW_TOKEN,
            _columns=columns,
            _index=index,
            _timestamp=timestamp,
        )

    @property
    def timestamp(self) -> np.datetime64:
        """The exact normalized union timestamp pinned by this view."""
        return self.__timestamp

    def __repr__(self) -> str:
        return f"FeatureView(timestamp={self.__timestamp!r})"

    def at(self, name: str, offset: int = 0) -> float:
        """Read at or before the pinned clock position."""
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
        ):
            raise ConfigurationError("feature offset must be an integer.")
        if offset > 0:
            raise LookaheadViolation(
                "positive feature offsets would read future data."
            )
        resolved = self.__index + offset
        if resolved < 0:
            raise ConfigurationError(
                "feature offset precedes the available history."
            )
        try:
            return float(self.__columns[name][resolved])
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown feature column: {name!r}."
            ) from exc


def _lag(
    values: NDArray[np.float64],
    periods: int,
) -> NDArray[np.float64]:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if periods < len(values):
        result[periods:] = values[:-periods]
    return result


def _sma(
    values: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if window > len(values):
        return result
    finite = np.isfinite(values)
    scaled = np.divide(
        values,
        window,
        out=np.zeros(len(values), dtype=np.float64),
        where=finite,
    )
    sums = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.cumsum(scaled, dtype=np.float64),
        )
    )
    counts = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(finite, dtype=np.int64),
        )
    )
    rolling_sums = sums[window:] - sums[:-window]
    rolling_counts = counts[window:] - counts[:-window]
    valid = rolling_counts == window
    result[window - 1:][valid] = rolling_sums[valid]
    return result


def _ema(
    values: NDArray[np.float64],
    span: int,
) -> NDArray[np.float64]:
    result = np.full(len(values), np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    previous = 0.0
    seeded = False
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if not seeded:
            previous = float(value)
            seeded = True
        else:
            previous = alpha * float(value) + (1.0 - alpha) * previous
        result[index] = previous
    return result
