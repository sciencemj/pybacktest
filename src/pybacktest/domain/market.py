"""Immutable market values exposed to strategy code."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .errors import DataValidationError
from .instruments import InstrumentId


@dataclass(frozen=True, slots=True)
class BarView:
    """A validated immutable OHLCV bar at one UTC timestamp."""

    timestamp: np.datetime64
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        try:
            timestamp = self.timestamp.astype("datetime64[ns]")
            open_ = float(self.open)
            high = float(self.high)
            low = float(self.low)
            close = float(self.close)
            volume = float(self.volume)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError(
                "BarView requires numeric OHLCV and a valid timestamp."
            ) from exc
        if np.isnat(timestamp):
            raise DataValidationError("BarView timestamp cannot be NaT.")
        values = np.array(
            [open_, high, low, close, volume],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise DataValidationError("BarView OHLCV values must be finite.")
        if any(value <= 0 for value in (open_, high, low, close)):
            raise DataValidationError("BarView prices must be positive.")
        if volume < 0:
            raise DataValidationError("BarView volume must be nonnegative.")
        if high < max(open_, close):
            raise DataValidationError("BarView high must include open and close.")
        if low > min(open_, close):
            raise DataValidationError("BarView low must include open and close.")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "open", open_)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)


@dataclass(frozen=True, slots=True)
class MarketSlice:
    """Bars tradable at exactly one timestamp; stale marks are excluded."""

    timestamp: np.datetime64
    bars: Mapping[InstrumentId, BarView]

    def __post_init__(self) -> None:
        try:
            timestamp = self.timestamp.astype("datetime64[ns]")
            bars = dict(self.bars)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DataValidationError(
                "MarketSlice requires a valid timestamp and bar mapping."
            ) from exc
        if np.isnat(timestamp):
            raise DataValidationError("MarketSlice timestamp cannot be NaT.")
        if not bars:
            raise DataValidationError("MarketSlice requires at least one current bar.")
        for instrument_id, bar in bars.items():
            if not isinstance(instrument_id, InstrumentId):
                raise DataValidationError(
                    "MarketSlice keys must be InstrumentId values."
                )
            if not isinstance(bar, BarView):
                raise DataValidationError("MarketSlice values must be BarView objects.")
            if bar.timestamp != timestamp:
                raise DataValidationError(
                    "MarketSlice bars must match the current timestamp."
                )
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "bars", MappingProxyType(bars))
