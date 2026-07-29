"""Immutable NumPy-backed market datasets."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from pybacktest.data.calendar import CalendarPolicy
from pybacktest.data.fingerprint import compute_dataset_fingerprint
from pybacktest.data.validation import normalize_ohlcv, normalize_timestamps
from pybacktest.domain.errors import ConfigurationError, DataValidationError
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.time import Timeframe


@dataclass(frozen=True, slots=True)
class BarSeries:
    """Validated UTC timestamps and OHLCV columns owned by the dataset."""

    timestamps: NDArray[np.datetime64]
    open: NDArray[np.float64]
    high: NDArray[np.float64]
    low: NDArray[np.float64]
    close: NDArray[np.float64]
    volume: NDArray[np.float64]

    def __post_init__(self) -> None:
        timestamps = normalize_timestamps(self.timestamps)
        open_, high, low, close, volume = normalize_ohlcv(
            open_=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            expected_length=len(timestamps),
        )
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "open", open_)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "close", close)
        object.__setattr__(self, "volume", volume)


@dataclass(frozen=True, slots=True)
class MarketDataSet:
    """A fixed validated universe whose fingerprint covers data and metadata."""

    series: Mapping[InstrumentId, BarSeries]
    instruments: Mapping[InstrumentId, Instrument]
    timeframe: Timeframe
    timestamps: NDArray[np.datetime64] = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            series = dict(self.series)
            instruments = dict(self.instruments)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                "series and instruments must be mappings."
            ) from exc
        if not series:
            raise DataValidationError(
                "MarketDataSet requires at least one instrument."
            )
        if set(series) != set(instruments):
            raise DataValidationError(
                "series and instrument metadata keys must match."
            )
        if not isinstance(self.timeframe, Timeframe):
            raise DataValidationError(
                "dataset timeframe must be a Timeframe."
            )
        for instrument_id, bar_series in series.items():
            if not isinstance(instrument_id, InstrumentId):
                raise DataValidationError(
                    "series keys must be InstrumentId values."
                )
            if not isinstance(bar_series, BarSeries):
                raise DataValidationError(
                    "series values must be BarSeries objects."
                )
            instrument = instruments[instrument_id]
            if (
                not isinstance(instrument, Instrument)
                or instrument.id != instrument_id
            ):
                raise DataValidationError(
                    "instrument metadata must match its mapping key."
                )

        timestamps = CalendarPolicy.union().build(
            {
                instrument_id: bar_series.timestamps
                for instrument_id, bar_series in series.items()
            }
        )
        fingerprint = compute_dataset_fingerprint(
            series,
            instruments,
            self.timeframe,
        )
        object.__setattr__(
            self,
            "series",
            MappingProxyType(series),
        )
        object.__setattr__(
            self,
            "instruments",
            MappingProxyType(instruments),
        )
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "fingerprint", fingerprint)

    def prefix(self, count: int) -> "MarketDataSet":
        """Return bars observed within the first ``count`` union timestamps."""
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or count > len(self.timestamps)
        ):
            raise ConfigurationError(
                "dataset prefix count must be within the union clock."
            )
        if count == len(self.timestamps):
            return self

        boundary = self.timestamps[count - 1]
        sliced: dict[InstrumentId, BarSeries] = {}
        for instrument_id, series in self.series.items():
            stop = int(
                np.searchsorted(series.timestamps, boundary, side="right")
            )
            if stop == 0:
                raise ConfigurationError(
                    "dataset prefix would leave an instrument empty."
                )
            sliced[instrument_id] = BarSeries(
                timestamps=series.timestamps[:stop],
                open=series.open[:stop],
                high=series.high[:stop],
                low=series.low[:stop],
                close=series.close[:stop],
                volume=series.volume[:stop],
            )
        return MarketDataSet(
            series=sliced,
            instruments=self.instruments,
            timeframe=self.timeframe,
        )
