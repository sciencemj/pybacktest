"""Pandas adapter for fixed in-memory OHLCV frames."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pandas as pd

from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.data.validation import copy_instrument_mapping
from pybacktest.domain.errors import (
    ConfigurationError,
    DataValidationError,
)
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.time import DateRange, Timeframe

_COLUMNS = frozenset(
    {"instrument", "open", "high", "low", "close", "volume"}
)
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")


class PandasDataSource:
    """Snapshot and load an exact-schema UTC-aware Pandas frame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        instruments: Mapping[InstrumentId, Instrument],
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise DataValidationError("frame must be a pandas DataFrame.")
        _validate_schema(frame)
        _validate_utc_index(frame.index)
        metadata = copy_instrument_mapping(instruments)
        copied = frame.copy(deep=True)
        copied["instrument"] = _canonical_instrument_values(
            copied["instrument"],
            metadata,
        )
        self._frame = copied
        self._instruments = MappingProxyType(metadata)

    def load(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        """Build a validated snapshot using exact ``[start, end)`` bounds."""
        requested = _validate_universe(universe, self._instruments)
        if not isinstance(period, DateRange):
            raise DataValidationError("period must be a DateRange.")
        if not isinstance(timeframe, Timeframe):
            raise DataValidationError("timeframe must be a Timeframe.")

        start = pd.Timestamp(period.start).tz_convert("UTC")
        end = pd.Timestamp(period.end).tz_convert("UTC")
        requested_values = {str(instrument_id) for instrument_id in requested}
        selected = self._frame.loc[
            (self._frame.index >= start)
            & (self._frame.index < end)
            & self._frame["instrument"].isin(requested_values)
        ]

        loaded: dict[InstrumentId, BarSeries] = {}
        for instrument_id in requested:
            instrument_frame = selected.loc[
                selected["instrument"] == str(instrument_id)
            ]
            if instrument_frame.empty:
                raise DataValidationError(
                    f"requested instrument {instrument_id} is missing "
                    "from the requested period."
                )
            loaded[instrument_id] = BarSeries(
                timestamps=instrument_frame.index.tz_localize(
                    None
                ).to_numpy(dtype="datetime64[ns]"),
                open=instrument_frame["open"].to_numpy(),
                high=instrument_frame["high"].to_numpy(),
                low=instrument_frame["low"].to_numpy(),
                close=instrument_frame["close"].to_numpy(),
                volume=instrument_frame["volume"].to_numpy(),
            )

        return MarketDataSet(
            series=loaded,
            instruments={
                instrument_id: self._instruments[instrument_id]
                for instrument_id in requested
            },
            timeframe=timeframe,
        )


def _validate_schema(frame: pd.DataFrame) -> None:
    if len(frame.columns) != len(_COLUMNS) or set(frame.columns) != _COLUMNS:
        expected = ", ".join(sorted(_COLUMNS))
        raise DataValidationError(
            f"frame columns must be exactly: {expected}."
        )
    for column in _NUMERIC_COLUMNS:
        dtype = frame[column].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            raise DataValidationError(
                f"{column} column must use a numeric dtype."
            )


def _validate_utc_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError(
            "frame index must be a UTC-aware DatetimeIndex."
        )
    if index.tz is None or str(index.tz) not in {"UTC", "Etc/UTC"}:
        raise DataValidationError(
            "frame index must use the UTC timezone."
        )
    if index.hasnans:
        raise DataValidationError(
            "frame index cannot contain NaT timestamps."
        )


def _canonical_instrument_values(
    values: pd.Series,
    instruments: Mapping[InstrumentId, Instrument],
) -> list[str]:
    canonical: list[str] = []
    for value in values:
        if not isinstance(value, (str, InstrumentId)):
            raise DataValidationError(
                "instrument column must contain canonical instrument ids."
            )
        try:
            instrument_id = (
                value if isinstance(value, InstrumentId) else InstrumentId.parse(value)
            )
        except ConfigurationError as exc:
            raise DataValidationError(
                "instrument column contains a malformed instrument id."
            ) from exc
        if instrument_id not in instruments:
            raise DataValidationError(
                f"instrument column contains unknown id {instrument_id}."
            )
        canonical.append(str(instrument_id))
    return canonical


def _validate_universe(
    universe: Sequence[InstrumentId],
    instruments: Mapping[InstrumentId, Instrument],
) -> tuple[InstrumentId, ...]:
    requested = tuple(universe)
    if not requested:
        raise DataValidationError(
            "universe requires at least one instrument."
        )
    if any(
        not isinstance(instrument_id, InstrumentId)
        for instrument_id in requested
    ):
        raise DataValidationError(
            "universe values must be InstrumentId objects."
        )
    if len(set(requested)) != len(requested):
        raise DataValidationError(
            "universe cannot contain duplicate instruments."
        )
    missing_metadata = [
        instrument_id
        for instrument_id in requested
        if instrument_id not in instruments
    ]
    if missing_metadata:
        missing = ", ".join(map(str, missing_metadata))
        raise DataValidationError(
            f"instrument metadata is missing for: {missing}."
        )
    return requested
