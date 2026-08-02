"""Normalization and validation for canonical columnar market data."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from pybacktest.domain.errors import DataValidationError
from pybacktest.domain.instruments import Instrument, InstrumentId

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")
_Scalar = TypeVar("_Scalar", bound=np.generic)


def normalize_timestamps(values: object) -> NDArray[np.datetime64]:
    """Return a copied one-dimensional UTC ``datetime64[ns]`` array."""
    source = np.asarray(values)
    if source.ndim != 1:
        raise DataValidationError("timestamps must be a one-dimensional array.")
    if source.size == 0:
        raise DataValidationError("BarSeries requires at least one timestamp.")

    try:
        if source.dtype.kind == "M":
            normalized = np.array(
                source,
                dtype="datetime64[ns]",
                order="C",
                copy=True,
            )
        elif source.dtype.kind == "O":
            normalized = _normalize_object_timestamps(source)
        else:
            normalized = np.array(
                source,
                dtype="datetime64[ns]",
                order="C",
                copy=True,
            )
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataValidationError(
            "timestamps must contain valid UTC datetime values."
        ) from exc

    if np.isnat(normalized).any():
        raise DataValidationError("timestamps cannot contain NaT.")
    if normalized.size > 1:
        differences = np.diff(normalized)
        if np.any(differences == np.timedelta64(0, "ns")):
            raise DataValidationError("timestamps cannot contain duplicates.")
        if np.any(differences < np.timedelta64(0, "ns")):
            raise DataValidationError(
                "timestamps must be strictly increasing."
            )
    return freeze_array(normalized)


def normalize_ohlcv(
    *,
    open_: object,
    high: object,
    low: object,
    close: object,
    volume: object,
    expected_length: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return copied finite float64 OHLCV arrays after invariant checks."""
    arrays = (
        _normalize_float_array("open", open_),
        _normalize_float_array("high", high),
        _normalize_float_array("low", low),
        _normalize_float_array("close", close),
        _normalize_float_array("volume", volume),
    )
    if any(len(values) != expected_length for values in arrays):
        raise DataValidationError(
            "timestamps and OHLCV columns must have equal length."
        )

    open_values, high_values, low_values, close_values, volume_values = arrays
    if any(
        np.any(values <= 0)
        for values in (open_values, high_values, low_values, close_values)
    ):
        raise DataValidationError("OHLC prices must be positive.")
    if np.any(volume_values < 0):
        raise DataValidationError("volume must be nonnegative.")
    if np.any(high_values < np.maximum(open_values, close_values)):
        raise DataValidationError(
            "high must be greater than or equal to open and close."
        )
    if np.any(low_values > np.minimum(open_values, close_values)):
        raise DataValidationError(
            "low must be less than or equal to open and close."
        )

    return (
        freeze_array(open_values),
        freeze_array(high_values),
        freeze_array(low_values),
        freeze_array(close_values),
        freeze_array(volume_values),
    )


def copy_instrument_mapping(
    instruments: Mapping[InstrumentId, Instrument],
) -> dict[InstrumentId, Instrument]:
    """Copy validated instrument metadata keyed by its canonical identity."""
    try:
        copied = dict(instruments)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "instruments must be a mapping of metadata."
        ) from exc
    if not copied:
        raise DataValidationError(
            "instrument metadata requires at least one instrument."
        )
    for instrument_id, instrument in copied.items():
        if not isinstance(instrument_id, InstrumentId):
            raise DataValidationError(
                "instrument metadata keys must be InstrumentId values."
            )
        if (
            not isinstance(instrument, Instrument)
            or instrument.id != instrument_id
        ):
            raise DataValidationError(
                "instrument metadata must be stored under its matching id."
            )
    return copied


def _normalize_object_timestamps(
    values: NDArray[np.object_],
) -> NDArray[np.datetime64]:
    normalized: list[np.datetime64] = []
    for value in values:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise DataValidationError(
                    "datetime timestamps must be timezone-aware."
                )
            utc_value = value.astimezone(UTC).replace(tzinfo=None)
            normalized.append(np.datetime64(utc_value, "ns"))
        else:
            normalized.append(np.datetime64(value, "ns"))
    return np.array(normalized, dtype="datetime64[ns]", order="C", copy=True)


def _normalize_float_array(
    name: str,
    values: object,
) -> NDArray[np.float64]:
    try:
        normalized = np.array(
            values,
            dtype=np.float64,
            order="C",
            copy=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataValidationError(
            f"{name} must contain numeric values."
        ) from exc
    if normalized.ndim != 1:
        raise DataValidationError(
            f"{name} must be a one-dimensional array."
        )
    if not np.isfinite(normalized).all():
        raise DataValidationError("OHLCV values must be finite.")
    return normalized


def freeze_array(
    values: NDArray[_Scalar],
) -> NDArray[_Scalar]:
    """Copy an array onto immutable bytes-backed storage."""
    immutable_bytes = values.tobytes(order="C")
    frozen = np.frombuffer(immutable_bytes, dtype=values.dtype).reshape(
        values.shape
    )
    frozen.setflags(write=False)
    return frozen
