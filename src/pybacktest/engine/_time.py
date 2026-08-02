"""Loss-aware conversions for the engine's fixed UTC market clock."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from pybacktest.domain.errors import DataValidationError


def _as_datetime(timestamp: np.datetime64) -> datetime:
    """Convert a representable NumPy timestamp to an aware UTC datetime."""
    normalized = timestamp.astype("datetime64[us]")
    value = normalized.item()
    if not isinstance(value, datetime):
        raise DataValidationError("market timestamp cannot be converted to datetime.")
    return value.replace(tzinfo=UTC)


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
    """Convert an aware datetime to the engine's nanosecond UTC scalar."""
    normalized = value.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(normalized, "ns")
