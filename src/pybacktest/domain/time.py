"""Time-window and timeframe value objects."""

from dataclasses import dataclass
from datetime import datetime

from .errors import ConfigurationError


def _require_aware(value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ConfigurationError("DateRange datetimes must be timezone-aware.")


@dataclass(frozen=True, slots=True)
class DateRange:
    """A timezone-aware time window using [start, end) semantics."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        _require_aware(self.start)
        _require_aware(self.end)
        if self.start >= self.end:
            raise ConfigurationError("DateRange requires start to be before end.")

    def contains(self, timestamp: datetime) -> bool:
        _require_aware(timestamp)
        return self.start <= timestamp < self.end


@dataclass(frozen=True, slots=True)
class Timeframe:
    """A positive count of minute or day bars."""

    unit: str
    count: int

    def __post_init__(self) -> None:
        if self.unit not in {"minute", "day"}:
            raise ConfigurationError("Timeframe unit must be 'minute' or 'day'.")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
        ):
            raise ConfigurationError("Timeframe count must be positive.")

    @classmethod
    def minutes(cls, count: int) -> "Timeframe":
        return cls(unit="minute", count=count)

    @classmethod
    def days(cls, count: int) -> "Timeframe":
        return cls(unit="day", count=count)
