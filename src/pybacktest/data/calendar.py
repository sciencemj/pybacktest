"""Explicit calendar policies for multi-instrument datasets."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from pybacktest.data.validation import normalize_timestamps
from pybacktest.domain.errors import ConfigurationError, DataValidationError
from pybacktest.domain.instruments import InstrumentId


class CalendarMode(StrEnum):
    """Supported timestamp-set operations for a market calendar."""

    UNION = "union"
    INTERSECTION = "intersection"


@dataclass(frozen=True, slots=True)
class CalendarPolicy:
    """Build trading timestamps while keeping marking staleness explicit."""

    mode: CalendarMode
    max_staleness_bars: int

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CalendarMode):
            raise ConfigurationError(
                "calendar mode must be a CalendarMode."
            )
        if (
            isinstance(self.max_staleness_bars, bool)
            or not isinstance(self.max_staleness_bars, int)
            or self.max_staleness_bars < 0
        ):
            raise ConfigurationError(
                "max_staleness_bars must be a nonnegative integer."
            )

    @classmethod
    def union(
        cls,
        max_staleness_bars: int = 0,
    ) -> "CalendarPolicy":
        return cls(
            mode=CalendarMode.UNION,
            max_staleness_bars=max_staleness_bars,
        )

    @classmethod
    def intersection(cls) -> "CalendarPolicy":
        return cls(
            mode=CalendarMode.INTERSECTION,
            max_staleness_bars=0,
        )

    def build(
        self,
        timestamps: Mapping[
            InstrumentId,
            NDArray[np.datetime64],
        ],
    ) -> NDArray[np.datetime64]:
        """Return a sorted read-only calendar from instrument timestamps."""
        if not timestamps:
            raise DataValidationError(
                "calendar requires at least one instrument."
            )
        normalized: list[NDArray[np.datetime64]] = []
        for instrument_id, values in timestamps.items():
            if not isinstance(instrument_id, InstrumentId):
                raise DataValidationError(
                    "calendar keys must be InstrumentId values."
                )
            normalized.append(normalize_timestamps(values))

        if self.mode is CalendarMode.UNION:
            calendar = np.unique(np.concatenate(normalized))
        else:
            calendar = normalized[0]
            for values in normalized[1:]:
                calendar = np.intersect1d(
                    calendar,
                    values,
                    assume_unique=True,
                )
        result = np.array(
            calendar,
            dtype="datetime64[ns]",
            order="C",
            copy=True,
        )
        result.setflags(write=False)
        return result
