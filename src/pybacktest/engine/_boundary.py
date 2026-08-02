"""DAY expiry policy derived only from a run's fixed dataset calendar."""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime

import numpy as np

from pybacktest.data.dataset import MarketDataSet
from pybacktest.domain.errors import AdapterContractError
from pybacktest.domain.orders import Order
from pybacktest.domain.time import TimeframeUnit
from pybacktest.engine._time import _as_datetime


class _FixedDatasetSessionBoundary:
    """Resolve DAY expiry on exact instrument-local calendar membership.

    Daily bars each define one session. Intraday bars share a session while
    their instrument-local calendar date is unchanged. The provider never
    reads wall-clock time and is injected only by the engine.
    """

    def __init__(
        self,
        dataset: MarketDataSet,
        calendar: np.ndarray,
    ) -> None:
        self._daily = dataset.timeframe.unit is TimeframeUnit.DAY
        self._instruments = dataset.instruments
        self._calendar = tuple(_as_datetime(timestamp) for timestamp in calendar)
        self._indices = {
            timestamp: index for index, timestamp in enumerate(self._calendar)
        }
        # Membership is decided on exact datetime64[ns] values before the
        # microsecond conversion. Converting first creates phantom DAY anchors.
        calendar_ns = frozenset(
            calendar.astype("datetime64[ns]").astype(np.int64).tolist()
        )
        self._instrument_calendars = {
            instrument_id: tuple(
                _as_datetime(value)
                for value, exact in zip(
                    dataset.series[instrument_id].timestamps,
                    dataset.series[instrument_id]
                    .timestamps.astype("datetime64[ns]")
                    .astype(np.int64)
                    .tolist(),
                    strict=True,
                )
                if exact in calendar_ns
            )
            for instrument_id in dataset.instruments
        }

    def day_order_expired(
        self,
        order: Order,
        timestamp: datetime,
    ) -> bool:
        """Return whether the order's anchored instrument session ended."""
        if order.instrument not in self._instruments:
            raise AdapterContractError(
                "DAY order instrument is outside the fixed dataset.",
                code="invalid_session_boundary_instrument",
            )
        current_index = self._indices.get(timestamp)
        if current_index is None or order.active_from not in self._indices:
            raise AdapterContractError(
                "DAY expiry timestamp is outside the fixed run calendar.",
                code="invalid_session_boundary_timestamp",
            )
        instrument_calendar = self._instrument_calendars[order.instrument]
        anchor_index = bisect_left(instrument_calendar, order.active_from)
        if anchor_index == len(instrument_calendar):
            return False
        anchor = instrument_calendar[anchor_index]
        if timestamp <= anchor:
            return False
        if self._daily:
            return True
        timezone = self._instruments[order.instrument].timezone
        return (
            timestamp.astimezone(timezone).date() > anchor.astimezone(timezone).date()
        )
