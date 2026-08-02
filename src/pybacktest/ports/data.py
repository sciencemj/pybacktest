"""Port for loading fixed validated market datasets."""

from collections.abc import Sequence
from typing import Protocol

from pybacktest.data.dataset import MarketDataSet
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.time import DateRange, Timeframe


class MarketDataSource(Protocol):
    """Load data without refreshing or mutating an external source."""

    def load(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        """Load a fixed, validated dataset without refreshing external data."""
        raise NotImplementedError
