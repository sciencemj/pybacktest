"""Deterministic builders shared by Pybacktest tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np

from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.domain.identifiers import OrderId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from pybacktest.domain.time import Timeframe

BASE_DATETIME = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
BASE_TIMESTAMP = np.datetime64("2024-01-02T14:30:00", "ns")


def timestamp(offset: int = 0) -> np.datetime64:
    """Return a stable nanosecond timestamp offset by whole days."""
    return BASE_TIMESTAMP + np.timedelta64(offset, "D")


def instrument(
    symbol: str = "AAPL",
    *,
    venue: str = "XNAS",
) -> Instrument:
    """Return deterministic instrument metadata."""
    return Instrument(
        id=InstrumentId(venue=venue, symbol=symbol),
        quote_currency="USD",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("1"),
        timezone=ZoneInfo("America/New_York"),
    )


def market_dataset(
    series_by_instrument: dict[Instrument, BarSeries],
) -> MarketDataSet:
    """Return a daily dataset from complete per-instrument series."""
    return MarketDataSet(
        series={
            item.id: series
            for item, series in series_by_instrument.items()
        },
        instruments={
            item.id: item
            for item in series_by_instrument
        },
        timeframe=Timeframe.days(1),
    )


def bar_series(
    *,
    closes: list[float | int],
    timestamps: list[np.datetime64],
) -> BarSeries:
    """Return deterministic flat OHLCV bars at the supplied observations."""
    close = np.asarray(closes, dtype=np.float64)
    return BarSeries(
        timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.full(len(close), 1_000.0),
    )


def one_instrument_dataset(
    *,
    closes: list[float | int],
    timestamps: list[np.datetime64] | None = None,
) -> tuple[MarketDataSet, InstrumentId]:
    """Return a deterministic single-instrument OHLCV dataset."""
    item = instrument()
    observed_at = np.asarray(
        timestamps
        if timestamps is not None
        else [timestamp(index) for index in range(len(closes))],
        dtype="datetime64[ns]",
    )
    series = bar_series(
        closes=closes,
        timestamps=observed_at.tolist(),
    )
    return market_dataset({item: series}), item.id


def order(
    *,
    item: Instrument | None = None,
    offset: int = 0,
) -> Order:
    """Return a stable pending limit order."""
    resolved = item or instrument()
    submitted_at = BASE_DATETIME + timedelta(days=offset)
    return Order.pending(
        id=OrderId.parse(f"order_{offset:032x}"),
        instrument=resolved.id,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Quantity.of("10"),
        quote_currency=resolved.quote_currency,
        limit_price=Money.of("100", resolved.quote_currency),
        time_in_force=TimeInForce.DAY,
        submitted_at=submitted_at,
        active_from=submitted_at,
        reason=DecisionReason.of("test_order"),
    )
