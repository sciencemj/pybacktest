"""Deterministic builders shared by Pybacktest tests."""

from collections.abc import Collection, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np

from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.market import BarView, MarketSlice
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from pybacktest.domain.portfolio import PortfolioSnapshot, Position
from pybacktest.domain.time import Timeframe
from pybacktest.ports.risk import RiskContext

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


def aapl() -> Instrument:
    """Return the canonical deterministic AAPL instrument."""
    return instrument()


def fill(
    instrument_id: InstrumentId,
    side: OrderSide,
    *,
    quantity: object = "10",
    price: object = "100",
    fee: object = "0",
    offset: int = 0,
    currency: str = "USD",
) -> Fill:
    """Return a deterministic fill at a whole-second offset."""
    observed_at = BASE_DATETIME + timedelta(seconds=offset)
    return Fill(
        id=FillId.parse(f"fill_{offset:032x}"),
        order_id=OrderId.parse(f"order_{offset:032x}"),
        instrument=instrument_id,
        side=side,
        quantity=Quantity.of(quantity),
        price=Money.of(price, currency),
        fee=Money.of(fee, currency),
        timestamp=observed_at,
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


def accepted_order(
    *,
    item: Instrument | None = None,
    order_id: OrderId | None = None,
    side: OrderSide | str = OrderSide.BUY,
    order_type: OrderType | str = OrderType.MARKET,
    quantity: object = "10",
    limit_price: object | None = None,
    time_in_force: TimeInForce | str = TimeInForce.GOOD_TIL_CANCELLED,
    submitted_at: str = "2024-01-02T21:00:00Z",
    active_from: str = "2024-01-03T14:30:00Z",
) -> Order:
    """Return a stable accepted order for broker contract tests."""
    resolved = item or instrument()
    resolved_side = OrderSide(side)
    resolved_type = OrderType(order_type)
    resolved_time_in_force = TimeInForce(time_in_force)
    submitted = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
    active = datetime.fromisoformat(active_from.replace("Z", "+00:00"))
    price = (
        Money.of(limit_price, resolved.quote_currency)
        if limit_price is not None
        else None
    )
    return Order(
        id=order_id or OrderId.parse("order_" + "8" * 32),
        instrument=resolved.id,
        side=resolved_side,
        type=resolved_type,
        quantity=Quantity.of(quantity),
        quote_currency=resolved.quote_currency,
        limit_price=price,
        time_in_force=resolved_time_in_force,
        submitted_at=submitted,
        active_from=active,
        reason=DecisionReason.of("broker_test"),
        status=OrderStatus.ACCEPTED,
        filled_quantity=Quantity.of("0"),
    )


def market_slice(
    observed_at: str,
    *,
    item: Instrument | None = None,
    open: object = "100",
    high: object | None = None,
    low: object | None = None,
    close: object | None = None,
    volume: object = "1000",
) -> MarketSlice:
    """Return one deterministic current OHLCV bar."""
    resolved = item or instrument()
    timestamp = np.datetime64(observed_at.replace("Z", ""), "ns")
    open_value = float(open)
    high_value = float(high if high is not None else open)
    low_value = float(low if low is not None else open)
    close_value = float(close if close is not None else open)
    bar = BarView(
        timestamp=timestamp,
        open=open_value,
        high=high_value,
        low=low_value,
        close=close_value,
        volume=float(volume),
    )
    return MarketSlice(timestamp=timestamp, bars={resolved.id: bar})


def rng(seed: int = 42) -> np.random.Generator:
    """Return an explicit deterministic engine-style random generator."""
    return np.random.default_rng(seed)


def portfolio_snapshot(
    *,
    cash: object = "10000",
    positions: Mapping[InstrumentId, object] | None = None,
    prices: Mapping[InstrumentId, object] | None = None,
    currency: str = "USD",
) -> PortfolioSnapshot:
    """Return a reconciled marked snapshot with the supplied current cash."""
    requested_positions = dict(positions or {})
    requested_prices = dict(prices or {})
    marks: dict[InstrumentId, Money] = {}
    for instrument_id in requested_positions:
        raw_mark = requested_prices.get(instrument_id, "100")
        marks[instrument_id] = (
            raw_mark
            if isinstance(raw_mark, Money)
            else Money.of(raw_mark, currency)
        )
    for instrument_id, raw_mark in requested_prices.items():
        marks[instrument_id] = (
            raw_mark
            if isinstance(raw_mark, Money)
            else Money.of(raw_mark, currency)
        )

    built_positions: dict[InstrumentId, Position] = {}
    market_value = Decimal("0")
    gross_exposure = Decimal("0")
    for instrument_id, raw_quantity in requested_positions.items():
        quantity = Quantity.of(raw_quantity)
        mark = marks[instrument_id]
        marked_value = quantity.value * mark.amount
        market_value += marked_value
        gross_exposure += abs(marked_value)
        built_positions[instrument_id] = Position(
            instrument=instrument_id,
            quantity=quantity,
            average_price=(
                None if quantity.value == Decimal("0") else mark
            ),
            book_cost=Money.of(
                abs(quantity.value) * mark.amount,
                currency,
            ),
            realized_pnl=Money.of("0", currency),
        )

    current_cash = Money.of(cash, currency)
    return PortfolioSnapshot(
        timestamp=BASE_DATETIME,
        cash=current_cash,
        positions=built_positions,
        realized_pnl=Money.of("0", currency),
        unrealized_pnl=Money.of("0", currency),
        total_fees=Money.of("0", currency),
        market_value=Money.of(market_value, currency),
        gross_exposure=Money.of(gross_exposure, currency),
        equity=Money.of(current_cash.amount + market_value, currency),
        valuation_prices=marks,
        cash_events=(),
    )


def risk_context(
    *,
    snapshot: PortfolioSnapshot | None = None,
    prices: Mapping[InstrumentId, object] | None = None,
    instruments: Mapping[InstrumentId, Instrument] | None = None,
    tradable: Collection[InstrumentId] | None = None,
    order_id: OrderId | None = None,
    submitted_at: datetime = BASE_DATETIME,
    active_from: datetime | None = None,
) -> RiskContext:
    """Return deterministic current data and proposed-order identity."""
    current_snapshot = snapshot or portfolio_snapshot()
    requested_prices = (
        dict(prices)
        if prices is not None
        else dict(current_snapshot.valuation_prices)
    )
    instrument_catalog = (
        dict(instruments)
        if instruments is not None
        else {
            instrument_id: instrument(
                instrument_id.symbol,
                venue=instrument_id.venue,
            )
            for instrument_id in {
                *current_snapshot.positions,
                *requested_prices,
            }
        }
    )
    if instruments is None and not instrument_catalog:
        default = aapl()
        instrument_catalog[default.id] = default
    if prices is None and not requested_prices:
        default_id = next(iter(instrument_catalog))
        requested_prices[default_id] = "100"
    normalized_prices = {
        instrument_id: (
            value
            if isinstance(value, Money)
            else Money.of(
                value,
                (
                    instrument_catalog[instrument_id].quote_currency
                    if instrument_id in instrument_catalog
                    else current_snapshot.cash.currency
                ),
            )
        )
        for instrument_id, value in requested_prices.items()
    }
    return RiskContext(
        snapshot=current_snapshot,
        prices=normalized_prices,
        instruments=instrument_catalog,
        tradable=frozenset(
            tradable
            if tradable is not None
            else instrument_catalog
        ),
        order_id=order_id
        or OrderId.parse("order_" + "7" * 32),
        submitted_at=submitted_at,
        active_from=active_from or submitted_at + timedelta(days=1),
    )
