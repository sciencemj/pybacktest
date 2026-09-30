"""yfinance history fetching and conversion into a core ``MarketDataSet``.

Only :func:`fetch_history` touches the network. Everything else is pure so it
can be tested with fabricated yfinance-shaped frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pybacktest import (
    BarSeries,
    Instrument,
    InstrumentId,
    MarketDataSet,
    Timeframe,
)
from streamlit_ui.errors import DemoInputError

MAX_TICKERS = 5
WARMUP_MARGIN_BARS = 5
ZERO_DECIMAL_CURRENCIES = frozenset({"KRW", "JPY"})
OHLCV = ("Open", "High", "Low", "Close", "Volume")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True, slots=True)
class TickerHistory:
    """One ticker's raw daily bars and quote currency."""

    frame: pd.DataFrame
    currency: str


def parse_tickers(raw: str) -> tuple[str, ...]:
    """Split comma-separated input into trimmed, upper-cased unique tickers."""
    seen: dict[str, None] = {}
    for part in raw.split(","):
        ticker = part.strip().upper()
        if ticker:
            seen.setdefault(ticker, None)
    tickers = tuple(seen)
    if not tickers:
        raise DemoInputError("no_tickers")
    if len(tickers) > MAX_TICKERS:
        raise DemoInputError("too_many_tickers", str(MAX_TICKERS))
    return tickers


def instrument_id(ticker: str) -> InstrumentId:
    """Map a yfinance ticker to the demo's instrument identifier."""
    return InstrumentId.parse(f"YF:{ticker}")


def tick_size_for(currency: str) -> Decimal:
    """Return the price increment used for ``currency``."""
    return Decimal("1") if currency in ZERO_DECIMAL_CURRENCIES else Decimal("0.01")


def fetch_history(
    tickers: Sequence[str],
    start: date,
    end: date,
) -> dict[str, TickerHistory]:
    """Download adjusted daily bars for each ticker (network access)."""
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError

    histories: dict[str, TickerHistory] = {}
    for ticker in tickers:
        handle = yf.Ticker(ticker)
        try:
            # yfinance treats ``end`` as exclusive; the chosen day is inclusive.
            frame = handle.history(
                start=start,
                end=end + timedelta(days=1),
                auto_adjust=True,
            )
        except YFRateLimitError as error:
            raise DemoInputError("rate_limited") from error
        try:
            currency = str(handle.fast_info["currency"] or "").upper()
        except Exception:  # yfinance raises assorted errors for unknown tickers
            currency = ""
        histories[ticker] = TickerHistory(frame=frame, currency=currency)
    return histories


def clean_frame(frame: pd.DataFrame, tick: Decimal) -> pd.DataFrame:
    """Keep OHLCV, drop incomplete rows, round prices, index by local date."""
    missing = [column for column in OHLCV if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=list(OHLCV))
    cleaned = frame.loc[:, list(OHLCV)].astype("float64").dropna()
    # Daily bars are labelled by their exchange-local calendar date. Converting
    # to UTC would move bars east of UTC (Seoul, Tokyo) to the previous day.
    cleaned.index = pd.DatetimeIndex(cleaned.index).tz_localize(None).normalize()
    prices = ["Open", "High", "Low", "Close"]
    for column in prices:
        cleaned[column] = [_round_to_tick(value, tick) for value in cleaned[column]]
    # Adjusted yfinance bars can leave open/close outside [low, high]; widen
    # the range so every bar satisfies the core OHLC invariant.
    cleaned["High"] = cleaned[prices].max(axis=1)
    cleaned["Low"] = cleaned[prices].min(axis=1)
    return cleaned[cleaned["Low"] > 0]


def _round_to_tick(value: float, tick: Decimal) -> float:
    steps = (Decimal(repr(value)) / tick).quantize(Decimal("1"), ROUND_HALF_EVEN)
    return float(steps * tick)


def require_data(histories: Mapping[str, TickerHistory]) -> None:
    """Raise ``empty_history`` for tickers with no bars or no currency."""
    missing = [
        ticker
        for ticker, item in histories.items()
        if item.frame.empty or not item.currency
    ]
    if missing:
        raise DemoInputError("empty_history", ", ".join(missing))


def validate_histories(histories: Mapping[str, TickerHistory]) -> str:
    """Return the single shared currency, or raise ``DemoInputError``."""
    require_data(histories)
    currencies = {ticker: item.currency for ticker, item in histories.items()}
    if len(set(currencies.values())) != 1:
        detail = ", ".join(f"{ticker}={code}" for ticker, code in currencies.items())
        raise DemoInputError("mixed_currency", detail)
    return next(iter(currencies.values()))


def to_market_dataset(
    histories: Mapping[str, TickerHistory],
    *,
    min_bars: int,
) -> tuple[MarketDataSet, str]:
    """Validate ``histories`` and convert them into one core dataset."""
    currency = validate_histories(histories)
    tick = tick_size_for(currency)
    series: dict[InstrumentId, BarSeries] = {}
    instruments: dict[InstrumentId, Instrument] = {}
    short: list[str] = []
    for ticker, item in histories.items():
        frame = clean_frame(item.frame, tick)
        if len(frame) < min_bars + WARMUP_MARGIN_BARS:
            short.append(f"{ticker}={len(frame)}")
            continue
        identifier = instrument_id(ticker)
        series[identifier] = BarSeries(
            timestamps=frame.index.to_numpy(dtype="datetime64[ns]"),
            open=frame["Open"].to_numpy(dtype=np.float64),
            high=frame["High"].to_numpy(dtype=np.float64),
            low=frame["Low"].to_numpy(dtype=np.float64),
            close=frame["Close"].to_numpy(dtype=np.float64),
            volume=frame["Volume"].to_numpy(dtype=np.float64),
        )
        instruments[identifier] = Instrument(
            id=identifier,
            quote_currency=currency,
            tick_size=tick,
            lot_size=Decimal("1"),
            timezone=UTC,
        )
    if short:
        raise DemoInputError(
            "insufficient_history",
            f"{', '.join(short)} < {min_bars + WARMUP_MARGIN_BARS}",
        )
    dataset = MarketDataSet(
        series=series,
        instruments=instruments,
        timeframe=Timeframe.days(1),
    )
    return dataset, currency
