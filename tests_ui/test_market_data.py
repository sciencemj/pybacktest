from datetime import date
from decimal import Decimal
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pybacktest import InstrumentId
from streamlit_ui.errors import DemoInputError
from streamlit_ui.market_data import (
    MAX_TICKERS,
    TickerHistory,
    clean_frame,
    fetch_history,
    parse_tickers,
    tick_size_for,
    to_market_dataset,
)
from tests_ui.support import make_frame, random_walk


def test_parse_tickers_trims_upper_cases_and_deduplicates():
    assert parse_tickers(" aapl, MSFT ,aapl,, 005930.ks ") == (
        "AAPL",
        "MSFT",
        "005930.KS",
    )


@pytest.mark.parametrize(
    ("raw", "code"),
    [("", "no_tickers"), (" , ,", "no_tickers"), ("A,B,C,D,E,F", "too_many_tickers")],
)
def test_parse_tickers_rejects_empty_and_too_many(raw, code):
    with pytest.raises(DemoInputError) as caught:
        parse_tickers(raw)
    assert caught.value.code == code


def test_max_tickers_is_five():
    assert MAX_TICKERS == 5


@pytest.mark.parametrize(
    ("currency", "tick"),
    [
        ("USD", Decimal("0.01")),
        ("EUR", Decimal("0.01")),
        ("KRW", Decimal("1")),
        ("JPY", Decimal("1")),
    ],
)
def test_tick_size_for_currency(currency, tick):
    assert tick_size_for(currency) == tick


def test_clean_frame_rounds_to_tick_and_labels_bars_by_local_date():
    frame = make_frame([100.004, 100.006], tz="America/New_York")
    cleaned = clean_frame(frame, Decimal("0.01"))
    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert cleaned["Close"].tolist() == [100.0, 100.01]
    assert cleaned.index.tz is None
    assert cleaned.index[0] == pd.Timestamp("2022-01-03 00:00")


def test_clean_frame_keeps_the_local_date_east_of_utc():
    # 2022-01-03 00:00 in Seoul is 2022-01-02 15:00 UTC; converting to UTC
    # would move the bar to the previous day, outside a period starting there.
    frame = make_frame([71000.0, 71100.0], tz="Asia/Seoul")
    cleaned = clean_frame(frame, Decimal("1"))
    assert cleaned.index[0] == pd.Timestamp("2022-01-03 00:00")


def test_clean_frame_rounds_krw_to_whole_won():
    frame = make_frame([71234.6, 71234.4], tz="Asia/Seoul")
    cleaned = clean_frame(frame, Decimal("1"))
    assert cleaned["Close"].tolist() == [71235.0, 71234.0]


def test_clean_frame_drops_rows_with_missing_values():
    frame = make_frame([100.0, 101.0, 102.0])
    frame.iloc[1, frame.columns.get_loc("Close")] = np.nan
    assert len(clean_frame(frame, Decimal("0.01"))) == 2


def test_clean_frame_widens_high_low_to_contain_open_and_close():
    frame = make_frame([100.0])
    frame.loc[frame.index[0], ["Open", "High", "Low", "Close"]] = [
        101.0,
        100.5,
        100.2,
        99.0,
    ]
    cleaned = clean_frame(frame, Decimal("0.01"))
    assert cleaned.iloc[0]["High"] == 101.0
    assert cleaned.iloc[0]["Low"] == 99.0


def test_clean_frame_of_frame_without_ohlcv_is_empty():
    empty = pd.DataFrame(columns=["Open", "Close"])
    assert clean_frame(empty, Decimal("0.01")).empty


def test_to_market_dataset_builds_one_series_per_ticker(two_usd_histories):
    dataset, currency = to_market_dataset(two_usd_histories, min_bars=10)
    assert currency == "USD"
    assert set(dataset.series) == {
        InstrumentId.parse("YF:AAPL"),
        InstrumentId.parse("YF:MSFT"),
    }
    instrument = dataset.instruments[InstrumentId.parse("YF:AAPL")]
    assert instrument.tick_size == Decimal("0.01")
    assert instrument.quote_currency == "USD"


def test_to_market_dataset_rejects_empty_history():
    histories = {
        "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
        "NOPE": TickerHistory(pd.DataFrame(), ""),
    }
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=10)
    assert caught.value.code == "empty_history"
    assert "NOPE" in caught.value.detail


def test_to_market_dataset_rejects_mixed_currency():
    histories = {
        "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
        "005930.KS": TickerHistory(make_frame(random_walk(2)), "KRW"),
    }
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=10)
    assert caught.value.code == "mixed_currency"
    assert "AAPL=USD" in caught.value.detail
    assert "005930.KS=KRW" in caught.value.detail


def test_to_market_dataset_rejects_history_shorter_than_warmup():
    histories = {"AAPL": TickerHistory(make_frame(random_walk(1, bars=20)), "USD")}
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=61)
    assert caught.value.code == "insufficient_history"
    assert "AAPL=20" in caught.value.detail


def test_to_market_dataset_treats_unknown_currency_as_missing_data():
    histories = {"AAPL": TickerHistory(make_frame(random_walk(1)), "")}
    with pytest.raises(DemoInputError) as caught:
        to_market_dataset(histories, min_bars=10)
    assert caught.value.code == "empty_history"
    assert caught.value.detail == "AAPL"


class _FakeTicker:
    """Stand-in for ``yfinance.Ticker`` that records how it was called."""

    calls: ClassVar[list[dict[str, object]]] = []
    error: ClassVar[Exception | None] = None

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.fast_info = {"currency": "USD"}

    def history(self, **kwargs):
        type(self).calls.append(kwargs)
        if type(self).error is not None:
            raise type(self).error
        return make_frame(random_walk(1, bars=30))


@pytest.fixture
def fake_yfinance(monkeypatch):
    _FakeTicker.calls = []
    _FakeTicker.error = None
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    return _FakeTicker


def test_fetch_history_end_date_is_inclusive(fake_yfinance):
    # yfinance treats ``end`` as exclusive, so the chosen day must be widened.
    fetch_history(("AAPL",), date(2024, 1, 2), date(2024, 3, 8))
    assert fake_yfinance.calls[0]["start"] == date(2024, 1, 2)
    assert fake_yfinance.calls[0]["end"] == date(2024, 3, 9)


def test_fetch_history_maps_rate_limit_to_a_typed_error(fake_yfinance):
    from yfinance.exceptions import YFRateLimitError

    fake_yfinance.error = YFRateLimitError()
    with pytest.raises(DemoInputError) as caught:
        fetch_history(("AAPL",), date(2024, 1, 2), date(2024, 3, 8))
    assert caught.value.code == "rate_limited"
