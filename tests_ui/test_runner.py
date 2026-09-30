from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from pybacktest import PybacktestError
from streamlit_ui.market_data import TickerHistory
from streamlit_ui.runner import METRIC_ROWS, RunConfig, run_backtest
from tests_ui.support import make_frame

START = date(2022, 1, 1)
END = date(2023, 12, 31)


def _config(strategy: str, params: dict[str, int], tickers=("AAPL", "MSFT")):
    return RunConfig(
        tickers=tickers,
        start=START,
        end=END,
        strategy=strategy,
        params=params,
        initial_cash=Decimal("10000"),
        commission_per_share=Decimal("0"),
    )


def _v_shape(bars: int = 120) -> np.ndarray:
    """Fall from 100 to 50, then rise back to 150."""
    half = bars // 2
    return np.concatenate(
        [np.linspace(100, 50, half), np.linspace(50, 150, bars - half)]
    )


def test_report_shape_and_benchmark(two_usd_histories):
    report = run_backtest(
        _config("ma_cross", {"fast": 5, "slow": 20}), two_usd_histories
    )
    assert list(report.metrics.index) == [name.value for name in METRIC_ROWS]
    assert list(report.metrics.columns) == ["strategy", "benchmark"]
    assert list(report.equity.columns) == ["strategy", "benchmark"]
    assert report.equity.index.is_monotonic_increasing
    assert report.equity.notna().all().all()
    assert report.info["currency"] == "USD"
    assert set(report.info["bars"]) == {"YF:AAPL", "YF:MSFT"}
    assert list(report.fills.columns) == [
        "timestamp",
        "instrument",
        "side",
        "quantity",
        "price",
        "fee",
    ]


def test_buy_and_hold_buys_each_instrument_once(two_usd_histories):
    report = run_backtest(_config("buy_and_hold", {}), two_usd_histories)
    assert sorted(report.fills["instrument"]) == ["YF:AAPL", "YF:MSFT"]
    assert set(report.fills["side"]) == {"buy"}
    # The chosen strategy is the benchmark, so both curves match exactly.
    assert report.equity["strategy"].equals(report.equity["benchmark"])


def test_ma_cross_buys_after_the_bottom_on_a_v_shape():
    histories = {"AAPL": TickerHistory(make_frame(_v_shape()), "USD")}
    report = run_backtest(
        _config("ma_cross", {"fast": 3, "slow": 10}, tickers=("AAPL",)), histories
    )
    buys = report.fills[report.fills["side"] == "buy"]
    assert len(buys) == 1
    assert buys.iloc[0]["price"] > 50.0


def test_rsi_buys_the_dip_and_sells_the_rally():
    histories = {"AAPL": TickerHistory(make_frame(_v_shape()), "USD")}
    report = run_backtest(
        _config("rsi", {"period": 5, "lower": 30, "upper": 70}, tickers=("AAPL",)),
        histories,
    )
    # A straight decline pins RSI at 0 and a straight rally pins it at 100, so
    # the buy lands during the fall and the sell right after the bottom.
    bottom = report.equity.index[len(report.equity) // 2]
    assert report.fills["side"].tolist() == ["buy", "sell"]
    assert report.fills.iloc[0]["timestamp"] < bottom
    assert report.fills.iloc[1]["timestamp"] >= bottom


def test_commission_is_charged_per_share(two_usd_histories):
    config = RunConfig(
        tickers=("AAPL", "MSFT"),
        start=START,
        end=END,
        strategy="buy_and_hold",
        initial_cash=Decimal("10000"),
        commission_per_share=Decimal("0.01"),
    )
    report = run_backtest(config, two_usd_histories)
    expected = (report.fills["quantity"] * 0.01).round(6).tolist()
    assert report.fills["fee"].round(6).tolist() == expected


def test_invalid_strategy_parameters_surface_as_core_errors(two_usd_histories):
    with pytest.raises(PybacktestError):
        run_backtest(_config("ma_cross", {"fast": 30, "slow": 10}), two_usd_histories)


def test_ticker_listed_mid_period_is_bought_when_its_data_starts():
    histories = {
        "AAPL": TickerHistory(make_frame(np.linspace(100, 120, 200)), "USD"),
        "NEWCO": TickerHistory(
            make_frame(np.linspace(20, 30, 100), start="2022-05-23"), "USD"
        ),
    }
    report = run_backtest(
        _config("buy_and_hold", {}, tickers=("AAPL", "NEWCO")), histories
    )
    newco = report.fills[report.fills["instrument"] == "YF:NEWCO"]
    assert len(newco) == 1
    assert newco.iloc[0]["timestamp"].date() > date(2022, 5, 23)


def test_holiday_in_one_series_does_not_break_the_union_calendar():
    gappy = make_frame(np.linspace(100, 120, 200))
    gappy = gappy.drop(gappy.index[50])
    histories = {
        "AAPL": TickerHistory(gappy, "USD"),
        "MSFT": TickerHistory(make_frame(np.linspace(50, 70, 200)), "USD"),
    }
    report = run_backtest(_config("ma_cross", {"fast": 3, "slow": 10}), histories)
    assert report.info["bars"] == {"YF:AAPL": 199, "YF:MSFT": 200}
    assert len(report.equity) == 200
