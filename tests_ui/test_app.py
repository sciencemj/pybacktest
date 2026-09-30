from datetime import date
from pathlib import Path
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from streamlit_ui.app import format_metric
from streamlit_ui.market_data import TickerHistory
from tests_ui.support import fake_fetch, make_frame, random_walk

ENTRYPOINT = str(Path(__file__).resolve().parents[1] / "streamlit_page.py")
FETCH = "streamlit_ui.market_data.fetch_history"


@pytest.fixture(autouse=True)
def _clear_cache():
    st.cache_data.clear()


def _app() -> AppTest:
    return AppTest.from_file(ENTRYPOINT, default_timeout=30).run()


def test_page_renders_intro_without_running():
    app = _app()
    assert not app.exception
    assert app.title[0].value == "Pybacktest demo"
    assert not app.metric


def test_submitting_the_form_shows_results():
    with mock.patch(FETCH, side_effect=fake_fetch):
        app = _app()
        app.sidebar.selectbox(key="strategy").set_value("ma_cross").run()
        app.sidebar.button[0].click().run()
    assert not app.exception
    assert not app.error, [element.value for element in app.error]
    assert len(app.metric) == 4
    assert len(app.tabs) == 4


def test_switching_language_keeps_the_last_report():
    with mock.patch(FETCH, side_effect=fake_fetch) as fetch:
        app = _app()
        app.sidebar.button[0].click().run()
        app.sidebar.radio(key="lang").set_value("ko").run()
    assert fetch.call_count == 1
    assert app.title[0].value == "Pybacktest 데모"
    assert len(app.metric) == 4


def test_mixed_currency_shows_localized_error():
    def mixed(tickers, start, end):
        del tickers, start, end
        return {
            "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
            "005930.KS": TickerHistory(make_frame(random_walk(2)), "KRW"),
        }

    with mock.patch(FETCH, side_effect=mixed):
        app = _app()
        app.sidebar.radio(key="lang").set_value("ko").run()
        app.sidebar.text_input(key="tickers").set_value("AAPL, 005930.KS")
        app.sidebar.button[0].click().run()
    assert "통화" in app.error[0].value


def test_start_after_end_is_rejected_before_fetching():
    with mock.patch(FETCH, side_effect=fake_fetch) as fetch:
        app = _app()
        app.sidebar.date_input(key="start").set_value(date(2024, 1, 2))
        app.sidebar.date_input(key="end").set_value(date(2024, 1, 1))
        app.sidebar.button[0].click().run()
    assert fetch.call_count == 0
    assert "start date" in app.error[0].value


def test_unexpected_failure_shows_generic_error_with_details():
    with mock.patch(FETCH, side_effect=RuntimeError("boom")):
        app = _app()
        app.sidebar.button[0].click().run()
    assert not app.exception
    assert "unexpected" in app.error[0].value
    assert "boom" in app.expander[0].code[0].value


def test_changing_only_the_strategy_reuses_cached_data():
    with mock.patch(FETCH, side_effect=fake_fetch) as fetch:
        app = _app()
        app.sidebar.button[0].click().run()
        app.sidebar.selectbox(key="strategy").set_value("rsi").run()
        app.sidebar.button[0].click().run()
    assert not app.error
    assert fetch.call_count == 1


@pytest.mark.parametrize(
    ("name", "value", "text"),
    [
        ("total_return", 0.1234, "12.34%"),
        ("maximum_drawdown", -0.5, "-50.00%"),
        ("sharpe", 1.234, "1.23"),
        ("win_rate", None, "—"),
    ],
)
def test_format_metric(name, value, text):
    assert format_metric(name, value) == text
