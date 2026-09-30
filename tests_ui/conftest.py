"""Fixtures for the UI tests."""

from __future__ import annotations

import pytest

from streamlit_ui.market_data import TickerHistory
from tests_ui.support import make_frame, random_walk


@pytest.fixture
def two_usd_histories() -> dict[str, TickerHistory]:
    return {
        "AAPL": TickerHistory(make_frame(random_walk(1)), "USD"),
        "MSFT": TickerHistory(make_frame(random_walk(2)), "USD"),
    }
