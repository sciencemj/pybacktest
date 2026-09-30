"""Fabricated yfinance-shaped data shared by the UI tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from streamlit_ui.market_data import TickerHistory


def make_frame(
    closes: list[float] | np.ndarray,
    *,
    start: str = "2022-01-03",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Build a yfinance ``history()``-shaped frame from closing prices."""
    close = np.asarray(closes, dtype=np.float64)
    index = pd.bdate_range(start, periods=len(close), tz=tz, name="Date")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(close), 1_000_000.0),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=index,
    )


def random_walk(seed: int, bars: int = 400) -> np.ndarray:
    """Seeded log random walk starting near 100."""
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.02, bars)))


def fake_fetch(
    tickers: Sequence[str],
    start: date,
    end: date,
) -> dict[str, TickerHistory]:
    """Stand-in for ``market_data.fetch_history`` with USD random walks."""
    del start, end
    return {
        ticker: TickerHistory(make_frame(random_walk(seed)), "USD")
        for seed, ticker in enumerate(tickers)
    }
