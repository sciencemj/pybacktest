import pandas as pd
import pytest

from pybacktest.data import DataError, normalize_ohlcv


def test_normalize_standard_ohlcv_columns():
    raw = pd.DataFrame(
        {
            "Open": [99.0, 101.0],
            "High": [101.0, 103.0],
            "Low": [98.0, 100.0],
            "Close": [100.0, 102.0],
            "Volume": [1000, 1500],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    normalized = normalize_ohlcv(raw)

    assert list(normalized.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Change",
        "Change_Pct",
    ]
    assert normalized.loc[pd.Timestamp("2024-01-02"), "Change"] == 2.0
    assert normalized.loc[pd.Timestamp("2024-01-02"), "Change_Pct"] == 2.0


def test_normalize_yfinance_adjusted_close_shape():
    raw = pd.DataFrame(
        {
            "Open": [99.0, 101.0],
            "High": [101.0, 103.0],
            "Low": [98.0, 100.0],
            "Close": [100.0, 102.0],
            "Adj Close": [99.5, 101.5],
            "Volume": [1000, 1500],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    normalized = normalize_ohlcv(raw)

    assert "Adj Close" not in normalized.columns
    assert normalized["Close"].tolist() == [100.0, 102.0]


def test_normalize_download_order_without_headers():
    raw = pd.DataFrame(
        [
            [100.0, 103.0, 99.0, 101.0, 2000],
            [102.0, 104.0, 101.0, 103.0, 3000],
        ],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    normalized = normalize_ohlcv(raw)

    assert normalized.loc[pd.Timestamp("2024-01-01"), "Close"] == 100.0
    assert normalized.loc[pd.Timestamp("2024-01-01"), "Volume"] == 2000


def test_normalize_empty_data_raises_clear_error():
    with pytest.raises(DataError, match="No price data"):
        normalize_ohlcv(pd.DataFrame(), ticker="BAD")
