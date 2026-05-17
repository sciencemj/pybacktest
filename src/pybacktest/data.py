from __future__ import annotations

import pandas as pd


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
YFINANCE_DOWNLOAD_ORDER = ["Close", "High", "Low", "Open", "Volume"]


class DataError(ValueError):
    """Raised when market data cannot be normalized for backtesting."""


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = [
            next(str(part) for part in column if str(part) in OHLCV_COLUMNS or str(part) == "Adj Close")
            for column in data.columns
        ]
    return data


def normalize_ohlcv(data: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    label = f" for {ticker}" if ticker else ""
    if data.empty:
        raise DataError(f"No price data{label}. Check ticker and date range.")

    normalized = _flatten_columns(data.copy())

    if set(OHLCV_COLUMNS).issubset(normalized.columns):
        normalized = normalized[OHLCV_COLUMNS]
    elif len(normalized.columns) == 5:
        normalized.columns = YFINANCE_DOWNLOAD_ORDER
        normalized = normalized[OHLCV_COLUMNS]
    elif set(OHLCV_COLUMNS + ["Adj Close"]).issubset(normalized.columns):
        normalized = normalized[OHLCV_COLUMNS]
    else:
        raise DataError(
            f"Price data{label} must contain Open, High, Low, Close, and Volume columns."
        )

    normalized = normalized.sort_index()
    normalized["Change"] = normalized["Close"] - normalized["Close"].shift(1)
    normalized["Change_Pct"] = (
        normalized["Change"] / normalized["Close"].shift(1) * 100
    )
    return normalized
