from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyResult:
    name: str
    equity_curve: dict[pd.Timestamp, float] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    daily_snapshots: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_cash: float = 0.0
    final_holdings: dict[str, int] = field(default_factory=dict)
    projection: pd.DataFrame | None = None

    def monthly_snapshots(self) -> pd.DataFrame:
        if not self.daily_snapshots:
            return pd.DataFrame()
        df = pd.DataFrame(self.daily_snapshots)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").resample("ME").last()


@dataclass
class BacktestResult:
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
