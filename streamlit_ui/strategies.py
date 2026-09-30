"""Demo strategies built only on the public ``pybacktest`` strategy contract.

Every strategy is a frozen, stateless dataclass that trades each instrument in
``instruments`` independently and targets an equal ``1/N`` weight. Feature
names are namespaced as ``<feature>:<instrument>`` so the plans of several
instruments never collide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from pybacktest import ConfigurationError, InstrumentId, StrategyContext
from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.orders import DecisionReason, TargetWeight

WEIGHT_QUANTUM = Decimal("0.000001")
RSI_PERIOD_RANGE = (2, 30)


def _instruments(value: Sequence[InstrumentId]) -> tuple[InstrumentId, ...]:
    instruments = tuple(value)
    if not instruments or not all(isinstance(i, InstrumentId) for i in instruments):
        raise ConfigurationError(
            "demo strategies need at least one InstrumentId.",
            code="invalid_demo_instruments",
        )
    if len(set(instruments)) != len(instruments):
        raise ConfigurationError(
            "demo strategy instruments must be unique.",
            code="invalid_demo_instruments",
        )
    return instruments


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"{name} must be an integer.",
            code="invalid_demo_parameter",
        )
    return value


def equal_weight(count: int) -> Decimal:
    """Return ``1/count`` quantized to six decimal places."""
    return (Decimal(1) / Decimal(count)).quantize(WEIGHT_QUANTUM)


def feature_name(feature: str, instrument: InstrumentId) -> str:
    """Namespace ``feature`` for one instrument."""
    return f"{feature}:{instrument}"


def rsi(closes: Sequence[float]) -> float:
    """Simple-average (Cutler) RSI over consecutive closes, oldest first."""
    gains = 0.0
    losses = 0.0
    for previous, current in pairwise(closes):
        change = current - previous
        if change > 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 50.0 if gains == 0 else 100.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


def _holding(context: StrategyContext, instrument: InstrumentId) -> bool:
    quantity = context.portfolio.positions.get(instrument)
    return quantity is not None and quantity.value != 0


def _pending(context: StrategyContext) -> frozenset[InstrumentId]:
    return frozenset(order.instrument for order in context.active_orders)


@dataclass(frozen=True, slots=True)
class BuyAndHold:
    """Buy ``1/N`` of each instrument once and hold it."""

    instruments: tuple[InstrumentId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instruments(self.instruments))

    @property
    def warmup_bars(self) -> int:
        return 1

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        for instrument in self.instruments:
            builder.source(feature_name("close", instrument), instrument, "close")
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[TargetWeight, ...]:
        del market
        pending = _pending(context)
        weight = equal_weight(len(self.instruments))
        intents = []
        for instrument in self.instruments:
            close = context.features.at(feature_name("close", instrument))
            if close != close or instrument in pending or _holding(context, instrument):
                continue
            intents.append(
                TargetWeight(
                    instrument=instrument,
                    weight=weight,
                    reason=DecisionReason.of("buy_and_hold"),
                )
            )
        return tuple(intents)


@dataclass(frozen=True, slots=True)
class MovingAverageCrossAll:
    """Per instrument: target ``1/N`` on a fast/slow SMA cross up, ``0`` down."""

    instruments: tuple[InstrumentId, ...]
    fast: int
    slow: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instruments(self.instruments))
        fast = _int(self.fast, "fast")
        slow = _int(self.slow, "slow")
        if not 0 < fast < slow:
            raise ConfigurationError(
                "moving-average cross requires 0 < fast < slow.",
                code="invalid_demo_parameter",
            )

    @property
    def warmup_bars(self) -> int:
        return self.slow + 1

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        for instrument in self.instruments:
            close = builder.source(
                feature_name("close", instrument), instrument, "close"
            )
            fast = builder.sma(feature_name("fast", instrument), close, self.fast)
            slow = builder.sma(feature_name("slow", instrument), close, self.slow)
            builder.lag(feature_name("fast_prev", instrument), fast, 1)
            builder.lag(feature_name("slow_prev", instrument), slow, 1)
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[TargetWeight, ...]:
        del market
        view = context.features
        weight = equal_weight(len(self.instruments))
        intents = []
        for instrument in self.instruments:
            fast = view.at(feature_name("fast", instrument))
            slow = view.at(feature_name("slow", instrument))
            fast_prev = view.at(feature_name("fast_prev", instrument))
            slow_prev = view.at(feature_name("slow_prev", instrument))
            if any(value != value for value in (fast, slow, fast_prev, slow_prev)):
                continue
            if fast_prev <= slow_prev and fast > slow:
                target, outcome = weight, "cross_up"
            elif fast_prev >= slow_prev and fast < slow:
                target, outcome = Decimal(0), "cross_down"
            else:
                continue
            intents.append(
                TargetWeight(
                    instrument=instrument,
                    weight=target,
                    reason=DecisionReason.of("ma_cross", outcome=outcome),
                )
            )
        return tuple(intents)


@dataclass(frozen=True, slots=True)
class RsiReversion:
    """Per instrument: target ``1/N`` below ``lower`` RSI, ``0`` above ``upper``."""

    instruments: tuple[InstrumentId, ...]
    period: int
    lower: int
    upper: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruments", _instruments(self.instruments))
        period = _int(self.period, "period")
        lower = _int(self.lower, "lower")
        upper = _int(self.upper, "upper")
        low, high = RSI_PERIOD_RANGE
        if not low <= period <= high:
            raise ConfigurationError(
                f"RSI period must be between {low} and {high}.",
                code="invalid_demo_parameter",
            )
        if not 1 <= lower < upper <= 99:
            raise ConfigurationError(
                "RSI thresholds require 1 <= lower < upper <= 99.",
                code="invalid_demo_parameter",
            )

    @property
    def warmup_bars(self) -> int:
        return self.period + 1

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        for instrument in self.instruments:
            close = builder.source(
                feature_name("close", instrument), instrument, "close"
            )
            for lag in range(1, self.period + 1):
                builder.lag(feature_name(f"close_lag{lag}", instrument), close, lag)
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[TargetWeight, ...]:
        del market
        view = context.features
        pending = _pending(context)
        weight = equal_weight(len(self.instruments))
        intents = []
        for instrument in self.instruments:
            if instrument in pending:
                continue
            closes = [
                view.at(feature_name(f"close_lag{lag}", instrument))
                for lag in range(self.period, 0, -1)
            ]
            closes.append(view.at(feature_name("close", instrument)))
            if any(value != value for value in closes):
                continue
            value = rsi(closes)
            holding = _holding(context, instrument)
            if value < self.lower and not holding:
                target, outcome = weight, "oversold"
            elif value > self.upper and holding:
                target, outcome = Decimal(0), "overbought"
            else:
                continue
            intents.append(
                TargetWeight(
                    instrument=instrument,
                    weight=target,
                    reason=DecisionReason.of(
                        "rsi_reversion", rsi=round(value, 4), outcome=outcome
                    ),
                )
            )
        return tuple(intents)


DemoStrategy = BuyAndHold | MovingAverageCrossAll | RsiReversion
"""Any demo strategy; each exposes ``instruments`` and ``warmup_bars``."""
