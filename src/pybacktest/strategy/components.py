"""Stateless reference strategies built on the public strategy contract."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from pybacktest.data.features import FeatureBuilder, FeaturePlan
from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.money import decimal_from
from pybacktest.domain.orders import DecisionReason, TargetWeight

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, not behaviour
    # ``ports.strategy`` imports ``strategy.intents``, so importing it here at
    # runtime would close a package-initialization cycle.
    from pybacktest.ports.strategy import StrategyContext

INVALID_WINDOWS = "invalid_moving_average_windows"
INVALID_WEIGHT = "invalid_moving_average_weight"
INVALID_INSTRUMENT = "invalid_strategy_instrument"
UNBOUND_INSTRUMENT = "unbound_strategy_instrument"


def _window(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"MovingAverageCross {field_name} must be an integer.",
            code=INVALID_WINDOWS,
        )
    return value


def _weight(value: object, field_name: str) -> Decimal:
    try:
        return decimal_from(value, field_name)
    except ConfigurationError as error:
        raise ConfigurationError(
            f"MovingAverageCross {field_name} must be a finite Decimal.",
            code=INVALID_WEIGHT,
        ) from error


@dataclass(frozen=True, slots=True)
class MovingAverageCross:
    """Target a long weight when a fast SMA crosses a slow SMA.

    The strategy keeps no run-specific state, so one frozen instance is safe
    to reuse across independent and concurrent sessions. Both the current and
    the previous fast/slow relationship are read from the declared plan: the
    lagged columns come from :meth:`FeatureBuilder.lag`, which is NaN during
    warm-up and lags on the instrument's own clock rather than the union
    clock, so a multi-instrument UNION calendar cannot shift the comparison.

    ``instrument`` is last and optional so the documented import-only form
    ``MovingAverageCross(fast=20, slow=60)`` stays constructible; declaring
    features or deciding a bar without a binding fails closed with
    ``unbound_strategy_instrument``. ``long_weight`` and ``flat_weight``
    default to ``1`` and ``0``; every executable example writes them out.

    A signal is emitted only on an actual transition. ``fast_prev <=
    slow_prev`` with ``fast > slow`` is ``cross_up``; ``fast_prev >=
    slow_prev`` with ``fast < slow`` is ``cross_down``. A bar whose fast and
    slow values are equal emits nothing, and because equality satisfies both
    previous-side tests it is a neutral pivot: the next strictly higher bar is
    a ``cross_up`` and the next strictly lower bar is a ``cross_down``. NaN
    warm-up and repeated bars on one side emit nothing.
    """

    fast: int
    slow: int
    long_weight: Decimal = Decimal("1")
    flat_weight: Decimal = Decimal("0")
    instrument: InstrumentId | None = None

    def __post_init__(self) -> None:
        fast = _window(self.fast, "fast")
        slow = _window(self.slow, "slow")
        if not 0 < fast < slow:
            raise ConfigurationError(
                "MovingAverageCross requires 0 < fast < slow.",
                code=INVALID_WINDOWS,
            )
        object.__setattr__(
            self,
            "long_weight",
            _weight(self.long_weight, "long_weight"),
        )
        object.__setattr__(
            self,
            "flat_weight",
            _weight(self.flat_weight, "flat_weight"),
        )
        if self.instrument is not None and not isinstance(
            self.instrument,
            InstrumentId,
        ):
            raise ConfigurationError(
                "MovingAverageCross instrument must be an InstrumentId or None.",
                code=INVALID_INSTRUMENT,
            )

    def build_features(self, builder: FeatureBuilder) -> FeaturePlan:
        """Declare close, both SMAs, and their one-bar lags for this instrument."""
        instrument = self._bound_instrument()
        close = builder.source("close", instrument, "close")
        fast = builder.sma("fast", close, self.fast)
        slow = builder.sma("slow", close, self.slow)
        builder.lag("fast_prev", fast, 1)
        builder.lag("slow_prev", slow, 1)
        return builder.plan()

    def on_bar(
        self,
        context: StrategyContext,
        market: MarketSlice,
    ) -> tuple[TargetWeight, ...]:
        """Emit one target weight when the pinned view shows a true crossover."""
        del market
        instrument = self._bound_instrument()
        view = context.features
        fast = view.at("fast")
        slow = view.at("slow")
        fast_prev = view.at("fast_prev")
        slow_prev = view.at("slow_prev")
        # NaN never compares equal to itself, which is exactly the warm-up
        # window in which no relationship has been observed yet.
        if any(value != value for value in (fast, slow, fast_prev, slow_prev)):
            return ()
        if fast_prev <= slow_prev and fast > slow:
            outcome = "cross_up"
            weight = self.long_weight
        elif fast_prev >= slow_prev and fast < slow:
            outcome = "cross_down"
            weight = self.flat_weight
        else:
            return ()
        return (
            TargetWeight(
                instrument=instrument,
                weight=weight,
                reason=DecisionReason.of(
                    "ma_cross",
                    fast=fast,
                    slow=slow,
                    fast_prev=fast_prev,
                    slow_prev=slow_prev,
                    outcome=outcome,
                ),
            ),
        )

    def _bound_instrument(self) -> InstrumentId:
        if self.instrument is None:
            raise ConfigurationError(
                "MovingAverageCross requires an instrument before execution.",
                code=UNBOUND_INSTRUMENT,
            )
        return self.instrument


__all__ = ["MovingAverageCross"]
