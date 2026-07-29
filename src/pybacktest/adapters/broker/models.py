"""Readable deterministic fill, fee, slippage, liquidity, and borrow models."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, localcontext
from enum import StrEnum

import numpy as np

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.market import BarView
from pybacktest.domain.money import Money, Quantity, decimal_from
from pybacktest.domain.orders import Order, OrderSide, OrderType

_ZERO = Decimal("0")
_ONE = Decimal("1")
_BASIS_POINTS = Decimal("10000")
_MAX_IMPACT_BPS = Decimal("10000")
_MIN_ARITHMETIC_PRECISION = 64


@contextmanager
def _arithmetic(values: Sequence[Decimal]) -> Iterator[Context]:
    coefficient_digits = sum(len(value.as_tuple().digits) for value in values)
    context = Context(
        prec=max(_MIN_ARITHMETIC_PRECISION, coefficient_digits + 16),
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
    )
    with localcontext(context) as active:
        yield active


def _parameter(
    value: object,
    field: str,
    *,
    maximum: Decimal | None = None,
) -> Decimal:
    if isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a finite Decimal.")
    result = decimal_from(value, field)
    if result < _ZERO:
        raise ConfigurationError(f"{field} must be nonnegative.")
    if maximum is not None and result > maximum:
        raise ConfigurationError(f"{field} cannot exceed {maximum}.")
    return result


def _bar_decimal(value: float) -> Decimal:
    return Decimal(str(value))


class IntrabarPolicy(StrEnum):
    """How ambiguous OHLC-only intrabar paths are interpreted."""

    CONSERVATIVE = "conservative"


@dataclass(frozen=True, slots=True)
class NextBarOpenFill:
    """Use an eligible bar open, with conservative limit-touch rules."""

    intrabar_policy: IntrabarPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.intrabar_policy, IntrabarPolicy):
            raise ConfigurationError("intrabar_policy must be an IntrabarPolicy.")

    def reference_price(self, order: Order, market: BarView) -> Money | None:
        open_price = _bar_decimal(market.open)
        if order.type is OrderType.MARKET:
            return Money.of(open_price, order.quote_currency)

        limit = order.limit_price
        if limit is None:
            raise ConfigurationError("limit orders require a limit_price.")
        if order.side is OrderSide.BUY:
            if open_price <= limit.amount:
                return Money.of(open_price, order.quote_currency)
            if _bar_decimal(market.low) <= limit.amount:
                return limit
            return None
        if open_price >= limit.amount:
            return Money.of(open_price, order.quote_currency)
        if _bar_decimal(market.high) >= limit.amount:
            return limit
        return None


@dataclass(frozen=True, slots=True)
class PerShareCommission:
    """Charge a quote-currency amount per actually filled share/unit."""

    rate_per_share: Decimal
    minimum_fee: Decimal = _ZERO

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rate_per_share",
            _parameter(self.rate_per_share, "rate_per_share"),
        )
        object.__setattr__(
            self,
            "minimum_fee",
            _parameter(self.minimum_fee, "minimum_fee"),
        )

    def calculate(
        self,
        order: Order,
        quantity: Quantity,
        price: Money,
    ) -> Money:
        del price
        with _arithmetic((quantity.value, self.rate_per_share)):
            fee = quantity.value * self.rate_per_share
        return Money.of(max(fee, self.minimum_fee), order.quote_currency)


@dataclass(frozen=True, slots=True)
class NoCommission:
    """Apply an explicit zero commission."""

    def calculate(
        self,
        order: Order,
        quantity: Quantity,
        price: Money,
    ) -> Money:
        del quantity, price
        return Money.of(_ZERO, order.quote_currency)


@dataclass(frozen=True, slots=True)
class VolumeShareSlippage:
    """Apply signed impact proportional to actual fill quantity/bar volume."""

    impact_bps: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "impact_bps",
            _parameter(
                self.impact_bps,
                "impact_bps",
                maximum=_MAX_IMPACT_BPS,
            ),
        )

    def apply(
        self,
        order: Order,
        quantity: Quantity,
        reference_price: Money,
        market: BarView,
        rng: np.random.Generator,
    ) -> Money:
        del rng
        volume = _bar_decimal(market.volume)
        if volume <= _ZERO:
            raise ConfigurationError(
                "VolumeShareSlippage requires positive bar volume."
            )
        values = (
            quantity.value,
            volume,
            self.impact_bps,
            reference_price.amount,
        )
        with _arithmetic(values):
            participation = quantity.value / volume
            impact = self.impact_bps / _BASIS_POINTS * participation
            multiplier = _ONE + impact if order.side is OrderSide.BUY else _ONE - impact
            amount = reference_price.amount * multiplier
        return Money.of(amount, reference_price.currency)


@dataclass(frozen=True, slots=True)
class NoSlippage:
    """Return the reference price unchanged."""

    def apply(
        self,
        order: Order,
        quantity: Quantity,
        reference_price: Money,
        market: BarView,
        rng: np.random.Generator,
    ) -> Money:
        del order, quantity, market, rng
        return reference_price


@dataclass(frozen=True, slots=True)
class VolumeParticipationLimit:
    """Cap shared instrument capacity to a ratio of current bar volume."""

    max_volume_ratio: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_volume_ratio",
            _parameter(
                self.max_volume_ratio,
                "max_volume_ratio",
                maximum=_ONE,
            ),
        )

    def available_quantity(
        self,
        order: Order,
        market: BarView,
    ) -> Quantity:
        del order
        volume = _bar_decimal(market.volume)
        with _arithmetic((volume, self.max_volume_ratio)):
            available = volume * self.max_volume_ratio
        return Quantity.of(available)


@dataclass(frozen=True, slots=True)
class NoLiquidityLimit:
    """Leave positive-volume capacity unlimited."""

    def available_quantity(
        self,
        order: Order,
        market: BarView,
    ) -> None:
        del order, market
        return None


@dataclass(frozen=True, slots=True)
class NoBorrowCost:
    """Apply an explicit zero borrow cost."""

    def calculate(
        self,
        order: Order,
        quantity: Quantity,
        price: Money,
    ) -> Money:
        del quantity, price
        return Money.of(_ZERO, order.quote_currency)


__all__ = [
    "IntrabarPolicy",
    "NextBarOpenFill",
    "NoBorrowCost",
    "NoCommission",
    "NoLiquidityLimit",
    "NoSlippage",
    "PerShareCommission",
    "VolumeParticipationLimit",
    "VolumeShareSlippage",
]
