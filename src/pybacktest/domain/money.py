"""Currency-safe monetary and quantity value objects."""

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from types import NotImplementedType

from .errors import ConfigurationError


def decimal_from(value: object, field_name: str) -> Decimal:
    """Convert a public numeric input without importing binary float noise."""
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ConfigurationError(f"{field_name} must be a finite Decimal.") from error
    if not decimal_value.is_finite():
        raise ConfigurationError(f"{field_name} must be a finite Decimal.")
    return decimal_value


def normalize_currency(currency: str) -> str:
    """Normalize a non-empty currency code for safe equality comparisons."""
    if not isinstance(currency, str):
        raise ConfigurationError("currency must be a non-empty string.")
    normalized = currency.strip().upper()
    if not normalized:
        raise ConfigurationError("currency must be a non-empty string.")
    return normalized


@dataclass(frozen=True, slots=True)
class Money:
    """An immutable finite amount in a normalized currency."""

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", decimal_from(self.amount, "amount"))
        object.__setattr__(self, "currency", normalize_currency(self.currency))

    @classmethod
    def of(cls, amount: object, currency: str) -> "Money":
        return cls(amount=decimal_from(amount, "amount"), currency=currency)

    @classmethod
    def usd(cls, amount: object) -> "Money":
        return cls.of(amount, "USD")

    def __add__(self, other: object) -> "Money | NotImplementedType":
        if not isinstance(other, Money):
            return NotImplemented
        self._require_matching_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: object) -> "Money | NotImplementedType":
        if not isinstance(other, Money):
            return NotImplemented
        self._require_matching_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def _require_matching_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ConfigurationError("Money currency values must match for arithmetic.")


@dataclass(frozen=True, slots=True)
class Quantity:
    """An immutable, non-negative finite absolute quantity."""

    value: Decimal

    def __post_init__(self) -> None:
        decimal_value = decimal_from(self.value, "value")
        if decimal_value < Decimal("0"):
            raise ConfigurationError("Quantity value must be non-negative.")
        object.__setattr__(self, "value", decimal_value)

    @classmethod
    def of(cls, value: object) -> "Quantity":
        return cls(value=decimal_from(value, "value"))

    def quantized(self, lot_size: object) -> "Quantity":
        lot = decimal_from(lot_size, "lot_size")
        if lot <= Decimal("0"):
            raise ConfigurationError("lot_size must be positive.")
        whole_lots = (self.value / lot).to_integral_value(rounding=ROUND_DOWN)
        return Quantity(whole_lots * lot)
