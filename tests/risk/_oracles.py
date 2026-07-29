"""Independent exact arithmetic oracles for risk tests."""

from decimal import Decimal
from fractions import Fraction


def fraction_decimal_lot_floor(
    numerator: int,
    denominator: int,
    decimal_places: int,
) -> Decimal:
    """Floor a literal fraction to a power-of-ten lot exactly."""
    scaled = Fraction(numerator * 10**decimal_places, denominator)
    lot_count = scaled.numerator // scaled.denominator
    digits = tuple(int(digit) for digit in str(lot_count))
    return Decimal((0, digits, -decimal_places))


__all__ = ["fraction_decimal_lot_floor"]
