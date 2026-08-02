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


def decimal_to_fraction(value: Decimal) -> Fraction:
    """Convert a finite Decimal to an exact Fraction without Decimal math."""
    decimal_tuple = value.as_tuple()
    exponent = decimal_tuple.exponent
    if not isinstance(exponent, int):
        raise AssertionError("test oracle requires a finite Decimal")
    coefficient = 0
    for digit in decimal_tuple.digits:
        coefficient = coefficient * 10 + digit
    if decimal_tuple.sign:
        coefficient = -coefficient
    if exponent >= 0:
        return Fraction(coefficient * 10**exponent, 1)
    return Fraction(coefficient, 10 ** (-exponent))


def fraction_lot_floor(
    notional: Decimal,
    price: Decimal,
    lot_size: Decimal,
) -> Fraction:
    """Floor a signed Decimal ratio to a lot using only Fractions."""
    signed_notional = decimal_to_fraction(notional)
    raw_lots = abs(signed_notional) / (
        decimal_to_fraction(price) * decimal_to_fraction(lot_size)
    )
    whole_lots = raw_lots.numerator // raw_lots.denominator
    quantity = whole_lots * decimal_to_fraction(lot_size)
    return -quantity if signed_notional < 0 else quantity


__all__ = [
    "decimal_to_fraction",
    "fraction_decimal_lot_floor",
    "fraction_lot_floor",
]
