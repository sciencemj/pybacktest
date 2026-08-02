"""Library-owned Decimal contexts for deterministic risk arithmetic."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    Inexact,
    InvalidOperation,
    localcontext,
)
from itertools import chain, repeat
from math import gcd

_ZERO = Decimal("0")
_MIN_WORKING_PRECISION = 64
_MAX_WORKING_PRECISION = 4096
_MAX_CANCELLATION_INPUT_DIGITS = 8192
_GUARD_DIGITS = 16
_INTEGER_CHUNK_DIGITS = 9
_INTEGER_CHUNK_BASE = 10**_INTEGER_CHUNK_DIGITS


@contextmanager
def decimal_context(values: Sequence[Decimal]) -> Iterator[Context]:
    """Use bounded significant-digit precision and the widest exponent range.

    Precision depends only on existing coefficients and is capped at 4096
    digits. Exponent distance never expands the working coefficient.
    """
    context = Context(
        prec=_arithmetic_precision(values),
        rounding=ROUND_HALF_EVEN,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
        capitals=1,
        clamp=0,
    )
    with localcontext(context) as active:
        yield active


def exact_add(left: Decimal, right: Decimal) -> Decimal:
    """Add exactly within the documented working bound."""
    if left == _ZERO:
        return right
    if right == _ZERO:
        return left
    with decimal_context((left, right)) as context:
        result = context.add(left, right)
        if context.flags[Inexact]:
            raise InvalidOperation("exact addition exceeds working precision")
        return result


def exact_subtract(left: Decimal, right: Decimal) -> Decimal:
    """Subtract exactly within the documented working bound."""
    if right == _ZERO:
        return left
    if left == _ZERO:
        return right.copy_negate()
    with decimal_context((left, right)) as context:
        result = context.subtract(left, right)
        if context.flags[Inexact]:
            raise InvalidOperation(
                "exact subtraction exceeds working precision"
            )
        return result


def exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    """Multiply exactly within the documented working bound."""
    if left == _ZERO or right == _ZERO:
        return _ZERO
    with decimal_context((left, right)) as context:
        result = context.multiply(left, right)
        if context.flags[Inexact]:
            raise InvalidOperation(
                "exact multiplication exceeds working precision"
            )
        return result


def floor_quantity_to_lot(
    notional: Decimal,
    price: Decimal,
    lot_size: Decimal,
) -> Decimal:
    """Floor ``notional / price`` to a positive lot from original operands."""
    if (
        not notional.is_finite()
        or not price.is_finite()
        or not lot_size.is_finite()
        or price <= _ZERO
        or lot_size <= _ZERO
    ):
        raise InvalidOperation("lot-floor operands are invalid")
    is_negative = notional.is_signed()
    if notional == _ZERO:
        return notional

    absolute_notional = notional.copy_abs()
    notional_digits, notional_exponent = _normalized_components(
        absolute_notional
    )
    price_digits, price_exponent = _normalized_components(price)
    lot_digits, lot_exponent = _normalized_components(lot_size)
    if any(
        len(digits) > _MAX_CANCELLATION_INPUT_DIGITS
        for digits in (notional_digits, price_digits, lot_digits)
    ):
        raise InvalidOperation(
            "lot-floor input exceeds cancellation resource bound"
        )
    with decimal_context(
        (notional, price, lot_size)
    ) as arithmetic:
        arithmetic.rounding = ROUND_DOWN
        raw_quantity = notional / price
        division_is_exact = not arithmetic.flags[Inexact]
    if division_is_exact and is_aligned(raw_quantity, lot_size):
        return raw_quantity

    numerator = _coefficient(notional_digits)
    price_coefficient = _coefficient(price_digits)
    lot_coefficient = _coefficient(lot_digits)
    common = gcd(numerator, price_coefficient)
    numerator //= common
    price_coefficient //= common
    common = gcd(numerator, lot_coefficient)
    numerator //= common
    lot_coefficient //= common
    _integer_digits(numerator)
    _integer_digits(price_coefficient)
    _integer_digits(lot_coefficient)
    denominator = price_coefficient * lot_coefficient
    _integer_digits(denominator)
    exponent_delta = (
        notional_exponent - price_exponent - lot_exponent
    )
    quotient_digits = _floor_scaled_ratio(
        numerator,
        denominator,
        exponent_delta,
    )
    whole_lots = Decimal((0, quotient_digits, 0))
    quantity = exact_multiply(whole_lots, lot_size)
    final_digits, _ = _normalized_components(quantity)
    if len(final_digits) > _MAX_WORKING_PRECISION:
        raise InvalidOperation(
            "lot-floor result exceeds working precision"
        )
    return (
        quantity.copy_negate() if is_negative else quantity
    )


def is_aligned(value: Decimal, increment: Decimal) -> bool:
    """Return exact Decimal divisibility without exponent expansion."""
    if (
        not value.is_finite()
        or not increment.is_finite()
        or increment <= _ZERO
    ):
        return False
    if value == _ZERO:
        return True

    value_tuple = value.as_tuple()
    increment_tuple = increment.as_tuple()
    value_exponent = value_tuple.exponent
    increment_exponent = increment_tuple.exponent
    if not isinstance(value_exponent, int) or not isinstance(
        increment_exponent,
        int,
    ):
        return False

    divisor = _coefficient(increment_tuple.digits)
    if divisor == 1:
        return value_exponent >= increment_exponent or (
            increment_exponent - value_exponent
            <= _trailing_zeros(value_tuple.digits)
        )

    exponent_delta = value_exponent - increment_exponent
    if exponent_delta >= 0:
        remainder = _digits_mod(value_tuple.digits, divisor)
        if remainder == 0:
            return True
        return (
            remainder
            * pow(10, exponent_delta, divisor)
            % divisor
            == 0
        )

    required_zeros = -exponent_delta
    if required_zeros > _trailing_zeros(value_tuple.digits):
        return False
    significant_digits = value_tuple.digits[:-required_zeros]
    return _digits_mod(significant_digits, divisor) == 0


def _arithmetic_precision(values: Sequence[Decimal]) -> int:
    nonzero = tuple(value for value in values if value != _ZERO)
    if not nonzero:
        return _MIN_WORKING_PRECISION
    operand_digits = sum(
        len(value.as_tuple().digits) for value in nonzero
    )
    carry_digits = (
        _small_integer_digit_count(len(nonzero)) + _GUARD_DIGITS
    )
    required = max(
        _MIN_WORKING_PRECISION,
        operand_digits + carry_digits,
    )
    return min(required, _MAX_WORKING_PRECISION)


def _coefficient(digits: Sequence[int]) -> int:
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    return coefficient


def _digits_mod(digits: Sequence[int], divisor: int) -> int:
    remainder = 0
    for digit in digits:
        remainder = (remainder * 10 + digit) % divisor
    return remainder


def _trailing_zeros(digits: Sequence[int]) -> int:
    count = 0
    for digit in reversed(digits):
        if digit != 0:
            break
        count += 1
    return count


def _normalized_components(
    value: Decimal,
) -> tuple[tuple[int, ...], int]:
    """Return context-free coefficient digits and exponent without zeros."""
    decimal_tuple = value.as_tuple()
    exponent = decimal_tuple.exponent
    if not value.is_finite() or not isinstance(exponent, int):
        raise InvalidOperation(
            "normalized components require a finite Decimal"
        )
    if value.is_zero():
        return (0,), 0
    trailing_zeros = _trailing_zeros(decimal_tuple.digits)
    if trailing_zeros == 0:
        return decimal_tuple.digits, exponent
    return (
        decimal_tuple.digits[:-trailing_zeros],
        exponent + trailing_zeros,
    )


def _floor_scaled_ratio(
    numerator: int,
    denominator: int,
    exponent_delta: int,
) -> tuple[int, ...]:
    numerator_digits = _integer_digits(numerator)
    denominator_digits = _integer_digits(denominator)
    if exponent_delta >= 0:
        scaled_length = len(numerator_digits) + exponent_delta
        if (
            scaled_length - len(denominator_digits)
            > _MAX_WORKING_PRECISION
        ):
            raise InvalidOperation(
                "lot count exceeds working precision"
            )
        remainder = 0
        quotient_digits: list[int] = []
        for digit in chain(
            numerator_digits,
            repeat(0, exponent_delta),
        ):
            quotient_digit, remainder = divmod(
                remainder * 10 + digit,
                denominator,
            )
            if quotient_digit or quotient_digits:
                quotient_digits.append(quotient_digit)
                if (
                    len(quotient_digits)
                    > _MAX_WORKING_PRECISION
                ):
                    raise InvalidOperation(
                        "lot count exceeds working precision"
                    )
        return tuple(quotient_digits or (0,))

    decimal_shift = -exponent_delta
    scaled_denominator_length = (
        len(denominator_digits) + decimal_shift
    )
    if scaled_denominator_length > len(numerator_digits):
        return (0,)
    if scaled_denominator_length == len(numerator_digits):
        scaled_denominator_digits = (
            denominator_digits + (0,) * decimal_shift
        )
        if numerator_digits < scaled_denominator_digits:
            return (0,)
    scaled_denominator = denominator * 10**decimal_shift
    return _integer_digits(numerator // scaled_denominator)


def _integer_digits(
    value: int,
    *,
    max_digits: int = _MAX_WORKING_PRECISION,
) -> tuple[int, ...]:
    """Convert a bounded integer to digits without Python string conversion."""
    remaining = abs(value)
    if remaining == 0:
        return (0,)
    maximum_chunks = (
        max_digits + _INTEGER_CHUNK_DIGITS - 1
    ) // _INTEGER_CHUNK_DIGITS
    chunks: list[int] = []
    while remaining and len(chunks) < maximum_chunks:
        remaining, chunk = divmod(
            remaining,
            _INTEGER_CHUNK_BASE,
        )
        chunks.append(chunk)
    if remaining:
        raise InvalidOperation(
            "integer exceeds working precision"
        )

    highest_digits = _small_integer_digits(chunks.pop())
    total_digits = (
        len(highest_digits)
        + len(chunks) * _INTEGER_CHUNK_DIGITS
    )
    if total_digits > max_digits:
        raise InvalidOperation(
            "integer exceeds working precision"
        )
    digits = list(highest_digits)
    for chunk in reversed(chunks):
        chunk_digits = _small_integer_digits(chunk)
        digits.extend(
            (0,) * (_INTEGER_CHUNK_DIGITS - len(chunk_digits))
        )
        digits.extend(chunk_digits)
    return tuple(digits)


def _small_integer_digits(value: int) -> tuple[int, ...]:
    if value == 0:
        return (0,)
    reversed_digits: list[int] = []
    while value:
        value, digit = divmod(value, 10)
        reversed_digits.append(digit)
    reversed_digits.reverse()
    return tuple(reversed_digits)


def _small_integer_digit_count(value: int) -> int:
    count = 1
    while value >= 10:
        value //= 10
        count += 1
    return count


__all__ = [
    "decimal_context",
    "exact_add",
    "exact_multiply",
    "exact_subtract",
    "floor_quantity_to_lot",
    "is_aligned",
]
