"""Library-owned Decimal contexts for deterministic risk arithmetic."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    Inexact,
    InvalidOperation,
    localcontext,
)

_ZERO = Decimal("0")
_MIN_WORKING_PRECISION = 64
_MAX_WORKING_PRECISION = 4096
_GUARD_DIGITS = 16


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
    carry_digits = len(str(len(nonzero))) + _GUARD_DIGITS
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


__all__ = [
    "decimal_context",
    "exact_add",
    "exact_multiply",
    "exact_subtract",
    "is_aligned",
]
