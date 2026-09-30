"""Bounded Decimal arithmetic independent of the ambient process context."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    Clamped,
    Context,
    Decimal,
    DecimalException,
    Inexact,
    Subnormal,
    Underflow,
    localcontext,
)

from pybacktest.domain.errors import PybacktestError

MAX_EXACT_DIGITS = 4096


class ExactDecimalError(PybacktestError, ArithmeticError):
    """Raised when exact arithmetic exceeds the supported finite envelope."""


def _require_finite(value: Decimal) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ExactDecimalError("exact arithmetic requires finite Decimals.")
    if len(value.as_tuple().digits) > MAX_EXACT_DIGITS:
        raise ExactDecimalError("Decimal coefficient exceeds the exact digit limit.")


def _context(precision: int, *, exact: bool = True) -> Context:
    if precision > MAX_EXACT_DIGITS:
        raise ExactDecimalError("exact arithmetic exceeds the digit limit.")
    context = Context(
        prec=max(1, precision),
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
    )
    if exact:
        context.traps[Inexact] = True
    for signal in (Underflow, Subnormal, Clamped):
        context.traps[signal] = True
    return context


def exact_add(left: Decimal, right: Decimal) -> Decimal:
    """Add two finite Decimals exactly within an explicit bounded envelope."""
    _require_finite(left)
    _require_finite(right)
    if left.is_zero():
        return right
    if right.is_zero():
        return left
    left_exponent = left.as_tuple().exponent
    right_exponent = right.as_tuple().exponent
    if not isinstance(left_exponent, int) or not isinstance(
        right_exponent,
        int,
    ):
        raise ExactDecimalError("exact arithmetic requires finite Decimals.")
    minimum_exponent = min(left_exponent, right_exponent)
    required = max(left.adjusted(), right.adjusted()) - minimum_exponent + 2
    try:
        with localcontext(_context(required)):
            result = left + right
    except DecimalException as error:
        raise ExactDecimalError("exact Decimal addition is unsupported.") from error
    if not result.is_finite():
        raise ExactDecimalError("exact Decimal addition produced a non-finite value.")
    return result


def exact_sum(values: Iterable[Decimal]) -> Decimal:
    """Sum finite Decimals exactly without consulting the ambient context."""
    total = Decimal("0")
    for value in values:
        total = exact_add(total, value)
    return total


def exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    """Multiply two finite Decimals exactly within an explicit digit limit."""
    _require_finite(left)
    _require_finite(right)
    if left.is_zero() or right.is_zero():
        return Decimal("0")
    required = len(left.as_tuple().digits) + len(right.as_tuple().digits)
    try:
        with localcontext(_context(required)):
            result = left * right
    except DecimalException as error:
        raise ExactDecimalError(
            "exact Decimal multiplication is unsupported."
        ) from error
    if not result.is_finite():
        raise ExactDecimalError(
            "exact Decimal multiplication produced a non-finite value."
        )
    return result


def calculation_context(*values: Decimal, minimum_precision: int = 64) -> Context:
    """Return a deterministic bounded context sized for supplied coefficients."""
    for value in values:
        _require_finite(value)
    precision = max(
        [minimum_precision] + [len(value.as_tuple().digits) + 16 for value in values]
    )
    return _context(precision, exact=False)


__all__ = [
    "MAX_EXACT_DIGITS",
    "ExactDecimalError",
    "calculation_context",
    "exact_add",
    "exact_multiply",
    "exact_sum",
]
