"""Library-owned Decimal contexts for deterministic risk arithmetic."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from decimal import (
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    localcontext,
)

_ZERO = Decimal("0")


@contextmanager
def decimal_context(values: Sequence[Decimal]) -> Iterator[Context]:
    """Use operand-derived precision and the widest Decimal exponent range."""
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


def _arithmetic_precision(values: Sequence[Decimal]) -> int:
    nonzero = tuple(value for value in values if value != _ZERO)
    if not nonzero:
        return 64
    highest_place = max(value.adjusted() for value in nonzero)
    exponents: list[int] = []
    for value in nonzero:
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int):
            return 64
        exponents.append(exponent)
    lowest_place = min(exponents)
    operand_digits = sum(
        len(value.as_tuple().digits) for value in nonzero
    )
    carry_digits = len(str(len(nonzero))) + 16
    required = max(
        64,
        highest_place - lowest_place + operand_digits + carry_digits,
    )
    return min(required, MAX_PREC)


__all__ = ["decimal_context"]
