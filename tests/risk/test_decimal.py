"""Focused tests for shared deterministic Decimal risk arithmetic."""

from decimal import Decimal, Inexact, Rounded, localcontext

import pytest

from pybacktest.risk._decimal import floor_quantity_to_lot
from tests.risk._oracles import fraction_decimal_lot_floor


@pytest.mark.parametrize(
    ("notional", "lot_size", "expected"),
    [
        ("1", "0.25", "1"),
        ("0.999999999999999999999999", "0.25", "0.75"),
        ("1.000000000000000000000001", "0.25", "1"),
        ("0.9", "0.3", "0.9"),
        ("0.899999999999999999999999", "0.3", "0.6"),
        ("0.900000000000000000000001", "0.3", "0.9"),
    ],
)
def test_lot_floor_handles_exact_and_adjacent_non_power_boundaries(
    notional: str,
    lot_size: str,
    expected: str,
):
    result = floor_quantity_to_lot(
        Decimal(notional),
        Decimal("1"),
        Decimal(lot_size),
    )

    assert result == Decimal(expected)


def test_lot_floor_owns_precision_and_traps_for_nonterminating_ratio():
    expected = fraction_decimal_lot_floor(1, 3, 100)

    with localcontext() as ambient:
        ambient.prec = 7
        ambient.traps[Inexact] = True
        ambient.traps[Rounded] = True
        result = floor_quantity_to_lot(
            Decimal("1"),
            Decimal("3"),
            Decimal("1E-100"),
        )

    assert result == expected
