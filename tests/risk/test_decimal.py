"""Focused tests for shared deterministic Decimal risk arithmetic."""

import random
from decimal import (
    Decimal,
    Inexact,
    InvalidOperation,
    Rounded,
    localcontext,
)

import pytest

from pybacktest.risk._decimal import floor_quantity_to_lot
from tests.risk._oracles import (
    decimal_to_fraction,
    fraction_decimal_lot_floor,
    fraction_lot_floor,
)


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


@pytest.mark.parametrize(
    ("notional", "expected", "is_signed"),
    [
        (Decimal("-0.6"), Decimal("-0.6"), True),
        (Decimal("-1"), Decimal("-0.3"), True),
        (Decimal("-0.01"), Decimal("-0"), True),
        (Decimal("0"), Decimal("0"), False),
        (Decimal("-0"), Decimal("-0"), True),
    ],
)
def test_lot_floor_preserves_direction_and_signed_zero(
    notional: Decimal,
    expected: Decimal,
    is_signed: bool,
):
    result = floor_quantity_to_lot(
        notional,
        Decimal("3") if notional.copy_abs() == Decimal("1") else Decimal("1"),
        Decimal("0.1"),
    )

    assert result == expected
    assert result.is_signed() is is_signed


def test_lot_floor_normalizes_before_common_factor_cancellation():
    common = Decimal("1" + "0" * 2999 + "1")
    three_common = Decimal("3" + "0" * 2999 + "3")

    result = floor_quantity_to_lot(
        common,
        three_common,
        Decimal("0.1"),
    )

    assert result == Decimal("0.3")
    assert decimal_to_fraction(result) == fraction_lot_floor(
        common,
        three_common,
        Decimal("0.1"),
    )


def test_lot_floor_strips_stored_trailing_zeroes_before_resource_guard():
    stored_trailing_zeroes = Decimal("1." + "0" * 4094)

    result = floor_quantity_to_lot(
        stored_trailing_zeroes,
        Decimal("3"),
        Decimal("0.1"),
    )

    assert result == Decimal("0.3")
    assert decimal_to_fraction(result) == fraction_lot_floor(
        stored_trailing_zeroes,
        Decimal("3"),
        Decimal("0.1"),
    )


def test_lot_floor_accepts_exact_work_bound_and_rejects_next_digit():
    at_bound = floor_quantity_to_lot(
        Decimal("1"),
        Decimal("3"),
        Decimal("1E-4096"),
    )

    assert at_bound == fraction_decimal_lot_floor(1, 3, 4096)
    with pytest.raises(InvalidOperation):
        floor_quantity_to_lot(
            Decimal("1"),
            Decimal("3"),
            Decimal("1E-4097"),
        )


def test_lot_floor_bounds_exact_aligned_result_significant_digits():
    at_bound = Decimal("9" * 4096)

    assert (
        floor_quantity_to_lot(
            at_bound,
            Decimal("1"),
            Decimal("1"),
        )
        == at_bound
    )
    with pytest.raises(InvalidOperation):
        floor_quantity_to_lot(
            Decimal("9" * 4097),
            Decimal("1"),
            Decimal("1"),
        )


def test_lot_floor_rejects_excessive_irreducible_input_deterministically():
    with pytest.raises(
        InvalidOperation,
        match="cancellation resource bound",
    ):
        floor_quantity_to_lot(
            Decimal("7" * 9000),
            Decimal("3"),
            Decimal("0.1"),
        )


def test_lot_floor_matches_independent_fraction_oracle_for_varied_inputs():
    generator = random.Random(20260729)
    lots = (
        Decimal("0.25"),
        Decimal("0.3"),
        Decimal("1E-3"),
        Decimal("2.5E+2"),
    )

    for _ in range(100):
        sign = "-" if generator.randrange(2) else ""
        notional = Decimal(
            f"{sign}{generator.randint(1, 999999)}E{generator.randint(-8, 8)}"
        )
        price = Decimal(f"{generator.randint(1, 999)}E{generator.randint(-4, 4)}")
        lot_size = generator.choice(lots)

        result = floor_quantity_to_lot(
            notional,
            price,
            lot_size,
        )

        assert decimal_to_fraction(result) == fraction_lot_floor(
            notional,
            price,
            lot_size,
        )


def test_positive_risk_cap_inputs_never_produce_a_negative_quantity():
    result = floor_quantity_to_lot(
        Decimal("1"),
        Decimal("3"),
        Decimal("0.1"),
    )

    assert result == Decimal("0.3")
    assert not result.is_signed()
