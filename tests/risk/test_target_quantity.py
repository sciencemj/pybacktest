"""Signed absolute-position contract for target-quantity sizing."""

from datetime import UTC
from decimal import Decimal

import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.events import OrderRejected
from pybacktest.domain.instruments import Instrument
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    Order,
    OrderSide,
    TargetQuantity,
)
from pybacktest.risk.sizing import DefaultOrderSizer
from tests.factories import instrument, portfolio_snapshot, risk_context


def _lot_two_instrument() -> Instrument:
    base = instrument()
    return Instrument(
        id=base.id,
        quote_currency=base.quote_currency,
        tick_size=Decimal("0.01"),
        lot_size=Decimal("2"),
        timezone=UTC,
    )


@pytest.mark.parametrize(
    ("current", "target", "expected_side", "expected_quantity"),
    [
        ("4", "4", None, None),
        ("4", "0", OrderSide.SELL, "4"),
        ("4", "-4", OrderSide.SELL, "8"),
        ("0", "4", OrderSide.BUY, "4"),
        ("0", "0", None, None),
        ("0", "-4", OrderSide.SELL, "4"),
        ("-4", "4", OrderSide.BUY, "8"),
        ("-4", "0", OrderSide.BUY, "4"),
        ("-4", "-4", None, None),
    ],
)
def test_target_quantity_sizes_signed_absolute_positions(
    current: str,
    target: str,
    expected_side: OrderSide | None,
    expected_quantity: str | None,
) -> None:
    item = _lot_two_instrument()
    snapshot = portfolio_snapshot(
        cash="10000",
        positions={item.id: current},
        prices={item.id: "100"},
    )
    context = risk_context(
        snapshot=snapshot,
        prices={item.id: Money.usd("100")},
        instruments={item.id: item},
    )

    result = DefaultOrderSizer().size(
        TargetQuantity(
            instrument=item.id,
            quantity=Quantity.of(target),
            reason=DecisionReason.of("absolute_position"),
        ),
        context,
    )

    if expected_side is None:
        assert isinstance(result, OrderRejected)
        assert result.reason.code == "no_op_target"
        return
    assert isinstance(result, Order)
    assert result.side is expected_side
    assert result.quantity == Quantity.of(expected_quantity)


def test_signed_target_must_align_to_the_instrument_lot() -> None:
    item = _lot_two_instrument()
    context = risk_context(
        snapshot=portfolio_snapshot(
            positions={item.id: "0"},
            prices={item.id: "100"},
        ),
        prices={item.id: Money.usd("100")},
        instruments={item.id: item},
    )

    result = DefaultOrderSizer().size(
        TargetQuantity(
            instrument=item.id,
            quantity=Quantity.of("-3"),
            reason=DecisionReason.of("unaligned_short_target"),
        ),
        context,
    )

    assert isinstance(result, OrderRejected)
    assert result.reason.code == "invalid_quantity_lot"


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "-Infinity"])
def test_target_numeric_values_cannot_bypass_quantity_validation(
    value: object,
) -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        Quantity.of(value)
