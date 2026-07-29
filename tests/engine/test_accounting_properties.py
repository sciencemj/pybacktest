"""Property tests for signed position and cash reconciliation."""

from datetime import timedelta
from decimal import Decimal
from itertools import pairwise

from hypothesis import given, settings
from hypothesis import strategies as st

from pybacktest.domain.money import Money
from pybacktest.domain.orders import Fill, OrderSide
from pybacktest.engine.accounting import PortfolioLedger
from tests.factories import BASE_DATETIME, aapl, fill

AAPL = aapl()

fill_terms = st.tuples(
    st.sampled_from(tuple(OrderSide)),
    st.integers(min_value=1, max_value=25),
    st.integers(min_value=1, max_value=50_000),
    st.integers(min_value=0, max_value=500),
)


def account() -> PortfolioLedger:
    return PortfolioLedger(
        base_currency="USD",
        initial_cash=Money.usd("100000"),
        instruments={AAPL.id: AAPL},
    )


def executions(
    terms: list[tuple[OrderSide, int, int, int]],
) -> list[Fill]:
    return [
        fill(
            AAPL.id,
            side,
            quantity=str(quantity),
            price=Decimal(price_cents) / Decimal("100"),
            fee=Decimal(fee_cents) / Decimal("100"),
            offset=index,
        )
        for index, (side, quantity, price_cents, fee_cents) in enumerate(terms)
    ]


@given(st.lists(fill_terms, min_size=1, max_size=40))
@settings(max_examples=80, deadline=None)
def test_fill_sequences_reconcile_signed_position_fees_cash_and_time(
    terms: list[tuple[OrderSide, int, int, int]],
) -> None:
    generated = executions(terms)
    ledger = account()
    snapshots = [ledger.apply_fill(item) for item in generated]

    expected_position = sum(
        (
            item.quantity.value
            if item.side is OrderSide.BUY
            else -item.quantity.value
        )
        for item in generated
    )
    expected_fees = sum((item.fee.amount for item in generated), Decimal("0"))
    expected_cash_delta = sum(
        (
            -item.quantity.value * item.price.amount
            if item.side is OrderSide.BUY
            else item.quantity.value * item.price.amount
        )
        - item.fee.amount
        for item in generated
    )

    assert snapshots[-1].positions[AAPL.id].quantity.value == expected_position
    assert snapshots[-1].total_fees.amount == expected_fees
    assert snapshots[-1].cash.amount == Decimal("100000") + expected_cash_delta
    assert all(
        previous.timestamp is not None
        and current.timestamp is not None
        and current.timestamp >= previous.timestamp
        for previous, current in pairwise(snapshots)
    )


@given(
    opening_side=st.sampled_from(tuple(OrderSide)),
    opening_quantity=st.integers(min_value=1, max_value=100),
    excess_quantity=st.integers(min_value=1, max_value=100),
    opening_price_cents=st.integers(min_value=1, max_value=50_000),
    crossing_price_cents=st.integers(min_value=1, max_value=50_000),
)
@settings(max_examples=80, deadline=None)
def test_zero_crossing_sequences_realize_only_closed_leg_and_rebase_open_leg(
    opening_side: OrderSide,
    opening_quantity: int,
    excess_quantity: int,
    opening_price_cents: int,
    crossing_price_cents: int,
) -> None:
    closing_side = (
        OrderSide.SELL
        if opening_side is OrderSide.BUY
        else OrderSide.BUY
    )
    opening_price = Decimal(opening_price_cents) / Decimal("100")
    crossing_price = Decimal(crossing_price_cents) / Decimal("100")
    ledger = account()
    ledger.apply_fill(
        fill(
            AAPL.id,
            opening_side,
            quantity=str(opening_quantity),
            price=opening_price,
            offset=0,
        )
    )
    snapshot = ledger.apply_fill(
        fill(
            AAPL.id,
            closing_side,
            quantity=str(opening_quantity + excess_quantity),
            price=crossing_price,
            offset=1,
        )
    )

    expected_position = (
        Decimal(-excess_quantity)
        if opening_side is OrderSide.BUY
        else Decimal(excess_quantity)
    )
    pnl_per_unit = (
        crossing_price - opening_price
        if opening_side is OrderSide.BUY
        else opening_price - crossing_price
    )
    assert snapshot.positions[AAPL.id].quantity.value == expected_position
    assert snapshot.positions[AAPL.id].average_price == Money.usd(crossing_price)
    assert snapshot.realized_pnl.amount == pnl_per_unit * opening_quantity


@given(st.lists(fill_terms, min_size=1, max_size=20))
@settings(max_examples=40, deadline=None)
def test_equal_timestamp_fills_are_monotonic_and_all_apply(
    terms: list[tuple[OrderSide, int, int, int]],
) -> None:
    ledger = account()
    snapshots = [
        ledger.apply_fill(
            fill(
                AAPL.id,
                side,
                quantity=str(quantity),
                price=Decimal(price_cents) / Decimal("100"),
                fee=Decimal(fee_cents) / Decimal("100"),
                offset=index,
            )
        )
        for index, (side, quantity, price_cents, fee_cents) in enumerate(terms)
    ]
    final = ledger.mark_to_market(
        timestamp=BASE_DATETIME + timedelta(seconds=len(terms) - 1),
        prices={AAPL.id: Money.usd("100")},
    )

    assert final.timestamp == snapshots[-1].timestamp
