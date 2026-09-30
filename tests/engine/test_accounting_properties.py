"""Property tests for signed position and cash reconciliation."""

from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Decimal
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


def independent_book_oracle(
    generated: list[Fill],
) -> tuple[Decimal, Decimal, Decimal]:
    quantity = Decimal("0")
    book_cost = Decimal("0")
    realized = Decimal("0")
    quantum = Decimal("0.01")
    for item in generated:
        signed_fill = (
            item.quantity.value if item.side is OrderSide.BUY else -item.quantity.value
        )
        new_quantity = quantity + signed_fill
        fill_notional = item.quantity.value * item.price.amount
        if quantity == 0:
            book_cost = fill_notional
        elif (quantity > 0) == (signed_fill > 0):
            book_cost += fill_notional
        else:
            closed = min(abs(quantity), abs(signed_fill))
            if closed == abs(quantity):
                allocated_book = book_cost
            else:
                allocated_book = (book_cost * closed / abs(quantity)).quantize(
                    quantum,
                    rounding=ROUND_HALF_EVEN,
                )
            direction = Decimal("1") if quantity > 0 else Decimal("-1")
            realized += direction * (closed * item.price.amount - allocated_book)
            if new_quantity == 0:
                book_cost = Decimal("0")
            elif (quantity > 0) == (new_quantity > 0):
                book_cost -= allocated_book
            else:
                book_cost = abs(new_quantity) * item.price.amount
        quantity = new_quantity
    return quantity, book_cost, realized


@given(st.lists(fill_terms, min_size=1, max_size=40))
@settings(max_examples=80, deadline=None)
def test_fill_sequences_reconcile_signed_position_fees_cash_and_time(
    terms: list[tuple[OrderSide, int, int, int]],
) -> None:
    generated = executions(terms)
    ledger = account()
    snapshots = [ledger.apply_fill(item) for item in generated]
    snapshot = ledger.mark_to_market(
        timestamp=BASE_DATETIME + timedelta(seconds=len(generated)),
        prices={AAPL.id: Money.usd("100")},
    )

    expected_position, expected_book, expected_realized = independent_book_oracle(
        generated
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
    expected_market_value = expected_position * Decimal("100")
    direction = (
        Decimal("1")
        if expected_position > 0
        else Decimal("-1")
        if expected_position < 0
        else Decimal("0")
    )
    expected_unrealized = expected_market_value - direction * expected_book

    assert snapshot.positions[AAPL.id].quantity.value == expected_position
    assert snapshot.positions[AAPL.id].book_cost.amount == expected_book
    assert snapshot.positions[AAPL.id].realized_pnl.amount == expected_realized
    assert snapshot.realized_pnl.amount == expected_realized
    assert snapshot.unrealized_pnl.amount == expected_unrealized
    assert snapshot.market_value.amount == expected_market_value
    assert snapshot.total_fees.amount == expected_fees
    assert snapshot.cash.amount == Decimal("100000") + expected_cash_delta
    assert snapshot.equity.amount == snapshot.cash.amount + expected_market_value
    assert all(
        previous.timestamp is not None
        and current.timestamp is not None
        and current.timestamp >= previous.timestamp
        for previous, current in pairwise(snapshots)
    )


@given(
    opening_side=st.sampled_from(tuple(OrderSide)),
    entries=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=25),
            st.integers(min_value=1, max_value=50_000),
        ),
        min_size=2,
        max_size=12,
    ),
    closing_price_cents=st.integers(min_value=1, max_value=50_000),
)
@settings(max_examples=80, deadline=None)
def test_repeated_entries_then_full_close_have_exact_realized_and_zero_unrealized(
    opening_side: OrderSide,
    entries: list[tuple[int, int]],
    closing_price_cents: int,
) -> None:
    closing_side = OrderSide.SELL if opening_side is OrderSide.BUY else OrderSide.BUY
    generated = [
        fill(
            AAPL.id,
            opening_side,
            quantity=str(quantity),
            price=Decimal(price_cents) / Decimal("100"),
            offset=index,
        )
        for index, (quantity, price_cents) in enumerate(entries)
    ]
    total_quantity = sum(
        (item.quantity.value for item in generated),
        Decimal("0"),
    )
    closing_price = Decimal(closing_price_cents) / Decimal("100")
    closing = fill(
        AAPL.id,
        closing_side,
        quantity=total_quantity,
        price=closing_price,
        offset=len(generated),
    )
    ledger = account()
    for item in generated:
        ledger.apply_fill(item)
    snapshot = ledger.apply_fill(closing)

    entry_book = sum(
        (item.quantity.value * item.price.amount for item in generated),
        Decimal("0"),
    )
    direction = Decimal("1") if opening_side is OrderSide.BUY else Decimal("-1")
    expected_realized = direction * (total_quantity * closing_price - entry_book)
    position = snapshot.positions[AAPL.id]
    assert position.quantity.value == Decimal("0")
    assert position.book_cost == Money.usd("0")
    assert position.realized_pnl.amount == expected_realized
    assert snapshot.realized_pnl.amount == expected_realized
    assert snapshot.unrealized_pnl == Money.usd("0")
    assert snapshot.market_value == Money.usd("0")


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
    closing_side = OrderSide.SELL if opening_side is OrderSide.BUY else OrderSide.BUY
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
