"""Exact examples and invariant tests for the portfolio ledger."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from pybacktest.domain.errors import AccountingInvariantError
from pybacktest.domain.identifiers import CashEventId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money
from pybacktest.domain.orders import OrderSide
from pybacktest.domain.portfolio import (
    CashEvent,
    CashEventCode,
    PortfolioSnapshot,
)
from pybacktest.engine.accounting import PortfolioLedger
from tests.factories import BASE_DATETIME, aapl, fill, instrument

AAPL = aapl()
MSFT = instrument("MSFT")


def ledger(
    *,
    items: tuple[Instrument, ...] = (AAPL,),
    initial_cash: object = "10000",
) -> PortfolioLedger:
    return PortfolioLedger(
        base_currency="USD",
        initial_cash=Money.usd(initial_cash),
        instruments={item.id: item for item in items},
    )


def test_long_entry_partial_exit_and_fees_reconcile() -> None:
    account = ledger()
    account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.BUY,
            quantity="10",
            price="100",
            fee="1",
            offset=0,
        )
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.SELL,
            quantity="4",
            price="110",
            fee="1",
            offset=1,
        )
    )

    assert snapshot.cash.amount == Decimal("9438")
    assert snapshot.positions[AAPL.id].quantity.value == Decimal("6")
    assert snapshot.positions[AAPL.id].average_price == Money.usd("100")
    assert snapshot.realized_pnl.amount == Decimal("40")
    assert snapshot.total_fees.amount == Decimal("2")


def test_short_entry_partial_exit_and_fees_reconcile() -> None:
    account = ledger()
    account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.SELL,
            quantity="10",
            price="100",
            fee="1",
            offset=0,
        )
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.BUY,
            quantity="4",
            price="90",
            fee="1",
            offset=1,
        )
    )

    assert snapshot.cash.amount == Decimal("10638")
    assert snapshot.positions[AAPL.id].quantity.value == Decimal("-6")
    assert snapshot.positions[AAPL.id].average_price == Money.usd("100")
    assert snapshot.realized_pnl.amount == Decimal("40")
    assert snapshot.total_fees.amount == Decimal("2")


def test_sell_crossing_zero_opens_short_at_fill_price() -> None:
    account = ledger()
    account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.BUY,
            quantity="10",
            price="100",
            offset=0,
        )
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.SELL,
            quantity="15",
            price="110",
            offset=1,
        )
    )

    position = snapshot.positions[AAPL.id]
    assert position.quantity.value == Decimal("-5")
    assert position.average_price == Money.usd("110")
    assert snapshot.realized_pnl.amount == Decimal("100")


def test_buy_crossing_zero_opens_long_at_fill_price() -> None:
    account = ledger()
    account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.SELL,
            quantity="10",
            price="100",
            offset=0,
        )
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.BUY,
            quantity="15",
            price="90",
            offset=1,
        )
    )

    position = snapshot.positions[AAPL.id]
    assert position.quantity.value == Decimal("5")
    assert position.average_price == Money.usd("90")
    assert snapshot.realized_pnl.amount == Decimal("100")


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_same_side_fills_use_quantity_weighted_average(side: OrderSide) -> None:
    account = ledger()
    account.apply_fill(
        fill(
            AAPL.id,
            side,
            quantity="10",
            price="100",
            offset=0,
        )
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            side,
            quantity="5",
            price="106",
            offset=1,
        )
    )

    position = snapshot.positions[AAPL.id]
    expected_quantity = Decimal("15") if side is OrderSide.BUY else Decimal("-15")
    assert position.quantity.value == expected_quantity
    assert position.average_price == Money.usd("102")
    assert position.realized_pnl == Money.usd("0")


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_flat_position_resets_average_price(side: OrderSide) -> None:
    closing_side = OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY
    account = ledger()
    account.apply_fill(
        fill(AAPL.id, side, quantity="3", price="100", offset=0)
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            closing_side,
            quantity="3",
            price="105",
            offset=1,
        )
    )

    assert snapshot.positions[AAPL.id].quantity.value == Decimal("0")
    assert snapshot.positions[AAPL.id].average_price is None


def test_fees_reduce_cash_once_but_do_not_reduce_realized_pnl() -> None:
    account = ledger()
    account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.BUY,
            quantity="2",
            price="100",
            fee="3",
            offset=0,
        )
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.SELL,
            quantity="2",
            price="110",
            fee="4",
            offset=1,
        )
    )

    assert snapshot.cash == Money.usd("10013")
    assert snapshot.realized_pnl == Money.usd("20")
    assert snapshot.total_fees == Money.usd("7")


def test_mark_to_market_does_not_mutate_previous_snapshot() -> None:
    account = ledger()
    before = account.snapshot()
    mark_timestamp = BASE_DATETIME
    after = account.mark_to_market(
        timestamp=mark_timestamp,
        prices={AAPL.id: Money.usd("105")},
    )

    assert before is not after
    assert before.timestamp is None
    assert after.timestamp == mark_timestamp
    assert before.valuation_prices == {}
    assert after.valuation_prices == {AAPL.id: Money.usd("105")}


def test_mark_to_market_changes_only_valuation_fields() -> None:
    account = ledger()
    before = account.apply_fill(
        fill(
            AAPL.id,
            OrderSide.BUY,
            quantity="10",
            price="100",
            fee="2",
            offset=0,
        )
    )
    after = account.mark_to_market(
        timestamp=BASE_DATETIME + timedelta(seconds=1),
        prices={AAPL.id: Money.usd("110")},
    )

    assert after.cash == before.cash
    assert after.positions == before.positions
    assert after.realized_pnl == before.realized_pnl
    assert after.total_fees == before.total_fees
    assert after.unrealized_pnl == Money.usd("100")
    assert after.market_value == Money.usd("1100")
    assert after.gross_exposure == Money.usd("1100")
    assert after.equity == Money.usd("10098")


def test_repeating_weighted_average_full_close_is_exactly_flat() -> None:
    account = ledger()
    account.apply_fill(
        fill(AAPL.id, OrderSide.BUY, quantity="1", price="100", offset=0)
    )
    account.apply_fill(
        fill(AAPL.id, OrderSide.BUY, quantity="2", price="101", offset=1)
    )
    snapshot = account.apply_fill(
        fill(AAPL.id, OrderSide.SELL, quantity="3", price="102", offset=2)
    )

    position = snapshot.positions[AAPL.id]
    assert position.quantity.value == Decimal("0")
    assert position.average_price is None
    assert position.book_cost == Money.usd("0")
    assert snapshot.equity == Money.usd("10004")
    assert snapshot.realized_pnl == Money.usd("4")
    assert snapshot.unrealized_pnl == Money.usd("0")


@pytest.mark.parametrize(
    ("opening_side", "closing_side", "close_price", "realized", "unrealized"),
    [
        (OrderSide.BUY, OrderSide.SELL, "102", "1.33", "2.67"),
        (OrderSide.SELL, OrderSide.BUY, "99", "1.67", "3.33"),
    ],
)
def test_partial_close_allocates_book_cost_at_notional_quantum(
    opening_side: OrderSide,
    closing_side: OrderSide,
    close_price: str,
    realized: str,
    unrealized: str,
) -> None:
    account = ledger()
    account.apply_fill(
        fill(AAPL.id, opening_side, quantity="1", price="100", offset=0)
    )
    account.apply_fill(
        fill(AAPL.id, opening_side, quantity="2", price="101", offset=1)
    )
    snapshot = account.apply_fill(
        fill(
            AAPL.id,
            closing_side,
            quantity="1",
            price=close_price,
            offset=2,
        )
    )

    position = snapshot.positions[AAPL.id]
    assert position.book_cost == Money.usd("201.33")
    assert snapshot.realized_pnl == Money.usd(realized)
    assert snapshot.unrealized_pnl == Money.usd(unrealized)


def test_mark_to_market_drops_missing_prices() -> None:
    account = ledger(items=(AAPL, MSFT))
    account.mark_to_market(
        timestamp=BASE_DATETIME,
        prices={
            AAPL.id: Money.usd("100"),
            MSFT.id: Money.usd("200"),
        },
    )
    snapshot = account.mark_to_market(
        timestamp=BASE_DATETIME + timedelta(seconds=1),
        prices={AAPL.id: Money.usd("101")},
    )

    assert snapshot.valuation_prices == {AAPL.id: Money.usd("101")}
    assert MSFT.id not in snapshot.valuation_prices


def test_cash_event_is_typed_reconciled_and_timestamped() -> None:
    account = ledger()
    event = CashEvent(
        id=CashEventId.parse("cash_event_00000000000000000000000000000000"),
        timestamp=BASE_DATETIME,
        amount=Money.usd("-250"),
        code=CashEventCode.EXTERNAL_FLOW,
    )

    snapshot = account.apply_cash_event(event)

    assert snapshot.cash == Money.usd("9750")
    assert snapshot.timestamp == BASE_DATETIME
    assert snapshot.cash_events == (event,)


def test_snapshots_and_cash_events_are_frozen_and_defensive() -> None:
    catalog = {AAPL.id: AAPL}
    account = PortfolioLedger(
        base_currency="USD",
        initial_cash=Money.usd("10000"),
        instruments=catalog,
    )
    catalog.clear()
    account.apply_fill(
        fill(AAPL.id, OrderSide.BUY, quantity="1", price="100")
    )
    snapshot = account.apply_cash_event(
        CashEvent(
            id=CashEventId.parse(
                "cash_event_00000000000000000000000000000000"
            ),
            timestamp=BASE_DATETIME,
            amount=Money.usd("1"),
            code=CashEventCode.EXTERNAL_FLOW,
        )
    )

    with pytest.raises(TypeError):
        snapshot.positions[AAPL.id] = snapshot.positions[AAPL.id]  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.valuation_prices[AAPL.id] = Money.usd("1")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.cash = Money.usd("0")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.cash_events[0].amount = Money.usd("0")  # type: ignore[misc]


def test_domain_snapshot_is_distinct_from_strategy_quantity_view() -> None:
    from pybacktest.ports.strategy import PortfolioSnapshot as StrategySnapshot

    snapshot = ledger().snapshot()

    assert type(snapshot) is PortfolioSnapshot
    assert not isinstance(snapshot, StrategySnapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("realized_pnl", Money.usd("1")),
        ("unrealized_pnl", Money.usd("99")),
        ("total_fees", Money.usd("-1")),
        ("market_value", Money.usd("1099")),
        ("gross_exposure", Money.usd("-1")),
        ("gross_exposure", Money.usd("1099")),
        ("equity", Money.usd("10099")),
    ],
)
def test_direct_snapshot_rejects_contradictory_derived_aggregates(
    field: str,
    value: Money,
) -> None:
    account = ledger()
    account.apply_fill(
        fill(AAPL.id, OrderSide.BUY, quantity="10", price="100", offset=0)
    )
    valid = account.mark_to_market(
        timestamp=BASE_DATETIME + timedelta(seconds=1),
        prices={AAPL.id: Money.usd("110")},
    )

    with pytest.raises(AccountingInvariantError):
        replace(valid, **{field: value})


@pytest.mark.parametrize(
    "history_kind",
    ["reversed", "duplicate", "foreign_currency"],
)
def test_direct_snapshot_rejects_invalid_cash_event_history(
    history_kind: str,
) -> None:
    first = CashEvent(
        id=CashEventId.parse("cash_event_00000000000000000000000000000000"),
        timestamp=BASE_DATETIME,
        amount=Money.usd("1"),
        code=CashEventCode.EXTERNAL_FLOW,
    )
    second = CashEvent(
        id=CashEventId.parse("cash_event_00000000000000000000000000000001"),
        timestamp=BASE_DATETIME + timedelta(seconds=1),
        amount=Money.usd("1"),
        code=CashEventCode.EXTERNAL_FLOW,
    )
    foreign = CashEvent(
        id=CashEventId.parse("cash_event_00000000000000000000000000000002"),
        timestamp=BASE_DATETIME + timedelta(seconds=1),
        amount=Money.of("1", "EUR"),
        code=CashEventCode.EXTERNAL_FLOW,
    )
    account = ledger()
    account.apply_cash_event(first)
    valid = account.apply_cash_event(second)

    invalid_histories = {
        "reversed": (second, first),
        "duplicate": (first, first),
        "foreign_currency": (first, foreign),
    }
    with pytest.raises(AccountingInvariantError):
        replace(valid, cash_events=invalid_histories[history_kind])


@pytest.mark.parametrize(
    ("bad_fill", "message"),
    [
        (
            fill(
                InstrumentId.parse("XNAS:UNKNOWN"),
                OrderSide.BUY,
                offset=0,
            ),
            "unknown instrument",
        ),
        (
            fill(
                AAPL.id,
                OrderSide.BUY,
                currency="EUR",
                offset=0,
            ),
            "currency",
        ),
        (
            fill(
                AAPL.id,
                OrderSide.BUY,
                fee="-0.01",
                offset=0,
            ),
            "fee",
        ),
        (
            fill(
                AAPL.id,
                OrderSide.BUY,
                quantity="1.5",
                offset=0,
            ),
            "lot",
        ),
        (
            fill(
                AAPL.id,
                OrderSide.BUY,
                price="100.005",
                offset=0,
            ),
            "tick",
        ),
    ],
)
def test_invalid_fill_is_rejected_before_state_mutation(
    bad_fill: object,
    message: str,
) -> None:
    account = ledger()
    before = account.snapshot()

    with pytest.raises(AccountingInvariantError, match=message):
        account.apply_fill(bad_fill)  # type: ignore[arg-type]

    assert account.snapshot() == before


def test_duplicate_fill_id_is_rejected_without_mutation() -> None:
    account = ledger()
    execution = fill(AAPL.id, OrderSide.BUY, offset=0)
    accepted = account.apply_fill(execution)

    with pytest.raises(AccountingInvariantError, match="duplicate"):
        account.apply_fill(execution)

    assert account.snapshot() == accepted


def test_decimal_alignment_failure_uses_stable_error_without_mutation() -> None:
    account = ledger()
    before = account.snapshot()
    extreme = fill(
        AAPL.id,
        OrderSide.BUY,
        price="1e999999",
        offset=0,
    )

    with pytest.raises(AccountingInvariantError, match="tick"):
        account.apply_fill(extreme)

    assert account.snapshot() == before


def test_non_monotonic_fill_cash_and_mark_timestamps_are_rejected() -> None:
    operations = (
        lambda account: account.apply_fill(
            fill(AAPL.id, OrderSide.BUY, offset=0)
        ),
        lambda account: account.apply_cash_event(
            CashEvent(
                id=CashEventId.parse(
                    "cash_event_00000000000000000000000000000000"
                ),
                timestamp=BASE_DATETIME,
                amount=Money.usd("1"),
                code=CashEventCode.EXTERNAL_FLOW,
            )
        ),
        lambda account: account.mark_to_market(
            timestamp=BASE_DATETIME,
            prices={AAPL.id: Money.usd("100")},
        ),
    )
    later = BASE_DATETIME + timedelta(days=1)
    for operation in operations:
        account = ledger()
        accepted = account.mark_to_market(
            timestamp=later,
            prices={AAPL.id: Money.usd("100")},
        )

        with pytest.raises(AccountingInvariantError, match="timestamp"):
            operation(account)

        assert account.snapshot() == accepted


@pytest.mark.parametrize(
    "prices",
    [
        {InstrumentId.parse("XNAS:UNKNOWN"): Money.usd("1")},
        {AAPL.id: Money.of("100", "EUR")},
        {AAPL.id: Money.usd("0")},
    ],
)
def test_invalid_mark_is_rejected_before_state_mutation(
    prices: dict[InstrumentId, Money],
) -> None:
    account = ledger()
    before = account.snapshot()

    with pytest.raises(AccountingInvariantError):
        account.mark_to_market(timestamp=BASE_DATETIME, prices=prices)

    assert account.snapshot() == before


def test_cash_event_rejects_naive_time_and_wrong_currency() -> None:
    with pytest.raises(AccountingInvariantError, match="timezone-aware"):
        CashEvent(
            id=CashEventId.parse(
                "cash_event_00000000000000000000000000000000"
            ),
            timestamp=datetime(2024, 1, 2),
            amount=Money.usd("1"),
            code=CashEventCode.EXTERNAL_FLOW,
        )

    account = ledger()
    before = account.snapshot()
    event = CashEvent(
        id=CashEventId.parse("cash_event_00000000000000000000000000000000"),
        timestamp=datetime(2024, 1, 2, tzinfo=UTC),
        amount=Money.of("1", "EUR"),
        code=CashEventCode.EXTERNAL_FLOW,
    )
    with pytest.raises(AccountingInvariantError, match="currency"):
        account.apply_cash_event(event)
    assert account.snapshot() == before


def test_duplicate_cash_event_id_is_rejected_without_mutation() -> None:
    account = ledger()
    event = CashEvent(
        id=CashEventId.parse("cash_event_00000000000000000000000000000000"),
        timestamp=BASE_DATETIME,
        amount=Money.usd("25"),
        code=CashEventCode.EXTERNAL_FLOW,
    )
    accepted = account.apply_cash_event(event)

    with pytest.raises(AccountingInvariantError, match="duplicate"):
        account.apply_cash_event(event)

    assert account.snapshot() == accepted


def test_cash_event_preserves_significance_beyond_default_decimal_context() -> None:
    account = ledger(initial_cash="9999999999999999999999999999")
    event = CashEvent(
        id=CashEventId.parse("cash_event_00000000000000000000000000000000"),
        timestamp=BASE_DATETIME,
        amount=Money.usd("0.01"),
        code=CashEventCode.EXTERNAL_FLOW,
    )

    snapshot = account.apply_cash_event(event)

    assert snapshot.cash.amount == Decimal(
        "9999999999999999999999999999.01"
    )


def test_constructor_requires_one_consistent_base_currency_catalog() -> None:
    eur_item = Instrument(
        id=InstrumentId.parse("XPAR:MC"),
        quote_currency="EUR",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("1"),
        timezone=AAPL.timezone,
    )
    with pytest.raises(AccountingInvariantError, match="initial cash"):
        PortfolioLedger(
            base_currency="USD",
            initial_cash=Money.of("10000", "EUR"),
            instruments={AAPL.id: AAPL},
        )
    with pytest.raises(AccountingInvariantError, match="quote currency"):
        PortfolioLedger(
            base_currency="USD",
            initial_cash=Money.usd("10000"),
            instruments={eur_item.id: eur_item},
        )
    with pytest.raises(AccountingInvariantError, match="catalog"):
        PortfolioLedger(
            base_currency="USD",
            initial_cash=Money.usd("10000"),
            instruments={AAPL.id: MSFT},
        )
