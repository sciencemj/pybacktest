"""Contract and behavior tests for deterministic intent sizing."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from typing import assert_type
from zoneinfo import ZoneInfo

import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.events import OrderRejected
from pybacktest.domain.identifiers import OrderId
from pybacktest.domain.instruments import Instrument
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    CancelOrderIntent,
    DecisionReason,
    LimitOrderIntent,
    MarketOrderIntent,
    Order,
    OrderSide,
    OrderType,
    TargetQuantity,
    TargetWeight,
    TimeInForce,
)
from pybacktest.domain.portfolio import PortfolioSnapshot
from pybacktest.ports.risk import OrderSizer, RiskContext
from pybacktest.risk.sizing import DefaultOrderSizer
from tests.factories import (
    BASE_DATETIME,
    aapl,
    instrument,
    portfolio_snapshot,
    risk_context,
)
from tests.risk._oracles import fraction_decimal_lot_floor


def snapshot(
    *,
    cash: object = "10000",
    position: object = "0",
    price: object = "100",
    item: Instrument | None = None,
):
    resolved = item or aapl()
    return portfolio_snapshot(
        cash=cash,
        positions=(
            {}
            if Decimal(str(position)) == Decimal("0")
            else {resolved.id: position}
        ),
        prices={resolved.id: price},
        currency=resolved.quote_currency,
    )


def context(
    *,
    portfolio=None,
    price: Money | None = None,
    item: Instrument | None = None,
    tradable: frozenset | None = None,
) -> RiskContext:
    resolved = item or aapl()
    current_snapshot = portfolio or snapshot(item=resolved)
    return risk_context(
        snapshot=current_snapshot,
        prices={
            resolved.id: price or Money.of("100", resolved.quote_currency)
        },
        instruments={resolved.id: resolved},
        tradable=(
            tradable
            if tradable is not None
            else frozenset({resolved.id})
        ),
    )


def test_risk_context_is_immutable_validated_and_owns_its_collections():
    item = aapl()
    prices = {item.id: Money.usd("100")}
    instruments = {item.id: item}
    tradable = {item.id}

    value = RiskContext(
        snapshot=snapshot(),
        prices=prices,
        instruments=instruments,
        tradable=tradable,
        order_id=OrderId.parse("order_" + "8" * 32),
        submitted_at=BASE_DATETIME,
        active_from=BASE_DATETIME + timedelta(days=1),
        active_orders=(),
    )
    prices.clear()
    instruments.clear()
    tradable.clear()

    assert value.prices == {item.id: Money.usd("100")}
    assert value.instruments == {item.id: item}
    assert value.tradable == frozenset({item.id})
    with pytest.raises(TypeError):
        value.prices[item.id] = Money.usd("1")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        value.order_id = OrderId.new()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"snapshot": object()}, "snapshot"),
        ({"prices": []}, "prices"),
        ({"instruments": []}, "instruments"),
        ({"tradable": "XNAS:AAPL"}, "tradable"),
        ({"order_id": "order_" + "1" * 32}, "order_id"),
        (
            {"submitted_at": datetime(2024, 1, 2)},
            "submitted_at",
        ),
        (
            {
                "submitted_at": datetime(2024, 1, 3, tzinfo=UTC),
                "active_from": datetime(2024, 1, 2, tzinfo=UTC),
            },
            "active_from",
        ),
    ],
)
def test_risk_context_rejects_invalid_direct_construction(
    override: dict[str, object],
    message: str,
):
    item = aapl()
    arguments: dict[str, object] = {
        "snapshot": snapshot(),
        "prices": {item.id: Money.usd("100")},
        "instruments": {item.id: item},
        "tradable": frozenset({item.id}),
        "order_id": OrderId.parse("order_" + "9" * 32),
        "submitted_at": BASE_DATETIME,
        "active_from": BASE_DATETIME + timedelta(days=1),
        "active_orders": (),
    }
    arguments.update(override)

    with pytest.raises(ConfigurationError, match=message):
        RiskContext(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("prices", "instruments", "tradable", "message"),
    [
        (
            {aapl().id: Money.usd("0")},
            {aapl().id: aapl()},
            frozenset({aapl().id}),
            "positive",
        ),
        (
            {aapl().id: Money.of("100", "KRW")},
            {aapl().id: aapl()},
            frozenset({aapl().id}),
            "currency",
        ),
        (
            {instrument("MSFT").id: Money.usd("100")},
            {aapl().id: aapl()},
            frozenset({aapl().id}),
            "catalog",
        ),
        (
            {aapl().id: Money.usd("100")},
            {aapl().id: aapl()},
            frozenset({instrument("MSFT").id}),
            "tradable",
        ),
    ],
)
def test_risk_context_rejects_inconsistent_current_data_relationships(
    prices: dict,
    instruments: dict,
    tradable: frozenset,
    message: str,
):
    with pytest.raises(ConfigurationError, match=message):
        RiskContext(
            snapshot=snapshot(),
            prices=prices,
            instruments=instruments,
            tradable=tradable,
            order_id=OrderId.parse("order_" + "6" * 32),
            submitted_at=BASE_DATETIME,
            active_from=BASE_DATETIME + timedelta(days=1),
            active_orders=(),
        )


def test_risk_context_rejects_future_snapshot_but_allows_initial_none_timestamp():
    future_snapshot = replace(
        snapshot(),
        timestamp=BASE_DATETIME + timedelta(days=1),
    )
    initial_snapshot = replace(snapshot(), timestamp=None)

    with pytest.raises(ConfigurationError, match=r"snapshot.*submitted_at"):
        RiskContext(
            snapshot=future_snapshot,
            prices={aapl().id: Money.usd("100")},
            instruments={aapl().id: aapl()},
            tradable=frozenset({aapl().id}),
            order_id=OrderId.parse("order_" + "5" * 32),
            submitted_at=BASE_DATETIME,
            active_from=BASE_DATETIME,
            active_orders=(),
        )

    value = RiskContext(
        snapshot=initial_snapshot,
        prices={aapl().id: Money.usd("100")},
        instruments={aapl().id: aapl()},
        tradable=frozenset({aapl().id}),
        order_id=OrderId.parse("order_" + "4" * 32),
        submitted_at=BASE_DATETIME,
        active_from=BASE_DATETIME,
        active_orders=(),
    )
    assert value.snapshot.timestamp is None


def test_default_sizer_is_immutable_and_satisfies_runtime_contract():
    sizer = DefaultOrderSizer()

    assert isinstance(sizer, OrderSizer)
    with pytest.raises((FrozenInstanceError, TypeError)):
        sizer.enabled = False  # type: ignore[attr-defined]


def test_target_weight_sizes_from_current_equity_and_position():
    item = aapl()
    order = DefaultOrderSizer().size(
        TargetWeight(
            instrument=item.id,
            weight=Decimal("0.50"),
            reason=DecisionReason.of("allocate_half"),
        ),
        context(portfolio=snapshot(cash="10000", position="0")),
    )

    assert isinstance(order, Order)
    assert order.side is OrderSide.BUY
    assert order.quantity.value == Decimal("50")


def test_target_sizing_includes_signed_active_remaining_orders() -> None:
    item = aapl()
    base = context(portfolio=snapshot(position="10"))
    pending = Order.pending(
        id=OrderId.parse("order_" + "a" * 32),
        instrument=item.id,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Quantity.of("20"),
        quote_currency=item.quote_currency,
        limit_price=None,
        time_in_force=TimeInForce.DAY,
        submitted_at=base.submitted_at,
        active_from=base.active_from,
        reason=DecisionReason.of("already_pending"),
    ).accept()
    with_reservation = replace(base, active_orders=(pending,))

    order = DefaultOrderSizer().size(
        TargetQuantity(
            instrument=item.id,
            quantity=Quantity.of("50"),
            reason=DecisionReason.of("target_fifty"),
        ),
        with_reservation,
    )

    assert isinstance(order, Order)
    assert order.side is OrderSide.BUY
    assert order.quantity == Quantity.of("20")


@pytest.mark.parametrize(
    ("position", "weight", "side", "quantity"),
    [
        ("20", "0.50", OrderSide.BUY, "30"),
        ("-20", "0.50", OrderSide.BUY, "70"),
        ("20", "-0.50", OrderSide.SELL, "70"),
    ],
)
def test_target_weight_uses_current_signed_position(
    position: str,
    weight: str,
    side: OrderSide,
    quantity: str,
):
    item = aapl()
    current = snapshot(
        cash="8000" if position == "20" else "12000",
        position=position,
    )
    assert current.cash == Money.usd(
        "8000" if position == "20" else "12000"
    )
    assert current.equity == Money.usd("10000")

    order = DefaultOrderSizer().size(
        TargetWeight(
            item.id,
            Decimal(weight),
            DecisionReason.of("rebalance"),
        ),
        context(portfolio=current),
    )

    assert isinstance(order, Order)
    assert order.side is side
    assert order.quantity == Quantity.of(quantity)


def test_target_quantity_is_an_absolute_positive_long_target():
    item = aapl()
    reason = DecisionReason.of("absolute_target")

    reduce_long = DefaultOrderSizer().size(
        TargetQuantity(item.id, Quantity.of("4"), reason),
        context(portfolio=snapshot(cash="9000", position="10")),
    )
    cross_short = DefaultOrderSizer().size(
        TargetQuantity(item.id, Quantity.of("4"), reason),
        context(portfolio=snapshot(cash="10500", position="-5")),
    )

    assert isinstance(reduce_long, Order)
    assert reduce_long.side is OrderSide.SELL
    assert reduce_long.quantity == Quantity.of("6")
    assert isinstance(cross_short, Order)
    assert cross_short.side is OrderSide.BUY
    assert cross_short.quantity == Quantity.of("9")


def test_target_quantity_preserves_exact_tiny_lots_under_low_ambient_precision():
    base = aapl()
    fine = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1E-20"),
        lot_size=Decimal("1E-20"),
        timezone=base.timezone,
    )
    target = Decimal("123456789.01234567890123456789")
    intent = TargetQuantity(
        fine.id,
        Quantity.of(target),
        DecisionReason.of("exact_target"),
    )
    current_context = context(
        portfolio=snapshot(cash="1000000000", price="1", item=fine),
        price=Money.usd("1"),
        item=fine,
    )

    with localcontext() as ambient:
        ambient.prec = 12
        result = DefaultOrderSizer().size(intent, current_context)

    assert isinstance(result, Order)
    assert result.quantity.value == target


def test_target_quantity_cancellation_keeps_one_exact_tiny_lot():
    base = aapl()
    fine = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1E-20"),
        lot_size=Decimal("1E-20"),
        timezone=base.timezone,
    )
    current_quantity = Decimal("123456789.01234567890123456788")
    target_quantity = Decimal("123456789.01234567890123456789")
    with localcontext() as setup:
        setup.prec = 100
        current_snapshot = snapshot(
            cash="1000000000",
            position=current_quantity,
            price="1",
            item=fine,
        )
    current_context = context(
        portfolio=current_snapshot,
        price=Money.usd("1"),
        item=fine,
    )

    with localcontext() as ambient:
        ambient.prec = 12
        result = DefaultOrderSizer().size(
            TargetQuantity(
                fine.id,
                Quantity.of(target_quantity),
                DecisionReason.of("one_lot_delta"),
            ),
            current_context,
        )

    assert isinstance(result, Order)
    assert result.side is OrderSide.BUY
    assert result.quantity == Quantity.of("1E-20")


def test_non_power_of_ten_weight_sizing_is_ambient_context_independent():
    base = aapl()
    odd_lot = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("0.03"),
        timezone=base.timezone,
    )
    current_context = context(
        portfolio=snapshot(cash="100", price="7", item=odd_lot),
        price=Money.usd("7"),
        item=odd_lot,
    )
    intent = TargetWeight(
        odd_lot.id,
        Decimal("0.5"),
        DecisionReason.of("odd_lot"),
    )

    with localcontext() as low:
        low.prec = 7
        low_result = DefaultOrderSizer().size(intent, current_context)
    with localcontext() as high:
        high.prec = 80
        high_result = DefaultOrderSizer().size(intent, current_context)

    assert isinstance(low_result, Order)
    assert isinstance(high_result, Order)
    assert low_result.quantity == high_result.quantity == Quantity.of("7.14")


def test_weight_sizing_widens_exponent_bounds_for_finite_result():
    base = aapl()
    huge = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1"),
        lot_size=Decimal("1"),
        timezone=base.timezone,
    )
    current_context = context(
        portfolio=snapshot(cash="1E+999998", price="1", item=huge),
        price=Money.usd("1"),
        item=huge,
    )

    result = DefaultOrderSizer().size(
        TargetWeight(
            huge.id,
            Decimal("100"),
            DecisionReason.of("large_exponent"),
        ),
        current_context,
    )

    assert isinstance(result, Order)
    assert result.quantity == Quantity.of("1E+1000000")


def test_weight_sizing_floors_the_original_ratio_to_hundred_decimal_lots():
    base = aapl()
    fine = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1"),
        lot_size=Decimal("1E-100"),
        timezone=base.timezone,
    )
    expected = fraction_decimal_lot_floor(1, 3, 100)

    result = DefaultOrderSizer().size(
        TargetWeight(
            instrument=fine.id,
            weight=Decimal("1"),
            reason=DecisionReason.of("exact_fraction_floor"),
        ),
        context(
            portfolio=snapshot(cash="1", price="3", item=fine),
            price=Money.usd("3"),
            item=fine,
        ),
    )

    assert isinstance(result, Order)
    assert result.side is OrderSide.BUY
    assert result.quantity == Quantity.of(expected)


def test_weight_sizing_rejects_a_nonterminating_floor_beyond_work_bound():
    base = aapl()
    excessive = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1"),
        lot_size=Decimal("1E-5000"),
        timezone=base.timezone,
    )

    result = DefaultOrderSizer().size(
        TargetWeight(
            instrument=excessive.id,
            weight=Decimal("1"),
            reason=DecisionReason.of("excessive_fraction_floor"),
        ),
        context(
            portfolio=snapshot(
                cash="1",
                price="3",
                item=excessive,
            ),
            price=Money.usd("3"),
            item=excessive,
        ),
    )

    assert isinstance(result, OrderRejected)
    assert result.reason.code == "invalid_sizing_arithmetic"


def test_negative_inexact_target_weight_creates_a_sell_order():
    base = aapl()
    fractional = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1"),
        lot_size=Decimal("0.1"),
        timezone=base.timezone,
    )

    result = DefaultOrderSizer().size(
        TargetWeight(
            instrument=fractional.id,
            weight=Decimal("-1"),
            reason=DecisionReason.of("negative_fractional_target"),
        ),
        context(
            portfolio=snapshot(
                cash="1",
                price="3",
                item=fractional,
            ),
            price=Money.usd("3"),
            item=fractional,
        ),
    )

    assert isinstance(result, Order)
    assert result.side is OrderSide.SELL
    assert result.quantity == Quantity.of("0.3")


def test_weight_sizing_cancels_large_common_factors_before_guarding():
    base = aapl()
    fractional = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1"),
        lot_size=Decimal("0.1"),
        timezone=base.timezone,
    )
    common = Decimal("1" + "0" * 2999 + "1")
    three_common = Decimal("3" + "0" * 2999 + "3")
    with localcontext() as setup:
        setup.prec = 7000
        current_snapshot = snapshot(
            cash=common,
            price=three_common,
            item=fractional,
        )

    result = DefaultOrderSizer().size(
        TargetWeight(
            instrument=fractional.id,
            weight=Decimal("1"),
            reason=DecisionReason.of("cancel_common_factor"),
        ),
        context(
            portfolio=current_snapshot,
            price=Money.usd(three_common),
            item=fractional,
        ),
    )

    assert isinstance(result, Order)
    assert result.side is OrderSide.BUY
    assert result.quantity == Quantity.of("0.3")


def test_zero_target_flattens_without_inventing_one_share():
    item = aapl()
    sizer = DefaultOrderSizer()
    reason = DecisionReason.of("flatten")

    flatten = sizer.size(
        TargetWeight(item.id, Decimal("0"), reason),
        context(portfolio=snapshot(cash="9500", position="5")),
    )
    no_position = sizer.size(
        TargetWeight(item.id, Decimal("0"), reason),
        context(portfolio=snapshot(cash="10000", position="0")),
    )

    assert isinstance(flatten, Order)
    assert flatten.side is OrderSide.SELL
    assert flatten.quantity == Quantity.of("5")
    assert isinstance(no_position, OrderRejected)
    assert no_position.reason.code == "no_op_target"


def test_missing_mark_and_lot_rounding_to_zero_are_typed_rejections():
    item = aapl()
    reason = DecisionReason.of("small_target")
    missing_context = context()
    missing_context = RiskContext(
        snapshot=missing_context.snapshot,
        prices={},
        instruments=missing_context.instruments,
        tradable=missing_context.tradable,
        order_id=missing_context.order_id,
        submitted_at=missing_context.submitted_at,
        active_from=missing_context.active_from,
        active_orders=missing_context.active_orders,
    )
    coarse = Instrument(
        id=item.id,
        quote_currency="USD",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("10"),
        timezone=ZoneInfo("America/New_York"),
    )

    missing = DefaultOrderSizer().size(
        TargetWeight(item.id, Decimal("0.1"), reason),
        missing_context,
    )
    rounded = DefaultOrderSizer().size(
        TargetWeight(item.id, Decimal("1"), reason),
        context(
            item=coarse,
            portfolio=snapshot(cash="500", item=coarse),
        ),
    )

    assert isinstance(missing, OrderRejected)
    assert missing.reason.code == "missing_price"
    assert isinstance(rounded, OrderRejected)
    assert rounded.reason.code == "lot_rounding_zero"


def test_explicit_market_and_limit_intents_preserve_all_execution_fields():
    item = aapl()
    reason = DecisionReason.of("explicit")
    current_context = context()
    market_intent = MarketOrderIntent(
        item.id,
        OrderSide.SELL,
        Quantity.of("3"),
        TimeInForce.IMMEDIATE_OR_CANCEL,
        reason,
    )
    limit_intent = LimitOrderIntent(
        item.id,
        OrderSide.BUY,
        Quantity.of("2"),
        Money.usd("99.50"),
        TimeInForce.GOOD_TIL_CANCELLED,
        reason,
    )

    market = DefaultOrderSizer().size(market_intent, current_context)
    limit = DefaultOrderSizer().size(limit_intent, current_context)

    assert isinstance(market, Order)
    assert (
        market.side,
        market.type,
        market.quantity,
        market.limit_price,
        market.time_in_force,
        market.reason,
    ) == (
        market_intent.side,
        OrderType.MARKET,
        market_intent.quantity,
        None,
        market_intent.time_in_force,
        market_intent.reason,
    )
    assert isinstance(limit, Order)
    assert (
        limit.side,
        limit.type,
        limit.quantity,
        limit.limit_price,
        limit.time_in_force,
        limit.reason,
    ) == (
        limit_intent.side,
        OrderType.LIMIT,
        limit_intent.quantity,
        limit_intent.limit_price,
        limit_intent.time_in_force,
        limit_intent.reason,
    )
    assert market.id == current_context.order_id == limit.id
    assert market.submitted_at == current_context.submitted_at
    assert market.active_from == current_context.active_from


def test_cancel_intent_genuinely_bypasses_sizing():
    intent = CancelOrderIntent(
        OrderId.parse("order_" + "a" * 32),
        DecisionReason.of("cancel_signal"),
    )

    result = DefaultOrderSizer().size(intent, context())

    assert_type(result, CancelOrderIntent)
    assert result is intent


def test_sizer_does_not_cap_a_target_to_current_cash():
    item = aapl()

    result = DefaultOrderSizer().size(
        TargetWeight(
            item.id,
            Decimal("2"),
            DecisionReason.of("leveraged_target"),
        ),
        context(portfolio=snapshot(cash="10000")),
    )

    assert isinstance(result, Order)
    assert result.side is OrderSide.BUY
    assert result.quantity == Quantity.of("200")


def test_absolute_target_that_already_matches_is_a_typed_noop():
    item = aapl()

    result = DefaultOrderSizer().size(
        TargetQuantity(
            item.id,
            Quantity.of("5"),
            DecisionReason.of("hold_target"),
        ),
        context(portfolio=snapshot(cash="9500", position="5")),
    )

    assert isinstance(result, OrderRejected)
    assert result.reason.code == "no_op_target"


def test_tick_misaligned_current_mark_is_a_typed_rejection():
    item = aapl()

    result = DefaultOrderSizer().size(
        TargetWeight(
            item.id,
            Decimal("0.1"),
            DecisionReason.of("invalid_mark"),
        ),
        context(price=Money.usd("100.005")),
    )

    assert isinstance(result, OrderRejected)
    assert result.reason.code == "invalid_price_tick"


def test_unknown_instrument_is_a_typed_sizing_rejection():
    item = aapl()
    base = context()
    other = instrument("MSFT")
    unknown_context = RiskContext(
        snapshot=base.snapshot,
        prices={other.id: Money.usd("200")},
        instruments={other.id: other},
        tradable=frozenset({other.id}),
        order_id=base.order_id,
        submitted_at=base.submitted_at,
        active_from=base.active_from,
        active_orders=(),
    )

    result = DefaultOrderSizer().size(
        TargetQuantity(
            item.id,
            Quantity.of("1"),
            DecisionReason.of("unknown"),
        ),
        unknown_context,
    )

    assert isinstance(result, OrderRejected)
    assert result.reason.code == "unknown_instrument"


def test_cross_currency_instrument_is_rejected_before_sizing():
    won = Instrument(
        id=aapl().id,
        quote_currency="KRW",
        tick_size=Decimal("1"),
        lot_size=Decimal("1"),
        timezone=ZoneInfo("Asia/Seoul"),
    )

    result = DefaultOrderSizer().size(
        TargetWeight(
            won.id,
            Decimal("0.1"),
            DecisionReason.of("cross_currency"),
        ),
        context(
            portfolio=snapshot(cash="10000"),
            price=Money.of("100", "KRW"),
            item=won,
        ),
    )

    assert isinstance(result, OrderRejected)
    assert result.reason.code == "cross_currency_instrument"


@pytest.mark.parametrize(
    ("intent", "code"),
    [
        (
            MarketOrderIntent(
                aapl().id,
                OrderSide.BUY,
                Quantity.of("1.5"),
                TimeInForce.DAY,
                DecisionReason.of("bad_lot"),
            ),
            "invalid_quantity_lot",
        ),
        (
            LimitOrderIntent(
                aapl().id,
                OrderSide.BUY,
                Quantity.of("1"),
                Money.of("100", "KRW"),
                TimeInForce.DAY,
                DecisionReason.of("bad_currency"),
            ),
            "limit_price_currency_mismatch",
        ),
        (
            LimitOrderIntent(
                aapl().id,
                OrderSide.BUY,
                Quantity.of("1"),
                Money.usd("100.005"),
                TimeInForce.DAY,
                DecisionReason.of("bad_tick"),
            ),
            "invalid_limit_tick",
        ),
    ],
)
def test_explicit_intents_reject_invalid_lot_or_limit_without_mutation(
    intent: MarketOrderIntent | LimitOrderIntent,
    code: str,
):
    result = DefaultOrderSizer().size(intent, context())

    assert isinstance(result, OrderRejected)
    assert result.reason.code == code


def test_zero_and_negative_equity_reject_nonzero_weight_but_zero_flattens():
    item = aapl()
    zero_equity = snapshot(cash="-1000", position="10")
    negative_equity = PortfolioSnapshot(
        timestamp=zero_equity.timestamp,
        cash=Money.usd("-1001"),
        positions=zero_equity.positions,
        realized_pnl=zero_equity.realized_pnl,
        unrealized_pnl=zero_equity.unrealized_pnl,
        total_fees=zero_equity.total_fees,
        market_value=zero_equity.market_value,
        gross_exposure=zero_equity.gross_exposure,
        equity=Money.usd("-1"),
        valuation_prices=zero_equity.valuation_prices,
        cash_events=zero_equity.cash_events,
    )
    reason = DecisionReason.of("equity_boundary")
    sizer = DefaultOrderSizer()

    zero_rejection = sizer.size(
        TargetWeight(item.id, Decimal("0.1"), reason),
        context(portfolio=zero_equity),
    )
    negative_rejection = sizer.size(
        TargetWeight(item.id, Decimal("0.1"), reason),
        context(portfolio=negative_equity),
    )
    flatten = sizer.size(
        TargetWeight(item.id, Decimal("0"), reason),
        context(portfolio=negative_equity),
    )

    assert isinstance(zero_rejection, OrderRejected)
    assert zero_rejection.reason.code == "non_positive_equity"
    assert isinstance(negative_rejection, OrderRejected)
    assert negative_rejection.reason.code == "non_positive_equity"
    assert isinstance(flatten, Order)
    assert flatten.side is OrderSide.SELL
    assert flatten.quantity == Quantity.of("10")


@pytest.mark.parametrize(
    ("cash", "price", "lot", "weight", "expected"),
    [
        ("1E+100", "1E+50", "1E-20", "1E-20", "1E+30"),
        ("1E-100", "1E-50", "1E-60", "1", "1E-50"),
    ],
)
def test_sizing_preserves_large_and_small_finite_decimal_precision(
    cash: str,
    price: str,
    lot: str,
    weight: str,
    expected: str,
):
    base = aapl()
    item = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal(lot),
        lot_size=Decimal(lot),
        timezone=base.timezone,
    )
    current = snapshot(cash=cash, price=price, item=item)

    result = DefaultOrderSizer().size(
        TargetWeight(
            item.id,
            Decimal(weight),
            DecisionReason.of("precision"),
        ),
        context(
            portfolio=current,
            price=Money.usd(price),
            item=item,
        ),
    )

    assert isinstance(result, Order)
    assert result.quantity == Quantity.of(expected)
