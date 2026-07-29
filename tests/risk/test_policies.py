"""Contract and core example tests for long/short risk decisions."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import Instrument
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TargetWeight,
    TimeInForce,
)
from pybacktest.ports.risk import (
    RiskContext,
    RiskDecision,
    RiskPolicy,
)
from pybacktest.risk.policies import LongShortRisk, RiskStatus
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


def proposed_order(
    *,
    side: OrderSide,
    quantity: object,
    item: Instrument | None = None,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Money | None = None,
    quote_currency: str | None = None,
) -> Order:
    resolved = item or aapl()
    return Order.pending(
        id=OrderId.parse("order_" + "b" * 32),
        instrument=resolved.id,
        side=side,
        type=order_type,
        quantity=Quantity.of(quantity),
        quote_currency=quote_currency or resolved.quote_currency,
        limit_price=limit_price,
        time_in_force=TimeInForce.DAY,
        submitted_at=BASE_DATETIME,
        active_from=BASE_DATETIME,
        reason=DecisionReason.of("proposal"),
    )


def active_order(
    *,
    side: OrderSide,
    quantity: object,
    item: Instrument | None = None,
    order_type: OrderType = OrderType.MARKET,
    limit_price: Money | None = None,
    filled: object = "0",
    order_id_digit: str = "a",
) -> Order:
    active = replace(
        proposed_order(
            side=side,
            quantity=quantity,
            item=item,
            order_type=order_type,
            limit_price=limit_price,
        ).accept(),
        id=OrderId.parse("order_" + order_id_digit * 32),
    )
    filled_quantity = Quantity.of(filled)
    if filled_quantity.value == Decimal("0"):
        return active
    return replace(
        active,
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=filled_quantity,
    )


def two_position_context(
    *,
    aapl_quantity: object = "0",
    msft_quantity: object = "-50",
    current_cash: object = "15000",
    include_msft_mark: bool = True,
) -> RiskContext:
    first = aapl()
    second = instrument("MSFT")
    aapl_value = Decimal(str(aapl_quantity))
    msft_value = Decimal(str(msft_quantity))
    marked = portfolio_snapshot(
        cash=current_cash,
        positions={
            first.id: aapl_value,
            second.id: msft_value,
        },
        prices={
            first.id: Money.usd("100"),
            second.id: Money.usd("100"),
        },
    )
    prices = {first.id: Money.usd("100")}
    if include_msft_mark:
        prices[second.id] = Money.usd("100")
    return risk_context(
        snapshot=marked,
        prices=prices,
        instruments={first.id: first, second.id: second},
        tradable=frozenset({first.id, second.id}),
        order_id=OrderId.parse("order_" + "d" * 32),
    )


def test_risk_decision_is_immutable_and_enforces_status_quantity_invariants():
    passed = RiskDecision(
        status=RiskStatus.PASSED,
        original_quantity=Quantity.of("10"),
        final_quantity=Quantity.of("10"),
        codes=(),
        message="order passed all risk constraints",
    )

    assert passed.codes == ()
    with pytest.raises(FrozenInstanceError):
        passed.message = "changed"  # type: ignore[misc]
    with pytest.raises(ConfigurationError, match="PASSED"):
        RiskDecision(
            status=RiskStatus.PASSED,
            original_quantity=Quantity.of("10"),
            final_quantity=Quantity.of("9"),
            codes=("max_leverage",),
            message="invalid pass",
        )
    with pytest.raises(ConfigurationError, match="ADJUSTED"):
        RiskDecision(
            status=RiskStatus.ADJUSTED,
            original_quantity=Quantity.of("10"),
            final_quantity=Quantity.of("0"),
            codes=("max_leverage",),
            message="invalid adjustment",
        )
    with pytest.raises(ConfigurationError, match="REJECTED"):
        RiskDecision(
            status=RiskStatus.REJECTED,
            original_quantity=Quantity.of("10"),
            final_quantity=Quantity.of("1"),
            codes=("short_not_allowed",),
            message="invalid rejection",
        )


def test_risk_decision_defensively_copies_ordered_codes():
    codes = ["max_position_weight", "max_leverage"]

    decision = RiskDecision(
        status=RiskStatus.REJECTED,
        original_quantity=Quantity.of("10"),
        final_quantity=Quantity.of("0"),
        codes=codes,  # type: ignore[arg-type]
        message="rejected by two constraints",
    )
    codes.clear()

    assert decision.codes == ("max_position_weight", "max_leverage")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_leverage", Decimal("0")),
        ("max_leverage", Decimal("-1")),
        ("max_leverage", Decimal("NaN")),
        ("max_leverage", Decimal("Infinity")),
        ("max_position_weight", Decimal("0")),
        ("max_position_weight", Decimal("-0.1")),
        ("max_position_weight", Decimal("NaN")),
        ("max_position_weight", Decimal("Infinity")),
    ],
)
def test_long_short_risk_rejects_invalid_direct_configuration(
    field: str,
    value: Decimal,
):
    arguments: dict[str, object] = {
        "max_leverage": Decimal("1"),
        "max_position_weight": None,
        "allow_short": True,
    }
    arguments[field] = value

    with pytest.raises(ConfigurationError, match=field):
        LongShortRisk(**arguments)  # type: ignore[arg-type]


def test_long_short_risk_requires_a_real_bool_for_short_permission():
    with pytest.raises(ConfigurationError, match="allow_short"):
        LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=1,  # type: ignore[arg-type]
        )


def test_long_short_risk_is_immutable_and_satisfies_runtime_contract():
    policy = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    )

    assert isinstance(policy, RiskPolicy)
    with pytest.raises(FrozenInstanceError):
        policy.allow_short = False  # type: ignore[misc]


def test_risk_adjustment_is_visible_not_silent():
    item = aapl()
    current = snapshot(cash="10000")
    current_context = context(portfolio=current)
    order = DefaultOrderSizer().size(
        TargetWeight(
            instrument=item.id,
            weight=Decimal("1"),
            reason=DecisionReason.of("full_allocation"),
        ),
        current_context,
    )
    assert isinstance(order, Order)

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("0.25"),
        allow_short=True,
    ).evaluate(order, current_context)

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.original_quantity == Quantity.of("100")
    assert decision.final_quantity == Quantity.of("25")
    assert decision.codes == ("max_position_weight",)
    assert "25" in decision.message


def test_short_order_from_flat_is_rejected_when_disabled():
    current_context = context(portfolio=snapshot(cash="10000"))
    order = proposed_order(side=OrderSide.SELL, quantity="50")

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(order, current_context)

    assert decision.status is RiskStatus.REJECTED
    assert decision.final_quantity == Quantity.of("0")
    assert decision.codes == ("short_not_allowed",)


def test_short_disabled_clamps_a_long_to_short_crossing_at_flat():
    order = proposed_order(side=OrderSide.SELL, quantity="20")

    decision = LongShortRisk(
        max_leverage=Decimal("2"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(
        order,
        context(portfolio=snapshot(cash="9000", position="10")),
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("10")
    assert decision.codes == ("short_not_allowed",)


def test_closing_an_existing_short_is_allowed_when_new_shorts_are_disabled():
    order = proposed_order(side=OrderSide.BUY, quantity="10")

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("0.25"),
        allow_short=False,
    ).evaluate(
        order,
        context(portfolio=snapshot(cash="11000", position="-10")),
    )

    assert decision.status is RiskStatus.PASSED
    assert decision.final_quantity == Quantity.of("10")
    assert decision.codes == ()


def test_increasing_an_existing_short_is_rejected_when_disabled():
    decision = LongShortRisk(
        max_leverage=Decimal("2"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(
        proposed_order(side=OrderSide.SELL, quantity="1"),
        context(portfolio=snapshot(cash="11000", position="-10")),
    )

    assert decision.status is RiskStatus.REJECTED
    assert decision.codes == ("short_not_allowed",)


def test_unknown_nontradable_and_missing_price_have_stable_first_codes():
    item = aapl()
    order = proposed_order(side=OrderSide.BUY, quantity="1")
    base = context()
    other = instrument("MSFT")
    unknown = type(base)(
        snapshot=base.snapshot,
        prices={other.id: Money.usd("200")},
        instruments={other.id: other},
        tradable=frozenset({other.id}),
        order_id=base.order_id,
        submitted_at=base.submitted_at,
        active_from=base.active_from,
    )
    nontradable = type(base)(
        snapshot=base.snapshot,
        prices=base.prices,
        instruments=base.instruments,
        tradable=frozenset(),
        order_id=base.order_id,
        submitted_at=base.submitted_at,
        active_from=base.active_from,
    )
    missing = type(base)(
        snapshot=base.snapshot,
        prices={},
        instruments=base.instruments,
        tradable=base.tradable,
        order_id=base.order_id,
        submitted_at=base.submitted_at,
        active_from=base.active_from,
    )
    policy = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    )

    assert policy.evaluate(order, unknown).codes == ("unknown_instrument",)
    assert policy.evaluate(order, nontradable).codes == (
        "instrument_not_tradable",
    )
    assert policy.evaluate(order, missing).codes == ("missing_price",)
    assert item.id == order.instrument


def test_available_cash_adjustment_uses_limit_price_and_never_returns_zero_order():
    item = aapl()
    order = Order.pending(
        id=OrderId.parse("order_" + "c" * 32),
        instrument=item.id,
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Quantity.of("10"),
        quote_currency="USD",
        limit_price=Money.usd("200"),
        time_in_force=TimeInForce.DAY,
        submitted_at=BASE_DATETIME,
        active_from=BASE_DATETIME,
        reason=DecisionReason.of("cash_cap"),
    )

    decision = LongShortRisk(
        max_leverage=Decimal("10"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        order,
        context(portfolio=snapshot(cash="1000")),
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("5")
    assert decision.codes == ("available_cash",)


def test_active_buy_reservations_reduce_available_cash_and_leverage() -> None:
    pending = active_order(
        side=OrderSide.BUY,
        quantity="60",
        order_type=OrderType.LIMIT,
        limit_price=Money.usd("100"),
    )
    current = replace(context(), active_orders=(pending,))

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="60"),
        current,
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("40")
    assert decision.codes == ("max_leverage", "available_cash")


def test_active_sell_reservations_cannot_rely_on_other_pending_buys() -> None:
    current = replace(
        context(portfolio=snapshot(cash="0", position="10")),
        active_orders=(
            active_order(side=OrderSide.BUY, quantity="100"),
            active_order(
                side=OrderSide.SELL,
                quantity="6",
                order_id_digit="c",
            ),
        ),
    )

    decision = LongShortRisk(
        max_leverage=Decimal("20"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(
        proposed_order(side=OrderSide.SELL, quantity="6"),
        current,
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("4")
    assert decision.codes == ("short_not_allowed",)


def test_partially_filled_order_reserves_only_its_remaining_quantity() -> None:
    partially_filled = active_order(
        side=OrderSide.BUY,
        quantity="10",
        filled="4",
    )
    current = replace(
        context(portfolio=snapshot(cash="1200", position="4")),
        active_orders=(partially_filled,),
    )

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="10"),
        current,
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("6")
    assert decision.codes == ("max_leverage", "available_cash")


def test_other_instrument_active_order_reserves_gross_capacity() -> None:
    first = aapl()
    second = instrument("MSFT")
    base = two_position_context(
        aapl_quantity="0",
        msft_quantity="0",
        current_cash="10000",
    )
    pending_other = active_order(
        side=OrderSide.BUY,
        quantity="60",
        item=second,
    )
    current = replace(base, active_orders=(pending_other,))

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=False,
    ).evaluate(
        proposed_order(
            side=OrderSide.BUY,
            quantity="60",
            item=first,
        ),
        current,
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("40")
    assert decision.codes == ("max_leverage", "available_cash")


def test_risk_rejects_tick_misaligned_current_mark_with_stable_code():
    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="1"),
        context(price=Money.usd("100.005")),
    )

    assert decision.status is RiskStatus.REJECTED
    assert decision.codes == ("invalid_price_tick",)


def test_risk_rejects_invalid_order_lot_and_limit_tick_or_currency():
    base = aapl()
    krw_order = proposed_order(
        side=OrderSide.BUY,
        quantity="1",
        order_type=OrderType.LIMIT,
        limit_price=Money.of("100", "KRW"),
        quote_currency="KRW",
    )
    tick_order = proposed_order(
        side=OrderSide.BUY,
        quantity="1",
        order_type=OrderType.LIMIT,
        limit_price=Money.usd("100.005"),
    )
    lot_order = proposed_order(side=OrderSide.BUY, quantity="1.5")
    policy = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    )

    assert policy.evaluate(krw_order, context()).codes == (
        "order_currency_mismatch",
    )
    assert policy.evaluate(tick_order, context()).codes == (
        "invalid_limit_tick",
    )
    assert policy.evaluate(lot_order, context()).codes == (
        "invalid_quantity_lot",
    )
    assert base.id == krw_order.instrument


def test_cross_currency_instrument_is_rejected_without_conversion():
    usd_snapshot = snapshot(cash="10000")
    won = Instrument(
        id=aapl().id,
        quote_currency="KRW",
        tick_size=Decimal("1"),
        lot_size=Decimal("1"),
        timezone=ZoneInfo("Asia/Seoul"),
    )
    won_context = RiskContext(
        snapshot=usd_snapshot,
        prices={won.id: Money.of("100", "KRW")},
        instruments={won.id: won},
        tradable=frozenset({won.id}),
        order_id=OrderId.parse("order_" + "e" * 32),
        submitted_at=BASE_DATETIME,
        active_from=BASE_DATETIME,
    )

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.BUY,
            quantity="1",
            item=won,
        ),
        won_context,
    )

    assert decision.codes == ("cross_currency_instrument",)


def test_position_cap_allows_reduction_while_over_cap_and_clamps_crossing():
    current_context = context(
        portfolio=snapshot(cash="0", position="100"),
    )
    policy = LongShortRisk(
        max_leverage=Decimal("3"),
        max_position_weight=Decimal("0.25"),
        allow_short=True,
    )

    reducing = policy.evaluate(
        proposed_order(side=OrderSide.SELL, quantity="50"),
        current_context,
    )
    crossing = policy.evaluate(
        proposed_order(side=OrderSide.SELL, quantity="150"),
        current_context,
    )

    assert reducing.status is RiskStatus.PASSED
    assert reducing.final_quantity == Quantity.of("50")
    assert crossing.status is RiskStatus.ADJUSTED
    assert crossing.final_quantity == Quantity.of("125")
    assert crossing.codes == ("max_position_weight",)


def test_gross_leverage_uses_other_long_or_short_absolute_exposure():
    current_context = two_position_context()

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="100"),
        current_context,
    )

    assert current_context.snapshot.equity == Money.usd("10000")
    assert current_context.snapshot.gross_exposure == Money.usd("5000")
    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("50")
    assert decision.codes == ("max_leverage",)


def test_missing_mark_for_an_existing_other_position_is_rejected():
    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="1"),
        two_position_context(include_msft_mark=False),
    )

    assert decision.status is RiskStatus.REJECTED
    assert decision.codes == ("missing_price",)
    assert "MSFT" in decision.message


def test_cross_currency_existing_position_is_rejected_before_gross_math():
    base = two_position_context()
    msft = instrument("MSFT")
    won_msft = Instrument(
        id=msft.id,
        quote_currency="KRW",
        tick_size=Decimal("1"),
        lot_size=Decimal("1"),
        timezone=ZoneInfo("Asia/Seoul"),
    )
    mixed_context = RiskContext(
        snapshot=base.snapshot,
        prices={
            aapl().id: Money.usd("100"),
            won_msft.id: Money.of("100", "KRW"),
        },
        instruments={
            aapl().id: aapl(),
            won_msft.id: won_msft,
        },
        tradable=base.tradable,
        order_id=base.order_id,
        submitted_at=base.submitted_at,
        active_from=base.active_from,
    )

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="1"),
        mixed_context,
    )

    assert decision.status is RiskStatus.REJECTED
    assert decision.codes == ("cross_currency_instrument",)


def test_leverage_cap_allows_reduction_when_starting_over_cap():
    current_context = context(
        portfolio=snapshot(cash="-10000", position="200"),
    )
    policy = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=None,
        allow_short=True,
    )

    reducing = policy.evaluate(
        proposed_order(side=OrderSide.SELL, quantity="50"),
        current_context,
    )
    crossing = policy.evaluate(
        proposed_order(side=OrderSide.SELL, quantity="350"),
        current_context,
    )

    assert reducing.status is RiskStatus.PASSED
    assert crossing.status is RiskStatus.ADJUSTED
    assert crossing.final_quantity == Quantity.of("300")
    assert crossing.codes == ("max_leverage",)


def test_nonpositive_equity_allows_flattening_but_rejects_new_exposure():
    item = aapl()
    zero_equity = snapshot(cash="-1000", position="10")
    negative_flat_base = snapshot(cash="0")
    negative_flat = type(negative_flat_base)(
        timestamp=negative_flat_base.timestamp,
        cash=Money.usd("-1"),
        positions=negative_flat_base.positions,
        realized_pnl=negative_flat_base.realized_pnl,
        unrealized_pnl=negative_flat_base.unrealized_pnl,
        total_fees=negative_flat_base.total_fees,
        market_value=negative_flat_base.market_value,
        gross_exposure=negative_flat_base.gross_exposure,
        equity=Money.usd("-1"),
        valuation_prices=negative_flat_base.valuation_prices,
        cash_events=negative_flat_base.cash_events,
    )
    policy = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("0.5"),
        allow_short=True,
    )

    flatten = policy.evaluate(
        proposed_order(side=OrderSide.SELL, quantity="10"),
        context(portfolio=zero_equity),
    )
    opening = policy.evaluate(
        proposed_order(side=OrderSide.SELL, quantity="1"),
        context(portfolio=negative_flat),
    )

    assert zero_equity.equity == Money.usd("0")
    assert flatten.status is RiskStatus.PASSED
    assert opening.status is RiskStatus.REJECTED
    assert opening.codes == ("non_positive_equity",)
    assert item.id in negative_flat.valuation_prices


def test_lot_rounding_a_cap_to_zero_rejects_instead_of_approving_zero():
    base = aapl()
    coarse = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("10"),
        timezone=base.timezone,
    )

    decision = LongShortRisk(
        max_leverage=Decimal("10"),
        max_position_weight=Decimal("0.1"),
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.BUY,
            quantity="10",
            item=coarse,
        ),
        context(
            portfolio=snapshot(cash="1000", item=coarse),
            item=coarse,
        ),
    )

    assert decision.status is RiskStatus.REJECTED
    assert decision.final_quantity == Quantity.of("0")
    assert decision.codes == ("max_position_weight",)


def test_simultaneous_caps_use_smallest_quantity_and_stable_code_order():
    order = proposed_order(
        side=OrderSide.BUY,
        quantity="100",
        order_type=OrderType.LIMIT,
        limit_price=Money.usd("250"),
    )

    decision = LongShortRisk(
        max_leverage=Decimal("0.5"),
        max_position_weight=Decimal("0.8"),
        allow_short=True,
    ).evaluate(
        order,
        context(portfolio=snapshot(cash="10000")),
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("40")
    assert decision.codes == (
        "max_position_weight",
        "max_leverage",
        "available_cash",
    )


def test_covering_a_short_is_not_blocked_only_by_negative_cash():
    short = snapshot(cash="1000", position="-10")
    insolvent_short = type(short)(
        timestamp=short.timestamp,
        cash=Money.usd("50"),
        positions=short.positions,
        realized_pnl=short.realized_pnl,
        unrealized_pnl=short.unrealized_pnl,
        total_fees=short.total_fees,
        market_value=short.market_value,
        gross_exposure=short.gross_exposure,
        equity=Money.usd("-950"),
        valuation_prices=short.valuation_prices,
        cash_events=short.cash_events,
    )

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("0.25"),
        allow_short=False,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="10"),
        context(portfolio=insolvent_short),
    )

    assert decision.status is RiskStatus.PASSED
    assert decision.final_quantity == Quantity.of("10")


@pytest.mark.parametrize(
    ("cash", "price", "lot", "max_weight", "requested", "expected"),
    [
        ("1E+100", "1E+50", "1E-20", "1E-20", "1E+40", "1E+30"),
        ("1E-100", "1E-50", "1E-60", "0.5", "1E-50", "5E-51"),
    ],
)
def test_risk_caps_preserve_large_and_small_finite_decimal_precision(
    cash: str,
    price: str,
    lot: str,
    max_weight: str,
    requested: str,
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

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal(max_weight),
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.BUY,
            quantity=requested,
            item=item,
        ),
        context(
            portfolio=snapshot(cash=cash, price=price, item=item),
            price=Money.usd(price),
            item=item,
        ),
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of(expected)
    assert decision.codes == ("max_position_weight",)


def test_position_cap_never_depends_on_ambient_precision_or_exceeds_exact_cap():
    base = aapl()
    fine = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1E-20"),
        lot_size=Decimal("1E-20"),
        timezone=base.timezone,
    )
    current_context = context(
        portfolio=snapshot(
            cash="10000000000",
            price="1",
            item=fine,
        ),
        price=Money.usd("1"),
        item=fine,
    )
    order = proposed_order(
        side=OrderSide.BUY,
        quantity="2000000000",
        item=fine,
    )
    policy = LongShortRisk(
        max_leverage=Decimal("10"),
        max_position_weight=Decimal(
            "0.12345678901234567890123456789"
        ),
        allow_short=True,
    )

    with localcontext() as low:
        low.prec = 12
        low_decision = policy.evaluate(order, current_context)
    with localcontext() as high:
        high.prec = 80
        high_decision = policy.evaluate(order, current_context)

    exact_cap = Quantity.of("1234567890.12345678901234567890")
    assert low_decision.final_quantity == exact_cap
    assert high_decision.final_quantity == exact_cap
    assert low_decision == high_decision


def test_risk_widens_exponent_bounds_for_finite_position_cap():
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

    decision = LongShortRisk(
        max_leverage=Decimal("1000"),
        max_position_weight=Decimal("100"),
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.SELL,
            quantity="1E+1000001",
            item=huge,
        ),
        current_context,
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("1E+1000000")
    assert decision.codes == ("max_position_weight",)


@pytest.mark.parametrize("requested", ["5", "10"])
def test_positive_cash_never_blocks_partial_or_full_short_cover(
    requested: str,
):
    current_context = context(
        portfolio=snapshot(cash="50", position="-10"),
    )

    decision = LongShortRisk(
        max_leverage=Decimal("100"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity=requested),
        current_context,
    )

    assert decision.status is RiskStatus.PASSED
    assert decision.final_quantity == Quantity.of(requested)


def test_market_short_to_long_crossing_does_not_count_short_cash_twice():
    current_context = context(
        portfolio=snapshot(cash="1500", position="-10"),
    )

    decision = LongShortRisk(
        max_leverage=Decimal("100"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(side=OrderSide.BUY, quantity="20"),
        current_context,
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("15")
    assert decision.codes == ("available_cash",)


def test_limit_short_to_long_crossing_uses_larger_of_cover_and_cash_capacity():
    current_context = context(
        portfolio=snapshot(cash="1500", position="-10"),
    )

    decision = LongShortRisk(
        max_leverage=Decimal("100"),
        max_position_weight=None,
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.BUY,
            quantity="20",
            order_type=OrderType.LIMIT,
            limit_price=Money.usd("200"),
        ),
        current_context,
    )

    assert decision.status is RiskStatus.ADJUSTED
    assert decision.final_quantity == Quantity.of("10")
    assert decision.codes == ("available_cash",)


def _order_in_status(status: OrderStatus) -> Order:
    pending = proposed_order(side=OrderSide.BUY, quantity="10")
    if status is OrderStatus.ACCEPTED:
        return pending.accept()
    if status in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}:
        accepted = pending.accept()
        fill_quantity = "4" if status is OrderStatus.PARTIALLY_FILLED else "10"
        return accepted.apply_fill(
            Fill(
                id=FillId.parse("fill_" + "f" * 32),
                order_id=accepted.id,
                instrument=accepted.instrument,
                side=accepted.side,
                quantity=Quantity.of(fill_quantity),
                price=Money.usd("100"),
                fee=Money.usd("0"),
                timestamp=BASE_DATETIME,
            )
        )
    if status is OrderStatus.CANCELLED:
        return pending.cancel("cancelled before risk re-evaluation")
    if status is OrderStatus.REJECTED:
        return pending.reject()
    raise AssertionError(f"unsupported test status {status}")


@pytest.mark.parametrize(
    "status",
    [
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    ],
)
def test_risk_evaluates_only_pending_proposed_orders(status: OrderStatus):
    order = _order_in_status(status)
    assert order.status is status

    with pytest.raises(ConfigurationError, match="PENDING"):
        LongShortRisk(
            max_leverage=Decimal("1"),
            max_position_weight=None,
            allow_short=True,
        ).evaluate(order, context())


def test_risk_uses_the_original_ratio_at_hundred_decimal_lot_boundary():
    base = aapl()
    fine = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1"),
        lot_size=Decimal("1E-100"),
        timezone=base.timezone,
    )
    expected = fraction_decimal_lot_floor(1, 3, 100)

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("1"),
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.SELL,
            quantity=expected,
            item=fine,
        ),
        context(
            portfolio=snapshot(cash="1", price="3", item=fine),
            price=Money.usd("3"),
            item=fine,
        ),
    )

    assert decision.status is RiskStatus.PASSED
    assert decision.final_quantity == Quantity.of(expected)
    assert decision.codes == ()


def test_risk_rejects_a_nonterminating_floor_beyond_work_bound():
    base = aapl()
    excessive = Instrument(
        id=base.id,
        quote_currency="USD",
        tick_size=Decimal("1"),
        lot_size=Decimal("1E-5000"),
        timezone=base.timezone,
    )

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("1"),
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.SELL,
            quantity=Decimal("1E-5000"),
            item=excessive,
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

    assert decision.status is RiskStatus.REJECTED
    assert decision.codes == ("invalid_risk_arithmetic",)


def test_risk_cancels_large_common_factors_before_guarding():
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

    decision = LongShortRisk(
        max_leverage=Decimal("1"),
        max_position_weight=Decimal("1"),
        allow_short=True,
    ).evaluate(
        proposed_order(
            side=OrderSide.SELL,
            quantity=Decimal("0.3"),
            item=fractional,
        ),
        context(
            portfolio=current_snapshot,
            price=Money.usd(three_common),
            item=fractional,
        ),
    )

    assert decision.status is RiskStatus.PASSED
    assert decision.final_quantity == Quantity.of("0.3")
    assert decision.codes == ()
