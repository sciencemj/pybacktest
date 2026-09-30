"""Long/short portfolio risk decisions with stable constraint ordering."""

from dataclasses import dataclass
from decimal import Decimal, DecimalException

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity, decimal_from
from pybacktest.domain.orders import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from pybacktest.ports.risk import (
    RiskContext,
    RiskDecision,
    RiskStatus,
)

from ._decimal import (
    decimal_context,
    exact_add,
    exact_multiply,
    exact_subtract,
    floor_quantity_to_lot,
    is_aligned,
)

_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class LongShortRisk:
    """Apply deterministic long/short, exposure, and cash constraints."""

    max_leverage: Decimal
    max_position_weight: Decimal | None
    allow_short: bool

    def __post_init__(self) -> None:
        leverage = decimal_from(self.max_leverage, "max_leverage")
        if leverage <= _ZERO:
            raise ConfigurationError("max_leverage must be positive.")
        position_weight = self.max_position_weight
        if position_weight is not None:
            position_weight = decimal_from(
                position_weight,
                "max_position_weight",
            )
            if position_weight <= _ZERO:
                raise ConfigurationError("max_position_weight must be positive.")
        if type(self.allow_short) is not bool:
            raise ConfigurationError("allow_short must be a bool.")
        object.__setattr__(self, "max_leverage", leverage)
        object.__setattr__(
            self,
            "max_position_weight",
            position_weight,
        )

    def evaluate(
        self,
        order: Order,
        context: RiskContext,
    ) -> RiskDecision:
        if not isinstance(order, Order):
            raise ConfigurationError("order must be an Order.")
        if not isinstance(context, RiskContext):
            raise ConfigurationError("context must be a RiskContext.")
        if order.status is not OrderStatus.PENDING:
            raise ConfigurationError(
                "risk evaluation requires a PENDING proposed order."
            )

        validation = _validate_inputs(order, context)
        if validation is not None:
            code, message = validation
            return _rejection(order.quantity, (code,), message)

        try:
            with decimal_context(_risk_arithmetic_values(self, order, context)):
                return self._evaluate_validated(order, context)
        except DecimalException:
            return _rejection(
                order.quantity,
                ("invalid_risk_arithmetic",),
                "Risk arithmetic could not produce a finite decision.",
            )

    def _evaluate_validated(
        self,
        order: Order,
        context: RiskContext,
    ) -> RiskDecision:
        original = order.quantity.value
        instrument = context.instruments[order.instrument]
        mark = context.prices[order.instrument]
        current = _position_quantity(context, order.instrument)
        reserved_buy = _active_quantity(
            context,
            order.instrument,
            OrderSide.BUY,
        )
        reserved_sell = _active_quantity(
            context,
            order.instrument,
            OrderSide.SELL,
        )
        direction = Decimal("1") if order.side is OrderSide.BUY else Decimal("-1")
        candidates: list[tuple[str, Decimal]] = []

        if not self.allow_short and order.side is OrderSide.SELL:
            sellable = max(
                exact_subtract(current, reserved_sell),
                _ZERO,
            )
            projected = exact_subtract(sellable, original)
            if projected < _ZERO:
                candidates.append(("short_not_allowed", sellable))

        equity = context.snapshot.equity.amount
        if self.max_position_weight is not None:
            position_cap = _ZERO
            position_code = "non_positive_equity"
            if equity > _ZERO:
                position_cap = _lot_quantity(
                    exact_multiply(
                        equity,
                        self.max_position_weight,
                    ),
                    mark.amount,
                    instrument.lot_size,
                )
                position_code = "max_position_weight"
            allowed = _reserved_directional_quantity_cap(
                current,
                direction,
                position_cap,
                reserved_buy=reserved_buy,
                reserved_sell=reserved_sell,
            )
            if allowed < original:
                candidates.append((position_code, allowed))

        gross_cap = _ZERO
        leverage_code = "non_positive_equity"
        if equity > _ZERO:
            gross_cap = exact_multiply(equity, self.max_leverage)
            leverage_code = "max_leverage"
        other_gross = _other_worst_case_gross_exposure(
            context,
            order.instrument,
        )
        remaining_gross = max(
            exact_subtract(gross_cap, other_gross),
            _ZERO,
        )
        leverage_position_cap = _lot_quantity(
            remaining_gross,
            mark.amount,
            instrument.lot_size,
        )
        leverage_allowed = _reserved_directional_quantity_cap(
            current,
            direction,
            leverage_position_cap,
            reserved_buy=reserved_buy,
            reserved_sell=reserved_sell,
        )
        if leverage_allowed < original:
            candidates.append((leverage_code, leverage_allowed))

        if order.side is OrderSide.BUY:
            affordability_price = (
                order.limit_price if order.type is OrderType.LIMIT else mark
            )
            if affordability_price is None:
                return _rejection(
                    order.quantity,
                    ("invalid_limit_price",),
                    "Limit order is missing its limit price.",
                )
            available_cash = max(
                exact_subtract(
                    context.snapshot.cash.amount,
                    _reserved_buy_cash(context),
                ),
                _ZERO,
            )
            cash_quantity = _lot_quantity(
                available_cash,
                affordability_price.amount,
                instrument.lot_size,
            )
            closing_short = max(
                exact_add(current, reserved_buy).copy_negate(),
                _ZERO,
            )
            cash_allowed = max(closing_short, cash_quantity)
            if cash_allowed < original:
                candidates.append(("available_cash", cash_allowed))

        codes: list[str] = []
        final = original
        for code, allowed in candidates:
            final = min(final, max(allowed, _ZERO))
            if code not in codes:
                codes.append(code)
        if not codes:
            return RiskDecision(
                status=RiskStatus.PASSED,
                original_quantity=order.quantity,
                final_quantity=order.quantity,
                codes=(),
                message="Order passed all risk constraints.",
            )
        if final == _ZERO:
            return _rejection(
                order.quantity,
                tuple(codes),
                "Order rejected because no positive lot-aligned quantity "
                f"is allowed ({', '.join(codes)}).",
            )
        final_quantity = Quantity.of(final)
        return RiskDecision(
            status=RiskStatus.ADJUSTED,
            original_quantity=order.quantity,
            final_quantity=final_quantity,
            codes=tuple(codes),
            message=(
                f"Order quantity adjusted from {original} to {final} "
                f"({', '.join(codes)})."
            ),
        )


def _validate_inputs(
    order: Order,
    context: RiskContext,
) -> tuple[str, str] | None:
    instrument = context.instruments.get(order.instrument)
    if instrument is None:
        return (
            "unknown_instrument",
            f"Unknown instrument {order.instrument}.",
        )
    if order.instrument not in context.tradable:
        return (
            "instrument_not_tradable",
            f"Instrument {order.instrument} is not currently tradable.",
        )
    if order.quote_currency != instrument.quote_currency:
        return (
            "order_currency_mismatch",
            "Order quote currency does not match the instrument.",
        )
    if instrument.quote_currency != context.snapshot.cash.currency:
        return (
            "cross_currency_instrument",
            "Instrument quote currency does not match portfolio currency.",
        )
    mark_validation = _validate_mark(
        context.prices.get(order.instrument),
        instrument,
    )
    if mark_validation is not None:
        return mark_validation
    if not _aligned(order.quantity.value, instrument.lot_size):
        return (
            "invalid_quantity_lot",
            "Order quantity does not align to the instrument lot size.",
        )
    if order.type is OrderType.LIMIT:
        limit = order.limit_price
        if limit is None:
            return (
                "invalid_limit_price",
                "Limit order is missing its limit price.",
            )
        if limit.currency != instrument.quote_currency:
            return (
                "limit_price_currency_mismatch",
                "Limit price currency does not match the instrument.",
            )
        if not _aligned(limit.amount, instrument.tick_size):
            return (
                "invalid_limit_tick",
                "Limit price does not align to the instrument tick size.",
            )
    for instrument_id, position in context.snapshot.positions.items():
        if position.quantity.value == _ZERO:
            continue
        held_instrument = context.instruments.get(instrument_id)
        if held_instrument is None:
            return (
                "unknown_instrument",
                f"Unknown held instrument {instrument_id}.",
            )
        if held_instrument.quote_currency != context.snapshot.cash.currency:
            return (
                "cross_currency_instrument",
                f"Held instrument {instrument_id} quote currency does not "
                "match portfolio currency.",
            )
        held_mark_validation = _validate_mark(
            context.prices.get(instrument_id),
            held_instrument,
        )
        if held_mark_validation is not None:
            return held_mark_validation
        if not _aligned(
            position.quantity.value,
            held_instrument.lot_size,
        ):
            return (
                "invalid_position_lot",
                f"Position {instrument_id} is not lot-aligned.",
            )
    for active in context.active_orders:
        reserved_instrument = context.instruments.get(active.instrument)
        if reserved_instrument is None:
            return (
                "unknown_instrument",
                f"Unknown reserved instrument {active.instrument}.",
            )
        reserved_mark_validation = _validate_mark(
            context.prices.get(active.instrument),
            reserved_instrument,
        )
        if reserved_mark_validation is not None:
            return reserved_mark_validation
    return None


def _validate_mark(
    mark: Money | None,
    instrument: Instrument,
) -> tuple[str, str] | None:
    if mark is None:
        return (
            "missing_price",
            f"Current mark is unavailable for {instrument.id}.",
        )
    if mark.amount <= _ZERO:
        return (
            "invalid_price",
            f"Current mark must be positive for {instrument.id}.",
        )
    if mark.currency != instrument.quote_currency:
        return (
            "price_currency_mismatch",
            f"Current mark currency does not match {instrument.id}.",
        )
    if not _aligned(mark.amount, instrument.tick_size):
        return (
            "invalid_price_tick",
            f"Current mark does not align to {instrument.id} tick size.",
        )
    return None


def _position_quantity(
    context: RiskContext,
    instrument_id: InstrumentId,
) -> Decimal:
    position = context.snapshot.positions.get(instrument_id)
    return _ZERO if position is None else position.quantity.value


def _active_quantity(
    context: RiskContext,
    instrument_id: InstrumentId,
    side: OrderSide,
) -> Decimal:
    """Sum the unfilled quantity already reserved on one instrument side."""
    total = _ZERO
    for order in context.active_orders:
        if order.instrument != instrument_id or order.side is not side:
            continue
        total = exact_add(total, order.remaining_quantity.value)
    return total


def _reserved_buy_cash(context: RiskContext) -> Decimal:
    """Cash already committed by every active buy across the portfolio."""
    total = _ZERO
    for order in context.active_orders:
        if order.side is not OrderSide.BUY:
            continue
        price = (
            order.limit_price
            if order.type is OrderType.LIMIT and order.limit_price is not None
            else context.prices.get(order.instrument)
        )
        if price is None:
            continue
        total = exact_add(
            total,
            exact_multiply(order.remaining_quantity.value, price.amount),
        )
    return total


def _other_worst_case_gross_exposure(
    context: RiskContext,
    excluded: InstrumentId,
) -> Decimal:
    """Gross exposure of other instruments including pending reservations."""
    instrument_ids: list[InstrumentId] = []
    for instrument_id in (
        *context.snapshot.positions,
        *(order.instrument for order in context.active_orders),
    ):
        if instrument_id != excluded and instrument_id not in instrument_ids:
            instrument_ids.append(instrument_id)
    total = _ZERO
    for instrument_id in instrument_ids:
        price = context.prices.get(instrument_id)
        if price is None:
            continue
        position = context.snapshot.positions.get(instrument_id)
        current = _ZERO if position is None else position.quantity.value
        long_case = exact_add(
            current,
            _active_quantity(context, instrument_id, OrderSide.BUY),
        )
        short_case = exact_subtract(
            current,
            _active_quantity(context, instrument_id, OrderSide.SELL),
        )
        exposure = exact_multiply(
            max(long_case.copy_abs(), short_case.copy_abs()),
            price.amount,
        )
        total = exact_add(total, exposure)
    return total


def _reserved_directional_quantity_cap(
    current: Decimal,
    direction: Decimal,
    target_absolute_cap: Decimal,
    *,
    reserved_buy: Decimal,
    reserved_sell: Decimal,
) -> Decimal:
    """Cap one new order against the worst-case projected position."""
    if direction > _ZERO:
        return max(
            exact_subtract(
                target_absolute_cap,
                exact_add(current, reserved_buy),
            ),
            _ZERO,
        )
    return max(
        exact_add(
            exact_subtract(current, reserved_sell),
            target_absolute_cap,
        ),
        _ZERO,
    )


def _lot_quantity(
    notional: Decimal,
    price: Decimal,
    lot_size: Decimal,
) -> Decimal:
    return floor_quantity_to_lot(notional, price, lot_size)


def _rejection(
    original_quantity: Quantity,
    codes: tuple[str, ...],
    message: str,
) -> RiskDecision:
    return RiskDecision(
        status=RiskStatus.REJECTED,
        original_quantity=original_quantity,
        final_quantity=Quantity.of(_ZERO),
        codes=codes,
        message=message,
    )


def _aligned(value: Decimal, increment: Decimal) -> bool:
    return is_aligned(value, increment)


def _risk_arithmetic_values(
    policy: LongShortRisk,
    order: Order,
    context: RiskContext,
) -> tuple[Decimal, ...]:
    values = [
        order.quantity.value,
        context.snapshot.cash.amount,
        context.snapshot.equity.amount,
        policy.max_leverage,
        *(position.quantity.value for position in context.snapshot.positions.values()),
        *(price.amount for price in context.prices.values()),
        *(
            value
            for instrument in context.instruments.values()
            for value in (instrument.tick_size, instrument.lot_size)
        ),
        *(
            value
            for active in context.active_orders
            for value in (
                active.quantity.value,
                active.remaining_quantity.value,
            )
        ),
        *(
            active.limit_price.amount
            for active in context.active_orders
            if active.limit_price is not None
        ),
    ]
    if policy.max_position_weight is not None:
        values.append(policy.max_position_weight)
    if order.limit_price is not None:
        values.append(order.limit_price.amount)
    return tuple(values)


__all__ = ["LongShortRisk", "RiskStatus"]
