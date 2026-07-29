"""Deterministic conversion of strategy intents into proposed orders."""

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, DecimalException
from typing import overload

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.events import OrderRejected
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
from pybacktest.ports.risk import RiskContext, SizedOrderIntent

from ._decimal import decimal_context

_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class DefaultOrderSizer:
    """Size target intents and preserve explicit execution instructions."""

    @overload
    def size(
        self,
        intent: CancelOrderIntent,
        context: RiskContext,
    ) -> CancelOrderIntent: ...

    @overload
    def size(
        self,
        intent: SizedOrderIntent,
        context: RiskContext,
    ) -> Order | OrderRejected: ...

    def size(
        self,
        intent: SizedOrderIntent | CancelOrderIntent,
        context: RiskContext,
    ) -> Order | OrderRejected | CancelOrderIntent:
        if not isinstance(context, RiskContext):
            raise ConfigurationError("context must be a RiskContext.")
        if isinstance(intent, CancelOrderIntent):
            return intent
        if not isinstance(
            intent,
            (
                TargetWeight,
                TargetQuantity,
                MarketOrderIntent,
                LimitOrderIntent,
            ),
        ):
            raise ConfigurationError("intent is not a supported order intent.")

        instrument = context.instruments.get(intent.instrument)
        if instrument is None:
            return _rejected(
                intent,
                context,
                "unknown_instrument",
                f"Unknown instrument {intent.instrument}.",
            )
        if instrument.quote_currency != context.snapshot.cash.currency:
            return _rejected(
                intent,
                context,
                "cross_currency_instrument",
                "Instrument quote currency does not match portfolio currency.",
            )
        if isinstance(intent, TargetWeight):
            return self._size_weight(intent, context, instrument)
        if isinstance(intent, TargetQuantity):
            return self._size_quantity(intent, context, instrument)
        return self._size_explicit(intent, context, instrument)

    def _size_weight(
        self,
        intent: TargetWeight,
        context: RiskContext,
        instrument: Instrument,
    ) -> Order | OrderRejected:
        current = _current_quantity(context, instrument)
        if not _aligned(current, instrument.lot_size):
            return _rejected(
                intent,
                context,
                "invalid_position_lot",
                "Current position does not align to the instrument lot size.",
            )
        if intent.weight == _ZERO:
            if current == _ZERO:
                return _rejected(
                    intent,
                    context,
                    "no_op_target",
                    "The target already matches the current flat position.",
                )
            return _pending_order(
                intent=intent,
                context=context,
                instrument=instrument,
                signed_quantity=current.copy_negate(),
                order_type=OrderType.MARKET,
                limit_price=None,
                time_in_force=TimeInForce.DAY,
            )

        mark_or_rejection = _validated_mark(intent, context, instrument)
        if isinstance(mark_or_rejection, OrderRejected):
            return mark_or_rejection
        equity = context.snapshot.equity.amount
        if equity <= _ZERO:
            return _rejected(
                intent,
                context,
                "non_positive_equity",
                "A non-zero target weight requires positive marked equity.",
            )
        mark = mark_or_rejection
        try:
            values = (
                equity,
                intent.weight,
                mark.amount,
                instrument.lot_size,
                current,
            )
            with decimal_context(values):
                desired_lots = (
                    equity
                    * intent.weight
                    / (mark.amount * instrument.lot_size)
                ).to_integral_value(rounding=ROUND_DOWN)
                target = desired_lots * instrument.lot_size
                signed_quantity = target - current
        except DecimalException:
            return _rejected(
                intent,
                context,
                "invalid_sizing_arithmetic",
                "Target weight arithmetic could not produce a finite order.",
            )
        if signed_quantity == _ZERO:
            code = (
                "lot_rounding_zero"
                if target == _ZERO and current == _ZERO
                else "no_op_target"
            )
            return _rejected(
                intent,
                context,
                code,
                "The target produces no lot-aligned position change.",
            )
        return _pending_order(
            intent=intent,
            context=context,
            instrument=instrument,
            signed_quantity=signed_quantity,
            order_type=OrderType.MARKET,
            limit_price=None,
            time_in_force=TimeInForce.DAY,
        )

    def _size_quantity(
        self,
        intent: TargetQuantity,
        context: RiskContext,
        instrument: Instrument,
    ) -> Order | OrderRejected:
        current = _current_quantity(context, instrument)
        if not _aligned(current, instrument.lot_size):
            return _rejected(
                intent,
                context,
                "invalid_position_lot",
                "Current position does not align to the instrument lot size.",
            )
        if not _aligned(intent.quantity.value, instrument.lot_size):
            return _rejected(
                intent,
                context,
                "invalid_quantity_lot",
                "Target quantity does not align to the instrument lot size.",
            )
        try:
            with decimal_context(
                (
                    intent.quantity.value,
                    current,
                    instrument.lot_size,
                )
            ):
                signed_quantity = intent.quantity.value - current
        except DecimalException:
            return _rejected(
                intent,
                context,
                "invalid_sizing_arithmetic",
                "Target quantity arithmetic could not produce a finite order.",
            )
        if signed_quantity == _ZERO:
            return _rejected(
                intent,
                context,
                "no_op_target",
                "The absolute target already matches the current position.",
            )
        return _pending_order(
            intent=intent,
            context=context,
            instrument=instrument,
            signed_quantity=signed_quantity,
            order_type=OrderType.MARKET,
            limit_price=None,
            time_in_force=TimeInForce.DAY,
        )

    def _size_explicit(
        self,
        intent: MarketOrderIntent | LimitOrderIntent,
        context: RiskContext,
        instrument: Instrument,
    ) -> Order | OrderRejected:
        if not _aligned(intent.quantity.value, instrument.lot_size):
            return _rejected(
                intent,
                context,
                "invalid_quantity_lot",
                "Requested quantity does not align to the instrument lot size.",
            )
        order_type = OrderType.MARKET
        limit_price = None
        if isinstance(intent, LimitOrderIntent):
            order_type = OrderType.LIMIT
            limit_price = intent.limit_price
            if limit_price.currency != instrument.quote_currency:
                return _rejected(
                    intent,
                    context,
                    "limit_price_currency_mismatch",
                    "Limit price currency does not match the instrument.",
                )
            if not _aligned(limit_price.amount, instrument.tick_size):
                return _rejected(
                    intent,
                    context,
                    "invalid_limit_tick",
                    "Limit price does not align to the instrument tick size.",
                )
        signed_quantity = (
            intent.quantity.value
            if intent.side is OrderSide.BUY
            else intent.quantity.value.copy_negate()
        )
        return _pending_order(
            intent=intent,
            context=context,
            instrument=instrument,
            signed_quantity=signed_quantity,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=intent.time_in_force,
        )


def _validated_mark(
    intent: SizedOrderIntent,
    context: RiskContext,
    instrument: Instrument,
) -> Money | OrderRejected:
    mark = context.prices.get(instrument.id)
    if mark is None:
        return _rejected(
            intent,
            context,
            "missing_price",
            f"Current mark is unavailable for {instrument.id}.",
        )
    if mark.amount <= _ZERO:
        return _rejected(
            intent,
            context,
            "invalid_price",
            f"Current mark must be positive for {instrument.id}.",
        )
    if mark.currency != instrument.quote_currency:
        return _rejected(
            intent,
            context,
            "price_currency_mismatch",
            "Current mark currency does not match the instrument.",
        )
    if not _aligned(mark.amount, instrument.tick_size):
        return _rejected(
            intent,
            context,
            "invalid_price_tick",
            "Current mark does not align to the instrument tick size.",
        )
    return mark


def _current_quantity(
    context: RiskContext,
    instrument: Instrument,
) -> Decimal:
    position = context.snapshot.positions.get(instrument.id)
    return _ZERO if position is None else position.quantity.value


def _pending_order(
    *,
    intent: SizedOrderIntent,
    context: RiskContext,
    instrument: Instrument,
    signed_quantity: Decimal,
    order_type: OrderType,
    limit_price: Money | None,
    time_in_force: TimeInForce,
) -> Order:
    side = OrderSide.BUY if signed_quantity > _ZERO else OrderSide.SELL
    return Order.pending(
        id=context.order_id,
        instrument=instrument.id,
        side=side,
        type=order_type,
        quantity=Quantity.of(signed_quantity.copy_abs()),
        quote_currency=instrument.quote_currency,
        limit_price=limit_price,
        time_in_force=time_in_force,
        submitted_at=context.submitted_at,
        active_from=context.active_from,
        reason=intent.reason,
    )


def _rejected(
    intent: SizedOrderIntent,
    context: RiskContext,
    code: str,
    message: str,
) -> OrderRejected:
    return OrderRejected(
        order_id=context.order_id,
        instrument=intent.instrument,
        timestamp=context.submitted_at,
        message=message,
        reason=DecisionReason.of(
            code,
            instrument=str(intent.instrument),
            intent_reason=intent.reason.code,
        ),
    )


def _aligned(value: Decimal, increment: Decimal) -> bool:
    try:
        with decimal_context((value, increment)):
            return value % increment == _ZERO
    except DecimalException:
        return False


__all__ = ["DefaultOrderSizer"]
