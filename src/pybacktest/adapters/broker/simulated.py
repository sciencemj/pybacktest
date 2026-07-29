"""Deterministic next-bar broker simulation with atomic state transitions."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_DOWN,
    Context,
    Decimal,
    Inexact,
    localcontext,
)
from types import MappingProxyType
from typing import Any

import numpy as np

from pybacktest.domain.errors import (
    ClockRegressionError,
    ConfigurationError,
    DataValidationError,
)
from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.market import MarketSlice
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    TimeInForce,
)
from pybacktest.ports.broker import (
    BorrowCostModel,
    BrokerEvent,
    BrokerRunContext,
    CommissionModel,
    FillIdSource,
    FillModel,
    LiquidityModel,
    OrderCancelledEvent,
    OrderExpiredEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    SessionBoundaryProvider,
    SlippageModel,
)

_ZERO = Decimal("0")
_UNIT_LOT = Decimal("1")
_MAX_EXACT_DIGITS = 4096
_DAY_EXPIRY_MESSAGE = "explicit session boundary reached"
_CANCELLATION_MESSAGE = "cancelled by request"
_UNSUPPORTED_IMMEDIATE_TIF = (
    "IOC and FOK are unsupported by SimulatedBroker; "
    "submit DAY or GOOD_TIL_CANCELLED explicitly."
)


@dataclass(frozen=True, slots=True)
class _ExecutionPlan:
    order: Order
    quantity: Quantity
    price: Money
    fee: Money
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class _ExpiryPlan:
    order: Order
    timestamp: datetime


_BrokerPlan = _ExecutionPlan | _ExpiryPlan


@dataclass(frozen=True, slots=True)
class _BrokerCommit:
    events: tuple[BrokerEvent, ...]
    active_orders: dict[OrderId, Order]
    known_orders: dict[OrderId, Order]
    used_fill_ids: frozenset[FillId]
    next_fill_sequence: int


@dataclass(frozen=True, slots=True)
class _IndexedFillIds:
    """Pure deterministic fallback indexed by broker-owned committed ordinal."""

    def fill_id(self, sequence: int) -> FillId:
        return FillId.parse(f"fill_{sequence:032x}")


def _require_model(value: object, protocol: type[object], field: str) -> None:
    if not isinstance(value, protocol):
        raise ConfigurationError(f"{field} does not implement its broker port.")


def _copy_instruments(
    instruments: Mapping[InstrumentId, Instrument] | None,
) -> Mapping[InstrumentId, Instrument]:
    if instruments is None:
        return MappingProxyType({})
    if not isinstance(instruments, Mapping):
        raise ConfigurationError("instruments must be a mapping.")
    copied = dict(instruments)
    if not all(
        isinstance(instrument_id, InstrumentId)
        and isinstance(instrument, Instrument)
        and instrument.id == instrument_id
        for instrument_id, instrument in copied.items()
    ):
        raise ConfigurationError(
            "instruments must map matching InstrumentId to Instrument."
        )
    return MappingProxyType(copied)


def _aware_datetime(timestamp: object) -> datetime:
    if (
        not isinstance(timestamp, datetime)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise ConfigurationError("timestamp must be timezone-aware.")
    return timestamp


def _market_datetime(timestamp: np.datetime64) -> datetime:
    nanoseconds = int(timestamp.astype("datetime64[ns]").astype(np.int64))
    microseconds, remainder = divmod(nanoseconds, 1_000)
    if remainder:
        raise DataValidationError(
            "MarketSlice timestamp must have microsecond precision "
            "to convert to an aware UTC datetime without precision loss."
        )
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=microseconds)
    except OverflowError as error:
        raise DataValidationError(
            "MarketSlice timestamp is outside Python datetime range."
        ) from error


def _floor_to_lot(quantity: Decimal, lot_size: Decimal) -> Decimal:
    if quantity <= _ZERO:
        return _ZERO
    quantity_numerator, quantity_denominator = quantity.as_integer_ratio()
    lot_numerator, lot_denominator = lot_size.as_integer_ratio()
    whole_lots = (quantity_numerator * lot_denominator) // (
        quantity_denominator * lot_numerator
    )
    if whole_lots <= 0:
        return _ZERO
    whole = Decimal(whole_lots)
    precision = max(
        64,
        whole.adjusted() + len(lot_size.as_tuple().digits) + 18,
    )
    context = Context(
        prec=precision,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
    )
    with localcontext(context):
        return whole * lot_size


def _exact_nonnegative_subtract(
    left: Decimal,
    right: Decimal,
    *,
    operation: str,
) -> Decimal:
    if right == _ZERO:
        return left
    if left < right:
        raise ConfigurationError(f"{operation} cannot become negative.")
    common_exponent = min(
        left.as_tuple().exponent,
        right.as_tuple().exponent,
    )
    if not isinstance(common_exponent, int):
        raise ConfigurationError(f"{operation} requires finite Decimals.")
    required_digits = max(left.adjusted(), right.adjusted()) - common_exponent + 1
    if required_digits > _MAX_EXACT_DIGITS:
        raise ConfigurationError(
            f"{operation} exceeds the {_MAX_EXACT_DIGITS}-digit exact-work bound."
        )
    context = Context(
        prec=max(64, required_digits),
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
    )
    with localcontext(context) as active:
        result = active.subtract(left, right)
        if active.flags[Inexact]:
            raise ConfigurationError(f"{operation} could not be represented exactly.")
        return result


def _coefficient(digits: tuple[int, ...]) -> int:
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    return coefficient


def _digits_mod(digits: tuple[int, ...], divisor: int) -> int:
    remainder = 0
    for digit in digits:
        remainder = (remainder * 10 + digit) % divisor
    return remainder


def _trailing_zeros(digits: tuple[int, ...]) -> int:
    count = 0
    for digit in reversed(digits):
        if digit != 0:
            break
        count += 1
    return count


def _is_aligned(value: Decimal, increment: Decimal) -> bool:
    if not value.is_finite() or not increment.is_finite() or increment <= _ZERO:
        return False
    if value == _ZERO:
        return True
    value_tuple = value.as_tuple()
    increment_tuple = increment.as_tuple()
    value_exponent = value_tuple.exponent
    increment_exponent = increment_tuple.exponent
    if not isinstance(value_exponent, int) or not isinstance(
        increment_exponent,
        int,
    ):
        return False
    divisor = _coefficient(increment_tuple.digits)
    if divisor == 1:
        return value_exponent >= increment_exponent or (
            increment_exponent - value_exponent <= _trailing_zeros(value_tuple.digits)
        )
    exponent_delta = value_exponent - increment_exponent
    if exponent_delta >= 0:
        remainder = _digits_mod(value_tuple.digits, divisor)
        if remainder == 0:
            return True
        return remainder * pow(10, exponent_delta, divisor) % divisor == 0
    required_zeros = -exponent_delta
    if required_zeros > _trailing_zeros(value_tuple.digits):
        return False
    return _digits_mod(value_tuple.digits[:-required_zeros], divisor) == 0


def _round_to_tick(
    amount: Decimal,
    tick_size: Decimal,
    side: OrderSide,
) -> Decimal:
    if _is_aligned(amount, tick_size):
        return amount
    quotient_digits = max(
        1,
        amount.adjusted() - tick_size.adjusted() + 1,
    )
    precision = (
        quotient_digits
        + len(amount.as_tuple().digits)
        + len(tick_size.as_tuple().digits)
        + 16
    )
    if precision > _MAX_EXACT_DIGITS:
        raise ConfigurationError(
            "tick rounding exceeds the 4096-digit exact-work bound."
        )
    context = Context(
        prec=max(64, precision),
        rounding=ROUND_DOWN,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
    )
    with localcontext(context) as active:
        whole_ticks = active.divide(amount, tick_size).to_integral_value(
            rounding=ROUND_DOWN
        )
        active.clear_flags()
        rounded = active.multiply(whole_ticks, tick_size)
        if side is OrderSide.BUY:
            rounded = active.add(rounded, tick_size)
        if active.flags[Inexact]:
            raise ConfigurationError("tick rounding could not be represented exactly.")
        return rounded


def _clone_model(model: object, field: str, protocol: type[object]) -> Any:
    try:
        cloned = deepcopy(model)
    except Exception as error:
        raise ConfigurationError(
            f"{field} model could not be cloned for a fresh broker."
        ) from error
    _require_model(cloned, protocol, field)
    return cloned


class SimulatedBroker:
    """Deterministic broker for already-active immutable orders.

    When constructed without an instrument catalog, quantity is floored to a
    documented unit-lot fallback. Passing any catalog makes metadata mandatory
    for every submitted instrument.

    When ``session_boundary`` is omitted, DAY orders do not expire
    automatically. This direct-construction behavior deliberately avoids
    guessing venue sessions from UTC or local calendar dates.
    """

    def __init__(
        self,
        *,
        fill_model: FillModel,
        commission: CommissionModel,
        slippage: SlippageModel,
        liquidity: LiquidityModel,
        borrow_cost: BorrowCostModel,
        instruments: Mapping[InstrumentId, Instrument] | None = None,
        session_boundary: SessionBoundaryProvider | None = None,
        fill_ids: FillIdSource | None = None,
    ) -> None:
        _require_model(fill_model, FillModel, "fill_model")
        _require_model(commission, CommissionModel, "commission")
        _require_model(slippage, SlippageModel, "slippage")
        _require_model(liquidity, LiquidityModel, "liquidity")
        _require_model(borrow_cost, BorrowCostModel, "borrow_cost")
        if session_boundary is not None:
            _require_model(
                session_boundary,
                SessionBoundaryProvider,
                "session_boundary",
            )
        if fill_ids is not None:
            _require_model(fill_ids, FillIdSource, "fill_ids")

        self._fill_model = fill_model
        self._commission = commission
        self._slippage = slippage
        self._liquidity = liquidity
        self._borrow_cost = borrow_cost
        self._instruments = _copy_instruments(instruments)
        self._catalog_is_required = instruments is not None
        self._session_boundary = session_boundary
        self._fill_ids = fill_ids or _IndexedFillIds()
        self._active_orders: dict[OrderId, Order] = {}
        self._known_orders: dict[OrderId, Order] = {}
        self._used_fill_ids: set[FillId] = set()
        self._next_fill_sequence = 0
        self._last_market_timestamp: np.datetime64 | None = None
        self._last_committed_timestamp: datetime | None = None

    @property
    def active_orders(self) -> Mapping[OrderId, Order]:
        """Return an immutable snapshot, not the mutable backing mapping."""
        return MappingProxyType(dict(self._active_orders))

    @property
    def expires_day_orders(self) -> bool:
        """Report whether an explicit session-boundary provider is active."""
        return self._session_boundary is not None

    def submit(self, order: Order) -> tuple[BrokerEvent, ...]:
        """Store one accepted active order in stable submission order."""
        if not isinstance(order, Order):
            raise ConfigurationError("order must be an Order.")
        if order.id in self._known_orders:
            raise ConfigurationError(f"duplicate order id: {order.id}.")
        if order.time_in_force in {
            TimeInForce.IMMEDIATE_OR_CANCEL,
            TimeInForce.FILL_OR_KILL,
        }:
            raise ConfigurationError(_UNSUPPORTED_IMMEDIATE_TIF)
        if order.status not in {
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        }:
            raise ConfigurationError(
                "SimulatedBroker consumes only ACCEPTED or PARTIALLY_FILLED orders."
            )
        instrument = self._instruments.get(order.instrument)
        if self._catalog_is_required and instrument is None:
            raise ConfigurationError(
                "submitted order instrument is missing from the catalog."
            )
        if instrument is not None and instrument.quote_currency != order.quote_currency:
            raise ConfigurationError(
                "order quote currency must match instrument metadata."
            )
        if instrument is not None:
            self._validate_order_metadata(order, instrument)
        self._known_orders[order.id] = order
        self._active_orders[order.id] = order
        return ()

    def cancel(
        self,
        order_id: OrderId,
        timestamp: datetime,
    ) -> tuple[BrokerEvent, ...]:
        """Cancel one active order; unknown and repeated requests are errors."""
        if not isinstance(order_id, OrderId):
            raise ConfigurationError("order_id must be an OrderId.")
        cancelled_at = _aware_datetime(timestamp)
        if order_id not in self._known_orders:
            raise ConfigurationError(f"unknown order: {order_id}.")
        order = self._active_orders.get(order_id)
        if order is None:
            raise ConfigurationError(f"order {order_id} is no longer active.")
        if cancelled_at < order.submitted_at:
            raise ConfigurationError(
                "cancellation timestamp cannot precede order submission."
            )
        if (
            self._last_committed_timestamp is not None
            and cancelled_at < self._last_committed_timestamp
        ):
            raise ClockRegressionError(
                "cancellation timestamp violates broker chronology."
            )
        cancelled = order.cancel(_CANCELLATION_MESSAGE)
        event = OrderCancelledEvent(
            order=cancelled,
            timestamp=cancelled_at,
            message=_CANCELLATION_MESSAGE,
        )
        self._known_orders[order_id] = cancelled
        del self._active_orders[order_id]
        self._last_committed_timestamp = cancelled_at
        return (event,)

    def process(
        self,
        market: MarketSlice,
        rng: np.random.Generator,
    ) -> tuple[BrokerEvent, ...]:
        """Calculate all actions first, then commit one bar atomically."""
        if not isinstance(market, MarketSlice):
            raise ConfigurationError("market must be a MarketSlice.")
        if not isinstance(rng, np.random.Generator):
            raise ConfigurationError("rng must be a numpy.random.Generator.")
        timestamp = _market_datetime(market.timestamp)
        if (
            self._last_market_timestamp is not None
            and market.timestamp <= self._last_market_timestamp
        ):
            raise ClockRegressionError(
                "MarketSlice timestamps must be strictly increasing."
            )
        if (
            self._last_committed_timestamp is not None
            and timestamp < self._last_committed_timestamp
        ):
            raise ClockRegressionError(
                "MarketSlice timestamp violates broker chronology."
            )
        staged_rng = deepcopy(rng)
        plans = self._calculate_plans(market, timestamp, staged_rng)
        commit = self._stage_commit(plans)

        rng.bit_generator.state = deepcopy(staged_rng.bit_generator.state)
        self._active_orders = commit.active_orders
        self._known_orders = commit.known_orders
        self._used_fill_ids = set(commit.used_fill_ids)
        self._next_fill_sequence = commit.next_fill_sequence
        self._last_market_timestamp = market.timestamp
        self._last_committed_timestamp = timestamp
        return commit.events

    def _calculate_plans(
        self,
        market: MarketSlice,
        timestamp: datetime,
        rng: np.random.Generator,
    ) -> tuple[_BrokerPlan, ...]:
        plans: list[_BrokerPlan] = []
        remaining_capacity: dict[InstrumentId, Decimal | None] = {}
        for order in self._active_orders.values():
            if order.active_from > timestamp or order.submitted_at >= timestamp:
                continue
            if self._day_order_expired(order, timestamp):
                plans.append(_ExpiryPlan(order=order, timestamp=timestamp))
                continue
            bar = market.bars.get(order.instrument)
            if bar is None or bar.volume <= 0:
                continue
            reference_price = self._fill_model.reference_price(order, bar)
            if reference_price is None:
                continue
            self._validate_price(order, reference_price, "reference price")

            if order.instrument not in remaining_capacity:
                available = self._liquidity.available_quantity(order, bar)
                self._validate_capacity(available)
                remaining_capacity[order.instrument] = (
                    None if available is None else available.value
                )
            capacity = remaining_capacity[order.instrument]
            requested = order.remaining_quantity.value
            candidate = requested if capacity is None else min(requested, capacity)
            quantity_value = _floor_to_lot(
                candidate,
                self._lot_size(order.instrument),
            )
            if quantity_value <= _ZERO:
                continue
            quantity = Quantity.of(quantity_value)
            slipped = self._slippage.apply(
                order,
                quantity,
                reference_price,
                bar,
                rng,
            )
            self._validate_price(order, slipped, "slippage price")
            price = self._execution_price(order, slipped)
            fee = self._commission.calculate(order, quantity, price)
            self._validate_fee(order, fee)
            plans.append(
                _ExecutionPlan(
                    order=order,
                    quantity=quantity,
                    price=price,
                    fee=fee,
                    timestamp=timestamp,
                )
            )
            if capacity is not None:
                remaining_capacity[order.instrument] = _exact_nonnegative_subtract(
                    capacity,
                    quantity_value,
                    operation="capacity subtraction",
                )
        return tuple(plans)

    def _stage_commit(
        self,
        plans: Sequence[_BrokerPlan],
    ) -> _BrokerCommit:
        next_active = dict(self._active_orders)
        next_known = dict(self._known_orders)
        execution_count = sum(isinstance(plan, _ExecutionPlan) for plan in plans)
        candidate_fill_ids = tuple(
            self._fill_ids.fill_id(self._next_fill_sequence + offset)
            for offset in range(execution_count)
        )
        self._validate_fill_ids(candidate_fill_ids)
        fill_ids = iter(candidate_fill_ids)
        events: list[BrokerEvent] = []
        for plan in plans:
            if isinstance(plan, _ExpiryPlan):
                expired = plan.order.cancel(_DAY_EXPIRY_MESSAGE)
                next_known[expired.id] = expired
                del next_active[expired.id]
                events.append(
                    OrderExpiredEvent(
                        order=expired,
                        timestamp=plan.timestamp,
                        message=_DAY_EXPIRY_MESSAGE,
                    )
                )
                continue

            fill_id = next(fill_ids)
            fill = Fill(
                id=fill_id,
                order_id=plan.order.id,
                instrument=plan.order.instrument,
                side=plan.order.side,
                quantity=plan.quantity,
                price=plan.price,
                fee=plan.fee,
                timestamp=plan.timestamp,
            )
            updated = plan.order.apply_fill(fill)
            next_known[updated.id] = updated
            if updated.status is OrderStatus.FILLED:
                del next_active[updated.id]
                events.append(OrderFilledEvent(fill=fill, order=updated))
            else:
                next_active[updated.id] = updated
                events.append(OrderPartiallyFilledEvent(fill=fill, order=updated))
        return _BrokerCommit(
            events=tuple(events),
            active_orders=next_active,
            known_orders=next_known,
            used_fill_ids=frozenset({*self._used_fill_ids, *candidate_fill_ids}),
            next_fill_sequence=(self._next_fill_sequence + execution_count),
        )

    def _day_order_expired(
        self,
        order: Order,
        timestamp: datetime,
    ) -> bool:
        if order.time_in_force is not TimeInForce.DAY or self._session_boundary is None:
            return False
        result = self._session_boundary.day_order_expired(order, timestamp)
        if not isinstance(result, bool):
            raise ConfigurationError(
                "session_boundary.day_order_expired() must return bool."
            )
        return result

    def _validate_fill_ids(
        self,
        candidate_fill_ids: Sequence[object],
    ) -> None:
        seen: set[FillId] = set()
        for fill_id in candidate_fill_ids:
            if not isinstance(fill_id, FillId):
                raise ConfigurationError("fill id source must return a FillId.")
            if fill_id in self._used_fill_ids or fill_id in seen:
                raise ConfigurationError(
                    f"duplicate fill id in staged process: {fill_id}."
                )
            seen.add(fill_id)

    def _validate_order_metadata(
        self,
        order: Order,
        instrument: Instrument,
    ) -> None:
        alignments = (
            ("order quantity", order.quantity.value, instrument.lot_size),
            (
                "filled quantity",
                order.filled_quantity.value,
                instrument.lot_size,
            ),
        )
        for field, value, increment in alignments:
            if not _is_aligned(value, increment):
                raise ConfigurationError(
                    f"{field} must align to the instrument lot size."
                )
        remaining = _exact_nonnegative_subtract(
            order.quantity.value,
            order.filled_quantity.value,
            operation="order remaining quantity",
        )
        if not _is_aligned(remaining, instrument.lot_size):
            raise ConfigurationError(
                "remaining quantity must align to the instrument lot size."
            )
        if order.limit_price is not None and not _is_aligned(
            order.limit_price.amount,
            instrument.tick_size,
        ):
            raise ConfigurationError(
                "limit price must align to the instrument tick size."
            )

    def _lot_size(self, instrument_id: InstrumentId) -> Decimal:
        instrument = self._instruments.get(instrument_id)
        return _UNIT_LOT if instrument is None else instrument.lot_size

    @staticmethod
    def _validate_capacity(available: Quantity | None) -> None:
        if available is not None and (
            not isinstance(available, Quantity) or available.value < _ZERO
        ):
            raise ConfigurationError(
                "liquidity model must return a nonnegative Quantity or None."
            )

    @staticmethod
    def _validate_price(
        order: Order,
        price: object,
        field: str,
    ) -> None:
        if (
            not isinstance(price, Money)
            or price.currency != order.quote_currency
            or not price.amount.is_finite()
            or price.amount <= _ZERO
        ):
            raise ConfigurationError(
                f"{field} must be positive finite Money in the order quote currency."
            )

    @staticmethod
    def _validate_fee(order: Order, fee: object) -> None:
        if (
            not isinstance(fee, Money)
            or fee.currency != order.quote_currency
            or not fee.amount.is_finite()
            or fee.amount < _ZERO
        ):
            raise ConfigurationError(
                "fee must be nonnegative finite Money in the order quote currency."
            )

    def _execution_price(self, order: Order, price: Money) -> Money:
        instrument = self._instruments.get(order.instrument)
        if instrument is not None:
            price = Money.of(
                _round_to_tick(
                    price.amount,
                    instrument.tick_size,
                    order.side,
                ),
                price.currency,
            )
        limit = order.limit_price
        if limit is not None:
            if order.side is OrderSide.BUY and price.amount > limit.amount:
                price = limit
            if order.side is OrderSide.SELL and price.amount < limit.amount:
                price = limit
        self._validate_price(order, price, "execution price")
        return price


@dataclass(frozen=True, slots=True)
class SimulatedBrokerFactory:
    """Immutable model configuration that creates fresh per-run brokers."""

    fill_model: FillModel
    commission: CommissionModel
    slippage: SlippageModel
    liquidity: LiquidityModel
    borrow_cost: BorrowCostModel

    def __post_init__(self) -> None:
        _require_model(self.fill_model, FillModel, "fill_model")
        _require_model(self.commission, CommissionModel, "commission")
        _require_model(self.slippage, SlippageModel, "slippage")
        _require_model(self.liquidity, LiquidityModel, "liquidity")
        _require_model(self.borrow_cost, BorrowCostModel, "borrow_cost")

    def create(self, run_context: BrokerRunContext) -> SimulatedBroker:
        """Create a broker with fresh order/counter state for one run."""
        if not isinstance(run_context, BrokerRunContext):
            raise ConfigurationError("run_context must be a BrokerRunContext.")
        return SimulatedBroker(
            fill_model=_clone_model(
                self.fill_model,
                "fill_model",
                FillModel,
            ),
            commission=_clone_model(
                self.commission,
                "commission",
                CommissionModel,
            ),
            slippage=_clone_model(
                self.slippage,
                "slippage",
                SlippageModel,
            ),
            liquidity=_clone_model(
                self.liquidity,
                "liquidity",
                LiquidityModel,
            ),
            borrow_cost=_clone_model(
                self.borrow_cost,
                "borrow_cost",
                BorrowCostModel,
            ),
            instruments=run_context.instruments,
            session_boundary=run_context.session_boundary,
            fill_ids=run_context.fill_ids,
        )


__all__ = ["SimulatedBroker", "SimulatedBrokerFactory"]
