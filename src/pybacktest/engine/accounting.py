"""Reconciled long/short portfolio accounting."""

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, DecimalException, localcontext
from types import MappingProxyType

from pybacktest.domain.errors import AccountingInvariantError, ConfigurationError
from pybacktest.domain.identifiers import FillId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity, normalize_currency
from pybacktest.domain.orders import Fill, OrderSide
from pybacktest.domain.portfolio import CashEvent, PortfolioSnapshot, Position

_ZERO = Decimal("0")


class PortfolioLedger:
    """The sole mutable owner of reconciled cash and position state."""

    def __init__(
        self,
        *,
        base_currency: str,
        initial_cash: Money,
        instruments: Mapping[InstrumentId, Instrument],
    ) -> None:
        try:
            normalized_currency = normalize_currency(base_currency)
        except ConfigurationError as error:
            raise AccountingInvariantError(
                "base currency is invalid."
            ) from error
        if not isinstance(initial_cash, Money):
            raise AccountingInvariantError("initial cash must be Money.")
        if initial_cash.currency != normalized_currency:
            raise AccountingInvariantError(
                "initial cash currency must match the base currency."
            )
        if initial_cash.amount < _ZERO:
            raise AccountingInvariantError(
                "initial cash cannot be negative."
            )
        if not isinstance(instruments, Mapping):
            raise AccountingInvariantError(
                "instrument catalog must be a mapping."
            )
        copied_catalog = dict(instruments)
        for instrument_id, instrument in copied_catalog.items():
            if (
                not isinstance(instrument_id, InstrumentId)
                or not isinstance(instrument, Instrument)
                or instrument.id != instrument_id
            ):
                raise AccountingInvariantError(
                    "instrument catalog keys must match Instrument metadata."
                )
            if instrument.quote_currency != normalized_currency:
                raise AccountingInvariantError(
                    "instrument quote currency must match the base currency."
                )

        self._base_currency = normalized_currency
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._instruments = MappingProxyType(copied_catalog)
        self._positions: dict[InstrumentId, Position] = {}
        self._total_fees = Money.of(_ZERO, normalized_currency)
        self._timestamp: datetime | None = None
        self._processed_fill_ids: set[FillId] = set()
        self._valuation_prices: dict[InstrumentId, Money] = {}
        self._cash_events: list[CashEvent] = []

    def snapshot(self) -> PortfolioSnapshot:
        """Return a new immutable view of the current private state."""
        return self._build_snapshot(
            timestamp=self._timestamp,
            cash=self._cash,
            positions=self._positions,
            total_fees=self._total_fees,
            valuation_prices=self._valuation_prices,
            cash_events=tuple(self._cash_events),
        )

    def apply_fill(self, fill: Fill) -> PortfolioSnapshot:
        """Validate, reconcile, and atomically apply one execution."""
        self._validate_fill(fill)
        previous = self._positions.get(fill.instrument)
        old_quantity = (
            previous.quantity.value if previous is not None else _ZERO
        )
        old_average = previous.average_price if previous is not None else None
        old_realized = (
            previous.realized_pnl
            if previous is not None
            else Money.of(_ZERO, self._base_currency)
        )
        signed_quantity = (
            fill.quantity.value
            if fill.side is OrderSide.BUY
            else -fill.quantity.value
        )

        try:
            new_quantity = old_quantity + signed_quantity
            average_price, realized_delta = _transition_cost_basis(
                old_quantity=old_quantity,
                old_average=old_average,
                signed_fill_quantity=signed_quantity,
                fill_price=fill.price,
            )
            new_realized = Money.of(
                old_realized.amount + realized_delta,
                self._base_currency,
            )
            new_position = Position(
                instrument=fill.instrument,
                quantity=Quantity.of(new_quantity),
                average_price=average_price,
                realized_pnl=new_realized,
            )
            cash_delta = -(signed_quantity * fill.price.amount) - fill.fee.amount
            candidate_cash = Money.of(
                self._cash.amount + cash_delta,
                self._base_currency,
            )
            candidate_fees = Money.of(
                self._total_fees.amount + fill.fee.amount,
                self._base_currency,
            )
        except (DecimalException, ConfigurationError) as error:
            raise AccountingInvariantError(
                "fill arithmetic produced an impossible value."
            ) from error

        if new_quantity - old_quantity != signed_quantity:
            raise AccountingInvariantError(
                "fill position delta did not reconcile."
            )
        if candidate_cash.amount - self._cash.amount != cash_delta:
            raise AccountingInvariantError(
                "fill cash delta did not reconcile."
            )
        if (
            candidate_fees.amount - self._total_fees.amount
            != fill.fee.amount
        ):
            raise AccountingInvariantError(
                "fill fee delta did not reconcile."
            )

        candidate_positions = dict(self._positions)
        candidate_positions[fill.instrument] = new_position
        candidate_prices = self._prices_at_transition(fill.timestamp)
        candidate_prices[fill.instrument] = fill.price
        candidate = self._build_snapshot(
            timestamp=fill.timestamp,
            cash=candidate_cash,
            positions=candidate_positions,
            total_fees=candidate_fees,
            valuation_prices=candidate_prices,
            cash_events=tuple(self._cash_events),
        )

        self._cash = candidate_cash
        self._positions = candidate_positions
        self._total_fees = candidate_fees
        self._timestamp = fill.timestamp
        self._processed_fill_ids.add(fill.id)
        self._valuation_prices = candidate_prices
        return candidate

    def apply_cash_event(self, event: CashEvent) -> PortfolioSnapshot:
        """Validate, reconcile, and atomically apply explicit external cash."""
        if not isinstance(event, CashEvent):
            raise AccountingInvariantError(
                "cash event must be a CashEvent."
            )
        self._validate_timestamp(event.timestamp)
        if event.amount.currency != self._base_currency:
            raise AccountingInvariantError(
                "cash event currency must match the base currency."
            )
        try:
            candidate_cash = Money.of(
                self._cash.amount + event.amount.amount,
                self._base_currency,
            )
        except (DecimalException, ConfigurationError) as error:
            raise AccountingInvariantError(
                "cash event arithmetic produced an impossible value."
            ) from error
        if (
            candidate_cash.amount - self._cash.amount
            != event.amount.amount
        ):
            raise AccountingInvariantError(
                "cash event delta did not reconcile."
            )

        candidate_events = (*self._cash_events, event)
        candidate_prices = self._prices_at_transition(event.timestamp)
        candidate = self._build_snapshot(
            timestamp=event.timestamp,
            cash=candidate_cash,
            positions=self._positions,
            total_fees=self._total_fees,
            valuation_prices=candidate_prices,
            cash_events=candidate_events,
        )

        self._cash = candidate_cash
        self._timestamp = event.timestamp
        self._cash_events.append(event)
        self._valuation_prices = candidate_prices
        return candidate

    def mark_to_market(
        self,
        *,
        timestamp: datetime,
        prices: Mapping[InstrumentId, Money],
    ) -> PortfolioSnapshot:
        """Replace current valuation marks without changing book accounting."""
        self._validate_timestamp(timestamp)
        if not isinstance(prices, Mapping):
            raise AccountingInvariantError(
                "valuation prices must be a mapping."
            )
        candidate_prices = dict(prices)
        for instrument_id, price in candidate_prices.items():
            if instrument_id not in self._instruments:
                raise AccountingInvariantError(
                    f"unknown instrument {instrument_id} in valuation prices."
                )
            if not isinstance(price, Money):
                raise AccountingInvariantError(
                    "valuation price must be Money."
                )
            if price.currency != self._base_currency:
                raise AccountingInvariantError(
                    "valuation price currency must match the base currency."
                )
            if price.amount <= _ZERO:
                raise AccountingInvariantError(
                    "valuation price must be positive."
                )

        candidate = self._build_snapshot(
            timestamp=timestamp,
            cash=self._cash,
            positions=self._positions,
            total_fees=self._total_fees,
            valuation_prices=candidate_prices,
            cash_events=tuple(self._cash_events),
        )
        self._timestamp = timestamp
        self._valuation_prices = candidate_prices
        return candidate

    def _validate_fill(self, fill: Fill) -> Instrument:
        if not isinstance(fill, Fill):
            raise AccountingInvariantError("fill must be a Fill.")
        instrument = self._instruments.get(fill.instrument)
        if instrument is None:
            raise AccountingInvariantError(
                f"fill references unknown instrument {fill.instrument}."
            )
        if fill.id in self._processed_fill_ids:
            raise AccountingInvariantError(
                f"duplicate fill id {fill.id}."
            )
        self._validate_timestamp(fill.timestamp)
        if (
            fill.price.currency != self._base_currency
            or fill.fee.currency != self._base_currency
            or fill.price.currency != instrument.quote_currency
        ):
            raise AccountingInvariantError(
                "fill currency must match the instrument and base currency."
            )
        if fill.quantity.value <= _ZERO:
            raise AccountingInvariantError(
                "fill quantity must be positive."
            )
        if fill.price.amount <= _ZERO:
            raise AccountingInvariantError(
                "fill price must be positive."
            )
        if fill.fee.amount < _ZERO:
            raise AccountingInvariantError(
                "fill fee cannot be negative."
            )
        if not _is_aligned(
            fill.quantity.value,
            instrument.lot_size,
            unit_name="lot",
        ):
            raise AccountingInvariantError(
                "fill quantity must align to the instrument lot size."
            )
        if not _is_aligned(
            fill.price.amount,
            instrument.tick_size,
            unit_name="tick",
        ):
            raise AccountingInvariantError(
                "fill price must align to the instrument tick size."
            )
        return instrument

    def _validate_timestamp(self, timestamp: datetime) -> None:
        if (
            not isinstance(timestamp, datetime)
            or timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise AccountingInvariantError(
                "ledger timestamp must be timezone-aware."
            )
        if self._timestamp is not None and timestamp < self._timestamp:
            raise AccountingInvariantError(
                "ledger timestamp cannot move backwards."
            )

    def _prices_at_transition(
        self,
        timestamp: datetime,
    ) -> dict[InstrumentId, Money]:
        if timestamp == self._timestamp:
            return dict(self._valuation_prices)
        return {}

    def _build_snapshot(
        self,
        *,
        timestamp: datetime | None,
        cash: Money,
        positions: Mapping[InstrumentId, Position],
        total_fees: Money,
        valuation_prices: Mapping[InstrumentId, Money],
        cash_events: tuple[CashEvent, ...],
    ) -> PortfolioSnapshot:
        try:
            realized_amount = sum(
                (
                    position.realized_pnl.amount
                    for position in positions.values()
                ),
                _ZERO,
            )
            market_value_amount = _ZERO
            gross_exposure_amount = _ZERO
            for instrument_id, position in positions.items():
                quantity = position.quantity.value
                if quantity == _ZERO:
                    continue
                if position.average_price is None:
                    raise AccountingInvariantError(
                        "open position is missing average price."
                    )
                mark = valuation_prices.get(
                    instrument_id,
                    position.average_price,
                )
                marked_value = quantity * mark.amount
                market_value_amount += marked_value
                gross_exposure_amount += abs(marked_value)
            equity_amount = cash.amount + market_value_amount
            reconciliation_values = (
                equity_amount,
                self._initial_cash.amount,
                total_fees.amount,
                realized_amount,
                *(event.amount.amount for event in cash_events),
            )
            with localcontext() as context:
                context.prec = _exact_sum_precision(reconciliation_values)
                cash_event_amount = sum(
                    (event.amount.amount for event in cash_events),
                    _ZERO,
                )
                accounting_basis = (
                    self._initial_cash.amount
                    + cash_event_amount
                    - total_fees.amount
                    + realized_amount
                )
                unrealized_amount = equity_amount - accounting_basis
                if equity_amount != accounting_basis + unrealized_amount:
                    raise AccountingInvariantError(
                        "portfolio equity and P&L did not reconcile."
                    )
            return PortfolioSnapshot(
                timestamp=timestamp,
                cash=cash,
                positions=positions,
                realized_pnl=Money.of(
                    realized_amount,
                    self._base_currency,
                ),
                unrealized_pnl=Money.of(
                    unrealized_amount,
                    self._base_currency,
                ),
                total_fees=total_fees,
                market_value=Money.of(
                    market_value_amount,
                    self._base_currency,
                ),
                gross_exposure=Money.of(
                    gross_exposure_amount,
                    self._base_currency,
                ),
                equity=Money.of(
                    equity_amount,
                    self._base_currency,
                ),
                valuation_prices=valuation_prices,
                cash_events=cash_events,
            )
        except (DecimalException, ConfigurationError) as error:
            raise AccountingInvariantError(
                "portfolio valuation produced an impossible value."
            ) from error


def _transition_cost_basis(
    *,
    old_quantity: Decimal,
    old_average: Money | None,
    signed_fill_quantity: Decimal,
    fill_price: Money,
) -> tuple[Money | None, Decimal]:
    new_quantity = old_quantity + signed_fill_quantity
    if old_quantity == _ZERO:
        return fill_price, _ZERO
    if old_average is None:
        raise AccountingInvariantError(
            "open position is missing average price."
        )
    if _same_sign(old_quantity, signed_fill_quantity):
        weighted_cost = (
            abs(old_quantity) * old_average.amount
            + abs(signed_fill_quantity) * fill_price.amount
        )
        return (
            Money.of(weighted_cost / abs(new_quantity), fill_price.currency),
            _ZERO,
        )

    closed_quantity = min(abs(old_quantity), abs(signed_fill_quantity))
    direction = Decimal("1") if old_quantity > _ZERO else Decimal("-1")
    realized_delta = (
        fill_price.amount - old_average.amount
    ) * closed_quantity * direction
    if new_quantity == _ZERO:
        return None, realized_delta
    if _same_sign(old_quantity, new_quantity):
        return old_average, realized_delta
    return fill_price, realized_delta


def _same_sign(left: Decimal, right: Decimal) -> bool:
    return (left > _ZERO and right > _ZERO) or (
        left < _ZERO and right < _ZERO
    )


def _is_aligned(
    value: Decimal,
    unit: Decimal,
    *,
    unit_name: str,
) -> bool:
    try:
        return value % unit == _ZERO
    except DecimalException as error:
        raise AccountingInvariantError(
            f"fill {unit_name} alignment could not be reconciled."
        ) from error


def _exact_sum_precision(values: tuple[Decimal, ...]) -> int:
    nonzero_values = tuple(value for value in values if value != _ZERO)
    if not nonzero_values:
        return 64
    highest_place = max(value.adjusted() for value in nonzero_values)
    exponents: list[int] = []
    for value in nonzero_values:
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int):
            raise AccountingInvariantError(
                "portfolio reconciliation requires finite Decimal values."
            )
        exponents.append(exponent)
    lowest_place = min(exponents)
    carry_digits = len(str(len(nonzero_values))) + 2
    return max(64, highest_place - lowest_place + carry_digits)
