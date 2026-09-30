"""Immutable portfolio accounting values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException, localcontext
from enum import StrEnum
from types import MappingProxyType

from .errors import AccountingInvariantError
from .identifiers import CashEventId
from .instruments import InstrumentId
from .money import Money, Quantity


class CashEventCode(StrEnum):
    """Stable reason codes for explicit non-fill cash movements."""

    EXTERNAL_FLOW = "external_flow"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    BORROW_FEE = "borrow_fee"


@dataclass(frozen=True, slots=True)
class CashEvent:
    """A timestamped, typed cash movement applied directly to the ledger."""

    id: CashEventId
    timestamp: datetime
    amount: Money
    code: CashEventCode

    def __post_init__(self) -> None:
        if not isinstance(self.id, CashEventId):
            raise AccountingInvariantError("CashEvent id must be a CashEventId.")
        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise AccountingInvariantError(
                "CashEvent timestamp must be timezone-aware."
            )
        if not isinstance(self.amount, Money):
            raise AccountingInvariantError("CashEvent amount must be Money.")
        if not isinstance(self.code, CashEventCode):
            raise AccountingInvariantError("CashEvent code must be a CashEventCode.")


@dataclass(frozen=True, slots=True)
class Position:
    """Signed quantity, average cost, and cumulative realized P&L."""

    instrument: InstrumentId
    quantity: Quantity
    average_price: Money | None
    book_cost: Money
    realized_pnl: Money

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, InstrumentId):
            raise AccountingInvariantError(
                "Position instrument must be an InstrumentId."
            )
        if not isinstance(self.quantity, Quantity):
            raise AccountingInvariantError("Position quantity must be a Quantity.")
        if not isinstance(self.realized_pnl, Money):
            raise AccountingInvariantError("Position realized_pnl must be Money.")
        if not isinstance(self.book_cost, Money):
            raise AccountingInvariantError("Position book_cost must be Money.")
        if self.book_cost.currency != self.realized_pnl.currency:
            raise AccountingInvariantError("Position currencies must match.")
        if self.quantity.value == Decimal("0"):
            if self.average_price is not None:
                raise AccountingInvariantError(
                    "A flat position cannot retain an average price."
                )
            if self.book_cost.amount != Decimal("0"):
                raise AccountingInvariantError(
                    "A flat position must have zero book cost."
                )
            return
        if not isinstance(self.average_price, Money):
            raise AccountingInvariantError(
                "An open position requires an average price."
            )
        if self.average_price.amount <= Decimal("0"):
            raise AccountingInvariantError("Position average price must be positive.")
        if self.average_price.currency != self.realized_pnl.currency:
            raise AccountingInvariantError("Position currencies must match.")
        if self.book_cost.amount <= Decimal("0"):
            raise AccountingInvariantError(
                "An open position requires positive book cost."
            )


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """A complete immutable ledger state and valuation."""

    timestamp: datetime | None
    cash: Money
    positions: Mapping[InstrumentId, Position]
    realized_pnl: Money
    unrealized_pnl: Money
    total_fees: Money
    market_value: Money
    gross_exposure: Money
    equity: Money
    valuation_prices: Mapping[InstrumentId, Money]
    cash_events: tuple[CashEvent, ...]

    def __post_init__(self) -> None:
        if self.timestamp is not None and (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise AccountingInvariantError(
                "PortfolioSnapshot timestamp must be timezone-aware."
            )
        monetary_values = (
            self.cash,
            self.realized_pnl,
            self.unrealized_pnl,
            self.total_fees,
            self.market_value,
            self.gross_exposure,
            self.equity,
        )
        if not all(isinstance(value, Money) for value in monetary_values):
            raise AccountingInvariantError(
                "PortfolioSnapshot monetary fields must be Money."
            )
        currencies = {value.currency for value in monetary_values}
        if len(currencies) != 1:
            raise AccountingInvariantError("PortfolioSnapshot must use one currency.")
        try:
            positions = dict(self.positions)
            prices = dict(self.valuation_prices)
        except (TypeError, ValueError) as error:
            raise AccountingInvariantError(
                "PortfolioSnapshot mappings are invalid."
            ) from error
        for instrument_id, position in positions.items():
            if (
                not isinstance(instrument_id, InstrumentId)
                or not isinstance(position, Position)
                or position.instrument != instrument_id
            ):
                raise AccountingInvariantError(
                    "PortfolioSnapshot positions are inconsistent."
                )
            if position.realized_pnl.currency not in currencies:
                raise AccountingInvariantError(
                    "PortfolioSnapshot position currency must match."
                )
            if (
                position.average_price is not None
                and position.average_price.currency not in currencies
            ):
                raise AccountingInvariantError(
                    "PortfolioSnapshot position currency must match."
                )
        for instrument_id, price in prices.items():
            if (
                not isinstance(instrument_id, InstrumentId)
                or not isinstance(price, Money)
                or price.currency not in currencies
                or price.amount <= Decimal("0")
            ):
                raise AccountingInvariantError(
                    "PortfolioSnapshot valuation prices are inconsistent."
                )
        if isinstance(self.cash_events, (str, bytes, bytearray)) or not isinstance(
            self.cash_events, Sequence
        ):
            raise AccountingInvariantError(
                "PortfolioSnapshot cash_events must be a sequence."
            )
        cash_events = tuple(self.cash_events)
        if not all(isinstance(event, CashEvent) for event in cash_events):
            raise AccountingInvariantError(
                "PortfolioSnapshot cash_events must contain CashEvent values."
            )
        if self.total_fees.amount < Decimal("0"):
            raise AccountingInvariantError(
                "PortfolioSnapshot total fees cannot be negative."
            )
        if self.gross_exposure.amount < Decimal("0"):
            raise AccountingInvariantError(
                "PortfolioSnapshot gross exposure cannot be negative."
            )
        event_ids: set[CashEventId] = set()
        previous_event_timestamp: datetime | None = None
        for event in cash_events:
            if event.amount.currency != self.cash.currency:
                raise AccountingInvariantError(
                    "PortfolioSnapshot cash event currency must match."
                )
            if event.id in event_ids:
                raise AccountingInvariantError(
                    "PortfolioSnapshot cash event IDs must be unique."
                )
            if (
                previous_event_timestamp is not None
                and event.timestamp < previous_event_timestamp
            ):
                raise AccountingInvariantError(
                    "PortfolioSnapshot cash events must be monotonic."
                )
            if self.timestamp is None or event.timestamp > self.timestamp:
                raise AccountingInvariantError(
                    "PortfolioSnapshot cash event cannot follow its timestamp."
                )
            event_ids.add(event.id)
            previous_event_timestamp = event.timestamp

        arithmetic_values = (
            self.cash.amount,
            self.realized_pnl.amount,
            self.unrealized_pnl.amount,
            self.market_value.amount,
            self.gross_exposure.amount,
            self.equity.amount,
            *(
                value
                for position in positions.values()
                for value in (
                    position.quantity.value,
                    position.book_cost.amount,
                    position.realized_pnl.amount,
                )
            ),
            *(price.amount for price in prices.values()),
        )
        try:
            with localcontext() as context:
                context.prec = _portfolio_arithmetic_precision(arithmetic_values)
                expected_realized = sum(
                    (position.realized_pnl.amount for position in positions.values()),
                    Decimal("0"),
                )
                expected_market = Decimal("0")
                expected_gross = Decimal("0")
                expected_unrealized = Decimal("0")
                for instrument_id, position in positions.items():
                    quantity = position.quantity.value
                    if quantity == Decimal("0"):
                        continue
                    direction = (
                        Decimal("1") if quantity > Decimal("0") else Decimal("-1")
                    )
                    mark = prices.get(instrument_id)
                    marked_value = (
                        direction * position.book_cost.amount
                        if mark is None
                        else quantity * mark.amount
                    )
                    expected_market += marked_value
                    expected_gross += abs(marked_value)
                    expected_unrealized += (
                        marked_value - direction * position.book_cost.amount
                    )
                expected_equity = self.cash.amount + expected_market
        except DecimalException as error:
            raise AccountingInvariantError(
                "PortfolioSnapshot aggregates cannot be reconciled."
            ) from error

        derived_values = (
            (self.realized_pnl.amount, expected_realized, "realized P&L"),
            (
                self.unrealized_pnl.amount,
                expected_unrealized,
                "unrealized P&L",
            ),
            (self.market_value.amount, expected_market, "market value"),
            (self.gross_exposure.amount, expected_gross, "gross exposure"),
            (self.equity.amount, expected_equity, "equity"),
        )
        for actual, expected, field in derived_values:
            if actual != expected:
                raise AccountingInvariantError(
                    f"PortfolioSnapshot {field} is inconsistent."
                )
        object.__setattr__(
            self,
            "positions",
            MappingProxyType(positions),
        )
        object.__setattr__(
            self,
            "valuation_prices",
            MappingProxyType(prices),
        )
        object.__setattr__(self, "cash_events", cash_events)


def _portfolio_arithmetic_precision(
    values: tuple[Decimal, ...],
) -> int:
    nonzero_values = tuple(value for value in values if value != Decimal("0"))
    if not nonzero_values:
        return 64
    highest_place = max(value.adjusted() for value in nonzero_values)
    exponents: list[int] = []
    for value in nonzero_values:
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int):
            raise AccountingInvariantError(
                "PortfolioSnapshot requires finite Decimal values."
            )
        exponents.append(exponent)
    lowest_place = min(exponents)
    operand_digits = sum(len(value.as_tuple().digits) for value in nonzero_values)
    carry_digits = len(str(len(nonzero_values))) + 4
    return max(
        64,
        highest_place - lowest_place + operand_digits + carry_digits,
    )
