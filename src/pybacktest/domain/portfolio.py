"""Immutable portfolio accounting values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType

from .errors import AccountingInvariantError
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

    timestamp: datetime
    amount: Money
    code: CashEventCode

    def __post_init__(self) -> None:
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
            raise AccountingInvariantError(
                "CashEvent code must be a CashEventCode."
            )


@dataclass(frozen=True, slots=True)
class Position:
    """Signed quantity, average cost, and cumulative realized P&L."""

    instrument: InstrumentId
    quantity: Quantity
    average_price: Money | None
    realized_pnl: Money

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, InstrumentId):
            raise AccountingInvariantError(
                "Position instrument must be an InstrumentId."
            )
        if not isinstance(self.quantity, Quantity):
            raise AccountingInvariantError(
                "Position quantity must be a Quantity."
            )
        if not isinstance(self.realized_pnl, Money):
            raise AccountingInvariantError(
                "Position realized_pnl must be Money."
            )
        if self.quantity.value == Decimal("0"):
            if self.average_price is not None:
                raise AccountingInvariantError(
                    "A flat position cannot retain an average price."
                )
            return
        if not isinstance(self.average_price, Money):
            raise AccountingInvariantError(
                "An open position requires an average price."
            )
        if self.average_price.amount <= Decimal("0"):
            raise AccountingInvariantError(
                "Position average price must be positive."
            )
        if self.average_price.currency != self.realized_pnl.currency:
            raise AccountingInvariantError(
                "Position currencies must match."
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
            raise AccountingInvariantError(
                "PortfolioSnapshot must use one currency."
            )
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
        if (
            isinstance(self.cash_events, (str, bytes, bytearray))
            or not isinstance(self.cash_events, Sequence)
        ):
            raise AccountingInvariantError(
                "PortfolioSnapshot cash_events must be a sequence."
            )
        cash_events = tuple(self.cash_events)
        if not all(isinstance(event, CashEvent) for event in cash_events):
            raise AccountingInvariantError(
                "PortfolioSnapshot cash_events must contain CashEvent values."
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
