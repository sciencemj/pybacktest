"""Instrument identities and trading configuration."""

from dataclasses import dataclass
from datetime import tzinfo
from decimal import Decimal

from .errors import ConfigurationError
from .money import decimal_from, normalize_currency


@dataclass(frozen=True, slots=True, order=True)
class InstrumentId:
    """Canonical venue and symbol identity for a tradable instrument."""

    venue: str
    symbol: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.venue, str)
            or not isinstance(self.symbol, str)
            or not self.venue
            or not self.symbol
        ):
            raise ConfigurationError("InstrumentId requires venue and symbol.")
        if ":" in self.venue or ":" in self.symbol:
            raise ConfigurationError("InstrumentId parts cannot contain ':'.")
        object.__setattr__(self, "venue", self.venue.upper())
        object.__setattr__(self, "symbol", self.symbol.upper())

    @classmethod
    def parse(cls, value: str) -> "InstrumentId":
        parts = value.split(":")
        if len(parts) != 2:
            raise ConfigurationError(
                "InstrumentId must use the canonical 'VENUE:SYMBOL' form."
            )
        return cls(venue=parts[0].upper(), symbol=parts[1].upper())

    def __str__(self) -> str:
        return f"{self.venue}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class Instrument:
    """Immutable market configuration for a tradable instrument."""

    id: InstrumentId
    quote_currency: str
    tick_size: Decimal
    lot_size: Decimal
    timezone: tzinfo

    def __post_init__(self) -> None:
        if not isinstance(self.id, InstrumentId):
            raise ConfigurationError("id must be an InstrumentId.")
        tick_size = decimal_from(self.tick_size, "tick_size")
        lot_size = decimal_from(self.lot_size, "lot_size")
        if tick_size <= Decimal("0"):
            raise ConfigurationError("tick_size must be positive.")
        if lot_size <= Decimal("0"):
            raise ConfigurationError("lot_size must be positive.")
        if not isinstance(self.timezone, tzinfo):
            raise ConfigurationError("timezone must be a timezone object.")
        object.__setattr__(
            self, "quote_currency", normalize_currency(self.quote_currency)
        )
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "lot_size", lot_size)
