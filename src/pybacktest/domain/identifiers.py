"""Canonical identifiers for core backtest entities."""

import re
from dataclasses import dataclass
from uuid import uuid4

from .errors import ConfigurationError


def _validate_identifier(value: str, pattern: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ConfigurationError(f"{name} is not in canonical form.")


@dataclass(frozen=True, slots=True)
class RunId:
    """Canonical identity for a backtest run."""

    value: str

    def __post_init__(self) -> None:
        _validate_identifier(self.value, r"run_[0-9a-f]{32}", "RunId")

    @classmethod
    def new(cls) -> "RunId":
        return cls(f"run_{uuid4().hex}")

    @classmethod
    def parse(cls, value: str) -> "RunId":
        return cls(value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class OrderId:
    """Canonical identity for an order."""

    value: str

    def __post_init__(self) -> None:
        _validate_identifier(self.value, r"order_[0-9a-f]{32}", "OrderId")

    @classmethod
    def new(cls) -> "OrderId":
        return cls(f"order_{uuid4().hex}")

    @classmethod
    def parse(cls, value: str) -> "OrderId":
        return cls(value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FillId:
    """Canonical identity for a fill."""

    value: str

    def __post_init__(self) -> None:
        _validate_identifier(self.value, r"fill_[0-9a-f]{32}", "FillId")

    @classmethod
    def new(cls) -> "FillId":
        return cls(f"fill_{uuid4().hex}")

    @classmethod
    def parse(cls, value: str) -> "FillId":
        return cls(value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class CashEventId:
    """Canonical identity for an explicit ledger cash event."""

    value: str

    def __post_init__(self) -> None:
        _validate_identifier(
            self.value,
            r"cash_event_[0-9a-f]{32}",
            "CashEventId",
        )

    @classmethod
    def new(cls) -> "CashEventId":
        return cls(f"cash_event_{uuid4().hex}")

    @classmethod
    def parse(cls, value: str) -> "CashEventId":
        return cls(value)

    def __str__(self) -> str:
        return self.value
