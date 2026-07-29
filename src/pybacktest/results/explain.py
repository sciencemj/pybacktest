"""Pure causal explanation over already-recorded engine events."""

from collections.abc import Sequence

from pybacktest.domain.identifiers import OrderId

from .models import (
    EngineEvent,
    ResultValidationError,
    TradeExplanation,
    UnknownTradeError,
)


def build_trade_explanation(
    events: Sequence[EngineEvent],
    order_id: OrderId,
) -> TradeExplanation:
    """Return only exact matching records, without inferring missing stages."""
    if not isinstance(order_id, OrderId):
        raise ResultValidationError("order_id must be an OrderId.")
    copied = tuple(events)
    if not all(isinstance(event, EngineEvent) for event in copied):
        raise ResultValidationError("events must contain EngineEvent values.")
    entries = tuple(event for event in copied if event.order_id == order_id)
    if not entries:
        raise UnknownTradeError(str(order_id))
    return TradeExplanation(order_id=order_id, entries=entries)


__all__ = ["build_trade_explanation"]
