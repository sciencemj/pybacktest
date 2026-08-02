"""Strategy-facing intent contracts and reference strategies."""

from .components import MovingAverageCross
from .intents import OrderIntent

__all__ = ["MovingAverageCross", "OrderIntent"]
