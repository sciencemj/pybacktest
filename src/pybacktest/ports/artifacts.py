"""Inward artifact persistence port."""

from typing import Protocol, runtime_checkable

from pybacktest.results.models import ArtifactRef, BacktestResult


@runtime_checkable
class ArtifactStore(Protocol):
    """Persist one immutable result and return finalized integrity metadata."""

    def write(self, result: BacktestResult) -> ArtifactRef:
        """Write a result atomically without mutating it."""
        raise NotImplementedError


__all__ = ["ArtifactStore"]
