"""Inward artifact persistence port."""

from typing import Protocol, runtime_checkable

from pybacktest.domain.errors import AdapterContractError
from pybacktest.results.models import ArtifactRef, BacktestResult


class ArtifactDurabilityError(AdapterContractError):
    """Publication committed, but its final durability could not be confirmed."""

    def __init__(
        self,
        message: str,
        *,
        artifact_ref: ArtifactRef,
    ) -> None:
        super().__init__(
            message,
            code="artifact_published_durability_uncertain",
        )
        self.artifact_ref = artifact_ref


@runtime_checkable
class ArtifactStore(Protocol):
    """Persist one immutable result and return finalized integrity metadata."""

    def write(self, result: BacktestResult) -> ArtifactRef:
        """Write a result atomically without mutating it."""
        raise NotImplementedError


__all__ = ["ArtifactDurabilityError", "ArtifactStore"]
