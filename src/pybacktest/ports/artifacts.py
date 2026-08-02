"""Inward artifact persistence port."""

from typing import Literal, Protocol, runtime_checkable

from pybacktest.domain.errors import AdapterContractError
from pybacktest.results.models import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    BacktestResult,
)


class ArtifactDurabilityError(AdapterContractError):
    """Publication committed, but its final durability could not be confirmed."""

    def __init__(
        self,
        message: str,
        *,
        committed_artifact: ArtifactRef,
        location_lost: bool,
    ) -> None:
        super().__init__(
            message,
            code="artifact_published_durability_uncertain",
        )
        self.committed: Literal[True] = True
        self.location_lost = location_lost
        self.intended_path = committed_artifact.path
        self.manifest: ArtifactManifest = committed_artifact.manifest
        self.manifest_checksum = committed_artifact.manifest_checksum
        self.files: tuple[ArtifactFile, ...] = committed_artifact.files
        self.artifact_ref: ArtifactRef | None = (
            None if location_lost else committed_artifact
        )


@runtime_checkable
class ArtifactStore(Protocol):
    """Persist one immutable result and return finalized integrity metadata."""

    def write(self, result: BacktestResult) -> ArtifactRef:
        """Write a result atomically without mutating it."""
        raise NotImplementedError


__all__ = ["ArtifactDurabilityError", "ArtifactStore"]
