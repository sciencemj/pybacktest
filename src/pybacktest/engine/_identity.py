"""Deterministic run-scoped order and fill identity allocation."""

from __future__ import annotations

from uuid import UUID, uuid5

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import FillId, OrderId, RunId


def _run_namespace(run_id: RunId) -> UUID:
    """Return the UUID5 namespace derived from one run identity."""
    return UUID(run_id.value.removeprefix("run_"))


def _derive_fill_id(namespace: UUID, sequence: int) -> FillId:
    """Derive one fill identity from a run namespace and a fill ordinal."""
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ConfigurationError(
            "fill sequence must be a nonnegative integer.",
            code="invalid_fill_sequence",
        )
    return FillId.parse(f"fill_{uuid5(namespace, f'fill:{sequence}').hex}")


class _RunIdSequence:
    """Run-namespaced UUID5 identities with independent kind counters."""

    def __init__(self, run_id: RunId) -> None:
        self._namespace = _run_namespace(run_id)
        self._order_counter = 0

    def next_order_id(self) -> OrderId:
        """Allocate the next deterministic order identity for this run."""
        value = uuid5(
            self._namespace,
            f"order:{self._order_counter}",
        )
        self._order_counter += 1
        return OrderId.parse(f"order_{value.hex}")

    def fill_id(self, sequence: int) -> FillId:
        """Return the deterministic fill identity at an explicit ordinal."""
        return _derive_fill_id(self._namespace, sequence)


class _FillIdFacade:
    """Expose fill derivation without retaining the order-ID allocator."""

    __slots__ = ("_namespace",)

    def __init__(self, run_id: RunId) -> None:
        self._namespace = _run_namespace(run_id)

    def fill_id(self, sequence: int) -> FillId:
        """Return the run-scoped identity for one committed fill ordinal."""
        return _derive_fill_id(self._namespace, sequence)
