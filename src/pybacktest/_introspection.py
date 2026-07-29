"""Internal deterministic reflection shared by provenance and fingerprints.

Strategy provenance and engine-component fingerprinting must agree exactly on
what counts as an object's configuration. Two separate implementations would
drift, and a gap in either one silently collapses two behaviourally different
objects onto a single digest, so both paths call the one helper here.
"""

from __future__ import annotations

from pybacktest.domain.errors import ConfigurationError

_PSEUDO_SLOTS = frozenset({"__dict__", "__weakref__"})


def _declared_slots(klass: type) -> tuple[str, ...]:
    declared = klass.__dict__.get("__slots__", ())
    if isinstance(declared, str):
        return (declared,)
    try:
        slots = tuple(declared)
    except TypeError as exc:
        raise ConfigurationError(
            f"{klass.__qualname__}.__slots__ is not an iterable of names.",
            code="unsupported_instance_state",
        ) from exc
    if not all(isinstance(slot, str) for slot in slots):
        raise ConfigurationError(
            f"{klass.__qualname__}.__slots__ must contain only names.",
            code="unsupported_instance_state",
        )
    return slots


def deterministic_instance_state(value: object) -> dict[str, object]:
    """Collect ``__dict__`` plus every slot declared anywhere in the MRO.

    Reading only ``type(value).__slots__`` misses attributes backed by a slot
    descriptor on a base class: a subclass that declares no ``__slots__`` of
    its own still gets a ``__dict__``, yet those attributes live in the base
    slot, so ``vars()`` returns them nowhere. Walking ``__mro__`` closes that
    hole. ``__dict__`` and ``__weakref__`` are layout pseudo-slots, not
    configuration, and are ignored.
    """
    state: dict[str, object] = dict(vars(value)) if hasattr(value, "__dict__") else {}
    seen: set[str] = set()
    for klass in type(value).__mro__:
        for slot in _declared_slots(klass):
            if slot in _PSEUDO_SLOTS or slot in seen:
                continue
            seen.add(slot)
            if not hasattr(value, slot):
                continue
            slotted = getattr(value, slot)
            if slot in state and state[slot] is not slotted:
                raise ConfigurationError(
                    f"instance state for {slot!r} is ambiguous between "
                    "__dict__ and a slot descriptor.",
                    code="ambiguous_instance_state",
                )
            state[slot] = slotted
    return state


__all__ = ["deterministic_instance_state"]
