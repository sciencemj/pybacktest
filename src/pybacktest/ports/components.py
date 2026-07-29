"""Typed, immutable identity for pluggable deterministic engine components."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pybacktest.domain.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """Explicit immutable configuration and version for one component.

    A component declares only the values that change its behaviour. Mutable
    runtime telemetry — call counters, caches, or accumulated statistics —
    must never appear here, because the descriptor is hashed into the run
    manifest and must be identical for two behaviourally identical runs.
    """

    identity: str
    version: str
    configuration: Mapping[str, str]

    def __post_init__(self) -> None:
        for field_name in ("identity", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"component descriptor {field_name} must be a "
                    "non-empty string.",
                    code="invalid_component_descriptor",
                )
        if not isinstance(self.configuration, Mapping):
            raise ConfigurationError(
                "component descriptor configuration must be a mapping.",
                code="invalid_component_descriptor",
            )
        copied = dict(self.configuration)
        if not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in copied.items()
        ):
            raise ConfigurationError(
                "component descriptor configuration must map strings "
                "to strings.",
                code="invalid_component_descriptor",
            )
        object.__setattr__(
            self,
            "configuration",
            MappingProxyType(copied),
        )


@runtime_checkable
class DeterministicComponent(Protocol):
    """Declare fingerprintable identity for a custom engine component.

    Wrappers, decorators, and factories that carry mutable runtime state
    implement this protocol so the engine fingerprints their declared
    configuration instead of their live attributes.
    """

    def component_descriptor(self) -> ComponentDescriptor:
        """Return this component's immutable configuration and version."""
        raise NotImplementedError


__all__ = ["ComponentDescriptor", "DeterministicComponent"]
