"""Fail-closed deterministic hashing for immutable engine configuration."""

from __future__ import annotations

import hashlib
import re

from pybacktest._introspection import deterministic_instance_state
from pybacktest.domain.errors import ConfigurationError
from pybacktest.ports.components import (
    ComponentDescriptor,
    DeterministicComponent,
)
from pybacktest.results.serialization import canonical_json_bytes


def _fingerprint(value: object) -> str:
    """Hash canonical deterministic state or translate unsupported values."""
    try:
        encoded = canonical_json_bytes(value)
    except Exception as exc:
        raise ConfigurationError(
            "value contains unsupported deterministic state.",
            code="unsupported_deterministic_state",
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _component_fingerprint(*components: object) -> str:
    """Hash component descriptors without admitting mutable telemetry."""
    try:
        descriptors = tuple(
            _deterministic_component_state(component) for component in components
        )
        return _fingerprint(descriptors)
    except Exception as exc:
        if isinstance(exc, ConfigurationError) and exc.code in {
            "unsupported_component_state",
            "duplicate_slot_name",
            "ambiguous_instance_state",
            "unsupported_instance_state",
        }:
            raise
        raise ConfigurationError(
            "engine component contains unsupported deterministic state.",
            code="unsupported_component_state",
        ) from exc


def _deterministic_component_state(value: object) -> dict[str, object]:
    """Describe declared immutable configuration, never live telemetry."""
    identity = _type_identity(value)
    if isinstance(value, DeterministicComponent):
        descriptor = value.component_descriptor()
        if not isinstance(descriptor, ComponentDescriptor):
            raise ConfigurationError(
                "component_descriptor() must return a ComponentDescriptor.",
                code="unsupported_component_state",
            )
        return {
            "identity": descriptor.identity,
            "version": descriptor.version,
            "configuration": dict(descriptor.configuration),
        }
    if _is_frozen_dataclass(value):
        return {"identity": identity, "configuration": value}
    if not _instance_state(value):
        return {"identity": identity, "configuration": {}}
    raise ConfigurationError(
        f"engine component {identity} carries mutable runtime state; "
        "implement DeterministicComponent to declare its immutable "
        "configuration.",
        code="unsupported_component_state",
    )


def _is_frozen_dataclass(value: object) -> bool:
    parameters = getattr(type(value), "__dataclass_params__", None)
    return bool(getattr(parameters, "frozen", False))


def _instance_state(value: object) -> dict[str, object]:
    return deterministic_instance_state(value)


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _version_identity(value: object) -> str:
    """Return a manifest-safe type identity without local-scope markers."""
    identity = _type_identity(value).replace("<locals>", "locals")
    sanitized = re.sub(r"[^A-Za-z0-9._+-]", "_", identity)
    return sanitized if sanitized and sanitized[0].isalnum() else f"type.{sanitized}"
