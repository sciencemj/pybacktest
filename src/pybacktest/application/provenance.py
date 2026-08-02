"""Public, immutable provenance descriptors for reproducible runs."""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass
from importlib import metadata

from pybacktest._introspection import deterministic_instance_state
from pybacktest.data.features import FeaturePlan
from pybacktest.domain.errors import ConfigurationError
from pybacktest.results.serialization import canonical_json_bytes

_HASH = re.compile(r"^[0-9a-f]{64}$")
_UNPACKAGED = "unpackaged"


def _identity(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(
            f"{field_name} must be a non-empty string.",
            code="invalid_provenance_descriptor",
        )
    return value


def _fingerprint_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ConfigurationError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest.",
            code="invalid_provenance_descriptor",
        )
    return value


@dataclass(frozen=True, slots=True)
class ProvenanceDescriptor:
    """A complete, immutable statement of what produced a run.

    Strategy compilers, ``StrategySpec`` front ends, and MCP servers supply
    this value through :class:`~pybacktest.application.requests.BacktestRequest`
    or ``BacktestEngine.create_session()`` when the engine cannot derive
    trustworthy provenance itself.
    """

    strategy_identity: str
    strategy_fingerprint: str
    spec_identity: str
    compiler_identity: str
    spec_fingerprint: str | None = None
    schema_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "strategy_identity",
            "spec_identity",
            "compiler_identity",
        ):
            object.__setattr__(
                self,
                field_name,
                _identity(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "strategy_fingerprint",
            _fingerprint_text(
                self.strategy_fingerprint,
                "strategy_fingerprint",
            ),
        )
        for field_name in ("spec_fingerprint", "schema_fingerprint"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _fingerprint_text(value, field_name),
                )

    def canonical_details(self) -> dict[str, str]:
        """Return the manifest-visible optional compiler fingerprints."""
        return {
            "spec_fingerprint": self.spec_fingerprint or "",
            "schema_fingerprint": self.schema_fingerprint or "",
        }


def external_action_provenance(
    feature_plan: FeaturePlan,
) -> ProvenanceDescriptor:
    """Describe a session driven by external actions instead of a strategy."""
    if not isinstance(feature_plan, FeaturePlan):
        raise ConfigurationError("feature_plan must be a FeaturePlan.")
    return ProvenanceDescriptor(
        strategy_identity="external.actions",
        strategy_fingerprint=_digest(
            {
                "identity": "external.actions",
                "feature_plan": feature_plan,
            }
        ),
        spec_identity="external.actions",
        compiler_identity="pybacktest.session.external.v1",
    )


def python_strategy_provenance(
    strategy: object,
    feature_plan: FeaturePlan,
) -> ProvenanceDescriptor:
    """Describe a Python strategy by identity, implementation, and state.

    The fingerprint covers the strategy class source and its distributing
    package, not only ``module.qualname`` and mutable instance state. When
    the implementation cannot be read the call fails closed; supply an
    explicit :class:`ProvenanceDescriptor` in that case.
    """
    if not isinstance(feature_plan, FeaturePlan):
        raise ConfigurationError("feature_plan must be a FeaturePlan.")
    strategy_type = type(strategy)
    identity = f"{strategy_type.__module__}.{strategy_type.__qualname__}"
    implementation = _implementation_digest(strategy_type)
    package = _package_identity(strategy_type)
    state = _instance_state(strategy)
    try:
        fingerprint = _digest(
            {
                "identity": identity,
                "implementation": implementation,
                "package": package,
                "configuration": state,
                "feature_plan": feature_plan,
            }
        )
    except Exception as exc:
        raise ConfigurationError(
            "strategy configuration contains unsupported deterministic state.",
            code="unsupported_strategy_state",
        ) from exc
    return ProvenanceDescriptor(
        strategy_identity=identity,
        strategy_fingerprint=fingerprint,
        spec_identity="python.strategy",
        compiler_identity="pybacktest.session.python.v1",
    )


def _is_behavioral(klass: type) -> bool:
    """Decide whether a base class carries the strategy's own behaviour.

    ``object`` and anything shipped by this library or the standard library
    is framework scaffolding: including it would make every fingerprint move
    when an unrelated internal changes, while excluding a user's own base
    class would let two different implementations collide.
    """
    if klass is object:
        return False
    module = getattr(klass, "__module__", "")
    root = module.partition(".")[0]
    if root in {"builtins", "abc", "typing", "dataclasses", "enum"}:
        return False
    return root != "pybacktest"


def _implementation_descriptor(
    strategy_type: type,
) -> tuple[dict[str, str], ...]:
    """Digest each behaviour-carrying class in the strategy's own MRO.

    Hashing only ``type(strategy)`` misses ``build_features``/``on_bar``
    implementations that live on a user base class, so two subclasses whose
    inherited behaviour differs would share one fingerprint. MRO order is
    deterministic, so the descriptor is stable across runs.
    """
    entries: list[dict[str, str]] = []
    for index, klass in enumerate(strategy_type.__mro__):
        # The strategy's own type always counts, whatever module it claims;
        # only inherited bases are filtered for framework scaffolding.
        if index > 0 and not _is_behavioral(klass):
            continue
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError) as exc:
            raise ConfigurationError(
                "strategy implementation source is unavailable for "
                f"{klass.__qualname__}; supply an explicit "
                "ProvenanceDescriptor instead.",
                code="untrusted_strategy_provenance",
            ) from exc
        entries.append(
            {
                "module": getattr(klass, "__module__", ""),
                "qualname": klass.__qualname__,
                "source": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            }
        )
    return tuple(entries)


def _implementation_digest(strategy_type: type) -> str:
    return _digest(_implementation_descriptor(strategy_type))


def _package_identity(strategy_type: type) -> dict[str, str]:
    root = strategy_type.__module__.partition(".")[0]
    try:
        version = metadata.version(root)
    except metadata.PackageNotFoundError:
        version = _UNPACKAGED
    return {"root_module": root, "version": version}


def _instance_state(strategy: object) -> object:
    return deterministic_instance_state(strategy)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "ProvenanceDescriptor",
    "external_action_provenance",
    "python_strategy_provenance",
]
