"""One strict canonical serializer for replay hashing and JSON persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from math import isfinite

from pybacktest.domain.errors import PybacktestError
from pybacktest.domain.identifiers import CashEventId, FillId, OrderId, RunId

from .models import BacktestResult


class SerializationError(PybacktestError, ValueError):
    """Raised when a value cannot be represented without ambiguity."""


class _IdentifierOrdinals:
    def __init__(self, result: BacktestResult) -> None:
        self._run = {result.run_id.value: "run:0"}
        self._orders = {
            item.id.value: f"order:{index}"
            for index, item in enumerate(result.orders)
        }
        self._fills = {
            item.id.value: f"fill:{index}"
            for index, item in enumerate(result.fills)
        }
        cash_ids: dict[str, str] = {}
        for snapshot in result.snapshots:
            for event in snapshot.cash_events:
                if event.id.value not in cash_ids:
                    cash_ids[event.id.value] = f"cash_event:{len(cash_ids)}"
        self._cash_events = cash_ids
        self._strings = {
            **self._run,
            **self._orders,
            **self._fills,
            **self._cash_events,
        }

    def identifier(self, value: object) -> str | None:
        if isinstance(value, RunId):
            return self._run.get(value.value, "run:unbound")
        if isinstance(value, OrderId):
            return self._orders.get(value.value, "order:unbound")
        if isinstance(value, FillId):
            return self._fills.get(value.value, "fill:unbound")
        if isinstance(value, CashEventId):
            return self._cash_events.get(value.value, "cash_event:unbound")
        return None

    def string(self, value: str) -> str:
        return self._strings.get(value, value)


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _aware_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SerializationError("datetime values must be timezone-aware.")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )


def _canonical_data(
    value: object,
    *,
    ordinals: _IdentifierOrdinals | None,
    active: set[int],
) -> object:
    normalized_id = ordinals.identifier(value) if ordinals is not None else None
    if normalized_id is not None:
        return {
            "$identifier": {
                "type": type(value).__name__,
                "value": normalized_id,
            }
        }
    if isinstance(value, (RunId, OrderId, FillId, CashEventId)):
        return {
            "$identifier": {
                "type": type(value).__name__,
                "value": value.value,
            }
        }
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return {
            "$enum": {
                "type": _type_name(value),
                "value": _canonical_data(
                    value.value,
                    ordinals=ordinals,
                    active=active,
                ),
            }
        }
    if isinstance(value, str):
        return ordinals.string(value) if ordinals is not None else value
    if type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise SerializationError("float values must be finite.")
        return {"$float": value.hex()}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise SerializationError("Decimal values must be finite.")
        return {"$decimal": str(value)}
    if isinstance(value, datetime):
        return {"$datetime": _aware_utc(value)}

    track_identity = (
        is_dataclass(value)
        or isinstance(value, Mapping)
        or (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        )
    )
    identity = id(value)
    if track_identity:
        if identity in active:
            raise SerializationError("cyclic values are not serializable.")
        active.add(identity)
    try:
        if isinstance(value, Mapping):
            if all(isinstance(key, str) for key in value):
                return {
                    key: _canonical_data(
                        item,
                        ordinals=ordinals,
                        active=active,
                    )
                    for key, item in value.items()
                }
            items: list[tuple[str, object, object]] = []
            seen_keys: set[str] = set()
            for key, item in value.items():
                encoded_key = _canonical_data(
                    key,
                    ordinals=ordinals,
                    active=active,
                )
                key_bytes = json.dumps(
                    encoded_key,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                if key_bytes in seen_keys:
                    raise SerializationError(
                        "mapping keys collide after canonicalization."
                    )
                seen_keys.add(key_bytes)
                items.append(
                    (
                        key_bytes,
                        encoded_key,
                        _canonical_data(
                            item,
                            ordinals=ordinals,
                            active=active,
                        ),
                    )
                )
            items.sort(key=lambda item: item[0])
            return {
                "$mapping": [
                    [encoded_key, encoded_value]
                    for _, encoded_key, encoded_value in items
                ]
            }
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "$type": _type_name(value),
                "fields": {
                    item.name: _canonical_data(
                        getattr(value, item.name),
                        ordinals=ordinals,
                        active=active,
                    )
                    for item in fields(value)
                },
            }
        if isinstance(value, tuple):
            return [
                _canonical_data(
                    item,
                    ordinals=ordinals,
                    active=active,
                )
                for item in value
            ]
        if isinstance(value, list):
            return [
                _canonical_data(
                    item,
                    ordinals=ordinals,
                    active=active,
                )
                for item in value
            ]
    finally:
        if track_identity:
            active.remove(identity)
    raise SerializationError(f"unsupported canonical type: {_type_name(value)}")


def to_canonical_data(value: object) -> object:
    """Return a JSON-compatible type-aware representation or fail closed."""
    return _canonical_data(value, ordinals=None, active=set())


def canonical_json_bytes(value: object) -> bytes:
    """Encode compact canonical sorted UTF-8 JSON with no non-finite numbers."""
    try:
        return json.dumps(
            to_canonical_data(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        if isinstance(error, SerializationError):
            raise
        raise SerializationError("value could not be encoded canonically.") from error


def canonical_json_text(value: object) -> str:
    """Return the canonical UTF-8 representation as text."""
    return canonical_json_bytes(value).decode("utf-8")


def _behavioral_payload(result: BacktestResult) -> Mapping[str, object]:
    manifest = result.manifest
    return {
        "manifest": {
            "library_version": manifest.library_version,
            "schema_version": manifest.schema_version,
            "canonical_request": manifest.canonical_request,
            "strategy_identity": manifest.strategy_identity,
            "strategy_fingerprint": manifest.strategy_fingerprint,
            "spec_identity": manifest.spec_identity,
            "compiler_identity": manifest.compiler_identity,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "seed": manifest.seed,
            "adapter_versions": manifest.adapter_versions,
            "model_versions": manifest.model_versions,
        },
        "summary": result.summary,
        "market_timestamps": result.market_timestamps,
        "snapshots": result.snapshots,
        "orders": result.orders,
        "fills": result.fills,
        "events": result.events,
        "warnings": result.warnings,
    }


def calculate_replay_fingerprint(result: BacktestResult) -> str:
    """Hash all behavior after stable run-scoped identifier normalization."""
    if not isinstance(result, BacktestResult):
        raise SerializationError("result must be a BacktestResult.")
    ordinals = _IdentifierOrdinals(result)
    data = _canonical_data(
        _behavioral_payload(result),
        ordinals=ordinals,
        active=set(),
    )
    encoded = json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SerializationError",
    "calculate_replay_fingerprint",
    "canonical_json_bytes",
    "canonical_json_text",
    "to_canonical_data",
]
