"""Atomic, bounded, symlink-safe local result artifact persistence."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pybacktest.domain.errors import AdapterContractError
from pybacktest.domain.orders import Fill, Order
from pybacktest.domain.portfolio import PortfolioSnapshot
from pybacktest.ports.artifacts import ArtifactDurabilityError
from pybacktest.results._decimal import ExactDecimalError, exact_multiply
from pybacktest.results.artifact_schema import (
    ARTIFACT_ALL_FILES,
    ARTIFACT_PAYLOAD_FILES,
)
from pybacktest.results.models import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    BacktestResult,
    EngineEvent,
    MetricResult,
    ResultValidationError,
    RunManifest,
    RunWarning,
    SummaryMetrics,
)
from pybacktest.results.serialization import (
    calculate_replay_fingerprint,
    canonical_json_bytes,
    canonical_json_text,
)

_ALL_FILES = frozenset(ARTIFACT_ALL_FILES)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _utc_text(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def _details(value: Mapping[str, object]) -> str:
    return canonical_json_text(value)


def _warning_document(warning: RunWarning) -> dict[str, object]:
    return {
        "code": warning.code.value,
        "message": warning.message,
        "metric": (warning.metric.value if warning.metric is not None else None),
        "details": warning.details,
    }


def _metric_document(metric: MetricResult) -> dict[str, object]:
    return {
        "name": metric.name.value,
        "value": str(metric.value) if metric.value is not None else None,
        "metadata": {
            "formula_id": metric.metadata.formula_id,
            "annualization_periods": metric.metadata.annualization_periods,
            "risk_free_rate": str(metric.metadata.risk_free_rate),
            "missing_policy": metric.metadata.missing_policy.value,
            "parameters": metric.metadata.parameters,
        },
    }


def _manifest_document(manifest: RunManifest) -> dict[str, object]:
    return {
        "run_id": str(manifest.run_id),
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
        "started_at": _utc_text(manifest.started_at),
        "ended_at": _utc_text(manifest.ended_at),
    }


def _config_bytes(result: BacktestResult) -> bytes:
    return canonical_json_bytes(
        {
            "run": _manifest_document(result.manifest),
            "request": result.manifest.canonical_request,
        }
    )


def _summary_bytes(summary: SummaryMetrics) -> bytes:
    return canonical_json_bytes(
        {
            "metrics": [_metric_document(metric) for metric in summary.results],
            "warnings": [_warning_document(warning) for warning in summary.warnings],
        }
    )


def _equity_rows(
    snapshots: Sequence[PortfolioSnapshot],
) -> list[dict[str, object]]:
    return [
        {
            "snapshot_sequence": sequence,
            "timestamp": _utc_text(snapshot.timestamp),
            "cash": str(snapshot.cash.amount),
            "realized_pnl": str(snapshot.realized_pnl.amount),
            "unrealized_pnl": str(snapshot.unrealized_pnl.amount),
            "total_fees": str(snapshot.total_fees.amount),
            "market_value": str(snapshot.market_value.amount),
            "gross_exposure": str(snapshot.gross_exposure.amount),
            "equity": str(snapshot.equity.amount),
            "currency": snapshot.equity.currency,
            "cash_events": canonical_json_text(
                [
                    {
                        "cash_event_id": str(event.id),
                        "timestamp": _utc_text(event.timestamp),
                        "amount": str(event.amount.amount),
                        "currency": event.amount.currency,
                        "code": event.code.value,
                    }
                    for event in snapshot.cash_events
                ]
            ),
            "valuation_prices": canonical_json_text(
                [
                    {
                        "instrument": str(instrument_id),
                        "amount": str(snapshot.valuation_prices[instrument_id].amount),
                        "currency": (snapshot.valuation_prices[instrument_id].currency),
                    }
                    for instrument_id in sorted(
                        snapshot.valuation_prices,
                        key=str,
                    )
                ]
            ),
        }
        for sequence, snapshot in enumerate(snapshots)
        if snapshot.timestamp is not None
    ]


def _position_rows(
    snapshots: Sequence[PortfolioSnapshot],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence, snapshot in enumerate(snapshots):
        if snapshot.timestamp is None:
            continue
        for instrument_id in sorted(snapshot.positions, key=str):
            position = snapshot.positions[instrument_id]
            valuation = snapshot.valuation_prices.get(instrument_id)
            if valuation is None:
                market_value = (
                    position.book_cost.amount
                    if position.quantity.value > Decimal("0")
                    else position.book_cost.amount.copy_negate()
                )
            else:
                try:
                    market_value = exact_multiply(
                        position.quantity.value,
                        valuation.amount,
                    )
                except ExactDecimalError as error:
                    raise AdapterContractError(
                        "position valuation exceeds the supported numeric range.",
                        code="artifact_numeric_range",
                    ) from error
            rows.append(
                {
                    "snapshot_sequence": sequence,
                    "timestamp": _utc_text(snapshot.timestamp),
                    "instrument": str(instrument_id),
                    "quantity": str(position.quantity.value),
                    "average_price": (
                        str(position.average_price.amount)
                        if position.average_price is not None
                        else None
                    ),
                    "book_cost": str(position.book_cost.amount),
                    "realized_pnl": str(position.realized_pnl.amount),
                    "valuation_price": (
                        str(valuation.amount) if valuation is not None else None
                    ),
                    "market_value": str(market_value),
                    "currency": snapshot.equity.currency,
                }
            )
    return rows


def _order_rows(orders: Sequence[Order]) -> list[dict[str, object]]:
    return [
        {
            "order_sequence": sequence,
            "order_id": str(order.id),
            "instrument": str(order.instrument),
            "side": order.side.value,
            "order_type": order.type.value,
            "quantity": str(order.quantity.value),
            "quote_currency": order.quote_currency,
            "limit_price": (
                str(order.limit_price.amount) if order.limit_price is not None else None
            ),
            "time_in_force": order.time_in_force.value,
            "submitted_at": _utc_text(order.submitted_at),
            "active_from": _utc_text(order.active_from),
            "reason_code": order.reason.code,
            "reason_details": _details(order.reason.details),
            "status": order.status.value,
            "filled_quantity": str(order.filled_quantity.value),
        }
        for sequence, order in enumerate(orders)
    ]


def _fill_rows(fills: Sequence[Fill]) -> list[dict[str, object]]:
    return [
        {
            "fill_sequence": sequence,
            "fill_id": str(fill.id),
            "order_id": str(fill.order_id),
            "instrument": str(fill.instrument),
            "side": fill.side.value,
            "quantity": str(fill.quantity.value),
            "price": str(fill.price.amount),
            "fee": str(fill.fee.amount),
            "currency": fill.price.currency,
            "timestamp": _utc_text(fill.timestamp),
        }
        for sequence, fill in enumerate(fills)
    ]


def _event_rows(
    events: Sequence[EngineEvent],
) -> list[dict[str, object]]:
    return [
        {
            "sequence": event.sequence,
            "timestamp": _utc_text(event.timestamp),
            "stage": event.stage.value,
            "code": event.code.value,
            "order_id": (str(event.order_id) if event.order_id is not None else None),
            "details": _details(event.details),
            "message": event.message,
        }
        for event in events
    ]


def _string_field(pa: Any, name: str, *, nullable: bool = False) -> Any:
    return pa.field(name, pa.string(), nullable=nullable)


def _schema(pa: Any, name: str) -> Any:
    integer = lambda field_name: pa.field(  # noqa: E731
        field_name,
        pa.int64(),
        nullable=False,
    )
    schemas = {
        "equity.parquet": pa.schema(
            [
                integer("snapshot_sequence"),
                _string_field(pa, "timestamp"),
                _string_field(pa, "cash"),
                _string_field(pa, "realized_pnl"),
                _string_field(pa, "unrealized_pnl"),
                _string_field(pa, "total_fees"),
                _string_field(pa, "market_value"),
                _string_field(pa, "gross_exposure"),
                _string_field(pa, "equity"),
                _string_field(pa, "currency"),
                _string_field(pa, "cash_events"),
                _string_field(pa, "valuation_prices"),
            ]
        ),
        "positions.parquet": pa.schema(
            [
                integer("snapshot_sequence"),
                _string_field(pa, "timestamp"),
                _string_field(pa, "instrument"),
                _string_field(pa, "quantity"),
                _string_field(pa, "average_price", nullable=True),
                _string_field(pa, "book_cost"),
                _string_field(pa, "realized_pnl"),
                _string_field(pa, "valuation_price", nullable=True),
                _string_field(pa, "market_value"),
                _string_field(pa, "currency"),
            ]
        ),
        "orders.parquet": pa.schema(
            [
                integer("order_sequence"),
                _string_field(pa, "order_id"),
                _string_field(pa, "instrument"),
                _string_field(pa, "side"),
                _string_field(pa, "order_type"),
                _string_field(pa, "quantity"),
                _string_field(pa, "quote_currency"),
                _string_field(pa, "limit_price", nullable=True),
                _string_field(pa, "time_in_force"),
                _string_field(pa, "submitted_at"),
                _string_field(pa, "active_from"),
                _string_field(pa, "reason_code"),
                _string_field(pa, "reason_details"),
                _string_field(pa, "status"),
                _string_field(pa, "filled_quantity"),
            ]
        ),
        "fills.parquet": pa.schema(
            [
                integer("fill_sequence"),
                _string_field(pa, "fill_id"),
                _string_field(pa, "order_id"),
                _string_field(pa, "instrument"),
                _string_field(pa, "side"),
                _string_field(pa, "quantity"),
                _string_field(pa, "price"),
                _string_field(pa, "fee"),
                _string_field(pa, "currency"),
                _string_field(pa, "timestamp"),
            ]
        ),
        "events.parquet": pa.schema(
            [
                integer("sequence"),
                _string_field(pa, "timestamp"),
                _string_field(pa, "stage"),
                _string_field(pa, "code"),
                _string_field(pa, "order_id", nullable=True),
                _string_field(pa, "details"),
                _string_field(pa, "message", nullable=True),
            ]
        ),
    }
    return schemas[name]


def _serialize_parquet(name: str, result: BacktestResult) -> bytes:
    try:
        pa = importlib.import_module("pyarrow")
        parquet = importlib.import_module("pyarrow.parquet")
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterContractError(
            "Parquet artifact writing requires the 'parquet' extra.",
            code="parquet_extra_unavailable",
        ) from error
    row_builders: dict[
        str,
        Callable[[BacktestResult], list[dict[str, object]]],
    ] = {
        "equity.parquet": lambda item: _equity_rows(item.snapshots),
        "positions.parquet": lambda item: _position_rows(item.snapshots),
        "orders.parquet": lambda item: _order_rows(item.orders),
        "fills.parquet": lambda item: _fill_rows(item.fills),
        "events.parquet": lambda item: _event_rows(item.events),
    }
    table = pa.Table.from_pylist(
        row_builders[name](result),
        schema=_schema(pa, name),
    )
    sink = pa.BufferOutputStream()
    parquet.write_table(
        table,
        sink,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
    )
    return sink.getvalue().to_pybytes()


def _artifact_manifest_bytes(
    manifest: ArtifactManifest,
) -> bytes:
    return canonical_json_bytes(
        {
            "artifact_schema_version": manifest.artifact_schema_version,
            "artifact_id": manifest.artifact_id,
            "replay_fingerprint": manifest.replay_fingerprint,
            "run_manifest": _manifest_document(manifest.run_manifest),
            "files": [
                {
                    "name": item.name,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in manifest.files
            ],
        }
    )


def _write_file(
    directory_fd: int,
    name: str,
    data: bytes,
) -> ArtifactFile:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        file_stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != len(data):
        raise AdapterContractError(
            "artifact file changed while it was being finalized.",
            code="artifact_file_race",
        )
    return ArtifactFile(
        name=name,
        size_bytes=file_stat.st_size,
        sha256=hashlib.sha256(data).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class _PinnedEntry:
    parent_fd: int
    name: str
    identity: tuple[int, int]


@dataclass(slots=True)
class _PinnedRoot:
    path: str
    descriptors: tuple[int, ...]
    entries: tuple[_PinnedEntry, ...]

    @property
    def fd(self) -> int:
        return self.descriptors[-1]

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            with suppress(OSError):
                os.close(descriptor)
        self.descriptors = ()


def _unsafe_root(message: str, error: OSError | None = None) -> None:
    contract_error = AdapterContractError(
        message,
        code="unsafe_artifact_root",
    )
    if error is None:
        raise contract_error
    raise contract_error from error


def _open_directory(name: str, *, dir_fd: int | None = None) -> int:
    flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
    try:
        if dir_fd is None:
            return os.open(name, flags)
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as error:
        if error.errno in {
            errno.ELOOP,
            errno.ENOTDIR,
        }:
            _unsafe_root(
                "artifact root must contain only real directories.",
                error,
            )
        raise


def _safe_root(root: str) -> _PinnedRoot:
    planned = Path(root)
    anchor = planned.anchor
    descriptors: list[int] = []
    entries: list[_PinnedEntry] = []
    try:
        current_fd = _open_directory(anchor)
        descriptors.append(current_fd)
        for part in planned.parts[1:]:
            try:
                child_fd = _open_directory(part, dir_fd=current_fd)
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                child_fd = _open_directory(part, dir_fd=current_fd)
            child_stat = os.fstat(child_fd)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child_fd)
                _unsafe_root("artifact root must be a directory.")
            entries.append(
                _PinnedEntry(
                    parent_fd=current_fd,
                    name=part,
                    identity=(child_stat.st_dev, child_stat.st_ino),
                )
            )
            descriptors.append(child_fd)
            current_fd = child_fd
    except BaseException:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise
    return _PinnedRoot(
        path=root,
        descriptors=tuple(descriptors),
        entries=tuple(entries),
    )


def _verify_root_identity(root: _PinnedRoot) -> None:
    for entry in root.entries:
        try:
            current = os.stat(
                entry.name,
                dir_fd=entry.parent_fd,
                follow_symlinks=False,
            )
        except OSError as error:
            _unsafe_root(
                "artifact root identity changed during publication.",
                error,
            )
        if (
            not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != entry.identity
        ):
            _unsafe_root("artifact root identity changed during publication.")


def _fsync_directory(descriptor: int) -> None:
    os.fsync(descriptor)


def _same_inode_at(
    directory_fd: int,
    name: str,
    identity: tuple[int, int],
) -> bool:
    try:
        current = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == identity


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _verify_committed_location(
    root: _PinnedRoot,
    target_name: str,
    target_identity: tuple[int, int],
) -> None:
    _verify_root_identity(root)
    if not _same_inode_at(root.fd, target_name, target_identity):
        raise AdapterContractError(
            "committed artifact no longer has its intended target identity.",
            code="committed_artifact_location_lost",
        )


def _committed_error(
    root: _PinnedRoot,
    target_name: str,
    target_identity: tuple[int, int],
    artifact_ref: ArtifactRef,
) -> ArtifactDurabilityError:
    try:
        _verify_committed_location(
            root,
            target_name,
            target_identity,
        )
    except (AdapterContractError, OSError):
        location_lost = True
        message = (
            f"artifact publication for {target_name} committed, but the "
            "intended path no longer identifies the committed artifact."
        )
    else:
        location_lost = False
        message = (
            f"artifact for {target_name} was published at its intended path, "
            "but final durability could not be confirmed."
        )
    return ArtifactDurabilityError(
        message,
        committed_artifact=artifact_ref,
        location_lost=location_lost,
    )


def _cleanup_temp(
    root_fd: int,
    temp_fd: int,
    temp_name: str,
    *,
    identity: tuple[int, int],
) -> None:
    names = os.listdir(temp_fd)
    for name in names:
        if name not in _ALL_FILES:
            return
        entry_stat = os.stat(
            name,
            dir_fd=temp_fd,
            follow_symlinks=False,
        )
        if not (stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode)):
            return
    for name in names:
        os.unlink(name, dir_fd=temp_fd)
    if _same_inode_at(root_fd, temp_name, identity):
        os.rmdir(temp_name, dir_fd=root_fd)


def _remove_lock(
    root_fd: int,
    lock_name: str,
    identity: tuple[int, int],
) -> None:
    if _same_inode_at(root_fd, lock_name, identity):
        os.unlink(lock_name, dir_fd=root_fd)


_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEATX_NP: Any | None = getattr(_LIBC, "renameatx_np", None)
_RENAMEAT2: Any | None = getattr(_LIBC, "renameat2", None)
if _RENAMEATX_NP is not None:
    _RENAMEATX_NP.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    _RENAMEATX_NP.restype = ctypes.c_int
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    _RENAMEAT2.restype = ctypes.c_int


def _rename_noreplace(
    root_fd: int,
    source_name: str,
    target_name: str,
) -> None:
    source = os.fsencode(source_name)
    target = os.fsencode(target_name)
    if sys.platform == "darwin" and _RENAMEATX_NP is not None:
        result = _RENAMEATX_NP(
            root_fd,
            source,
            root_fd,
            target,
            0x00000004,
        )
    elif sys.platform.startswith("linux") and _RENAMEAT2 is not None:
        result = _RENAMEAT2(
            root_fd,
            source,
            root_fd,
            target,
            0x00000001,
        )
    else:
        raise AdapterContractError(
            "atomic no-replace publication is unavailable on this platform.",
            code="atomic_noreplace_unavailable",
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise AdapterContractError(
            f"artifact target {target_name} already exists.",
            code="duplicate_artifact",
        )
    raise OSError(error_number, os.strerror(error_number))


class LocalArtifactStore:
    """Reentrant atomic local adapter with no constructor filesystem I/O."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        max_bytes: int | None = None,
    ) -> None:
        try:
            raw_root = os.fspath(root)
        except TypeError as error:
            raise ResultValidationError("artifact root must be path-like.") from error
        if not isinstance(raw_root, str) or not raw_root or "\x00" in raw_root:
            raise ResultValidationError("artifact root must be a non-empty text path.")
        normalized = os.path.abspath(os.path.normpath(raw_root))
        if normalized == Path(normalized).anchor:
            raise ResultValidationError("filesystem root is not a safe artifact root.")
        if max_bytes is not None and (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            raise ResultValidationError(
                "max_bytes must be a nonnegative integer or None."
            )
        self._root = normalized
        self._max_bytes = max_bytes

    def write(self, result: BacktestResult) -> ArtifactRef:
        """Write payloads, finalize trust files, and publish by atomic rename."""
        if not isinstance(result, BacktestResult):
            raise ResultValidationError("result must be a BacktestResult.")
        run_name = str(result.run_id)
        target_path = os.path.join(self._root, run_name)
        lock_name = f".{run_name}.lock"
        temp_name = f".{run_name}.tmp"
        root: _PinnedRoot | None = None
        lock_descriptor: int | None = None
        lock_identity: tuple[int, int] | None = None
        temp_descriptor: int | None = None
        temp_identity: tuple[int, int] | None = None
        committed = False
        artifact_ref: ArtifactRef | None = None
        try:
            root = _safe_root(self._root)
            _verify_root_identity(root)
            if _entry_exists(root.fd, run_name):
                raise AdapterContractError(
                    f"artifact for {run_name} already exists.",
                    code="duplicate_artifact",
                )
            try:
                lock_descriptor = os.open(
                    lock_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
                    0o600,
                    dir_fd=root.fd,
                )
            except FileExistsError as error:
                raise AdapterContractError(
                    f"artifact for {run_name} is already being written.",
                    code="duplicate_artifact",
                ) from error
            lock_stat = os.fstat(lock_descriptor)
            lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
            os.write(lock_descriptor, run_name.encode("ascii"))
            os.fsync(lock_descriptor)
            os.close(lock_descriptor)
            lock_descriptor = None
            if _entry_exists(root.fd, run_name):
                raise AdapterContractError(
                    f"artifact for {run_name} already exists.",
                    code="duplicate_artifact",
                )
            try:
                os.mkdir(temp_name, 0o700, dir_fd=root.fd)
            except FileExistsError as error:
                raise AdapterContractError(
                    "run-specific artifact temporary path already exists.",
                    code="unsafe_temporary_artifact",
                ) from error
            temp_descriptor = _open_directory(
                temp_name,
                dir_fd=root.fd,
            )
            temp_stat = os.fstat(temp_descriptor)
            temp_identity = (temp_stat.st_dev, temp_stat.st_ino)
            if not _same_inode_at(root.fd, temp_name, temp_identity):
                raise AdapterContractError(
                    "artifact temporary directory changed after creation.",
                    code="unsafe_temporary_artifact",
                )

            written: list[ArtifactFile] = []
            total_bytes = 0

            def write_one(name: str, data: bytes) -> ArtifactFile:
                nonlocal total_bytes
                artifact_file = _write_file(
                    temp_descriptor,
                    name,
                    data,
                )
                written.append(artifact_file)
                total_bytes += artifact_file.size_bytes
                if self._max_bytes is not None and total_bytes > self._max_bytes:
                    raise AdapterContractError(
                        (f"artifact exceeds max_bytes={self._max_bytes} after {name}."),
                        code="artifact_size_limit",
                    )
                return artifact_file

            write_one("config.json", _config_bytes(result))
            write_one("summary.json", _summary_bytes(result.summary))
            for name in ARTIFACT_PAYLOAD_FILES[2:]:
                write_one(name, _serialize_parquet(name, result))

            replay_fingerprint = calculate_replay_fingerprint(result)
            artifact_manifest = ArtifactManifest(
                run_manifest=result.manifest,
                artifact_schema_version="artifact.v1",
                artifact_id=f"artifact_{replay_fingerprint}",
                replay_fingerprint=replay_fingerprint,
                files=tuple(written),
            )
            manifest_file = write_one(
                "manifest.json",
                _artifact_manifest_bytes(artifact_manifest),
            )
            sidecar_file = write_one(
                "manifest.sha256",
                f"{manifest_file.sha256}\n".encode("ascii"),
            )
            artifact_ref = ArtifactRef(
                path=target_path,
                manifest=artifact_manifest,
                manifest_checksum=manifest_file.sha256,
                files=(
                    *artifact_manifest.files,
                    manifest_file,
                    sidecar_file,
                ),
            )
            _fsync_directory(temp_descriptor)
            if not _same_inode_at(root.fd, temp_name, temp_identity):
                raise AdapterContractError(
                    "artifact temporary directory changed before publication.",
                    code="unsafe_temporary_artifact",
                )
            _verify_root_identity(root)
            _rename_noreplace(root.fd, temp_name, run_name)
            committed = True
            try:
                _fsync_directory(root.fd)
                _verify_committed_location(
                    root,
                    run_name,
                    temp_identity,
                )
                _remove_lock(root.fd, lock_name, lock_identity)
                lock_identity = None
                _fsync_directory(root.fd)
                _verify_committed_location(
                    root,
                    run_name,
                    temp_identity,
                )
            except Exception as error:
                if lock_identity is not None:
                    with suppress(OSError):
                        _remove_lock(root.fd, lock_name, lock_identity)
                raise _committed_error(
                    root,
                    run_name,
                    temp_identity,
                    artifact_ref,
                ) from error
            return artifact_ref
        except OSError as error:
            if committed and artifact_ref is not None:
                if root is None or temp_identity is None:
                    raise
                raise _committed_error(
                    root,
                    run_name,
                    temp_identity,
                    artifact_ref,
                ) from error
            raise AdapterContractError(
                f"artifact I/O failed for {run_name}.",
                code="artifact_io_error",
            ) from error
        finally:
            if lock_descriptor is not None:
                with suppress(OSError):
                    os.close(lock_descriptor)
            if (
                not committed
                and root is not None
                and temp_descriptor is not None
                and temp_identity is not None
            ):
                with suppress(OSError):
                    _cleanup_temp(
                        root.fd,
                        temp_descriptor,
                        temp_name,
                        identity=temp_identity,
                    )
            if temp_descriptor is not None:
                with suppress(OSError):
                    os.close(temp_descriptor)
            if root is not None and lock_identity is not None:
                with suppress(OSError):
                    _remove_lock(root.fd, lock_name, lock_identity)
            if root is not None:
                root.close()


__all__ = ["LocalArtifactStore"]
