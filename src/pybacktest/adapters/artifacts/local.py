"""Atomic, bounded, symlink-safe local result artifact persistence."""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pybacktest.domain.errors import AdapterContractError
from pybacktest.domain.orders import Fill, Order
from pybacktest.domain.portfolio import PortfolioSnapshot
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

_PAYLOAD_ORDER = (
    "config.json",
    "summary.json",
    "equity.parquet",
    "positions.parquet",
    "orders.parquet",
    "fills.parquet",
    "events.parquet",
)
_TRUST_ORDER = ("manifest.json", "manifest.sha256")
_ALL_FILES = frozenset((*_PAYLOAD_ORDER, *_TRUST_ORDER))
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )


def _details(value: Mapping[str, object]) -> str:
    return canonical_json_text(value)


def _warning_document(warning: RunWarning) -> dict[str, object]:
    return {
        "code": warning.code.value,
        "message": warning.message,
        "metric": (
            warning.metric.value if warning.metric is not None else None
        ),
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
            "metrics": [
                _metric_document(metric)
                for metric in summary.results
            ],
            "warnings": [
                _warning_document(warning)
                for warning in summary.warnings
            ],
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
                direction = (
                    Decimal("1")
                    if position.quantity.value > Decimal("0")
                    else Decimal("-1")
                )
                market_value = direction * position.book_cost.amount
            else:
                market_value = (
                    position.quantity.value * valuation.amount
                )
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
                        str(valuation.amount)
                        if valuation is not None
                        else None
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
                str(order.limit_price.amount)
                if order.limit_price is not None
                else None
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
            "order_id": (
                str(event.order_id)
                if event.order_id is not None
                else None
            ),
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


def _write_file(path: str, data: bytes) -> ArtifactFile:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    file_stat = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != len(data):
        raise AdapterContractError(
            "artifact file changed while it was being finalized.",
            code="artifact_file_race",
        )
    return ArtifactFile(
        name=os.path.basename(path),
        size_bytes=file_stat.st_size,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _safe_root(root: str) -> str:
    planned = Path(root)
    anchor = planned.anchor
    current = Path(anchor)
    for part in planned.parts[1:]:
        current = current / part
        path = os.fspath(current)
        if os.path.lexists(path):
            current_stat = os.lstat(path)
            if stat.S_ISLNK(current_stat.st_mode):
                raise AdapterContractError(
                    "artifact root cannot contain symlinks.",
                    code="unsafe_artifact_root",
                )
            if not stat.S_ISDIR(current_stat.st_mode):
                raise AdapterContractError(
                    "artifact root must be a directory.",
                    code="unsafe_artifact_root",
                )
            continue
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            current_stat = os.lstat(path)
            if stat.S_ISLNK(current_stat.st_mode) or not stat.S_ISDIR(
                current_stat.st_mode
            ):
                raise AdapterContractError(
                    "artifact root creation raced with an unsafe path.",
                    code="unsafe_artifact_root",
                ) from None
    return root


def _fsync_directory(path: str) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_inode(path: str, identity: tuple[int, int]) -> bool:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == identity


def _cleanup_temp(
    path: str,
    *,
    identity: tuple[int, int],
) -> None:
    if not _same_inode(path, identity):
        return
    current = os.lstat(path)
    if not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode):
        return
    for entry in os.scandir(path):
        if entry.name not in _ALL_FILES:
            return
        entry_stat = entry.stat(follow_symlinks=False)
        if not (
            stat.S_ISREG(entry_stat.st_mode)
            or stat.S_ISLNK(entry_stat.st_mode)
        ):
            return
    for entry in os.scandir(path):
        os.unlink(entry.path)
    os.rmdir(path)


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
            raise ResultValidationError(
                "artifact root must be path-like."
            ) from error
        if (
            not isinstance(raw_root, str)
            or not raw_root
            or "\x00" in raw_root
        ):
            raise ResultValidationError(
                "artifact root must be a non-empty text path."
            )
        normalized = os.path.abspath(os.path.normpath(raw_root))
        if normalized == Path(normalized).anchor:
            raise ResultValidationError(
                "filesystem root is not a safe artifact root."
            )
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
        root = _safe_root(self._root)
        run_name = str(result.run_id)
        target = os.path.join(root, run_name)
        lock_path = os.path.join(root, f".{run_name}.lock")
        temp_path = os.path.join(root, f".{run_name}.tmp")
        if os.path.lexists(target):
            raise AdapterContractError(
                f"artifact for {run_name} already exists.",
                code="duplicate_artifact",
            )

        lock_descriptor: int | None = None
        lock_identity: tuple[int, int] | None = None
        temp_identity: tuple[int, int] | None = None
        renamed = False
        try:
            try:
                lock_descriptor = os.open(
                    lock_path,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | _O_NOFOLLOW,
                    0o600,
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
            if os.path.lexists(target):
                raise AdapterContractError(
                    f"artifact for {run_name} already exists.",
                    code="duplicate_artifact",
                )
            try:
                os.mkdir(temp_path, 0o700)
            except FileExistsError as error:
                raise AdapterContractError(
                    "run-specific artifact temporary path already exists.",
                    code="unsafe_temporary_artifact",
                ) from error
            temp_stat = os.lstat(temp_path)
            temp_identity = (temp_stat.st_dev, temp_stat.st_ino)

            written: list[ArtifactFile] = []
            total_bytes = 0

            def write_one(name: str, data: bytes) -> ArtifactFile:
                nonlocal total_bytes
                artifact_file = _write_file(
                    os.path.join(temp_path, name),
                    data,
                )
                written.append(artifact_file)
                total_bytes += artifact_file.size_bytes
                if (
                    self._max_bytes is not None
                    and total_bytes > self._max_bytes
                ):
                    raise AdapterContractError(
                        (
                            f"artifact exceeds max_bytes={self._max_bytes} "
                            f"after {name}."
                        ),
                        code="artifact_size_limit",
                    )
                return artifact_file

            write_one("config.json", _config_bytes(result))
            write_one("summary.json", _summary_bytes(result.summary))
            for name in _PAYLOAD_ORDER[2:]:
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
            _fsync_directory(temp_path)
            if os.path.lexists(target):
                raise AdapterContractError(
                    f"artifact for {run_name} already exists.",
                    code="duplicate_artifact",
                )
            os.rename(temp_path, target)
            renamed = True
            _fsync_directory(root)
            return ArtifactRef(
                path=target,
                manifest=artifact_manifest,
                manifest_checksum=manifest_file.sha256,
                files=(
                    *artifact_manifest.files,
                    manifest_file,
                    sidecar_file,
                ),
            )
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            if (
                not renamed
                and temp_identity is not None
                and os.path.lexists(temp_path)
            ):
                _cleanup_temp(temp_path, identity=temp_identity)
            if (
                lock_identity is not None
                and _same_inode(lock_path, lock_identity)
            ):
                os.unlink(lock_path)


__all__ = ["LocalArtifactStore"]
