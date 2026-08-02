"""Atomic local artifact persistence with independently verified integrity."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pybacktest.adapters.artifacts import LocalArtifactStore
from pybacktest.domain.errors import AdapterContractError
from pybacktest.domain.identifiers import CashEventId, FillId, OrderId, RunId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import (
    DecisionReason,
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from pybacktest.domain.portfolio import (
    CashEvent,
    CashEventCode,
    PortfolioSnapshot,
    Position,
)
from pybacktest.ports.artifacts import ArtifactDurabilityError, ArtifactStore
from pybacktest.results.metrics import MetricsConfig, calculate_metrics
from pybacktest.results.models import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    BacktestResult,
    CausalStage,
    EngineEvent,
    EngineEventCode,
    ResultValidationError,
    RunManifest,
)

_FILES = {
    "manifest.json",
    "manifest.sha256",
    "config.json",
    "summary.json",
    "equity.parquet",
    "positions.parquet",
    "orders.parquet",
    "fills.parquet",
    "events.parquet",
}
_PAYLOAD_FILES = _FILES - {"manifest.json", "manifest.sha256"}
_BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
_AAPL = InstrumentId.parse("XNAS:AAPL")
_MSFT = InstrumentId.parse("XNAS:MSFT")


def _manifest(run_digit: str) -> RunManifest:
    run_id = RunId.parse("run_" + run_digit * 32)
    return RunManifest(
        run_id=run_id,
        library_version="0.2.0",
        schema_version="results.v1",
        canonical_request={
            "simulation": {
                "initial_cash": "1000",
                "run_id": str(run_id),
            }
        },
        strategy_identity="tests.ArtifactStrategy",
        strategy_fingerprint="a" * 64,
        spec_identity="python",
        compiler_identity="none",
        dataset_fingerprint="b" * 64,
        seed=42,
        adapter_versions={"broker": "simulated.v1"},
        model_versions={"fill": "next_open.v1"},
        started_at=_BASE + timedelta(days=int(run_digit)),
        ended_at=_BASE + timedelta(days=int(run_digit), seconds=2),
    )


def _flat_snapshot(timestamp: datetime) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=timestamp,
        cash=Money.usd("1000"),
        positions={},
        realized_pnl=Money.usd("0"),
        unrealized_pnl=Money.usd("0"),
        total_fees=Money.usd("0"),
        market_value=Money.usd("0"),
        gross_exposure=Money.usd("0"),
        equity=Money.usd("1000"),
        valuation_prices={},
        cash_events=(),
    )


def _position_snapshot(timestamp: datetime) -> PortfolioSnapshot:
    price = Money.usd("100")
    position = Position(
        instrument=_AAPL,
        quantity=Quantity.of("1"),
        average_price=price,
        book_cost=Money.usd("100"),
        realized_pnl=Money.usd("0"),
    )
    return PortfolioSnapshot(
        timestamp=timestamp,
        cash=Money.usd("900"),
        positions={_AAPL: position},
        realized_pnl=Money.usd("0"),
        unrealized_pnl=Money.usd("0"),
        total_fees=Money.usd("0"),
        market_value=Money.usd("100"),
        gross_exposure=Money.usd("100"),
        equity=Money.usd("1000"),
        valuation_prices={_AAPL: price},
        cash_events=(),
    )


def _result(run_digit: str = "1", *, empty: bool = False) -> BacktestResult:
    manifest = _manifest(run_digit)
    if empty:
        summary = calculate_metrics(
            equity=(),
            fills=(),
            snapshots=(),
            config=MetricsConfig(
                risk_free_rate="0",
                annualization_periods=252,
            ),
        )
        return BacktestResult(
            manifest=manifest,
            summary=summary,
            market_timestamps=(),
            snapshots=(),
            orders=(),
            fills=(),
            events=(),
            warnings=summary.warnings,
        )
    order_id = OrderId.parse("order_" + "2" * 32)
    fill_id = FillId.parse("fill_" + "3" * 32)
    order = Order(
        id=order_id,
        instrument=_AAPL,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Quantity.of("1"),
        quote_currency="USD",
        limit_price=None,
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        submitted_at=_BASE,
        active_from=_BASE + timedelta(seconds=1),
        reason=DecisionReason.of("artifact.buy", source="test"),
        status=OrderStatus.FILLED,
        filled_quantity=Quantity.of("1"),
    )
    fill = Fill(
        id=fill_id,
        order_id=order_id,
        instrument=_AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        price=Money.usd("100"),
        fee=Money.usd("0"),
        timestamp=_BASE + timedelta(seconds=1),
    )
    snapshots = (
        _flat_snapshot(_BASE),
        _position_snapshot(_BASE + timedelta(seconds=1)),
    )
    summary = calculate_metrics(
        equity=["1000", "1000"],
        fills=(fill,),
        snapshots=snapshots,
        config=MetricsConfig(
            risk_free_rate="0",
            annualization_periods=252,
        ),
    )
    event = EngineEvent(
        timestamp=_BASE + timedelta(seconds=1),
        sequence=0,
        stage=CausalStage.of("fill"),
        code=EngineEventCode.of("broker.filled"),
        order_id=order_id,
        details={"execution": {"price": "100", "path": ["next", "open"]}},
        message="filled at next bar open",
    )
    return BacktestResult(
        manifest=manifest,
        summary=summary,
        market_timestamps=(_BASE, _BASE + timedelta(seconds=1)),
        snapshots=snapshots,
        orders=(order,),
        fills=(fill,),
        events=(event,),
        warnings=summary.warnings,
    )


def _result_with_complete_snapshot_state() -> BacktestResult:
    baseline = _result()
    cash_event = CashEvent(
        id=CashEventId.parse("cash_event_" + "4" * 32),
        timestamp=_BASE,
        amount=Money.usd("10.25"),
        code=CashEventCode.DIVIDEND,
    )
    first = replace(
        baseline.snapshots[0],
        cash_events=(cash_event,),
    )
    second = replace(
        baseline.snapshots[1],
        valuation_prices={
            **baseline.snapshots[1].valuation_prices,
            _MSFT: Money.usd("250.125"),
        },
        cash_events=(cash_event,),
    )
    snapshots = (first, second)
    summary = calculate_metrics(
        equity=[snapshot.equity.amount for snapshot in snapshots],
        fills=baseline.fills,
        snapshots=snapshots,
        config=MetricsConfig(
            risk_free_rate="0",
            annualization_periods=252,
        ),
    )
    return replace(
        baseline,
        snapshots=snapshots,
        summary=summary,
        warnings=summary.warnings,
    )


def _file_view(path: Path) -> dict[str, tuple[int, str]]:
    return {
        item.name: (
            item.stat().st_size,
            hashlib.sha256(item.read_bytes()).hexdigest(),
        )
        for item in path.iterdir()
    }


def _assert_no_private_write_state(root: Path) -> None:
    if root.exists():
        assert not any(
            item.name.endswith(".tmp") or item.name.endswith(".lock")
            for item in root.iterdir()
        )


def test_constructor_performs_no_io_and_write_creates_missing_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing" / "artifacts"
    store = LocalArtifactStore(root)

    assert isinstance(store, ArtifactStore)
    assert not root.exists()

    ref = store.write(_result())

    assert root.is_dir()
    assert Path(ref.path).is_dir()


@pytest.mark.parametrize("bad", [True, 1.5, "10", -1])
def test_constructor_rejects_invalid_max_bytes_without_creating_root(
    tmp_path: Path,
    bad: object,
) -> None:
    root = tmp_path / "never-created"

    with pytest.raises(ResultValidationError):
        LocalArtifactStore(root, max_bytes=bad)  # type: ignore[arg-type]

    assert not root.exists()


def test_write_persists_exact_files_and_noncyclic_checksums(
    tmp_path: Path,
) -> None:
    result = _result()
    before = result.replay_fingerprint()

    ref = LocalArtifactStore(tmp_path / "artifacts").write(result)

    artifact_path = Path(ref.path)
    assert {item.name for item in artifact_path.iterdir()} == _FILES
    actual = _file_view(artifact_path)
    returned = {
        item.name: (item.size_bytes, item.sha256)
        for item in ref.files
    }
    assert returned == actual
    assert ref.manifest_checksum == actual["manifest.json"][1]
    assert (artifact_path / "manifest.sha256").read_text("ascii") == (
        ref.manifest_checksum + "\n"
    )
    embedded = {
        item.name: (item.size_bytes, item.sha256)
        for item in ref.manifest.files
    }
    assert set(embedded) == _PAYLOAD_FILES
    assert embedded == {
        name: actual[name]
        for name in _PAYLOAD_FILES
    }
    manifest_document = json.loads(
        (artifact_path / "manifest.json").read_text("utf-8")
    )
    assert manifest_document["artifact_id"] == ref.manifest.artifact_id
    assert {
        item["name"]
        for item in manifest_document["files"]
    } == _PAYLOAD_FILES
    assert "manifest.json" not in manifest_document["files"]
    assert "manifest.sha256" not in manifest_document["files"]
    assert before == result.replay_fingerprint()


def test_parquet_schemas_and_content_are_explicit_and_round_trip(
    tmp_path: Path,
) -> None:
    ref = LocalArtifactStore(tmp_path / "artifacts").write(_result())
    artifact_path = Path(ref.path)

    expected_columns = {
        "equity.parquet": [
            "snapshot_sequence",
            "timestamp",
            "cash",
            "realized_pnl",
            "unrealized_pnl",
            "total_fees",
            "market_value",
            "gross_exposure",
            "equity",
            "currency",
            "cash_events",
            "valuation_prices",
        ],
        "positions.parquet": [
            "snapshot_sequence",
            "timestamp",
            "instrument",
            "quantity",
            "average_price",
            "book_cost",
            "realized_pnl",
            "valuation_price",
            "market_value",
            "currency",
        ],
        "orders.parquet": [
            "order_sequence",
            "order_id",
            "instrument",
            "side",
            "order_type",
            "quantity",
            "quote_currency",
            "limit_price",
            "time_in_force",
            "submitted_at",
            "active_from",
            "reason_code",
            "reason_details",
            "status",
            "filled_quantity",
        ],
        "fills.parquet": [
            "fill_sequence",
            "fill_id",
            "order_id",
            "instrument",
            "side",
            "quantity",
            "price",
            "fee",
            "currency",
            "timestamp",
        ],
        "events.parquet": [
            "sequence",
            "timestamp",
            "stage",
            "code",
            "order_id",
            "details",
            "message",
        ],
    }
    for name, columns in expected_columns.items():
        table = pq.read_table(artifact_path / name)
        assert table.column_names == columns

    equity = pq.read_table(artifact_path / "equity.parquet").to_pylist()
    positions = pq.read_table(
        artifact_path / "positions.parquet"
    ).to_pylist()
    events = pq.read_table(artifact_path / "events.parquet").to_pylist()
    assert [row["equity"] for row in equity] == ["1000", "1000"]
    assert positions == [
        {
            "snapshot_sequence": 1,
            "timestamp": "2024-01-02T14:30:01.000000Z",
            "instrument": "XNAS:AAPL",
            "quantity": "1",
            "average_price": "100",
            "book_cost": "100",
            "realized_pnl": "0",
            "valuation_price": "100",
            "market_value": "100",
            "currency": "USD",
        }
    ]
    assert json.loads(events[0]["details"]) == {
        "execution": {
            "path": ["next", "open"],
            "price": "100",
        }
    }


def test_snapshot_cash_events_and_all_valuation_prices_round_trip(
    tmp_path: Path,
) -> None:
    result = _result_with_complete_snapshot_state()
    ref = LocalArtifactStore(tmp_path / "artifacts").write(result)

    equity = pq.read_table(
        Path(ref.path) / "equity.parquet"
    ).to_pylist()
    expected_cash_events = [
        {
            "cash_event_id": "cash_event_" + "4" * 32,
            "timestamp": "2024-01-02T14:30:00.000000Z",
            "amount": "10.25",
            "currency": "USD",
            "code": "dividend",
        }
    ]
    assert json.loads(equity[0]["cash_events"]) == expected_cash_events
    assert json.loads(equity[1]["cash_events"]) == expected_cash_events
    assert json.loads(equity[0]["valuation_prices"]) == []
    assert json.loads(equity[1]["valuation_prices"]) == [
        {
            "instrument": "XNAS:AAPL",
            "amount": "100",
            "currency": "USD",
        },
        {
            "instrument": "XNAS:MSFT",
            "amount": "250.125",
            "currency": "USD",
        },
    ]


def test_empty_results_keep_all_stable_parquet_schemas(tmp_path: Path) -> None:
    ref = LocalArtifactStore(tmp_path / "artifacts").write(
        _result("4", empty=True)
    )

    for name in (
        "equity.parquet",
        "positions.parquet",
        "orders.parquet",
        "fills.parquet",
        "events.parquet",
    ):
        table = pq.read_table(Path(ref.path) / name)
        assert table.num_rows == 0
        assert table.num_columns > 0


def test_duplicate_target_is_rejected_without_modifying_existing_artifact(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    result = _result()
    first = store.write(result)
    before = _file_view(Path(first.path))

    with pytest.raises(AdapterContractError) as captured:
        store.write(result)

    assert captured.value.code == "duplicate_artifact"
    assert _file_view(Path(first.path)) == before
    _assert_no_private_write_state(tmp_path / "artifacts")


def test_concurrent_duplicate_writers_never_overwrite_each_other(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    result = _result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                lambda _: _write_outcome(store, result),
                range(2),
            )
        )

    assert sorted(outcomes) == ["duplicate_artifact", "success"]
    target = tmp_path / "artifacts" / str(result.run_id)
    assert {item.name for item in target.iterdir()} == _FILES
    _assert_no_private_write_state(tmp_path / "artifacts")


def _write_outcome(
    store: LocalArtifactStore,
    result: BacktestResult,
) -> str:
    try:
        store.write(result)
    except AdapterContractError as error:
        return error.code
    return "success"


def test_max_bytes_accepts_exact_boundary_and_rejects_one_byte_less(
    tmp_path: Path,
) -> None:
    baseline = LocalArtifactStore(tmp_path / "baseline").write(_result("1"))
    total = sum(item.size_bytes for item in baseline.files)

    exact = LocalArtifactStore(
        tmp_path / "exact",
        max_bytes=total,
    ).write(_result("2"))
    assert sum(item.size_bytes for item in exact.files) == total

    limited_root = tmp_path / "limited"
    with pytest.raises(AdapterContractError) as captured:
        LocalArtifactStore(
            limited_root,
            max_bytes=total - 1,
        ).write(_result("3"))
    assert captured.value.code == "artifact_size_limit"
    assert not (limited_root / str(_result("3").run_id)).exists()
    _assert_no_private_write_state(limited_root)


def test_zero_max_bytes_rejects_first_nonempty_file(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"

    with pytest.raises(AdapterContractError) as captured:
        LocalArtifactStore(root, max_bytes=0).write(_result())

    assert captured.value.code == "artifact_size_limit"
    _assert_no_private_write_state(root)


def test_missing_pyarrow_extra_has_clear_error_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    real_import = local.importlib.import_module

    def missing(name: str):
        if name.startswith("pyarrow"):
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(local.importlib, "import_module", missing)
    root = tmp_path / "artifacts"

    with pytest.raises(AdapterContractError) as captured:
        LocalArtifactStore(root).write(_result())

    assert captured.value.code == "parquet_extra_unavailable"
    _assert_no_private_write_state(root)


@pytest.mark.parametrize("failure", ["serialize", "write", "rename"])
def test_injected_failures_leave_no_target_temp_or_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    root = tmp_path / "artifacts"
    result = _result()
    if failure == "serialize":
        monkeypatch.setattr(
            local,
            "_serialize_parquet",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("serialization failed")
            ),
        )
    elif failure == "write":
        real_write = local._write_file
        calls = 0

        def fail_write(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("write failed")
            return real_write(*args, **kwargs)

        monkeypatch.setattr(local, "_write_file", fail_write)
    else:
        monkeypatch.setattr(
            local,
            "_rename_noreplace",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("rename failed")
            ),
        )

    expected_error = (
        RuntimeError if failure == "serialize" else AdapterContractError
    )
    with pytest.raises(expected_error) as captured:
        LocalArtifactStore(root).write(result)

    if failure != "serialize":
        assert captured.value.code == "artifact_io_error"
    assert not (root / str(result.run_id)).exists()
    _assert_no_private_write_state(root)


def test_root_file_and_root_symlink_are_rejected_at_write(
    tmp_path: Path,
) -> None:
    root_file = tmp_path / "root-file"
    root_file.write_text("not a directory", encoding="utf-8")
    store_for_file = LocalArtifactStore(root_file)
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_directory, target_is_directory=True)
    store_for_link = LocalArtifactStore(root_link)

    with pytest.raises(AdapterContractError):
        store_for_file.write(_result("1"))
    with pytest.raises(AdapterContractError):
        store_for_link.write(_result("2"))

    assert list(real_directory.iterdir()) == []


@pytest.mark.parametrize("broken", [False, True])
def test_existing_target_symlink_or_broken_symlink_is_rejected(
    tmp_path: Path,
    broken: bool,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    result = _result()
    target = root / str(result.run_id)
    destination = tmp_path / "missing" if broken else tmp_path / "outside"
    if not broken:
        destination.mkdir()
    target.symlink_to(destination, target_is_directory=True)

    with pytest.raises(AdapterContractError) as captured:
        LocalArtifactStore(root).write(result)

    assert captured.value.code == "duplicate_artifact"
    assert target.is_symlink()
    _assert_no_private_write_state(root)


def test_unsafe_filesystem_root_is_rejected_lexically() -> None:
    with pytest.raises(ResultValidationError):
        LocalArtifactStore(Path("/"))


def test_store_is_reentrant_and_artifact_id_depends_on_behavior(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    first = store.write(_result("1"))
    second = store.write(_result("2"))

    assert Path(first.path).name != Path(second.path).name
    assert first.manifest.artifact_id == second.manifest.artifact_id
    assert first.manifest.replay_fingerprint == (
        second.manifest.replay_fingerprint
    )


def test_fsync_temp_rename_and_root_fsync_order_is_observable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    events: list[str] = []
    real_fsync = local.os.fsync
    real_publish = local._rename_noreplace

    def recording_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def recording_publish(
        root_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        events.append("publish")
        real_publish(root_fd, source_name, target_name)

    monkeypatch.setattr(local.os, "fsync", recording_fsync)
    monkeypatch.setattr(local, "_rename_noreplace", recording_publish)

    LocalArtifactStore(tmp_path / "artifacts").write(_result())

    assert events.count("publish") == 1
    publish_index = events.index("publish")
    assert events[publish_index + 1 :] == ["fsync", "fsync"]
    assert events[:publish_index].count("fsync") >= 10


@pytest.mark.parametrize("swap", ["root", "ancestor"])
def test_root_chain_swap_after_pinning_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    ancestor = tmp_path / "owned"
    root = ancestor / "artifacts"
    root.mkdir(parents=True)
    outside_ancestor = tmp_path / "outside"
    outside_root = outside_ancestor / "artifacts"
    outside_root.mkdir(parents=True)
    displaced = tmp_path / "displaced"
    real_safe_root = local._safe_root

    def swap_after_pinning(path: str):
        pinned = real_safe_root(path)
        if swap == "root":
            root.rename(displaced)
            root.symlink_to(outside_root, target_is_directory=True)
        else:
            ancestor.rename(displaced)
            ancestor.symlink_to(outside_ancestor, target_is_directory=True)
        return pinned

    monkeypatch.setattr(local, "_safe_root", swap_after_pinning)

    with pytest.raises(AdapterContractError) as captured:
        LocalArtifactStore(root).write(_result())

    assert captured.value.code == "unsafe_artifact_root"
    assert list(outside_root.iterdir()) == []


@pytest.mark.parametrize("nonempty", [False, True])
def test_target_created_at_publication_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nonempty: bool,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    root = tmp_path / "artifacts"
    result = _result()
    sentinel = b"owned-by-racer"
    if hasattr(local, "_rename_noreplace"):
        real_publish = local._rename_noreplace

        def race_publish(
            root_fd: int,
            source_name: str,
            target_name: str,
        ) -> None:
            os.mkdir(target_name, 0o700, dir_fd=root_fd)
            if nonempty:
                target_descriptor = os.open(
                    target_name,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    dir_fd=root_fd,
                )
                try:
                    descriptor = os.open(
                        "sentinel",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=target_descriptor,
                    )
                    try:
                        os.write(descriptor, sentinel)
                    finally:
                        os.close(descriptor)
                finally:
                    os.close(target_descriptor)
            real_publish(root_fd, source_name, target_name)

        monkeypatch.setattr(local, "_rename_noreplace", race_publish)
    else:
        real_publish = local.os.rename

        def race_publish(source: str, target: str) -> None:
            target_path = Path(target)
            target_path.mkdir()
            if nonempty:
                (target_path / "sentinel").write_bytes(sentinel)
            real_publish(source, target)

        monkeypatch.setattr(local.os, "rename", race_publish)

    with pytest.raises(AdapterContractError) as captured:
        LocalArtifactStore(root).write(result)

    assert captured.value.code == "duplicate_artifact"
    target = root / str(result.run_id)
    assert target.is_dir()
    if nonempty:
        assert (target / "sentinel").read_bytes() == sentinel
    else:
        assert list(target.iterdir()) == []
    _assert_no_private_write_state(root)


def test_operations_after_root_pinning_are_descriptor_relative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    after_pinning = False
    real_safe_root = local._safe_root
    real_open = local.os.open

    def record_pinning(path: str):
        nonlocal after_pinning
        pinned = real_safe_root(path)
        after_pinning = True
        return pinned

    def reject_absolute_reopen(path: object, *args, **kwargs):
        if after_pinning and os.path.isabs(os.fspath(path)):
            raise AssertionError("artifact path reopened after root pinning")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(local, "_safe_root", record_pinning)
    monkeypatch.setattr(local.os, "open", reject_absolute_reopen)

    ref = LocalArtifactStore(tmp_path / "artifacts").write(_result())

    assert Path(ref.path).is_dir()


@pytest.mark.parametrize(
    "failure",
    ["root_fsync", "root_identity", "lock_removal"],
)
def test_post_publish_failures_return_committed_ref_without_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    root = tmp_path / "artifacts"
    result = _result()
    if failure == "root_fsync":
        real_fsync = local._fsync_directory
        calls = 0

        def fail_root_fsync(directory: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("root fsync failed")
            real_fsync(directory)

        monkeypatch.setattr(local, "_fsync_directory", fail_root_fsync)
    elif failure == "root_identity":
        assert hasattr(local, "_verify_root_identity")
        real_verify = local._verify_root_identity
        calls = 0

        def fail_post_publish_verify(pinned: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("root identity check failed")
            real_verify(pinned)

        monkeypatch.setattr(
            local,
            "_verify_root_identity",
            fail_post_publish_verify,
        )
    else:
        assert hasattr(local, "_remove_lock")

        def fail_lock_removal(*args, **kwargs) -> None:
            raise OSError("lock removal failed")

        monkeypatch.setattr(local, "_remove_lock", fail_lock_removal)

    with pytest.raises(ArtifactDurabilityError) as captured:
        LocalArtifactStore(root).write(result)

    error = captured.value
    assert error.code == "artifact_published_durability_uncertain"
    assert error.committed is True
    assert error.location_lost is False
    assert error.intended_path == str(root / str(result.run_id))
    assert error.artifact_ref == ArtifactRef(
        path=error.intended_path,
        manifest=error.manifest,
        manifest_checksum=error.manifest_checksum,
        files=error.files,
    )
    assert Path(error.artifact_ref.path).is_dir()
    assert "published" in str(error)
    assert "durability" in str(error)


@pytest.mark.parametrize("substitution", ["root", "target"])
def test_post_publish_namespace_loss_never_claims_a_valid_artifact_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    root = tmp_path / "artifacts"
    root.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    displaced_root = tmp_path / "displaced-root"
    displaced_target_name = ".committed-target"
    result = _result()
    intended_path = root / str(result.run_id)
    real_publish = local._rename_noreplace

    def publish_then_substitute(
        root_fd: int,
        source_name: str,
        target_name: str,
    ) -> None:
        real_publish(root_fd, source_name, target_name)
        if substitution == "root":
            root.rename(displaced_root)
            root.symlink_to(outside_root, target_is_directory=True)
        else:
            os.rename(
                target_name,
                displaced_target_name,
                src_dir_fd=root_fd,
                dst_dir_fd=root_fd,
            )
            os.mkdir(target_name, 0o700, dir_fd=root_fd)

    monkeypatch.setattr(local, "_rename_noreplace", publish_then_substitute)

    with pytest.raises(ArtifactDurabilityError) as captured:
        LocalArtifactStore(root).write(result)

    error = captured.value
    assert error.code == "artifact_published_durability_uncertain"
    assert error.committed is True
    assert error.location_lost is True
    assert error.artifact_ref is None
    assert error.intended_path == str(intended_path)
    assert error.manifest.run_manifest == result.manifest
    assert error.manifest_checksum == next(
        item.sha256 for item in error.files if item.name == "manifest.json"
    )
    assert tuple(item.name for item in error.files) == (
        "config.json",
        "summary.json",
        "equity.parquet",
        "positions.parquet",
        "orders.parquet",
        "fills.parquet",
        "events.parquet",
        "manifest.json",
        "manifest.sha256",
    )
    assert "intended path" in str(error)
    assert "no longer identifies" in str(error)
    if substitution == "root":
        assert not intended_path.exists()
        assert (displaced_root / str(result.run_id)).is_dir()
        assert list(outside_root.iterdir()) == []
    else:
        assert intended_path.is_dir()
        assert list(intended_path.iterdir()) == []
        assert (root / displaced_target_name).is_dir()


def test_write_oserror_is_wrapped_as_typed_adapter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pybacktest.adapters.artifacts.local as local

    monkeypatch.setattr(
        local,
        "_write_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("write failed")
        ),
    )

    with pytest.raises(AdapterContractError) as captured:
        LocalArtifactStore(tmp_path / "artifacts").write(_result())

    assert type(captured.value) is AdapterContractError
    assert captured.value.code == "artifact_io_error"


def _artifact_file(name: str, marker: int) -> ArtifactFile:
    return ArtifactFile(
        name=name,
        size_bytes=marker,
        sha256=f"{marker:064x}",
    )


def _payload_files() -> tuple[ArtifactFile, ...]:
    return tuple(
        _artifact_file(name, index + 1)
        for index, name in enumerate(
            (
                "config.json",
                "summary.json",
                "equity.parquet",
                "positions.parquet",
                "orders.parquet",
                "fills.parquet",
                "events.parquet",
            )
        )
    )


@pytest.mark.parametrize(
    "files",
    [
        (),
        _payload_files()[:-1],
        (*_payload_files(), _artifact_file("extra.json", 20)),
        (
            _payload_files()[1],
            _payload_files()[0],
            *_payload_files()[2:],
        ),
    ],
)
def test_artifact_manifest_requires_exact_ordered_payload_graph(
    files: tuple[ArtifactFile, ...],
) -> None:
    with pytest.raises(ResultValidationError):
        ArtifactManifest(
            run_manifest=_manifest("1"),
            artifact_schema_version="artifact.v1",
            artifact_id="artifact_" + "a" * 64,
            replay_fingerprint="a" * 64,
            files=files,
        )


@pytest.mark.parametrize("change", ["reordered", "mismatched"])
def test_artifact_ref_requires_manifest_payload_entries_by_identity_and_order(
    change: str,
) -> None:
    payload = _payload_files()
    manifest = ArtifactManifest(
        run_manifest=_manifest("1"),
        artifact_schema_version="artifact.v1",
        artifact_id="artifact_" + "a" * 64,
        replay_fingerprint="a" * 64,
        files=payload,
    )
    manifest_file = _artifact_file("manifest.json", 30)
    sidecar_file = _artifact_file("manifest.sha256", 31)
    if change == "reordered":
        files = (
            payload[1],
            payload[0],
            *payload[2:],
            manifest_file,
            sidecar_file,
        )
    else:
        files = (
            _artifact_file(payload[0].name, 40),
            *payload[1:],
            manifest_file,
            sidecar_file,
        )

    with pytest.raises(ResultValidationError):
        ArtifactRef(
            path="/safe/run",
            manifest=manifest,
            manifest_checksum=manifest_file.sha256,
            files=files,
        )
