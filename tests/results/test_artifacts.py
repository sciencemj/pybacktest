"""Atomic local artifact persistence with independently verified integrity."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pybacktest.adapters.artifacts import LocalArtifactStore
from pybacktest.domain.errors import AdapterContractError
from pybacktest.domain.identifiers import FillId, OrderId, RunId
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
from pybacktest.domain.portfolio import PortfolioSnapshot, Position
from pybacktest.ports.artifacts import ArtifactStore
from pybacktest.results.metrics import MetricsConfig, calculate_metrics
from pybacktest.results.models import (
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
            local.os,
            "rename",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("rename failed")
            ),
        )

    with pytest.raises((RuntimeError, OSError)):
        LocalArtifactStore(root).write(result)

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
    real_rename = local.os.rename

    def recording_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def recording_rename(source: object, target: object) -> None:
        events.append("rename")
        real_rename(source, target)

    monkeypatch.setattr(local.os, "fsync", recording_fsync)
    monkeypatch.setattr(local.os, "rename", recording_rename)

    LocalArtifactStore(tmp_path / "artifacts").write(_result())

    assert events.count("rename") == 1
    assert events[-2:] == ["rename", "fsync"]
    assert events[: events.index("rename")].count("fsync") >= 10
