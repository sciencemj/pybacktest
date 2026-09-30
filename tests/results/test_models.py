"""Result-boundary validation, immutability, and replay normalization."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

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
from pybacktest.domain.portfolio import PortfolioSnapshot
from pybacktest.results.metrics import MetricsConfig, calculate_metrics
from pybacktest.results.models import (
    BacktestResult,
    CausalStage,
    EngineEvent,
    EngineEventCode,
    FrozenMapping,
    ResultValidationError,
    RunManifest,
)
from pybacktest.results.serialization import (
    SerializationError,
    calculate_replay_fingerprint,
    canonical_json_bytes,
)

_BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
_AAPL = InstrumentId.parse("XNAS:AAPL")


def test_task9_submodule_exports_are_curated_for_later_tasks() -> None:
    from pybacktest import engine as engine_api
    from pybacktest import ports as ports_api
    from pybacktest import results as results_api

    assert engine_api.RunRecorder.__name__ == "RunRecorder"
    assert ports_api.ArtifactStore.__name__ == "ArtifactStore"
    assert results_api.BacktestResult is BacktestResult
    assert results_api.RunManifest is RunManifest
    assert results_api.calculate_replay_fingerprint is calculate_replay_fingerprint


def _run_id(digit: str = "1") -> RunId:
    return RunId.parse("run_" + digit * 32)


def _order_id(digit: str = "2") -> OrderId:
    return OrderId.parse("order_" + digit * 32)


def _fill_id(digit: str = "3") -> FillId:
    return FillId.parse("fill_" + digit * 32)


def _manifest(
    *,
    run_id: RunId | None = None,
    started_at: datetime = _BASE,
    ended_at: datetime = _BASE + timedelta(seconds=2),
) -> RunManifest:
    resolved = run_id or _run_id()
    return RunManifest(
        run_id=resolved,
        library_version="0.2.0",
        schema_version="results.v1",
        canonical_request={
            "simulation": {
                "initial_cash": "1000",
                "run_id": str(resolved),
            }
        },
        strategy_identity="tests.BuyAndHold",
        strategy_fingerprint="a" * 64,
        spec_identity="python",
        compiler_identity="none",
        dataset_fingerprint="b" * 64,
        seed=42,
        adapter_versions={"broker": "simulated.v1", "data": "memory.v1"},
        model_versions={"fill": "next_open.v1"},
        started_at=started_at,
        ended_at=ended_at,
    )


def _snapshot(timestamp: datetime, equity: object = "1000") -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=timestamp,
        cash=Money.usd(equity),
        positions={},
        realized_pnl=Money.usd("0"),
        unrealized_pnl=Money.usd("0"),
        total_fees=Money.usd("0"),
        market_value=Money.usd("0"),
        gross_exposure=Money.usd("0"),
        equity=Money.usd(equity),
        valuation_prices={},
        cash_events=(),
    )


def _filled_order(
    order_id: OrderId,
    *,
    quantity: object = "1",
    reason_details: dict[str, object] | None = None,
) -> Order:
    return Order(
        id=order_id,
        instrument=_AAPL,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Quantity.of(quantity),
        quote_currency="USD",
        limit_price=None,
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        submitted_at=_BASE,
        active_from=_BASE + timedelta(seconds=1),
        reason=DecisionReason(
            code="test.buy",
            details=reason_details or {},
        ),
        status=OrderStatus.FILLED,
        filled_quantity=Quantity.of(quantity),
    )


def _fill(fill_id: FillId, order_id: OrderId, *, price: object = "100") -> Fill:
    return Fill(
        id=fill_id,
        order_id=order_id,
        instrument=_AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        price=Money.usd(price),
        fee=Money.usd("1"),
        timestamp=_BASE + timedelta(seconds=1),
    )


def _summary(equity: list[object]):
    return calculate_metrics(
        equity=equity,
        fills=(),
        snapshots=(),
        config=MetricsConfig(risk_free_rate="0", annualization_periods=252),
    )


def _result(
    *,
    manifest: RunManifest | None = None,
    order_id: OrderId | None = None,
    fill_id: FillId | None = None,
    price: object = "100",
    stage: str = "fill",
    code: str = "broker.filled",
    event_message: str | None = None,
    event_details: dict[str, object] | None = None,
    reason_details: dict[str, object] | None = None,
) -> BacktestResult:
    resolved_order = order_id or _order_id()
    resolved_fill = fill_id or _fill_id()
    timestamps = (_BASE, _BASE + timedelta(seconds=1))
    snapshots = (_snapshot(timestamps[0]), _snapshot(timestamps[1]))
    summary = _summary(["1000", "1000"])
    return BacktestResult(
        manifest=manifest or _manifest(),
        summary=summary,
        market_timestamps=timestamps,
        snapshots=snapshots,
        orders=(
            _filled_order(
                resolved_order,
                reason_details=reason_details,
            ),
        ),
        fills=(_fill(resolved_fill, resolved_order, price=price),),
        events=(
            EngineEvent(
                timestamp=timestamps[1],
                sequence=0,
                stage=CausalStage.of(stage),
                code=EngineEventCode.of(code),
                order_id=resolved_order,
                details=event_details
                or {"price": str(price), "nested": {"path": [1, 2]}},
                message=event_message,
            ),
        ),
        warnings=summary.warnings,
    )


def test_event_recursively_copies_and_freezes_nested_details() -> None:
    details = {"nested": {"path": [1, 2]}}
    event = EngineEvent(
        timestamp=_BASE,
        sequence=0,
        stage=CausalStage.of("feature"),
        code=EngineEventCode.of("decision.feature"),
        details=details,
    )
    details["nested"]["path"].append(3)  # type: ignore[index, union-attr]

    assert isinstance(event.details, FrozenMapping)
    nested = event.details["nested"]
    assert isinstance(nested, FrozenMapping)
    assert nested["path"] == (1, 2)
    with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
        event.details["new"] = "value"  # type: ignore[index]


def test_direct_frozen_mapping_construction_refreezes_nested_values() -> None:
    nested = {"path": [1, 2]}
    frozen = FrozenMapping((("nested", nested),))
    nested["path"].append(3)

    copied = frozen["nested"]
    assert isinstance(copied, FrozenMapping)
    assert copied["path"] == (1, 2)


@pytest.mark.parametrize(
    "items",
    [
        (("duplicate", 1), ("duplicate", 2)),
        ((1, "not-a-string-key"),),
        (("missing-value",),),
        ("not-a-pair",),
    ],
)
def test_direct_frozen_mapping_rejects_malformed_or_duplicate_items(
    items: object,
) -> None:
    with pytest.raises(ResultValidationError):
        FrozenMapping(items)  # type: ignore[arg-type]


def test_frozen_mapping_rejects_recursive_mutable_input() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(ResultValidationError):
        FrozenMapping((("recursive", recursive),))


def test_event_refreezes_an_already_created_frozen_mapping() -> None:
    nested = [1, 2]
    direct = FrozenMapping((("nested", nested),))
    event = EngineEvent(
        timestamp=_BASE,
        sequence=0,
        stage=CausalStage.of("feature"),
        code=EngineEventCode.of("decision.feature"),
        details=direct,
    )
    nested.append(3)

    assert event.details["nested"] == (1, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestamp": datetime(2024, 1, 1), "sequence": 0},
        {"timestamp": _BASE, "sequence": True},
        {"timestamp": _BASE, "sequence": -1},
        {"timestamp": _BASE, "sequence": 0, "details": {"bad": float("nan")}},
    ],
)
def test_event_rejects_naive_bool_negative_and_nonfinite_values(
    kwargs: dict[str, object],
) -> None:
    values = {
        "timestamp": _BASE,
        "sequence": 0,
        "stage": CausalStage.of("feature"),
        "code": EngineEventCode.of("decision.feature"),
        "details": {},
    }
    values.update(kwargs)

    with pytest.raises(ResultValidationError):
        EngineEvent(**values)  # type: ignore[arg-type]


def test_manifest_defensively_freezes_request_and_version_mappings() -> None:
    request = {"nested": {"items": [1, 2]}}
    adapters = {"broker": "simulated.v1"}
    manifest = replace(
        _manifest(),
        canonical_request=request,
        adapter_versions=adapters,
    )
    request["nested"]["items"].append(3)  # type: ignore[index, union-attr]
    adapters["broker"] = "mutated"

    nested = manifest.canonical_request["nested"]
    assert isinstance(nested, FrozenMapping)
    assert nested["items"] == (1, 2)
    assert manifest.adapter_versions["broker"] == "simulated.v1"


@pytest.mark.parametrize(
    "change",
    [
        {"seed": True},
        {"seed": -1},
        {"dataset_fingerprint": "not-a-hash"},
        {"strategy_fingerprint": "A" * 64},
        {"schema_version": "../v1"},
        {"started_at": datetime(2024, 1, 1)},
        {"ended_at": _BASE - timedelta(seconds=1)},
    ],
)
def test_manifest_rejects_invalid_seed_hash_version_and_time_metadata(
    change: dict[str, object],
) -> None:
    baseline = _manifest()

    with pytest.raises(ResultValidationError):
        replace(baseline, **change)


def test_backtest_result_copies_sequences_and_delegates_run_id_and_explain() -> None:
    baseline = _result()
    timestamps = list(baseline.market_timestamps)
    snapshots = list(baseline.snapshots)
    orders = list(baseline.orders)
    fills = list(baseline.fills)
    events = list(baseline.events)
    warnings = list(baseline.warnings)
    copied = BacktestResult(
        manifest=baseline.manifest,
        summary=baseline.summary,
        market_timestamps=timestamps,
        snapshots=snapshots,
        orders=orders,
        fills=fills,
        events=events,
        warnings=warnings,
    )
    timestamps.append(_BASE + timedelta(days=1))
    snapshots.clear()
    orders.clear()
    fills.clear()
    events.clear()
    warnings.clear()

    assert copied.run_id == baseline.manifest.run_id
    assert copied.market_timestamps == baseline.market_timestamps
    assert copied.snapshots == baseline.snapshots
    assert copied.orders == baseline.orders
    assert copied.fills == baseline.fills
    assert copied.events == baseline.events
    assert copied.warnings == baseline.warnings
    assert copied.explain_trade(baseline.orders[0].id).entries == baseline.events


def test_backtest_result_allows_a_valid_empty_no_trade_run() -> None:
    summary = _summary([])
    result = BacktestResult(
        manifest=_manifest(),
        summary=summary,
        market_timestamps=(),
        snapshots=(),
        orders=(),
        fills=(),
        events=(),
        warnings=summary.warnings,
    )

    assert result.orders == ()
    assert result.fills == ()


@pytest.mark.parametrize(
    "change",
    [
        {"market_timestamps": (_BASE + timedelta(seconds=1), _BASE)},
        {
            "snapshots": (
                _snapshot(_BASE),
                _snapshot(_BASE + timedelta(seconds=2)),
            )
        },
        {"orders": (_filled_order(_order_id()), _filled_order(_order_id()))},
        {
            "fills": (
                _fill(_fill_id(), _order_id()),
                _fill(_fill_id(), _order_id()),
            )
        },
        {
            "fills": (
                replace(
                    _fill(_fill_id(), _order_id()),
                    timestamp=_BASE + timedelta(seconds=5),
                ),
            )
        },
        {
            "events": (
                EngineEvent(
                    timestamp=_BASE + timedelta(seconds=1),
                    sequence=1,
                    stage=CausalStage.of("fill"),
                    code=EngineEventCode.of("broker.filled"),
                    order_id=_order_id(),
                ),
            )
        },
        {"warnings": ()},
    ],
)
def test_backtest_result_rejects_inconsistent_sequences_and_relationships(
    change: dict[str, object],
) -> None:
    baseline = _result()

    with pytest.raises(ResultValidationError):
        replace(baseline, **change)


def test_replay_fingerprint_normalizes_run_order_fill_ids_and_run_times() -> None:
    first = _result()
    second = _result(
        manifest=_manifest(
            run_id=_run_id("9"),
            started_at=_BASE.astimezone(timezone(timedelta(hours=9))),
            ended_at=(_BASE + timedelta(hours=3)).astimezone(
                timezone(timedelta(hours=-5))
            ),
        ),
        order_id=_order_id("8"),
        fill_id=_fill_id("7"),
    )

    assert first.replay_fingerprint() == second.replay_fingerprint()
    assert calculate_replay_fingerprint(first) == first.replay_fingerprint()
    assert first.replay_fingerprint() == first.replay_fingerprint()


def test_replay_fingerprint_does_not_collide_id_shaped_prose_with_ordinal() -> None:
    raw_order_id = _order_id("2")
    first = _result(
        order_id=raw_order_id,
        event_message=str(raw_order_id),
    )
    second = _result(
        manifest=_manifest(
            run_id=_run_id("9"),
            started_at=_BASE + timedelta(days=5),
            ended_at=_BASE + timedelta(days=5, seconds=2),
        ),
        order_id=_order_id("8"),
        fill_id=_fill_id("7"),
        event_message="order:0",
    )

    assert first.replay_fingerprint() != second.replay_fingerprint()


def test_replay_fingerprint_preserves_id_shaped_ordinary_notes() -> None:
    first_order = _order_id("2")
    second_order = _order_id("8")
    first = _result(
        order_id=first_order,
        event_details={"note": str(first_order)},
        reason_details={"note": str(first_order)},
    )
    second = _result(
        manifest=_manifest(
            run_id=_run_id("9"),
            started_at=_BASE + timedelta(days=5),
            ended_at=_BASE + timedelta(days=5, seconds=2),
        ),
        order_id=second_order,
        fill_id=_fill_id("7"),
        event_details={"note": str(second_order)},
        reason_details={"note": str(second_order)},
    )

    assert first.replay_fingerprint() != second.replay_fingerprint()


def test_replay_fingerprint_normalizes_only_exact_designated_id_fields() -> None:
    first_order = _order_id("2")
    second_order = _order_id("8")
    first = _result(
        order_id=first_order,
        event_details={
            "nested": {"order_id": str(first_order)},
            "OrderId": "literal",
        },
        reason_details={"fill_id": str(_fill_id("3"))},
    )
    second = _result(
        manifest=_manifest(
            run_id=_run_id("9"),
            started_at=_BASE + timedelta(days=5),
            ended_at=_BASE + timedelta(days=5, seconds=2),
        ),
        order_id=second_order,
        fill_id=_fill_id("7"),
        event_details={
            "nested": {"order_id": str(second_order)},
            "OrderId": "literal",
        },
        reason_details={"fill_id": str(_fill_id("7"))},
    )

    assert first.replay_fingerprint() == second.replay_fingerprint()


@pytest.mark.parametrize(
    "changed",
    [
        _result(price="101"),
        _result(code="broker.partial_fill"),
        _result(stage="risk"),
        _result(event_message="different recorded reason"),
    ],
)
def test_replay_fingerprint_changes_for_behavioral_differences(
    changed: BacktestResult,
) -> None:
    assert changed.replay_fingerprint() != _result().replay_fingerprint()


def test_canonical_serializer_normalizes_aware_time_and_decimal_exactly() -> None:
    utc = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    seoul = datetime(
        2024,
        1,
        2,
        9,
        0,
        tzinfo=timezone(timedelta(hours=9)),
    )

    assert canonical_json_bytes({"at": utc}) == canonical_json_bytes({"at": seoul})
    assert b'"$decimal":"1.2300"' in canonical_json_bytes({"amount": Decimal("1.2300")})


@pytest.mark.parametrize("unknown", [object(), float("nan"), datetime(2024, 1, 1)])
def test_canonical_serializer_rejects_unknown_nonfinite_and_naive_values(
    unknown: object,
) -> None:
    with pytest.raises(SerializationError):
        canonical_json_bytes({"value": unknown})
