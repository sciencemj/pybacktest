"""RunRecorder owns one chronological mutable stream and finalizes once."""

from datetime import UTC, datetime, timedelta

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
from pybacktest.engine.recorder import RecorderStateError, RunRecorder
from pybacktest.results.metrics import MetricsConfig
from pybacktest.results.models import (
    CausalStage,
    EngineEvent,
    EngineEventCode,
    ResultValidationError,
    RunManifest,
)

_BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
_AAPL = InstrumentId.parse("XNAS:AAPL")
_ORDER_ID = OrderId.parse("order_" + "2" * 32)


def _manifest() -> RunManifest:
    return RunManifest(
        run_id=RunId.parse("run_" + "1" * 32),
        library_version="0.2.0",
        schema_version="results.v1",
        canonical_request={"initial_cash": "1000"},
        strategy_identity="tests.Strategy",
        strategy_fingerprint="a" * 64,
        spec_identity="python",
        compiler_identity="none",
        dataset_fingerprint="b" * 64,
        seed=42,
        adapter_versions={"broker": "simulated.v1"},
        model_versions={"fill": "next_open.v1"},
        started_at=_BASE,
        ended_at=_BASE + timedelta(seconds=2),
    )


def _snapshot(timestamp: datetime) -> PortfolioSnapshot:
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


def _order(status: OrderStatus, filled: object) -> Order:
    return Order(
        id=_ORDER_ID,
        instrument=_AAPL,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Quantity.of("1"),
        quote_currency="USD",
        limit_price=None,
        time_in_force=TimeInForce.GOOD_TIL_CANCELLED,
        submitted_at=_BASE,
        active_from=_BASE + timedelta(seconds=1),
        reason=DecisionReason.of("test.buy"),
        status=status,
        filled_quantity=Quantity.of(filled),
    )


def _fill() -> Fill:
    return Fill(
        id=FillId.parse("fill_" + "3" * 32),
        order_id=_ORDER_ID,
        instrument=_AAPL,
        side=OrderSide.BUY,
        quantity=Quantity.of("1"),
        price=Money.usd("100"),
        fee=Money.usd("0"),
        timestamp=_BASE + timedelta(seconds=1),
    )


def _recorder() -> RunRecorder:
    return RunRecorder(
        manifest=_manifest(),
        metrics_config=MetricsConfig(
            risk_free_rate="0",
            annualization_periods=252,
        ),
    )


def test_recorder_assigns_one_sequence_and_preserves_final_order_state() -> None:
    recorder = _recorder()
    recorder.record_market_timestamp(_BASE)
    recorder.record_snapshot(_snapshot(_BASE))
    recorder.record_order(_order(OrderStatus.ACCEPTED, "0"))
    first = recorder.emit_event(
        timestamp=_BASE,
        stage=CausalStage.of("risk"),
        code=EngineEventCode.of("risk.approved"),
        order_id=_ORDER_ID,
        details={"approved": True},
    )
    recorder.record_market_timestamp(_BASE + timedelta(seconds=1))
    recorder.record_fill(_fill())
    recorder.record_order(_order(OrderStatus.FILLED, "1"))
    recorder.record_snapshot(_snapshot(_BASE + timedelta(seconds=1)))
    second = recorder.emit_event(
        timestamp=_BASE + timedelta(seconds=1),
        stage=CausalStage.of("fill"),
        code=EngineEventCode.of("broker.filled"),
        order_id=_ORDER_ID,
        details={"price": "100"},
    )

    result = recorder.finalize()

    assert (first.sequence, second.sequence) == (0, 1)
    assert result.events == (first, second)
    assert result.orders == (_order(OrderStatus.FILLED, "1"),)
    assert result.fills == (_fill(),)
    assert result.explain_trade(_ORDER_ID).entries == (first, second)


def test_recorder_accepts_only_the_next_explicit_event_sequence() -> None:
    recorder = _recorder()
    event = EngineEvent(
        timestamp=_BASE,
        sequence=1,
        stage=CausalStage.of("feature"),
        code=EngineEventCode.of("decision.feature"),
    )

    with pytest.raises(ResultValidationError):
        recorder.record_event(event)


@pytest.mark.parametrize(
    "operation",
    [
        "market_regression",
        "snapshot_without_market",
        "snapshot_wrong_timestamp",
        "order_without_market",
        "fill_before_order",
        "fill_after_snapshot",
    ],
)
def test_recorder_rejects_chronology_and_relationship_violations(
    operation: str,
) -> None:
    recorder = _recorder()
    if operation == "market_regression":
        recorder.record_market_timestamp(_BASE + timedelta(seconds=1))
        with pytest.raises(ResultValidationError):
            recorder.record_market_timestamp(_BASE)
    elif operation == "snapshot_without_market":
        with pytest.raises(ResultValidationError):
            recorder.record_snapshot(_snapshot(_BASE))
    elif operation == "snapshot_wrong_timestamp":
        recorder.record_market_timestamp(_BASE)
        with pytest.raises(ResultValidationError):
            recorder.record_snapshot(_snapshot(_BASE + timedelta(seconds=1)))
    elif operation == "order_without_market":
        with pytest.raises(ResultValidationError):
            recorder.record_order(_order(OrderStatus.ACCEPTED, "0"))
    elif operation == "fill_before_order":
        recorder.record_market_timestamp(_BASE + timedelta(seconds=1))
        with pytest.raises(ResultValidationError):
            recorder.record_fill(_fill())
    else:
        recorder.record_market_timestamp(_BASE)
        recorder.record_order(_order(OrderStatus.ACCEPTED, "0"))
        recorder.record_snapshot(_snapshot(_BASE))
        recorder.record_market_timestamp(_BASE + timedelta(seconds=1))
        recorder.record_snapshot(_snapshot(_BASE + timedelta(seconds=1)))
        with pytest.raises(ResultValidationError):
            recorder.record_fill(_fill())


def test_recorder_finalizes_once_and_rejects_all_later_mutation() -> None:
    recorder = _recorder()
    recorder.finalize()

    with pytest.raises(RecorderStateError):
        recorder.finalize()
    with pytest.raises(RecorderStateError):
        recorder.record_market_timestamp(_BASE)
    with pytest.raises(RecorderStateError):
        recorder.emit_event(
            timestamp=_BASE,
            stage=CausalStage.of("feature"),
            code=EngineEventCode.of("decision.feature"),
        )
