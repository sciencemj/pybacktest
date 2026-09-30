"""Decimal arithmetic is library-owned, bounded, and ambient-context neutral."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import (
    MAX_EMAX,
    Clamped,
    Context,
    Decimal,
    Inexact,
    Subnormal,
    Underflow,
    localcontext,
)
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from pybacktest.adapters.artifacts import LocalArtifactStore
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
from pybacktest.engine.recorder import RunRecorder
from pybacktest.results.metrics import MetricsConfig, calculate_metrics
from pybacktest.results.models import (
    BacktestResult,
    MetricName,
    ResultValidationError,
    RunManifest,
    SummaryMetrics,
)

_BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
_AAPL = InstrumentId.parse("XNAS:AAPL")
_ORDER_ID = OrderId.parse("order_" + "2" * 32)
_CONFIG = MetricsConfig(risk_free_rate="0", annualization_periods=252)


def _manifest() -> RunManifest:
    return RunManifest(
        run_id=RunId.parse("run_" + "1" * 32),
        library_version="0.2.0",
        schema_version="results.v1",
        canonical_request={"initial_cash": "1000"},
        strategy_identity="tests.NumericStrategy",
        strategy_fingerprint="a" * 64,
        spec_identity="python",
        compiler_identity="none",
        dataset_fingerprint="b" * 64,
        seed=42,
        adapter_versions={"broker": "simulated.v1"},
        model_versions={"fill": "next_open.v1"},
        started_at=_BASE,
        ended_at=_BASE + timedelta(seconds=1),
    )


def _flat_snapshot(equity: object = "1000") -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=_BASE,
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


def _fill(
    sequence: int,
    *,
    quantity: object,
    price: object = "100",
    side: OrderSide = OrderSide.BUY,
) -> Fill:
    return Fill(
        id=FillId.parse(f"fill_{sequence:032x}"),
        order_id=OrderId.parse(f"order_{sequence:032x}"),
        instrument=_AAPL,
        side=side,
        quantity=Quantity.of(quantity),
        price=Money.usd(price),
        fee=Money.usd("0"),
        timestamp=_BASE,
    )


def _order(
    quantity: object,
    *,
    status: OrderStatus,
    filled: object,
    order_id: OrderId = _ORDER_ID,
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
        active_from=_BASE,
        reason=DecisionReason.of("numeric.test"),
        status=status,
        filled_quantity=Quantity.of(filled),
    )


def _summary(equity: tuple[object, ...] = ("1000",)):
    return calculate_metrics(
        equity=equity,
        fills=(),
        snapshots=(),
        config=_CONFIG,
    )


def _result_with_order_fills(
    order: Order,
    fills: tuple[Fill, ...],
) -> BacktestResult:
    summary = _summary()
    return BacktestResult(
        manifest=_manifest(),
        summary=summary,
        market_timestamps=(_BASE,),
        snapshots=(_flat_snapshot(),),
        orders=(order,),
        fills=fills,
        events=(),
        warnings=summary.warnings,
    )


def _hostile_context(precision: int) -> Context:
    context = Context(prec=precision)
    for signal in (Inexact, Underflow, Subnormal, Clamped):
        context.traps[signal] = True
        context.flags[signal] = True
    return context


def _has_warning(
    metrics: SummaryMetrics,
    metric: MetricName,
    code: str,
) -> bool:
    return any(
        warning.metric is metric and warning.code.value == code
        for warning in metrics.warnings
    )


def test_backtest_result_fill_aggregation_ignores_hostile_ambient_context() -> None:
    large = Decimal("1e50")
    total = Decimal("1" + "0" * 49 + "1")
    order = _order(total, status=OrderStatus.FILLED, filled=total)
    fills = (
        replace(_fill(1, quantity=large), order_id=_ORDER_ID),
        replace(_fill(2, quantity="1"), order_id=_ORDER_ID),
    )

    with localcontext(_hostile_context(1)):
        result = _result_with_order_fills(order, fills)

    assert result.orders[0].filled_quantity.value == total


def test_recorder_fill_aggregation_ignores_hostile_ambient_context() -> None:
    large = Decimal("1e50")
    total = Decimal("1" + "0" * 49 + "1")
    recorder = RunRecorder(manifest=_manifest(), metrics_config=_CONFIG)
    recorder.record_market_timestamp(_BASE)
    recorder.record_order(_order(total, status=OrderStatus.ACCEPTED, filled="0"))
    first = _fill(1, quantity=large)
    second = _fill(2, quantity="1")
    first = Fill(
        id=first.id,
        order_id=_ORDER_ID,
        instrument=first.instrument,
        side=first.side,
        quantity=first.quantity,
        price=first.price,
        fee=first.fee,
        timestamp=first.timestamp,
    )
    second = Fill(
        id=second.id,
        order_id=_ORDER_ID,
        instrument=second.instrument,
        side=second.side,
        quantity=second.quantity,
        price=second.price,
        fee=second.fee,
        timestamp=second.timestamp,
    )
    recorder.record_fill(first)

    with localcontext(_hostile_context(1)):
        recorder.record_fill(second)

    assert recorder.events_for(_ORDER_ID) == ()


def test_result_rejects_beyond_supported_decimal_range_with_typed_error() -> None:
    quantity = Decimal(f"9e{MAX_EMAX}")
    order = _order(quantity, status=OrderStatus.FILLED, filled=quantity)
    fills = tuple(
        Fill(
            id=FillId.parse(f"fill_{sequence:032x}"),
            order_id=_ORDER_ID,
            instrument=_AAPL,
            side=OrderSide.BUY,
            quantity=Quantity.of(quantity),
            price=Money.usd("1"),
            fee=Money.usd("0"),
            timestamp=_BASE,
        )
        for sequence in (1, 2)
    )

    with pytest.raises(ResultValidationError):
        _result_with_order_fills(order, fills)


def test_artifact_position_value_is_exact_under_hostile_ambient_context(
    tmp_path: Path,
) -> None:
    quantity = Decimal("12345678901234567890")
    price = Decimal("98765432109876543210")
    with localcontext(Context(prec=100)):
        market_value = quantity * price
        cash = Decimal("1000") - market_value
    position = Position(
        instrument=_AAPL,
        quantity=Quantity.of(quantity),
        average_price=Money.usd(price),
        book_cost=Money.usd(market_value),
        realized_pnl=Money.usd("0"),
    )
    snapshot = PortfolioSnapshot(
        timestamp=_BASE,
        cash=Money.usd(cash),
        positions={_AAPL: position},
        realized_pnl=Money.usd("0"),
        unrealized_pnl=Money.usd("0"),
        total_fees=Money.usd("0"),
        market_value=Money.usd(market_value),
        gross_exposure=Money.usd(market_value),
        equity=Money.usd("1000"),
        valuation_prices={_AAPL: Money.usd(price)},
        cash_events=(),
    )
    summary = _summary(("1000",))
    result = BacktestResult(
        manifest=_manifest(),
        summary=summary,
        market_timestamps=(_BASE,),
        snapshots=(snapshot,),
        orders=(),
        fills=(),
        events=(),
        warnings=summary.warnings,
    )

    with localcontext(_hostile_context(1)):
        ref = LocalArtifactStore(tmp_path / "artifacts").write(result)

    rows = pq.read_table(Path(ref.path) / "positions.parquet").to_pylist()
    assert rows[0]["market_value"] == str(market_value)


def test_turnover_preserves_exact_fill_notional_before_division() -> None:
    operand = Decimal("1234567890" * 4)
    with localcontext(Context(prec=200)):
        expected = operand * operand
    fill = _fill(1, quantity=operand, price=operand)

    metrics = calculate_metrics(
        equity=["1", "1"],
        fills=(fill,),
        snapshots=(),
        config=_CONFIG,
    )

    assert metrics.turnover == expected


def test_tiny_closing_loss_is_not_rounded_into_a_win() -> None:
    base = "1" + "0" * 100
    plus_two = "1" + "0" * 99 + "2"
    plus_half = "1" + "0" * 99 + "0.5"
    fills = (
        _fill(1, quantity="1", price=base),
        _fill(2, quantity="1", price=plus_two),
        _fill(
            3,
            quantity="2",
            price=plus_half,
            side=OrderSide.SELL,
        ),
    )

    metrics = calculate_metrics(
        equity=["100", "100"],
        fills=fills,
        snapshots=(),
        config=_CONFIG,
    )

    assert metrics.win_rate == Decimal("0")


def test_metrics_isolate_beyond_context_range_as_typed_warnings() -> None:
    huge = Decimal(f"9e{MAX_EMAX}")
    tiny = Decimal(f"1e{-MAX_EMAX}")

    metrics = calculate_metrics(
        equity=(tiny, huge),
        fills=(),
        snapshots=(),
        config=_CONFIG,
    )

    assert all(
        result.value is None or result.value.is_finite() for result in metrics.results
    )
    assert any(
        warning.metric is MetricName.TOTAL_RETURN
        and warning.code.value == "metric.unsupported_numeric_range"
        for warning in metrics.warnings
    )


def test_metric_formula_groups_reject_silent_underflow() -> None:
    huge = Decimal(f"1e{MAX_EMAX}")
    tiny = Decimal(f"1e{-MAX_EMAX}")

    metrics = calculate_metrics(
        equity=(huge, tiny, tiny),
        fills=(),
        snapshots=(),
        config=_CONFIG,
    )

    for metric in (
        MetricName.TOTAL_RETURN,
        MetricName.CAGR,
        MetricName.MAXIMUM_DRAWDOWN,
        MetricName.VOLATILITY,
        MetricName.SHARPE,
        MetricName.SORTINO,
    ):
        assert metrics.result_for(metric).value is None
        assert _has_warning(
            metrics,
            metric,
            "metric.unsupported_numeric_range",
        )


def test_turnover_rejects_nonzero_value_that_underflows_to_zero() -> None:
    huge = Decimal(f"1e{MAX_EMAX}")
    tiny = Decimal(f"1e{-MAX_EMAX}")
    fill = _fill(1, quantity=tiny, price="1")

    metrics = calculate_metrics(
        equity=(huge, huge),
        fills=(fill,),
        snapshots=(),
        config=_CONFIG,
    )

    assert metrics.turnover is None
    assert _has_warning(
        metrics,
        MetricName.TURNOVER,
        "metric.unsupported_numeric_range",
    )


def test_exposure_metrics_reject_nonzero_values_that_underflow_to_zero() -> None:
    huge = Decimal(f"1e{MAX_EMAX}")
    tiny = Decimal(f"1e{-MAX_EMAX}")
    snapshot = _flat_snapshot("1")
    object.__setattr__(snapshot, "cash", Money.usd(huge))
    object.__setattr__(snapshot, "equity", Money.usd(huge))
    object.__setattr__(snapshot, "gross_exposure", Money.usd(tiny))
    object.__setattr__(snapshot, "market_value", Money.usd(tiny))

    metrics = calculate_metrics(
        equity=(huge,),
        fills=(),
        snapshots=(snapshot,),
        config=_CONFIG,
    )

    for metric in (
        MetricName.GROSS_EXPOSURE,
        MetricName.NET_EXPOSURE,
    ):
        assert metrics.result_for(metric).value is None
        assert _has_warning(
            metrics,
            metric,
            "metric.unsupported_numeric_range",
        )


def test_exact_decimal_boundary_and_true_zero_metrics_remain_valid() -> None:
    huge = Decimal(f"1e{MAX_EMAX}")
    tiny = Decimal(f"1e{-MAX_EMAX}")
    boundary = calculate_metrics(
        equity=("1", "1"),
        fills=(_fill(1, quantity=tiny, price="1"),),
        snapshots=(),
        config=_CONFIG,
    )
    true_zero = calculate_metrics(
        equity=(huge, huge),
        fills=(),
        snapshots=(),
        config=_CONFIG,
    )

    assert boundary.turnover == tiny
    assert not _has_warning(
        boundary,
        MetricName.TURNOVER,
        "metric.unsupported_numeric_range",
    )
    assert true_zero.total_return == Decimal("0")
    assert true_zero.turnover == Decimal("0")
    assert not _has_warning(
        true_zero,
        MetricName.TOTAL_RETURN,
        "metric.unsupported_numeric_range",
    )
    assert not _has_warning(
        true_zero,
        MetricName.TURNOVER,
        "metric.unsupported_numeric_range",
    )


def test_win_rate_extreme_exact_input_is_typed_not_raw() -> None:
    tiny = Decimal(f"1e{-MAX_EMAX}")
    fills = (
        _fill(1, quantity="1", price="1"),
        _fill(2, quantity="1", price=tiny, side=OrderSide.SELL),
    )

    metrics = calculate_metrics(
        equity=("1", "1"),
        fills=fills,
        snapshots=(),
        config=_CONFIG,
    )

    assert metrics.win_rate is None
    assert _has_warning(
        metrics,
        MetricName.WIN_RATE,
        "metric.unsupported_numeric_range",
    )


@pytest.mark.parametrize("precision", [1, 28, 64])
def test_metrics_are_independent_of_hostile_ambient_context(
    precision: int,
) -> None:
    with localcontext(_hostile_context(precision)):
        actual = calculate_metrics(
            equity=["100", "120", "90", "110"],
            fills=(),
            snapshots=(),
            config=_CONFIG,
        )

    assert actual == calculate_metrics(
        equity=["100", "120", "90", "110"],
        fills=(),
        snapshots=(),
        config=_CONFIG,
    )
