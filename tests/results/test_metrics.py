"""Independent formula checks for immutable result metrics."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from pybacktest.domain.identifiers import FillId, OrderId
from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.orders import Fill, OrderSide
from pybacktest.domain.portfolio import PortfolioSnapshot, Position
from pybacktest.results.metrics import MetricsConfig, calculate_metrics
from pybacktest.results.models import MetricName, ResultValidationError

_BASE = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
_AAPL = InstrumentId.parse("XNAS:AAPL")
_MSFT = InstrumentId.parse("XNAS:MSFT")


def _fill(
    sequence: int,
    *,
    instrument: InstrumentId = _AAPL,
    side: OrderSide,
    quantity: object,
    price: object,
    fee: object = "0",
) -> Fill:
    return Fill(
        id=FillId.parse(f"fill_{sequence:032x}"),
        order_id=OrderId.parse(f"order_{sequence:032x}"),
        instrument=instrument,
        side=side,
        quantity=Quantity.of(quantity),
        price=Money.usd(price),
        fee=Money.usd(fee),
        timestamp=_BASE + timedelta(seconds=sequence),
    )


def _snapshot(
    sequence: int,
    *,
    equity: object,
    positions: dict[InstrumentId, tuple[object, object]] | None = None,
) -> PortfolioSnapshot:
    built_positions: dict[InstrumentId, Position] = {}
    marks: dict[InstrumentId, Money] = {}
    market_value = Decimal("0")
    gross_exposure = Decimal("0")
    for instrument_id, (raw_quantity, raw_price) in (positions or {}).items():
        quantity = Quantity.of(raw_quantity)
        price = Money.usd(raw_price)
        value = quantity.value * price.amount
        market_value += value
        gross_exposure += abs(value)
        marks[instrument_id] = price
        built_positions[instrument_id] = Position(
            instrument=instrument_id,
            quantity=quantity,
            average_price=price,
            book_cost=Money.usd(abs(value)),
            realized_pnl=Money.usd("0"),
        )
    equity_value = Decimal(str(equity))
    return PortfolioSnapshot(
        timestamp=_BASE + timedelta(seconds=sequence),
        cash=Money.usd(equity_value - market_value),
        positions=built_positions,
        realized_pnl=Money.usd("0"),
        unrealized_pnl=Money.usd("0"),
        total_fees=Money.usd("0"),
        market_value=Money.usd(market_value),
        gross_exposure=Money.usd(gross_exposure),
        equity=Money.usd(equity_value),
        valuation_prices=marks,
        cash_events=(),
    )


def _config(
    *,
    risk_free_rate: object = "0",
    annualization_periods: int = 252,
) -> MetricsConfig:
    return MetricsConfig(
        risk_free_rate=risk_free_rate,
        annualization_periods=annualization_periods,
    )


def test_declared_return_and_drawdown_formulas_use_the_supplied_equity() -> None:
    metrics = calculate_metrics(
        equity=[Decimal("100"), Decimal("120"), Decimal("90"), Decimal("110")],
        fills=(),
        snapshots=(),
        config=MetricsConfig(
            risk_free_rate=Decimal("0"),
            annualization_periods=252,
        ),
    )

    assert metrics.total_return == Decimal("0.1")
    assert metrics.maximum_drawdown == Decimal("-0.25")
    assert metrics.metadata_for("total_return").formula_id == "total_return.v1"
    assert metrics.metadata_for("maximum_drawdown").formula_id == (
        "maximum_drawdown.v1"
    )
    assert not any(
        warning.metric is not None
        and warning.metric.value in {"total_return", "maximum_drawdown"}
        for warning in metrics.warnings
    )


def test_sample_volatility_and_ratios_match_literal_oracles() -> None:
    metrics = calculate_metrics(
        equity=["100", "110", "99"],
        fills=(),
        snapshots=(),
        config=MetricsConfig(
            risk_free_rate="0",
            annualization_periods=4,
        ),
    )

    # Returns are exactly +0.1 and -0.1. Sample std is sqrt(0.02);
    # annualized volatility is sqrt(0.02) * 2, and both ratio numerators are 0.
    assert float(metrics.volatility) == pytest.approx(0.282842712474619, rel=1e-14)
    assert metrics.sharpe == Decimal("0")
    assert metrics.sortino == Decimal("0")


def test_cagr_uses_intervals_and_rejects_nonpositive_endpoints() -> None:
    metrics = calculate_metrics(
        equity=["100", "110", "121"],
        fills=(),
        snapshots=(),
        config=_config(annualization_periods=2),
    )
    invalid = calculate_metrics(
        equity=["100", "50", "0"],
        fills=(),
        snapshots=(),
        config=_config(annualization_periods=2),
    )

    assert metrics.cagr == Decimal("0.21")
    assert invalid.cagr is None
    assert any(
        warning.metric is MetricName.CAGR
        and warning.code.value == "metric.invalid_cagr_endpoints"
        for warning in invalid.warnings
    )


def test_one_two_and_three_observations_have_declared_missing_behavior() -> None:
    one = calculate_metrics(
        equity=["100"],
        fills=(),
        snapshots=(),
        config=_config(),
    )
    two = calculate_metrics(
        equity=["100", "110"],
        fills=(),
        snapshots=(),
        config=_config(),
    )
    three = calculate_metrics(
        equity=["100", "110", "121"],
        fills=(),
        snapshots=(),
        config=_config(),
    )

    assert one.total_return is None
    assert two.total_return == Decimal("0.1")
    assert two.volatility is None
    assert three.volatility == Decimal("0")
    assert three.sharpe is None
    assert three.sortino is None


@pytest.mark.parametrize(
    ("fills", "expected"),
    [
        (
            (
                _fill(0, side=OrderSide.BUY, quantity="10", price="100"),
                _fill(1, side=OrderSide.SELL, quantity="4", price="110"),
                _fill(2, side=OrderSide.SELL, quantity="6", price="90"),
            ),
            Decimal("0.5"),
        ),
        (
            (
                _fill(0, side=OrderSide.SELL, quantity="5", price="100"),
                _fill(1, side=OrderSide.BUY, quantity="5", price="90"),
            ),
            Decimal("1"),
        ),
        (
            (
                _fill(0, side=OrderSide.BUY, quantity="3", price="100"),
                _fill(
                    1,
                    side=OrderSide.SELL,
                    quantity="5",
                    price="110",
                    fee="1000",
                ),
                _fill(2, side=OrderSide.BUY, quantity="1", price="110"),
                _fill(3, side=OrderSide.BUY, quantity="1", price="90"),
            ),
            Decimal("1"),
        ),
        (
            (
                _fill(
                    0,
                    instrument=_AAPL,
                    side=OrderSide.BUY,
                    quantity="2",
                    price="100",
                ),
                _fill(
                    1,
                    instrument=_MSFT,
                    side=OrderSide.SELL,
                    quantity="2",
                    price="50",
                ),
                _fill(
                    2,
                    instrument=_MSFT,
                    side=OrderSide.BUY,
                    quantity="1",
                    price="60",
                ),
                _fill(
                    3,
                    instrument=_AAPL,
                    side=OrderSide.SELL,
                    quantity="1",
                    price="120",
                ),
            ),
            Decimal("0.5"),
        ),
    ],
)
def test_win_rate_counts_nonzero_gross_closing_legs(
    fills: tuple[Fill, ...],
    expected: Decimal,
) -> None:
    metrics = calculate_metrics(
        equity=["1000", "1000"],
        fills=fills,
        snapshots=(),
        config=_config(),
    )

    assert metrics.win_rate == expected
    metadata = metrics.metadata_for(MetricName.WIN_RATE)
    assert metadata.parameters["closing_leg_policy"] == (
        "each opposing fill closes up to the open quantity"
    )
    assert metadata.parameters["zero_profit_policy"] == "excluded"
    assert metadata.parameters["fee_policy"] == "gross_pnl_excludes_fees"


def test_win_rate_warns_when_no_nonzero_closing_leg_is_available() -> None:
    metrics = calculate_metrics(
        equity=["100", "100"],
        fills=(
            _fill(0, side=OrderSide.BUY, quantity="1", price="100"),
            _fill(1, side=OrderSide.SELL, quantity="1", price="100"),
        ),
        snapshots=(),
        config=_config(),
    )

    assert metrics.win_rate is None
    assert any(
        warning.metric is MetricName.WIN_RATE
        and warning.code.value == "metric.unavailable_closing_legs"
        for warning in metrics.warnings
    )


def test_turnover_uses_actual_fill_notional_over_mean_equity() -> None:
    metrics = calculate_metrics(
        equity=["100", "200"],
        fills=(
            _fill(0, side=OrderSide.BUY, quantity="2", price="10"),
            _fill(1, side=OrderSide.SELL, quantity="1", price="30"),
        ),
        snapshots=(),
        config=_config(),
    )

    assert metrics.turnover == Decimal(
        "0.3333333333333333333333333333333333333333333333333333333333333333"
    )


def test_turnover_rejects_a_nonpositive_mean_equity_denominator() -> None:
    metrics = calculate_metrics(
        equity=["100", "-100"],
        fills=(),
        snapshots=(),
        config=_config(),
    )

    assert metrics.turnover is None
    assert any(
        warning.metric is MetricName.TURNOVER
        and warning.code.value == "metric.nonpositive_denominator"
        for warning in metrics.warnings
    )


def test_exposure_averages_explicit_snapshots_and_excludes_nonpositive_equity() -> None:
    snapshots = (
        _snapshot(
            0,
            equity="200",
            positions={
                _AAPL: ("1", "100"),
                _MSFT: ("-1", "50"),
            },
        ),
        _snapshot(
            1,
            equity="100",
            positions={_AAPL: ("-1", "20")},
        ),
        _snapshot(
            2,
            equity="0",
            positions={_AAPL: ("1", "10")},
        ),
        _snapshot(
            3,
            equity="-10",
            positions={_AAPL: ("1", "10")},
        ),
    )
    metrics = calculate_metrics(
        equity=["200", "100", "0", "-10"],
        fills=(),
        snapshots=snapshots,
        config=_config(),
    )

    # Valid observations: gross (150/200, 20/100) and net (50/200, -20/100).
    assert metrics.gross_exposure == Decimal("0.475")
    assert metrics.net_exposure == Decimal("0.025")
    for name in (MetricName.GROSS_EXPOSURE, MetricName.NET_EXPOSURE):
        assert any(
            warning.metric is name
            and warning.code.value == "metric.excluded_nonpositive_equity"
            and warning.details["excluded_observations"] == 2
            for warning in metrics.warnings
        )


def test_metric_warnings_are_stable_deduplicated_and_values_never_nonfinite() -> None:
    first = calculate_metrics(
        equity=[],
        fills=(),
        snapshots=(),
        config=_config(),
    )
    second = calculate_metrics(
        equity=[],
        fills=(),
        snapshots=(),
        config=_config(),
    )

    assert first == second
    assert len(first.warnings) == len(set(first.warnings))
    assert all(
        item.value is None or item.value.is_finite()
        for item in first.results
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("risk_free_rate", True),
        ("risk_free_rate", "NaN"),
        ("risk_free_rate", "Infinity"),
        ("annualization_periods", True),
        ("annualization_periods", 0),
    ],
)
def test_metrics_config_rejects_bool_nonfinite_and_nonpositive_values(
    field: str,
    value: object,
) -> None:
    values = {
        "risk_free_rate": "0",
        "annualization_periods": 252,
    }
    values[field] = value

    with pytest.raises(ResultValidationError):
        MetricsConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), Decimal("NaN")])
def test_equity_rejects_bool_and_nonfinite_values(bad: object) -> None:
    with pytest.raises(ResultValidationError):
        calculate_metrics(
            equity=["100", bad],
            fills=(),
            snapshots=(),
            config=_config(),
        )


def test_overflow_sized_finite_equity_does_not_produce_infinity() -> None:
    metrics = calculate_metrics(
        equity=["1e4000", "2e4000", "3e4000"],
        fills=(),
        snapshots=(),
        config=_config(),
    )

    assert metrics.total_return == Decimal("2")
    assert all(
        result.value is None or result.value.is_finite()
        for result in metrics.results
    )
