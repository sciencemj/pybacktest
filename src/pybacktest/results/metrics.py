"""Readable versioned performance formulas without NaN or infinity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction
from itertools import pairwise
from typing import cast

from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.orders import Fill
from pybacktest.domain.portfolio import PortfolioSnapshot

from ._decimal import (
    MAX_EXACT_DIGITS,
    ExactDecimalError,
    calculation_context,
    exact_multiply,
    exact_sum,
)
from .models import (
    FrozenMapping,
    MetricMetadata,
    MetricName,
    MetricResult,
    ResultValidationError,
    RunWarning,
    SummaryMetrics,
    WarningCode,
)

_FORMULA_IDS = {
    MetricName.TOTAL_RETURN: "total_return.v1",
    MetricName.CAGR: "cagr.v1",
    MetricName.VOLATILITY: "volatility.v1",
    MetricName.SHARPE: "sharpe.v1",
    MetricName.SORTINO: "sortino.v1",
    MetricName.MAXIMUM_DRAWDOWN: "maximum_drawdown.v1",
    MetricName.TURNOVER: "turnover.v1",
    MetricName.WIN_RATE: "win_rate.v1",
    MetricName.GROSS_EXPOSURE: "gross_exposure.v1",
    MetricName.NET_EXPOSURE: "net_exposure.v1",
}
_ZERO = Decimal("0")
_ONE = Decimal("1")
_MAX_FRACTION_BITS = MAX_EXACT_DIGITS * 4


def _finite_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ResultValidationError(f"{field_name} must be a finite number.")
    try:
        result = Decimal(str(value))
    except (TypeError, ValueError, DecimalException) as error:
        raise ResultValidationError(f"{field_name} must be a finite number.") from error
    if not result.is_finite():
        raise ResultValidationError(f"{field_name} must be a finite number.")
    return result


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Explicit inputs shared by all versioned metric formulas."""

    risk_free_rate: Decimal
    annualization_periods: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "risk_free_rate",
            _finite_decimal(self.risk_free_rate, "risk_free_rate"),
        )
        if (
            isinstance(self.annualization_periods, bool)
            or not isinstance(self.annualization_periods, int)
            or self.annualization_periods <= 0
        ):
            raise ResultValidationError(
                "annualization_periods must be a positive integer."
            )


def _warning(
    metric: MetricName,
    code: str,
    message: str,
    **details: object,
) -> RunWarning:
    return RunWarning(
        code=WarningCode.of(code),
        message=message,
        metric=metric,
        details=FrozenMapping.from_mapping(details),
    )


def _numeric_warning(metric: MetricName) -> RunWarning:
    return _warning(
        metric,
        "metric.unsupported_numeric_range",
        f"{metric.value} exceeds the supported numeric range.",
    )


def _deduplicate_warnings(
    warnings: Sequence[RunWarning],
) -> tuple[RunWarning, ...]:
    result: list[RunWarning] = []
    seen: set[RunWarning] = set()
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            result.append(warning)
    return tuple(result)


def _bounded_fraction(value: Fraction) -> Fraction:
    if (
        value.numerator.bit_length() > _MAX_FRACTION_BITS
        or value.denominator.bit_length() > _MAX_FRACTION_BITS
    ):
        raise ExactDecimalError(
            "win-rate arithmetic exceeds the supported exact range."
        )
    return value


def _exact_fraction(value: Decimal) -> Fraction:
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ExactDecimalError("win-rate arithmetic requires a finite Decimal.")
    if (
        len(value.as_tuple().digits) > MAX_EXACT_DIGITS
        or abs(exponent) > MAX_EXACT_DIGITS
    ):
        raise ExactDecimalError(
            "win-rate arithmetic exceeds the supported exact range."
        )
    return _bounded_fraction(Fraction(value))


def _closing_leg_win_rate(fills: Sequence[Fill]) -> Decimal | None:
    positions: dict[InstrumentId, tuple[Fraction, Fraction]] = {}
    profitable = 0
    nonzero = 0
    for fill in fills:
        old_quantity, old_book_cost = positions.get(
            fill.instrument,
            (Fraction(0), Fraction(0)),
        )
        fill_quantity = _exact_fraction(fill.quantity.value)
        fill_price = _exact_fraction(fill.price.amount)
        signed_fill = fill_quantity if fill.side.value == "buy" else -fill_quantity
        if (
            old_quantity == 0
            or (old_quantity > 0 and signed_fill > 0)
            or (old_quantity < 0 and signed_fill < 0)
        ):
            new_quantity = _bounded_fraction(old_quantity + signed_fill)
            new_book_cost = _bounded_fraction(
                old_book_cost + abs(signed_fill) * fill_price
            )
            positions[fill.instrument] = (new_quantity, new_book_cost)
            continue
        exit_value = _bounded_fraction(fill_price * abs(old_quantity))
        gross_profit_sign = (
            exit_value - old_book_cost
            if old_quantity > 0
            else old_book_cost - exit_value
        )
        if gross_profit_sign != 0:
            nonzero += 1
            if gross_profit_sign > 0:
                profitable += 1
        new_quantity = _bounded_fraction(old_quantity + signed_fill)
        if new_quantity == 0:
            positions[fill.instrument] = (Fraction(0), Fraction(0))
        elif (new_quantity > 0) == (old_quantity > 0):
            positions[fill.instrument] = (
                new_quantity,
                _bounded_fraction(
                    old_book_cost * abs(new_quantity) / abs(old_quantity)
                ),
            )
        else:
            positions[fill.instrument] = (
                new_quantity,
                _bounded_fraction(abs(new_quantity) * fill_price),
            )
    if nonzero == 0:
        return None
    with localcontext(calculation_context(Decimal(profitable), Decimal(nonzero))):
        return Decimal(profitable) / Decimal(nonzero)


def calculate_metrics(
    *,
    equity: object,
    fills: Sequence[Fill],
    snapshots: Sequence[PortfolioSnapshot],
    config: MetricsConfig,
) -> SummaryMetrics:
    """Calculate metrics only from the explicitly supplied typed series."""
    if not isinstance(config, MetricsConfig):
        raise ResultValidationError("config must be a MetricsConfig.")
    if isinstance(equity, (str, bytes, bytearray)):
        raise ResultValidationError("equity must be a numeric sequence.")
    try:
        values = tuple(
            _finite_decimal(value, f"equity[{index}]")
            for index, value in enumerate(cast("Sequence[object]", equity))
        )
    except TypeError as error:
        raise ResultValidationError("equity must be a numeric sequence.") from error
    if isinstance(fills, (str, bytes, bytearray)) or not isinstance(
        fills,
        Sequence,
    ):
        raise ResultValidationError("fills must be a sequence.")
    copied_fills = tuple(fills)
    if not all(isinstance(fill, Fill) for fill in copied_fills):
        raise ResultValidationError("fills must contain Fill values.")
    if len({fill.id for fill in copied_fills}) != len(copied_fills):
        raise ResultValidationError("fill IDs must be unique.")
    if any(
        current.timestamp < previous.timestamp
        for previous, current in pairwise(copied_fills)
    ):
        raise ResultValidationError("fills must be timestamp-monotonic.")
    if isinstance(snapshots, (str, bytes, bytearray)) or not isinstance(
        snapshots,
        Sequence,
    ):
        raise ResultValidationError("snapshots must be a sequence.")
    copied_snapshots = tuple(snapshots)
    if not all(
        isinstance(snapshot, PortfolioSnapshot) for snapshot in copied_snapshots
    ):
        raise ResultValidationError("snapshots must contain PortfolioSnapshot values.")
    snapshot_timestamps = tuple(snapshot.timestamp for snapshot in copied_snapshots)
    if any(timestamp is None for timestamp in snapshot_timestamps):
        raise ResultValidationError("metric snapshots must have aware timestamps.")
    if any(
        current < previous
        for previous, current in pairwise(snapshot_timestamps)
        if previous is not None and current is not None
    ):
        raise ResultValidationError("snapshots must be timestamp-monotonic.")

    metric_values: dict[MetricName, Decimal | None] = dict.fromkeys(
        MetricName,
        None,
    )
    warnings: list[RunWarning] = []
    context = calculation_context()
    with localcontext(context):
        if len(values) < 2:
            for metric in (
                MetricName.TOTAL_RETURN,
                MetricName.MAXIMUM_DRAWDOWN,
                MetricName.VOLATILITY,
                MetricName.SHARPE,
                MetricName.SORTINO,
            ):
                warnings.append(
                    _warning(
                        metric,
                        "metric.insufficient_observations",
                        f"{metric.value} requires more equity observations.",
                    )
                )
        else:
            initial = values[0]
            if initial == _ZERO:
                warnings.append(
                    _warning(
                        MetricName.TOTAL_RETURN,
                        "metric.zero_denominator",
                        "total_return requires nonzero initial equity.",
                    )
                )
            else:
                try:
                    metric_values[MetricName.TOTAL_RETURN] = values[-1] / initial - _ONE
                except DecimalException:
                    warnings.append(_numeric_warning(MetricName.TOTAL_RETURN))

            if values[0] <= _ZERO or values[-1] <= _ZERO:
                warnings.append(
                    _warning(
                        MetricName.CAGR,
                        "metric.invalid_cagr_endpoints",
                        "cagr requires positive initial and final equity.",
                    )
                )
            else:
                intervals = Decimal(len(values) - 1)
                exponent = Decimal(config.annualization_periods) / intervals
                try:
                    cagr = (values[-1] / values[0]) ** exponent - _ONE
                    if not cagr.is_finite():
                        raise ArithmeticError
                    metric_values[MetricName.CAGR] = cagr
                except DecimalException:
                    warnings.append(_numeric_warning(MetricName.CAGR))
                except ArithmeticError:
                    warnings.append(
                        _warning(
                            MetricName.CAGR,
                            "metric.nonfinite_result",
                            "cagr could not be represented as a finite value.",
                        )
                    )

            try:
                running_peak = values[0]
                drawdowns: list[Decimal] = []
                drawdown_is_valid = True
                for value in values:
                    running_peak = max(running_peak, value)
                    if running_peak <= _ZERO:
                        drawdown_is_valid = False
                        break
                    drawdowns.append(value / running_peak - _ONE)
                if drawdown_is_valid:
                    metric_values[MetricName.MAXIMUM_DRAWDOWN] = min(drawdowns)
                else:
                    warnings.append(
                        _warning(
                            MetricName.MAXIMUM_DRAWDOWN,
                            "metric.nonpositive_denominator",
                            ("maximum_drawdown requires positive running peak equity."),
                        )
                    )
            except DecimalException:
                warnings.append(_numeric_warning(MetricName.MAXIMUM_DRAWDOWN))

            returns: list[Decimal] = []
            invalid_return_denominator = False
            returns_numeric_error = False
            try:
                for previous, current in pairwise(values):
                    if previous == _ZERO:
                        returns = []
                        invalid_return_denominator = True
                        break
                    returns.append(current / previous - _ONE)
            except DecimalException:
                returns = []
                returns_numeric_error = True
            if returns_numeric_error:
                for metric in (
                    MetricName.VOLATILITY,
                    MetricName.SHARPE,
                    MetricName.SORTINO,
                ):
                    warnings.append(_numeric_warning(metric))
            elif len(returns) < 2:
                for metric in (
                    MetricName.VOLATILITY,
                    MetricName.SHARPE,
                ):
                    warnings.append(
                        _warning(
                            metric,
                            (
                                "metric.zero_denominator"
                                if invalid_return_denominator
                                else "metric.insufficient_observations"
                            ),
                            (
                                f"{metric.value} encountered zero prior equity."
                                if invalid_return_denominator
                                else (f"{metric.value} requires at least two returns.")
                            ),
                        )
                    )
            else:
                try:
                    count = Decimal(len(returns))
                    mean_return = sum(returns, _ZERO) / count
                    variance = sum(
                        ((item - mean_return) ** 2 for item in returns),
                        _ZERO,
                    ) / Decimal(len(returns) - 1)
                    sample_std = variance.sqrt()
                    annual_root = Decimal(config.annualization_periods).sqrt()
                    metric_values[MetricName.VOLATILITY] = sample_std * annual_root
                except DecimalException:
                    sample_std = None
                    warnings.append(_numeric_warning(MetricName.VOLATILITY))
                    warnings.append(_numeric_warning(MetricName.SHARPE))
                if sample_std is not None:
                    if sample_std == _ZERO:
                        warnings.append(
                            _warning(
                                MetricName.SHARPE,
                                "metric.zero_denominator",
                                "sharpe requires nonzero sample volatility.",
                            )
                        )
                    else:
                        try:
                            periodic_rf = config.risk_free_rate / Decimal(
                                config.annualization_periods
                            )
                            mean_excess = (
                                sum(
                                    (item - periodic_rf for item in returns),
                                    _ZERO,
                                )
                                / count
                            )
                            metric_values[MetricName.SHARPE] = (
                                mean_excess / sample_std * annual_root
                            )
                        except DecimalException:
                            warnings.append(_numeric_warning(MetricName.SHARPE))

            if returns_numeric_error:
                pass
            elif not returns:
                warnings.append(
                    _warning(
                        MetricName.SORTINO,
                        (
                            "metric.zero_denominator"
                            if invalid_return_denominator
                            else "metric.insufficient_observations"
                        ),
                        (
                            "sortino encountered zero prior equity."
                            if invalid_return_denominator
                            else "sortino requires at least one valid return."
                        ),
                    )
                )
            else:
                try:
                    periodic_rf = config.risk_free_rate / Decimal(
                        config.annualization_periods
                    )
                    excess = tuple(item - periodic_rf for item in returns)
                    downside_rms = (
                        sum(
                            (min(item, _ZERO) ** 2 for item in excess),
                            _ZERO,
                        )
                        / Decimal(len(excess))
                    ).sqrt()
                    if downside_rms == _ZERO:
                        warnings.append(
                            _warning(
                                MetricName.SORTINO,
                                "metric.zero_denominator",
                                ("sortino requires nonzero downside deviation."),
                            )
                        )
                    else:
                        metric_values[MetricName.SORTINO] = (
                            sum(excess, _ZERO)
                            / Decimal(len(excess))
                            / downside_rms
                            * Decimal(config.annualization_periods).sqrt()
                        )
                except DecimalException:
                    warnings.append(_numeric_warning(MetricName.SORTINO))

        if values:
            try:
                total_equity = exact_sum(values)
                with localcontext(calculation_context(total_equity)):
                    mean_equity = total_equity / Decimal(len(values))
                if mean_equity <= _ZERO:
                    warnings.append(
                        _warning(
                            MetricName.TURNOVER,
                            "metric.nonpositive_denominator",
                            "turnover requires positive mean equity.",
                        )
                    )
                else:
                    fill_notional = exact_sum(
                        exact_multiply(
                            fill.quantity.value,
                            fill.price.amount,
                        ).copy_abs()
                        for fill in copied_fills
                    )
                    with localcontext(calculation_context(fill_notional, mean_equity)):
                        metric_values[MetricName.TURNOVER] = fill_notional / mean_equity
            except (DecimalException, ExactDecimalError):
                warnings.append(_numeric_warning(MetricName.TURNOVER))
        else:
            warnings.append(
                _warning(
                    MetricName.TURNOVER,
                    "metric.insufficient_observations",
                    "turnover requires at least one equity observation.",
                )
            )

        try:
            win_rate = _closing_leg_win_rate(copied_fills)
        except (DecimalException, ExactDecimalError):
            warnings.append(_numeric_warning(MetricName.WIN_RATE))
        else:
            if win_rate is None:
                warnings.append(
                    _warning(
                        MetricName.WIN_RATE,
                        "metric.unavailable_closing_legs",
                        ("win_rate requires at least one nonzero gross closing leg."),
                    )
                )
            else:
                metric_values[MetricName.WIN_RATE] = win_rate

        valid_snapshots = tuple(
            snapshot for snapshot in copied_snapshots if snapshot.equity.amount > _ZERO
        )
        excluded = len(copied_snapshots) - len(valid_snapshots)
        for metric in (
            MetricName.GROSS_EXPOSURE,
            MetricName.NET_EXPOSURE,
        ):
            if excluded:
                warnings.append(
                    _warning(
                        metric,
                        "metric.excluded_nonpositive_equity",
                        (f"{metric.value} excluded snapshots with nonpositive equity."),
                        excluded_observations=excluded,
                    )
                )
        if valid_snapshots:
            count = Decimal(len(valid_snapshots))
            for metric, numerator_name in (
                (MetricName.GROSS_EXPOSURE, "gross_exposure"),
                (MetricName.NET_EXPOSURE, "market_value"),
            ):
                try:
                    metric_values[metric] = (
                        sum(
                            (
                                getattr(snapshot, numerator_name).amount
                                / snapshot.equity.amount
                                for snapshot in valid_snapshots
                            ),
                            _ZERO,
                        )
                        / count
                    )
                except DecimalException:
                    warnings.append(_numeric_warning(metric))
        else:
            for metric in (
                MetricName.GROSS_EXPOSURE,
                MetricName.NET_EXPOSURE,
            ):
                warnings.append(
                    _warning(
                        metric,
                        "metric.insufficient_observations",
                        (f"{metric.value} requires a positive-equity snapshot."),
                    )
                )

    results = tuple(
        MetricResult(
            name=name,
            value=metric_values[name],
            metadata=MetricMetadata(
                formula_id=_FORMULA_IDS[name],
                annualization_periods=config.annualization_periods,
                risk_free_rate=config.risk_free_rate,
                parameters=FrozenMapping.from_mapping(
                    {
                        "closing_leg_policy": (
                            "each opposing fill closes up to the open quantity"
                        ),
                        "zero_profit_policy": "excluded",
                        "fee_policy": "gross_pnl_excludes_fees",
                    }
                    if name is MetricName.WIN_RATE
                    else {}
                ),
            ),
        )
        for name in MetricName
    )
    return SummaryMetrics(
        results=results,
        warnings=_deduplicate_warnings(warnings),
    )


__all__ = ["MetricsConfig", "calculate_metrics"]
