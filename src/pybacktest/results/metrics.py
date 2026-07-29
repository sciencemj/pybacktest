"""Readable versioned performance formulas without NaN or infinity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, DecimalException, localcontext
from itertools import pairwise
from typing import cast

from pybacktest.domain.instruments import InstrumentId
from pybacktest.domain.orders import Fill
from pybacktest.domain.portfolio import PortfolioSnapshot

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


def _finite_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ResultValidationError(f"{field_name} must be a finite number.")
    try:
        result = Decimal(str(value))
    except (TypeError, ValueError, DecimalException) as error:
        raise ResultValidationError(
            f"{field_name} must be a finite number."
        ) from error
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


def _closing_leg_win_rate(fills: Sequence[Fill]) -> Decimal | None:
    positions: dict[InstrumentId, tuple[Decimal, Decimal | None]] = {}
    profitable = 0
    nonzero = 0
    for fill in fills:
        old_quantity, old_average = positions.get(
            fill.instrument,
            (_ZERO, None),
        )
        signed_fill = (
            fill.quantity.value
            if fill.side.value == "buy"
            else -fill.quantity.value
        )
        if old_quantity == _ZERO or (
            old_quantity > _ZERO and signed_fill > _ZERO
        ) or (old_quantity < _ZERO and signed_fill < _ZERO):
            new_quantity = old_quantity + signed_fill
            old_notional = (
                abs(old_quantity) * old_average
                if old_average is not None
                else _ZERO
            )
            new_average = (
                old_notional
                + abs(signed_fill) * fill.price.amount
            ) / abs(new_quantity)
            positions[fill.instrument] = (new_quantity, new_average)
            continue
        if old_average is None:
            raise ResultValidationError(
                "open fill position is missing an average price."
            )
        closed_quantity = min(abs(old_quantity), abs(signed_fill))
        gross_profit = (
            (fill.price.amount - old_average) * closed_quantity
            if old_quantity > _ZERO
            else (old_average - fill.price.amount) * closed_quantity
        )
        if gross_profit != _ZERO:
            nonzero += 1
            if gross_profit > _ZERO:
                profitable += 1
        new_quantity = old_quantity + signed_fill
        if new_quantity == _ZERO:
            positions[fill.instrument] = (_ZERO, None)
        elif (new_quantity > _ZERO) == (old_quantity > _ZERO):
            positions[fill.instrument] = (new_quantity, old_average)
        else:
            positions[fill.instrument] = (
                new_quantity,
                fill.price.amount,
            )
    if nonzero == 0:
        return None
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
        isinstance(snapshot, PortfolioSnapshot)
        for snapshot in copied_snapshots
    ):
        raise ResultValidationError(
            "snapshots must contain PortfolioSnapshot values."
        )
    snapshot_timestamps = tuple(
        snapshot.timestamp for snapshot in copied_snapshots
    )
    if any(timestamp is None for timestamp in snapshot_timestamps):
        raise ResultValidationError(
            "metric snapshots must have aware timestamps."
        )
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
    context = Context(prec=64, Emin=MIN_EMIN, Emax=MAX_EMAX)
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
                metric_values[MetricName.TOTAL_RETURN] = values[-1] / initial - _ONE

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
                exponent = (
                    Decimal(config.annualization_periods) / intervals
                )
                try:
                    cagr = (values[-1] / values[0]) ** exponent - _ONE
                except DecimalException:
                    cagr = Decimal("NaN")
                if cagr.is_finite():
                    metric_values[MetricName.CAGR] = cagr
                else:
                    warnings.append(
                        _warning(
                            MetricName.CAGR,
                            "metric.nonfinite_result",
                            "cagr could not be represented as a finite value.",
                        )
                    )

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
                        "maximum_drawdown requires positive running peak equity.",
                    )
                )

            returns: list[Decimal] = []
            invalid_return_denominator = False
            for previous, current in pairwise(values):
                if previous == _ZERO:
                    returns = []
                    invalid_return_denominator = True
                    break
                returns.append(current / previous - _ONE)
            if len(returns) < 2:
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
                                else (
                                    f"{metric.value} requires at least two returns."
                                )
                            ),
                        )
                    )
            else:
                count = Decimal(len(returns))
                mean_return = sum(returns, _ZERO) / count
                variance = sum(
                    ((item - mean_return) ** 2 for item in returns),
                    _ZERO,
                ) / Decimal(len(returns) - 1)
                sample_std = variance.sqrt()
                annual_root = Decimal(config.annualization_periods).sqrt()
                metric_values[MetricName.VOLATILITY] = sample_std * annual_root
                if sample_std == _ZERO:
                    warnings.append(
                        _warning(
                            MetricName.SHARPE,
                            "metric.zero_denominator",
                            "sharpe requires nonzero sample volatility.",
                        )
                    )
                else:
                    periodic_rf = (
                        config.risk_free_rate
                        / Decimal(config.annualization_periods)
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

            if not returns:
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
                periodic_rf = (
                    config.risk_free_rate
                    / Decimal(config.annualization_periods)
                )
                excess = tuple(item - periodic_rf for item in returns)
                downside_rms = (
                    sum((min(item, _ZERO) ** 2 for item in excess), _ZERO)
                    / Decimal(len(excess))
                ).sqrt()
                if downside_rms == _ZERO:
                    warnings.append(
                        _warning(
                            MetricName.SORTINO,
                            "metric.zero_denominator",
                            "sortino requires nonzero downside deviation.",
                        )
                    )
                else:
                    metric_values[MetricName.SORTINO] = (
                        sum(excess, _ZERO)
                        / Decimal(len(excess))
                        / downside_rms
                        * Decimal(config.annualization_periods).sqrt()
                    )

        if values:
            mean_equity = sum(values, _ZERO) / Decimal(len(values))
            if mean_equity <= _ZERO:
                warnings.append(
                    _warning(
                        MetricName.TURNOVER,
                        "metric.nonpositive_denominator",
                        "turnover requires positive mean equity.",
                    )
                )
            else:
                fill_notional = sum(
                    (
                        abs(fill.quantity.value * fill.price.amount)
                        for fill in copied_fills
                    ),
                    _ZERO,
                )
                metric_values[MetricName.TURNOVER] = (
                    fill_notional / mean_equity
                )
        else:
            warnings.append(
                _warning(
                    MetricName.TURNOVER,
                    "metric.insufficient_observations",
                    "turnover requires at least one equity observation.",
                )
            )

        win_rate = _closing_leg_win_rate(copied_fills)
        if win_rate is None:
            warnings.append(
                _warning(
                    MetricName.WIN_RATE,
                    "metric.unavailable_closing_legs",
                    "win_rate requires at least one nonzero gross closing leg.",
                )
            )
        else:
            metric_values[MetricName.WIN_RATE] = win_rate

        valid_snapshots = tuple(
            snapshot
            for snapshot in copied_snapshots
            if snapshot.equity.amount > _ZERO
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
                        (
                            f"{metric.value} excluded snapshots with "
                            "nonpositive equity."
                        ),
                        excluded_observations=excluded,
                    )
                )
        if valid_snapshots:
            count = Decimal(len(valid_snapshots))
            metric_values[MetricName.GROSS_EXPOSURE] = (
                sum(
                    (
                        snapshot.gross_exposure.amount
                        / snapshot.equity.amount
                        for snapshot in valid_snapshots
                    ),
                    _ZERO,
                )
                / count
            )
            metric_values[MetricName.NET_EXPOSURE] = (
                sum(
                    (
                        snapshot.market_value.amount
                        / snapshot.equity.amount
                        for snapshot in valid_snapshots
                    ),
                    _ZERO,
                )
                / count
            )
        else:
            for metric in (
                MetricName.GROSS_EXPOSURE,
                MetricName.NET_EXPOSURE,
            ):
                warnings.append(
                    _warning(
                        metric,
                        "metric.insufficient_observations",
                        (
                            f"{metric.value} requires a positive-equity "
                            "snapshot."
                        ),
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
