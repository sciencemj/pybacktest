"""Streamlit page: widgets, charts, and tables only."""

from __future__ import annotations

import traceback
from datetime import date, timedelta
from decimal import Decimal

import streamlit as st

from pybacktest import PybacktestError
from streamlit_ui import market_data
from streamlit_ui.errors import DemoInputError
from streamlit_ui.i18n import LANGUAGES, t
from streamlit_ui.runner import STRATEGY_NAMES, RunConfig, RunReport, run_backtest

PERCENT_METRICS = frozenset(
    {"total_return", "cagr", "volatility", "maximum_drawdown", "win_rate"}
)
HEADLINE_METRICS = ("total_return", "cagr", "sharpe", "maximum_drawdown")
DEFAULT_TICKERS = "AAPL, MSFT"
DEFAULT_YEARS = 5


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_history(
    tickers: tuple[str, ...],
    start: date,
    end: date,
) -> dict[str, market_data.TickerHistory]:
    # Looked up on the module at call time so tests can patch it.
    return market_data.fetch_history(tickers, start, end)


def format_metric(name: str, value: float | None) -> str:
    """Render one metric value for display."""
    if value is None:
        return "—"
    if name in PERCENT_METRICS:
        return f"{value:.2%}"
    return f"{value:.2f}"


def _strategy_params(strategy: str, lang: str) -> dict[str, int]:
    if strategy == "ma_cross":
        fast = st.number_input(t("param.fast", lang), 2, 200, 20, key="fast")
        slow = st.number_input(t("param.slow", lang), 3, 400, 60, key="slow")
        return {"fast": int(fast), "slow": int(slow)}
    if strategy == "rsi":
        period = st.number_input(t("param.period", lang), 2, 30, 14, key="period")
        lower = st.number_input(t("param.lower", lang), 1, 98, 30, key="lower")
        upper = st.number_input(t("param.upper", lang), 2, 99, 70, key="upper")
        return {"period": int(period), "lower": int(lower), "upper": int(upper)}
    return {}


def _language() -> str:
    with st.sidebar:
        return st.radio(
            "Language / 언어",
            list(LANGUAGES),
            format_func=LANGUAGES.__getitem__,
            horizontal=True,
            key="lang",
        )


def _sidebar(lang: str) -> RunConfig | None:
    """Collect the run configuration; ``None`` until the form is submitted."""
    with st.sidebar:
        strategy = st.selectbox(
            t("sidebar.strategy", lang),
            STRATEGY_NAMES,
            format_func=lambda name: t(f"strategy.{name}", lang),
            key="strategy",
        )
        with st.form("run"):
            raw_tickers = st.text_input(
                t("sidebar.tickers", lang),
                DEFAULT_TICKERS,
                help=t("sidebar.tickers_help", lang),
                key="tickers",
            )
            today = date.today()
            start = st.date_input(
                t("sidebar.start", lang),
                today - timedelta(days=365 * DEFAULT_YEARS),
                key="start",
            )
            end = st.date_input(t("sidebar.end", lang), today, key="end")
            params = _strategy_params(strategy, lang)
            cash = st.number_input(
                t("sidebar.initial_cash", lang), 100.0, 1e12, 10000.0, key="cash"
            )
            commission = st.number_input(
                t("sidebar.commission", lang), 0.0, 100.0, 0.0, key="commission"
            )
            submitted = st.form_submit_button(t("sidebar.run", lang), type="primary")
    if not submitted:
        return None
    if start >= end:
        raise DemoInputError("invalid_period")
    return RunConfig(
        tickers=market_data.parse_tickers(raw_tickers),
        start=start,
        end=end,
        strategy=strategy,
        params=params,
        initial_cash=Decimal(str(cash)),
        commission_per_share=Decimal(str(commission)),
    )


def _render_report(report: RunReport, lang: str) -> None:
    st.subheader(t("result.metrics", lang))
    columns = st.columns(len(HEADLINE_METRICS))
    for column, name in zip(columns, HEADLINE_METRICS, strict=True):
        strategy_value = report.metrics.loc[name, "strategy"]
        benchmark_value = report.metrics.loc[name, "benchmark"]
        column.metric(
            t(f"metric.{name}", lang),
            format_metric(name, _optional(strategy_value)),
            f"{t('column.benchmark', lang)} "
            f"{format_metric(name, _optional(benchmark_value))}",
            delta_color="off",
        )
    st.subheader(t("result.equity", lang))
    st.line_chart(
        report.equity.rename(
            columns={
                "strategy": t("column.strategy", lang),
                "benchmark": t("column.benchmark", lang),
            }
        )
    )
    st.table(
        {
            t("column.strategy", lang): _formatted_column(report, "strategy", lang),
            t("column.benchmark", lang): _formatted_column(report, "benchmark", lang),
        }
    )
    orders, fills, warnings, info = st.tabs(
        [
            t("result.orders", lang),
            t("result.fills", lang),
            t("result.warnings", lang),
            t("result.info", lang),
        ]
    )
    with orders:
        st.dataframe(report.orders, hide_index=True, width="stretch")
    with fills:
        st.dataframe(report.fills, hide_index=True, width="stretch")
    with warnings:
        if report.warnings.empty:
            st.write(t("result.none", lang))
        else:
            st.dataframe(report.warnings, hide_index=True, width="stretch")
    with info:
        st.json(report.info)


def _formatted_column(report: RunReport, column: str, lang: str) -> dict[str, str]:
    return {
        t(f"metric.{name}", lang): format_metric(
            name, _optional(report.metrics.loc[name, column])
        )
        for name in report.metrics.index
    }


def _optional(value: object) -> float | None:
    if value is None or value != value:
        return None
    return float(value)


def main() -> None:
    """Render the demo page."""
    st.set_page_config(page_title="Pybacktest demo", layout="wide")
    lang = _language()
    st.title(t("page.title", lang))
    try:
        config = _sidebar(lang)
        if config is not None:
            with st.spinner(t("result.spinner", lang)):
                histories = _cached_history(config.tickers, config.start, config.end)
                st.session_state["report"] = run_backtest(config, histories)
    except DemoInputError as error:
        st.error(t(f"error.{error.code}", lang, detail=error.detail))
        return
    except PybacktestError as error:
        st.error(t("error.engine", lang, detail=str(error)))
        return
    except Exception:
        st.error(t("error.unexpected", lang))
        with st.expander(t("error.details", lang)):
            st.code(traceback.format_exc())
        return
    report = st.session_state.get("report")
    if report is None:
        st.write(t("page.intro", lang))
        st.markdown(t("page.steps", lang))
        return
    _render_report(report, lang)
