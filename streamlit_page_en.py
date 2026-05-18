"""English-language Streamlit page for the pybacktest strategy editor."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from pybacktest.backtest import Backtest
from pybacktest.models import Stock
from pybacktest.strategy import StrategyManager, StrategyWrapper
from streamlit_ui import (
    T,
    _apply_uploaded_json,
    _maybe_reseed_widgets,
    _save_strategy_callback,
    inject_global_styles,
    input_strategy_details,
    render_buy_sell_selector,
    render_metric,
    render_page_header,
    render_section_header,
    render_strategy_card,
)


_LANG = "en"


def _sidebar() -> None:
    with st.sidebar:
        uploaded_file = st.file_uploader(
            T("load_json_label", _LANG), type=["json"], key="en_upload"
        )
        if uploaded_file is not None:
            if st.button(
                T("apply_data_button", _LANG),
                type="primary",
                use_container_width=True,
                key="en_apply",
            ):
                try:
                    loaded = json.load(uploaded_file)
                    if isinstance(loaded, dict):
                        _apply_uploaded_json(loaded)
                        st.success(T("json_loaded_success", _LANG))
                        st.rerun()
                    else:
                        st.error(T("invalid_json_format", _LANG))
                except json.JSONDecodeError:
                    st.error(T("invalid_json_file", _LANG))
                except Exception as exc:  # noqa: BLE001 — user-visible diagnostic
                    st.error(T("error_occurred", _LANG).format(error=exc))
        st.markdown("---")
        if st.button(
            T("reset_all_button", _LANG),
            use_container_width=True,
            key="en_reset",
        ):
            _apply_uploaded_json({})
            st.rerun()


def _editor_left(main_ticker: str) -> None:
    render_section_header(T("edit_strategy_header", _LANG))
    main_ticker_input = st.text_input(
        T("main_ticker_label", _LANG),
        value=main_ticker,
        key=f"{_LANG}_main_ticker",
        help=T("main_ticker_help", _LANG),
    ).upper()
    if not main_ticker_input:
        st.warning(T("please_enter_ticker_warning", _LANG))
        return
    _maybe_reseed_widgets(_LANG, main_ticker_input)
    current = st.session_state["strategies"].get(main_ticker_input, {})
    if current:
        st.caption(T("editing_existing", _LANG).format(ticker=main_ticker_input))
    else:
        st.caption(T("creating_new", _LANG).format(ticker=main_ticker_input))

    st.slider(
        T("portfolio_weight_label", _LANG),
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key=f"{_LANG}_weight_{main_ticker_input}",
    )

    active_side = render_buy_sell_selector(f"{_LANG}_active_side", _LANG)
    if active_side == "buy":
        input_strategy_details(
            f"{_LANG}_buy",
            allowed_qty_types=("count", "percent", "value", "split"),
            lang=_LANG,
        )
    else:
        input_strategy_details(
            f"{_LANG}_sell",
            allowed_qty_types=("count", "percent", "value"),
            lang=_LANG,
        )

    btn_label = (
        T("save_changes_button", _LANG) if current
        else T("add_strategy_button", _LANG)
    )
    st.button(
        btn_label,
        use_container_width=True,
        type="primary",
        key=f"en_save_{main_ticker_input}",
        on_click=_save_strategy_callback,
        args=(_LANG, main_ticker_input),
    )


def _editor_right() -> None:
    render_section_header(T("portfolio_composition_header", _LANG))
    strategies = st.session_state.get("strategies", {})
    total_weight = sum(s.get("portfolio_weight", 0.0) for s in strategies.values())
    count = len(strategies)
    if count == 0:
        st.info(T("json_empty_message", _LANG))
        return
    col_count, col_weight = st.columns(2)
    with col_count:
        render_metric(T("summary_strategies_label", _LANG), str(count), kind="accent")
    with col_weight:
        if abs(total_weight - 1.0) < 1e-6:
            kind = "up"
        elif total_weight > 1.0:
            kind = "down"
        else:
            kind = "accent"
        render_metric(
            T("summary_weight_label", _LANG),
            f"{total_weight:.2f} / 1.00",
            kind=kind,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    for ticker, strategy in strategies.items():
        render_strategy_card(ticker, strategy, _LANG)
    with st.expander(T("view_raw_json_expander", _LANG)):
        json_str = json.dumps(strategies, indent=4, ensure_ascii=False)
        st.code(json_str, language="json")
        st.download_button(
            label=T("download_json_button", _LANG),
            data=json_str,
            file_name="trading_strategies.json",
            mime="application/json",
            key="en_download",
        )


def _backtest_tab() -> None:
    render_section_header(T("backtest_header", _LANG))
    col_form, col_results = st.columns([0.4, 0.6])
    with col_form:
        with st.form("backtest_form_en"):
            c1, c2 = st.columns(2)
            start = c1.date_input(
                T("start_date_label", _LANG),
                value=pd.to_datetime("2023-01-01"),
            )
            end = c2.date_input(T("end_date_label", _LANG))
            initial_cash = st.number_input(T("initial_capital_label", _LANG), value=10000)
            run_button = st.form_submit_button(
                T("start_backtest_button", _LANG),
                use_container_width=True,
                type="primary",
            )
            if run_button:
                strategies_dict = st.session_state["strategies"]
                stocks = [
                    Stock(t, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                    for t in strategies_dict
                ]
                manager = StrategyManager(
                    "strategy", StrategyWrapper(**strategies_dict)
                )
                backtest = Backtest(stocks, [manager], initial_cash)
                backtest.run()
                st.session_state["backtest"] = backtest
        backtest = st.session_state.get("backtest")
        if backtest:
            with st.container(border=True):
                st.subheader(T("trade_history_header", _LANG))
                for ticker, trades in backtest.trades.items():
                    df = pd.DataFrame(trades)
                    if not df.empty:
                        df["value"] = df["quantity"] * df["price"]
                        st.dataframe(
                            df,
                            column_config={
                                "date": st.column_config.DateColumn("date"),
                                "price": st.column_config.NumberColumn(
                                    "price", format="$%.2f"
                                ),
                                "value": st.column_config.NumberColumn(
                                    "value", format="$%.2f"
                                ),
                            },
                        )
                st.dataframe(
                    {"CASH": backtest.portfolio.cash}
                    | backtest.portfolio.stock_count
                )
                st.subheader(T("portfolio_value_at_date_header", _LANG))
                slider_date = st.slider(
                    "date", start, end, end,
                    label_visibility="hidden",
                    key="en_slider",
                )
                value_at = backtest.get_protfolio_value(slider_date.strftime("%Y-%m-%d"))
                st.markdown(
                    T("value_at_date_template", _LANG).format(
                        date=slider_date, value=value_at,
                    )
                )
                st.subheader(T("monthly_snapshot_header", _LANG))
                monthly_df = backtest.get_monthly_snapshots()
                if not monthly_df.empty:
                    st.dataframe(monthly_df.style.format("{:,.2f}"))
                else:
                    st.info(T("no_monthly_data_message", _LANG))
    with col_results:
        backtest = st.session_state.get("backtest")
        if backtest:
            final_value = backtest.get_protfolio_value(end.strftime("%Y-%m-%d"))
            profit_rate = final_value / initial_cash
            trade_count = sum(len(t) for t in backtest.trades.values())
            ticker_count = len(backtest.trades)

            cols = st.columns(4)
            with cols[0]:
                render_metric(
                    T("final_value_metric", _LANG),
                    f"${final_value:,.2f}",
                    kind="accent",
                )
            with cols[1]:
                render_metric(
                    T("profit_rate_metric", _LANG),
                    f"{profit_rate:.3f}x",
                    kind="up" if profit_rate >= 1.0 else "down",
                )
            with cols[2]:
                render_metric(T("trade_count_metric", _LANG), str(trade_count))
            with cols[3]:
                render_metric(T("ticker_count_metric", _LANG), str(ticker_count))
            st.pyplot(backtest.plot_performance(instance_show=False))


def show_english_page() -> None:
    inject_global_styles()
    _sidebar()
    render_page_header(T("page_title", _LANG))

    tab1, tab2 = st.tabs([T("edit_strategy_tab", _LANG), T("backtest_tab", _LANG)])
    with tab1:
        left, right = st.columns([1.2, 1])
        with left:
            current_ticker = st.session_state.get(f"{_LANG}_main_ticker", "AAPL")
            _editor_left(current_ticker)
        with right:
            _editor_right()
    with tab2:
        _backtest_tab()
