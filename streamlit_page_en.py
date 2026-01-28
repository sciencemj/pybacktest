import streamlit as st
from pybacktest.backtest import Backtest
from pybacktest.models import Stock
from pybacktest.strategy import StrategyManager, StrategyWrapper
import json
import pandas as pd

def show_english_page():
    # Page basic settings
    st.title("📈 Automated Trading Strategy")
    st.markdown("---")

    # -----------------------------------------------------------------------------
    # Sidebar: Data Management (Upload and Reset)
    # -----------------------------------------------------------------------------
    with st.sidebar:
        # 1. File Uploader
        uploaded_file = st.file_uploader("Load JSON Configuration File", type=["json"], key="en_upload")

        if uploaded_file is not None:
            # Show button after file upload to prevent accidental overwrites
            if st.button("Apply Data", type="primary", use_container_width=True, key="en_apply"):
                try:
                    loaded_data = json.load(uploaded_file)
                    if isinstance(loaded_data, dict):
                        st.session_state['strategies'] = loaded_data
                        st.success("JSON file loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid JSON format. (Root must be a dictionary)")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        st.markdown("---")

        # 2. Reset Button
        if st.button("🗑️ Reset All", use_container_width=True, key="en_reset"):
            st.session_state['strategies'] = {}
            st.rerun()

    # -----------------------------------------------------------------------------
    # Main input form function (use saved data as defaults if available)
    # -----------------------------------------------------------------------------
    def input_strategy_details(key_prefix, default_ticker, saved_data=None):
        """
        Function to input buy/sell details.
        If saved_data exists, use its values as the form's defaults.
        """
        # Default value logic (use defaults if no saved data)
        def_ticker = saved_data.get('ticker', default_ticker) if saved_data else default_ticker

        # Extract 'By' settings
        saved_by = saved_data.get('by', ["current", "Change_Pct"]) if saved_data else ["current", "Change_Pct"]
        def_by_agg = saved_by[0]
        def_by_field = saved_by[1]

        # Extract 'Period' settings
        saved_period = saved_data.get('period', False) if saved_data else False
        def_use_period = isinstance(saved_period, (int, float)) and saved_period is not False
        def_period_val = saved_period if def_use_period else 3

        # Extract 'Criteria' settings
        saved_crit = saved_data.get('criteria', ["percent-change", -0.5]) if saved_data else ["percent-change", -0.5]
        def_crit_type = saved_crit[0]
        def_crit_val = float(saved_crit[1])

        # Extract 'Quantity' settings
        saved_qty = saved_data.get('quantity', ["count", 10]) if saved_data else ["count", 10]
        def_qty_type = saved_qty[0]
        def_qty_val = float(saved_qty[1])

        # Extract 'Trade_as' settings
        def_trade_as = saved_data.get('trade_as', "Close") if saved_data else "Close"

        # --- UI Configuration ---
        col1, col2 = st.columns(2)
        with col1:
            target_ticker = st.text_input("Target Ticker", value=def_ticker, key=f"{key_prefix}_ticker")

            st.caption("Base (By)")
            c1, c2 = st.columns(2)
            # Find index for selectbox default value
            opts_agg = ["current", "average"]
            opts_field = ["Close", "Change_Pct", "Change", "Open", "High", "Low"]
            opts_trade_as = ["Close", "Open", "High", "Low"]

            idx_agg = opts_agg.index(def_by_agg) if def_by_agg in opts_agg else 0
            idx_field = opts_field.index(def_by_field) if def_by_field in opts_field else 0
            idx_trade_as = opts_trade_as.index(def_trade_as) if def_trade_as in opts_trade_as else 0

            by_agg = c1.selectbox("Aggregation Method", opts_agg, index=idx_agg, key=f"{key_prefix}_by_agg")
            by_field = c2.selectbox("Field", opts_field, index=idx_field, key=f"{key_prefix}_by_field")
            trade_as = st.selectbox("Purchase Price Basis", opts_trade_as, index=idx_trade_as, key=f"{key_prefix}_trade_as")

        with col2:
            st.caption("Period")
            use_period = st.checkbox("Use Period Setting", value=def_use_period, key=f"{key_prefix}_use_period")
            if use_period:
                period_val = st.number_input("Period (days)", min_value=1, value=int(def_period_val), step=1, key=f"{key_prefix}_period_val")
                period_final = int(period_val)
            else:
                period_final = False

        col3, col4 = st.columns(2)
        with col3:
            st.caption("Criteria")
            c3, c4 = st.columns(2)
            opts_crit = ["percent-change", "profit-rate", "point", "value"]
            idx_crit = opts_crit.index(def_crit_type) if def_crit_type in opts_crit else 0

            crit_type = c3.selectbox("Criteria Type", opts_crit, index=idx_crit, key=f"{key_prefix}_crit_type")
            crit_val = c4.number_input("Criteria Value", value=def_crit_val, step=0.1, format="%.2f", key=f"{key_prefix}_crit_val")

        with col4:
            st.caption("Order Quantity")
            c5, c6 = st.columns(2)
            opts_qty = ["count", "percent", "value"]
            idx_qty = opts_qty.index(def_qty_type) if def_qty_type in opts_qty else 0

            qty_type = c5.selectbox("Unit", opts_qty, index=idx_qty, key=f"{key_prefix}_qty_type")
            qty_val = c6.number_input("Quantity Value", value=def_qty_val, step=1.0, key=f"{key_prefix}_qty_val")
            if qty_type == "count":
                qty_val = int(qty_val)

        return {
            "ticker": target_ticker,
            "by": [by_agg, by_field],
            "period": period_final,
            "criteria": [crit_type, crit_val],
            "quantity": [qty_type, qty_val],
            "trade_as": trade_as
        }

    # -----------------------------------------------------------------------------
    # Main Layout
    # -----------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["Edit Strategy", "Backtest"])
    with tab1:
        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            st.subheader("📝 Edit Strategy")

            # Main Ticker input
            main_ticker = st.text_input("Main Ticker (Enter ticker to edit)", value="AAPL", key="en_main_ticker").upper()

            if not main_ticker:
                st.warning("Please enter a Ticker.")
            else:
                # Check if data for the ticker exists in the current session (or uploaded file)
                current_data = st.session_state['strategies'].get(main_ticker, {})

                if current_data:
                    st.info(f"💾 Loaded existing data for **[{main_ticker}]**.")
                else:
                    st.caption(f"Creating a new strategy for **[{main_ticker}]**.")

                tab_buy, tab_sell = st.tabs(["🔵 Buy", "🔴 Sell"])

                # Pass saved_data to the form function to apply presets
                with tab_buy:
                    buy_strategy = input_strategy_details(
                        f"buy_{main_ticker}_en",
                        main_ticker,
                        saved_data=current_data.get('buy')
                    )

                with tab_sell:
                    sell_strategy = input_strategy_details(
                        f"sell_{main_ticker}_en",
                        main_ticker,
                        saved_data=current_data.get('sell')
                    )

                # Save/Update button
                btn_label = "💾 Save Changes" if current_data else "➕ Add Strategy"
                if st.button(btn_label, use_container_width=True, key=f"en_save_{main_ticker}"):
                    st.session_state['strategies'][main_ticker] = {
                        "buy": buy_strategy,
                        "sell": sell_strategy
                    }
                    st.success(f"[{main_ticker}] strategy has been updated!")
                    # Rerun to update the JSON view
                    st.rerun()

        with right_col:
            st.subheader("💻 Current JSON Data")

            if st.session_state['strategies']:
                json_str = json.dumps(st.session_state['strategies'], indent=4, ensure_ascii=False)
                st.code(json_str, language="json")

                st.download_button(
                    label="Download JSON File",
                    data=json_str,
                    file_name="trading_strategies.json",
                    mime="application/json",
                    key="en_download"
                )
            else:
                st.info("Data is empty. Add a strategy from the left or upload a JSON file.")
    with tab2:
        st.subheader("Backtest")
        col21, col22 = st.columns([0.5, 1])
        with col21:
            with st.form("backtest_form_en"):
                col31, col32 = st.columns([1,1])
                start = col31.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
                end = col32.date_input('End Date')
                initial_cash = st.number_input('Initial Capital', value=10000)
                run_button = st.form_submit_button('Start Backtest!', use_container_width=True)
                if run_button:
                    stocks = []
                    for ticker in st.session_state['strategies']:
                        stocks.append(Stock(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
                    strategy = StrategyManager("strategy", StrategyWrapper(**st.session_state['strategies']))
                    backtest = Backtest(stocks, [strategy], initial_cash)
                    backtest.run()
                    st.session_state['backtest'] = backtest
            if 'backtest' in st.session_state and st.session_state['backtest']:
                with st.container(border=True):
                    st.subheader("Trade History")
                    backtest = st.session_state['backtest']
                    for ticker in backtest.trades:
                        trade_data = pd.DataFrame(backtest.trades[ticker])
                        trade_data['value'] = trade_data['quantity'] * trade_data['price']
                        st.dataframe(trade_data, column_config={
                            "date": st.column_config.DateColumn("date"),
                            "price": st.column_config.NumberColumn("price", format="$%.2f"),
                            "value": st.column_config.NumberColumn("value", format="$%.2f")
                        })
                    st.dataframe({"CASH":backtest.portfolio.cash} | backtest.portfolio.stock_count)
                    st.subheader('Portfolio Value at a Specific Point in Time')
                    date = st.slider('date', start, end, end, label_visibility="hidden", key="en_slider")
                    st.markdown(f"Value at {date}: **${backtest.get_protfolio_value(date.strftime('%Y-%m-%d')):,.2f}**")
        with col22:
            if 'backtest' in st.session_state and st.session_state['backtest']:
                backtest: Backtest = st.session_state['backtest']
                final_value = backtest.get_protfolio_value(end.strftime('%Y-%m-%d'))
                profit_rate = (final_value / initial_cash)
                st.markdown(f"## Final Profit Rate: {profit_rate:.3f}")
                st.pyplot(backtest.plot_performance(instance_show=False))