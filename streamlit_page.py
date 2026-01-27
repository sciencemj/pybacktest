import streamlit as st
from pybacktest.backtest import Backtest
from pybacktest.models import Stock
from pybacktest.strategy import StrategyManager, StrategyWrapper
import json
import pandas as pd

# 페이지 기본 설정
st.set_page_config(layout="wide", page_title="Trading Strategy Generator")

st.title("📈 주식 자동매매 전략")
st.markdown("---")

# 세션 상태 초기화
if 'strategies' not in st.session_state:
    st.session_state['strategies'] = {}
if 'backtest' not in st.session_state:
    st.session_state['backtest'] = None

# -----------------------------------------------------------------------------
# 사이드바: 데이터 관리 (업로드 및 초기화)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 파일 관리")
    
    # 1. 파일 업로더
    uploaded_file = st.file_uploader("JSON 설정 파일 불러오기", type=["json"])
    
    if uploaded_file is not None:
        # 파일이 업로드되면 버튼을 표시하여 의도치 않은 덮어쓰기 방지
        if st.button("데이터 적용하기", type="primary", use_container_width=True):
            try:
                loaded_data = json.load(uploaded_file)
                if isinstance(loaded_data, dict):
                    st.session_state['strategies'] = loaded_data
                    st.success("JSON 파일을 성공적으로 불러왔습니다!")
                    st.rerun()
                else:
                    st.error("JSON 형식이 올바르지 않습니다. (Root가 dict여야 함)")
            except json.JSONDecodeError:
                st.error("유효하지 않은 JSON 파일입니다.")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    st.markdown("---")
    
    # 2. 초기화 버튼
    if st.button("🗑️ 전체 초기화", use_container_width=True):
        st.session_state['strategies'] = {}
        st.rerun()

# -----------------------------------------------------------------------------
# 메인 입력 폼 함수 (저장된 데이터가 있으면 기본값으로 사용)
# -----------------------------------------------------------------------------
def input_strategy_details(key_prefix, default_ticker, saved_data=None):
    """
    매수/매도 세부 설정을 입력받는 함수.
    saved_data가 존재하면 해당 값을 폼의 기본값(value)으로 설정합니다.
    """
    # 기본값 설정 로직 (저장된 데이터가 없으면 기본값 사용)
    def_ticker = saved_data.get('ticker', default_ticker) if saved_data else default_ticker
    
    # By 설정 추출
    saved_by = saved_data.get('by', ["current", "Change_Pct"]) if saved_data else ["current", "Change_Pct"]
    def_by_agg = saved_by[0]
    def_by_field = saved_by[1]

    # Period 설정 추출
    saved_period = saved_data.get('period', False) if saved_data else False
    def_use_period = isinstance(saved_period, (int, float)) and saved_period is not False
    def_period_val = saved_period if def_use_period else 3

    # Criteria 설정 추출
    saved_crit = saved_data.get('criteria', ["percent-change", -0.5]) if saved_data else ["percent-change", -0.5]
    def_crit_type = saved_crit[0]
    def_crit_val = float(saved_crit[1])

    # Quantity 설정 추출
    saved_qty = saved_data.get('quantity', ["count", 10]) if saved_data else ["count", 10]
    def_qty_type = saved_qty[0]
    def_qty_val = float(saved_qty[1])

    # Trade_as 설정 추출
    def_trade_as = saved_data.get('trade_as', "Close") if saved_data else "Close"

    # --- UI 구성 ---
    col1, col2 = st.columns(2)
    with col1:
        target_ticker = st.text_input("대상 Ticker", value=def_ticker, key=f"{key_prefix}_ticker")
        
        st.caption("기준 (By)")
        c1, c2 = st.columns(2)
        # index 찾기 (selectbox 기본값 설정을 위해)
        opts_agg = ["current", "average"]
        opts_field = ["Close", "Change_Pct", "Change", "Open", "High", "Low"]
        opts_trade_as = ["Close", "Open", "High", "Low"]
        
        idx_agg = opts_agg.index(def_by_agg) if def_by_agg in opts_agg else 0
        idx_field = opts_field.index(def_by_field) if def_by_field in opts_field else 0
        idx_trade_as = opts_trade_as.index(def_trade_as) if def_trade_as in opts_trade_as else 0
        
        by_agg = c1.selectbox("집계 방식", opts_agg, index=idx_agg, key=f"{key_prefix}_by_agg")
        by_field = c2.selectbox("필드", opts_field, index=idx_field, key=f"{key_prefix}_by_field")
        trade_as = st.selectbox("구매가 기준", opts_trade_as, index=idx_trade_as, key=f"{key_prefix}_trade_as")
        
    with col2:
        st.caption("기간 (Period)")
        use_period = st.checkbox("기간 설정 사용", value=def_use_period, key=f"{key_prefix}_use_period")
        if use_period:
            period_val = st.number_input("기간 (일)", min_value=1, value=int(def_period_val), step=1, key=f"{key_prefix}_period_val")
            period_final = int(period_val)
        else:
            period_final = False

    col3, col4 = st.columns(2)
    with col3:
        st.caption("조건 (Criteria)")
        c3, c4 = st.columns(2)
        opts_crit = ["percent-change", "profit-rate", "point", "value"]
        idx_crit = opts_crit.index(def_crit_type) if def_crit_type in opts_crit else 0
        
        crit_type = c3.selectbox("조건 타입", opts_crit, index=idx_crit, key=f"{key_prefix}_crit_type")
        crit_val = c4.number_input("조건 값", value=def_crit_val, step=0.1, format="%.2f", key=f"{key_prefix}_crit_val")

    with col4:
        st.caption("주문 수량 (Quantity)")
        c5, c6 = st.columns(2)
        opts_qty = ["count", "percent", "value"]
        idx_qty = opts_qty.index(def_qty_type) if def_qty_type in opts_qty else 0
        
        qty_type = c5.selectbox("단위", opts_qty, index=idx_qty, key=f"{key_prefix}_qty_type")
        qty_val = c6.number_input("수량 값", value=def_qty_val, step=1.0, key=f"{key_prefix}_qty_val")
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
# 메인 레이아웃
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["전략 편집", "백테스트"])
with tab1:
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("📝 전략 편집")
        
        # 메인 Ticker 입력
        main_ticker = st.text_input("메인 Ticker (편집할 종목명 입력)", value="AAPL").upper()
        
        if not main_ticker:
            st.warning("Ticker를 입력해주세요.")
        else:
            # 현재 세션(또는 업로드된 파일)에 해당 Ticker 데이터가 있는지 확인
            current_data = st.session_state['strategies'].get(main_ticker, {})
            
            if current_data:
                st.info(f"💾 기존에 저장된 **[{main_ticker}]** 데이터를 불러왔습니다.")
            else:
                st.caption(f"새로운 **[{main_ticker}]** 전략을 생성합니다.")

            tab_buy, tab_sell = st.tabs(["🔵 매수 (Buy)", "🔴 매도 (Sell)"])
            
            # 저장된 데이터(saved_data)를 폼 함수에 전달하여 프리셋 적용
            with tab_buy:
                buy_strategy = input_strategy_details(
                    f"buy_{main_ticker}", 
                    main_ticker, 
                    saved_data=current_data.get('buy')
                )
                
            with tab_sell:
                sell_strategy = input_strategy_details(
                    f"sell_{main_ticker}", 
                    main_ticker,
                    saved_data=current_data.get('sell')
                )

            # 저장/수정 버튼
            btn_label = "💾 수정사항 저장" if current_data else "➕ 전략 추가"
            if st.button(btn_label, use_container_width=True):
                st.session_state['strategies'][main_ticker] = {
                    "buy": buy_strategy,
                    "sell": sell_strategy
                }
                st.success(f"[{main_ticker}] 전략이 업데이트되었습니다!")
                # JSON 뷰 갱신을 위해 rerun
                st.rerun()

    with right_col:
        st.subheader("💻 현재 JSON 데이터")
        
        if st.session_state['strategies']:
            json_str = json.dumps(st.session_state['strategies'], indent=4, ensure_ascii=False)
            st.code(json_str, language="json")
            
            st.download_button(
                label="JSON 파일 다운로드",
                data=json_str,
                file_name="trading_strategies.json",
                mime="application/json"
            )
        else:
            st.info("데이터가 비어있습니다. 왼쪽에서 추가하거나 JSON 파일을 업로드하세요.")
with tab2:
    st.subheader("백테스트")
    col21, col22 = st.columns([0.5, 1])
    with col21:
        with st.form("backtest_form"):
            col31, col32 = st.columns([1,1])
            start = col31.date_input('시작일', value='2025-01-01')
            end = col32.date_input('종료일')
            initial_cash = st.number_input('초기 자금', value=10000)
            run_button = st.form_submit_button('백테스트 시작!', use_container_width=True)
            if run_button:
                stocks = []
                for ticker in st.session_state['strategies']:
                    stocks.append(Stock(ticker, start, end))
                strategy = StrategyManager("strategy", StrategyWrapper(**st.session_state['strategies']))
                backtest = Backtest(stocks, [strategy], initial_cash)
                backtest.run()
                st.session_state['backtest'] = backtest
        if st.session_state['backtest']:
            with st.container(border=True):
                st.subheader("거래 기록")
                backtest = st.session_state['backtest']
                for ticker in backtest.trades:
                    trade_data = pd.DataFrame(backtest.trades[ticker])
                    trade_data['value'] = trade_data['quantity'] * trade_data['price']
                    st.dataframe(trade_data, column_config={
                        "date": st.column_config.DateColumn("date"),
                        "price": st.column_config.NumberColumn("price", format="$%d"),
                        "value": st.column_config.NumberColumn("value", format="$%d")
                    })
                st.dataframe({"CASH":backtest.portfolio.cash} | backtest.portfolio.stock_count)
                st.subheader('특점 시점에서의 포트폴리오 가치')
                date = st.slider('', start, end, end, label_visibility= "hidden")
                st.markdown(f"Value at {date}: **{backtest.get_protfolio_value(date):0.0f}**")
    with col22:
        if st.session_state['backtest']:
            backtest: Backtest = st.session_state['backtest']
            st.markdown(f"## 최종 이익률: {backtest.get_protfolio_value(end)/initial_cash:0.3f}")
            st.pyplot(backtest.plot_performance(instance_show=False))