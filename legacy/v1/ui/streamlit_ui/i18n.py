"""Localized strings for the Streamlit pages.

LABELS["en"] and LABELS["ko"] MUST have identical key sets (enforced by
tests/test_streamlit_i18n.py).
"""

from __future__ import annotations

LABELS: dict[str, dict[str, str]] = {
    "en": {
        # Page chrome
        "page_title": "Automated Trading Strategy",
        "edit_strategy_tab": "Edit Strategy",
        "backtest_tab": "Backtest",
        # Sidebar
        "load_json_label": "Load JSON Configuration File",
        "apply_data_button": "Apply Data",
        "reset_all_button": "Reset All",
        "json_loaded_success": "JSON file loaded successfully!",
        "invalid_json_format": "Invalid JSON format. (Root must be a dictionary)",
        "invalid_json_file": "Invalid JSON file.",
        "error_occurred": "An error occurred: {error}",
        # Editor
        "edit_strategy_header": "Edit Strategy",
        "main_ticker_label": "Main Ticker",
        "main_ticker_help": "Enter ticker to edit",
        "editing_existing": "Editing existing data for [{ticker}]",
        "creating_new": "Creating a new strategy for [{ticker}]",
        "please_enter_ticker_warning": "Please enter a Ticker.",
        "portfolio_weight_label": "Target Portfolio Weight (for Rebalancing)",
        "buy_pill_label": "🔵 Buy",
        "sell_pill_label": "🔴 Sell",
        # Form fields
        "target_ticker_label": "Target Ticker",
        "base_group_label": "BASE",
        "aggregation_method_label": "Aggregation Method",
        "field_label": "Field",
        "price_point_label": "Purchase Price Basis",
        "period_group_label": "PERIOD",
        "use_period_label": "Use Period Setting",
        "period_days_label": "Period (days)",
        "criteria_group_label": "CRITERIA",
        "criteria_type_label": "Criteria Type",
        "criteria_value_label": "Criteria Value",
        "quantity_group_label": "QUANTITY",
        "quantity_unit_label": "Unit",
        "quantity_value_label": "Quantity Value",
        "save_changes_button": "💾 Save Changes",
        "add_strategy_button": "➕ Add Strategy",
        "strategy_updated_toast": "[{ticker}] strategy updated",
        # Right column
        "portfolio_composition_header": "Portfolio Composition",
        "summary_strategies_label": "STRATEGIES",
        "summary_weight_label": "TOTAL WEIGHT",
        "card_edit_button": "Edit",
        "card_delete_button": "Delete",
        "view_raw_json_expander": "View Raw JSON",
        "download_json_button": "Download JSON File",
        "filename_label": "Filename",
        "filename_placeholder": "trading_strategies.json",
        "json_empty_message": "Data is empty. Add a strategy from the left or upload a JSON file.",
        # Rule descriptions
        "describe_when": "When",
        "describe_buy_prefix": "Buy",
        "describe_sell_prefix": "Sell",
        "describe_at_price": "at",
        "describe_drops": "drops",
        "describe_rises": "rises",
        "describe_reaches": "reaches",
        "describe_current": "current",
        "describe_average": "average",
        "describe_shares": "shares",
        "describe_split_parts": "split into {parts} parts",
        # Backtest
        "backtest_header": "Backtest",
        "start_date_label": "Start Date",
        "end_date_label": "End Date",
        "initial_capital_label": "Initial Capital",
        "start_backtest_button": "Start Backtest!",
        "trade_history_header": "Trade History",
        "portfolio_value_at_date_header": "Portfolio Value at a Specific Point in Time",
        "monthly_snapshot_header": "Monthly Portfolio Snapshot",
        "value_at_date_template": "Value at {date}: **${value:,.2f}**",
        "no_monthly_data_message": "No monthly data available.",
        # Backtest metrics
        "final_value_metric": "FINAL VALUE",
        "profit_rate_metric": "PROFIT RATE",
        "trade_count_metric": "TRADES",
        "ticker_count_metric": "TICKERS",
        "final_profit_rate_label": "Final Profit Rate: {rate:.3f}",
    },
    "ko": {
        # Page chrome
        "page_title": "자동매매 전략",
        "edit_strategy_tab": "전략 편집",
        "backtest_tab": "백테스트",
        # Sidebar
        "load_json_label": "JSON 설정 파일 불러오기",
        # Existing: "데이터 적용하기" (streamlit_page_ko.py line 29)
        "apply_data_button": "데이터 적용하기",
        "reset_all_button": "전체 초기화",
        "json_loaded_success": "JSON 파일을 성공적으로 불러왔습니다!",
        # Existing: "JSON 형식이 올바르지 않습니다. (Root가 dict여야 함)"
        "invalid_json_format": "JSON 형식이 올바르지 않습니다. (Root가 dict여야 함)",
        # Existing: "유효하지 않은 JSON 파일입니다."
        "invalid_json_file": "유효하지 않은 JSON 파일입니다.",
        # Existing: "오류 발생: {error}" (f"오류 발생: {e}")
        "error_occurred": "오류 발생: {error}",
        # Editor
        "edit_strategy_header": "전략 편집",
        "main_ticker_label": "메인 Ticker",
        # Existing: "메인 Ticker (편집할 종목명 입력)"
        "main_ticker_help": "편집할 종목명 입력",
        # Existing info: "💾 기존에 저장된 **[{ticker}]** 데이터를 불러왔습니다."
        "editing_existing": "기존에 저장된 [{ticker}] 데이터를 불러왔습니다",
        # Existing caption: "새로운 **[{ticker}]** 전략을 생성합니다."
        "creating_new": "새로운 [{ticker}] 전략을 생성합니다",
        # Existing: "Ticker를 입력해주세요."
        "please_enter_ticker_warning": "Ticker를 입력해주세요.",
        "portfolio_weight_label": "목표 포트폴리오 비중 (리밸런싱용)",
        "buy_pill_label": "🔵 매수",
        "sell_pill_label": "🔴 매도",
        # Form fields
        "target_ticker_label": "대상 Ticker",
        # Existing caption: "기준 (By)"
        "base_group_label": "기준 (By)",
        "aggregation_method_label": "집계 방식",
        "field_label": "필드",
        # Existing: "구매가 기준"
        "price_point_label": "구매가 기준",
        # Existing caption: "기간 (Period)"
        "period_group_label": "기간 (Period)",
        "use_period_label": "기간 설정 사용",
        "period_days_label": "기간 (일)",
        # Existing caption: "조건 (Criteria)"
        "criteria_group_label": "조건 (Criteria)",
        # Existing: "조건 타입"
        "criteria_type_label": "조건 타입",
        "criteria_value_label": "조건 값",
        # Existing caption: "주문 수량 (Quantity)"
        "quantity_group_label": "주문 수량 (Quantity)",
        "quantity_unit_label": "단위",
        "quantity_value_label": "수량 값",
        # Existing: "💾 수정사항 저장"
        "save_changes_button": "💾 수정사항 저장",
        "add_strategy_button": "➕ 전략 추가",
        # Existing: "[{ticker}] 전략이 업데이트되었습니다!"
        "strategy_updated_toast": "[{ticker}] 전략이 업데이트되었습니다!",
        # Right column
        "portfolio_composition_header": "포트폴리오 구성",
        "summary_strategies_label": "전략 수",
        "summary_weight_label": "총 비중",
        "card_edit_button": "편집",
        "card_delete_button": "삭제",
        "view_raw_json_expander": "원본 JSON 보기",
        "download_json_button": "JSON 파일 다운로드",
        "filename_label": "파일명",
        "filename_placeholder": "trading_strategies.json",
        # Existing: "데이터가 비어있습니다. 왼쪽에서 추가하거나 JSON 파일을 업로드하세요."
        "json_empty_message": "데이터가 비어있습니다. 왼쪽에서 추가하거나 JSON 파일을 업로드하세요.",
        # Rule descriptions
        "describe_when": "조건:",
        "describe_buy_prefix": "매수",
        "describe_sell_prefix": "매도",
        "describe_at_price": "기준",
        "describe_drops": "하락",
        "describe_rises": "상승",
        "describe_reaches": "도달",
        "describe_current": "현재",
        "describe_average": "평균",
        "describe_shares": "주",
        "describe_split_parts": "{parts}분할",
        # Backtest
        "backtest_header": "백테스트",
        "start_date_label": "시작일",
        "end_date_label": "종료일",
        # Existing: "초기 자금"
        "initial_capital_label": "초기 자금",
        "start_backtest_button": "백테스트 시작!",
        "trade_history_header": "거래 기록",
        # Existing: "특점 시점에서의 포트폴리오 가치" (note: "특점" appears to be a typo for "특정" in the source)
        "portfolio_value_at_date_header": "특정 시점에서의 포트폴리오 가치",
        # Existing: "월별 포트폴리오 현황"
        "monthly_snapshot_header": "월별 포트폴리오 현황",
        "value_at_date_template": "{date} 가치: **${value:,.2f}**",
        # Existing: "월별 데이터가 없습니다."
        "no_monthly_data_message": "월별 데이터가 없습니다.",
        # Backtest metrics
        "final_value_metric": "최종 가치",
        "profit_rate_metric": "수익률",
        "trade_count_metric": "거래 수",
        "ticker_count_metric": "종목 수",
        # Existing: "## 최종 이익률: {profit_rate:.3f}" — extracted label portion
        "final_profit_rate_label": "최종 이익률: {rate:.3f}",
    },
}


def T(key: str, lang: str) -> str:
    """Return the label for ``key`` in ``lang``. Raises KeyError if missing."""
    return LABELS[lang][key]
