"""Korean and English UI strings."""

from __future__ import annotations

LANGUAGES = {"en": "English", "ko": "한국어"}

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "page.title": "Pybacktest demo",
        "page.intro": (
            "Backtest a simple strategy on Yahoo Finance daily data with the "
            "deterministic Pybacktest 0.2 engine."
        ),
        "page.steps": (
            "1. Enter up to five tickers that share one currency.\n"
            "2. Pick a period and a strategy.\n"
            "3. Press **Run backtest**."
        ),
        "sidebar.language": "Language",
        "sidebar.tickers": "Tickers (comma-separated)",
        "sidebar.tickers_help": "Examples: AAPL, MSFT or 005930.KS, 000660.KS",
        "sidebar.start": "Start date",
        "sidebar.end": "End date",
        "sidebar.strategy": "Strategy",
        "sidebar.initial_cash": "Initial cash",
        "sidebar.initial_cash_help": (
            "In the tickers' currency. Cash is split equally, so it must cover "
            "at least one share of each ticker (KRW and JPY stocks need "
            "millions)."
        ),
        "sidebar.commission": "Commission per share",
        "sidebar.run": "Run backtest",
        "strategy.buy_and_hold": "Buy & Hold",
        "strategy.ma_cross": "Moving-average cross",
        "strategy.rsi": "RSI mean reversion",
        "param.fast": "Fast SMA window",
        "param.slow": "Slow SMA window",
        "param.period": "RSI period",
        "param.lower": "Buy below RSI",
        "param.upper": "Sell above RSI",
        "result.spinner": "Downloading data and running the engine…",
        "result.metrics": "Strategy vs. buy & hold",
        "result.equity": "Equity curve",
        "result.orders": "Orders",
        "result.fills": "Fills",
        "result.warnings": "Warnings",
        "result.info": "Run info",
        "result.none": "None",
        "column.strategy": "Strategy",
        "column.benchmark": "Buy & Hold",
        "metric.total_return": "Total return",
        "metric.cagr": "CAGR",
        "metric.volatility": "Volatility",
        "metric.sharpe": "Sharpe",
        "metric.sortino": "Sortino",
        "metric.maximum_drawdown": "Max drawdown",
        "metric.win_rate": "Win rate",
        "metric.turnover": "Turnover",
        "error.no_tickers": "Enter at least one ticker.",
        "error.too_many_tickers": "Enter at most {detail} tickers.",
        "error.empty_history": "No data was found for: {detail}.",
        "error.mixed_currency": (
            "All tickers must share one currency (found {detail})."
        ),
        "error.insufficient_history": (
            "Not enough bars for this strategy ({detail}). Choose a longer period."
        ),
        "error.invalid_period": "The start date must be before the end date.",
        "error.rate_limited": (
            "Yahoo Finance is rate-limiting requests. Please try again in a minute."
        ),
        "error.insufficient_cash": (
            "Initial cash is too small to buy one share per ticker "
            "(price > cash per ticker): {detail}. Increase the initial cash."
        ),
        "error.engine": "The engine rejected this configuration: {detail}",
        "error.unexpected": "Something unexpected went wrong.",
        "error.details": "Details",
    },
    "ko": {
        "page.title": "Pybacktest 데모",
        "page.intro": (
            "야후 파이낸스 일봉 데이터로 간단한 전략을 결정적(deterministic) "
            "Pybacktest 0.2 엔진에서 백테스트해 보세요."
        ),
        "page.steps": (
            "1. 같은 통화의 티커를 최대 5개까지 입력하세요.\n"
            "2. 기간과 전략을 고르세요.\n"
            "3. **백테스트 실행**을 누르세요."
        ),
        "sidebar.language": "언어",
        "sidebar.tickers": "티커 (쉼표로 구분)",
        "sidebar.tickers_help": "예: AAPL, MSFT 또는 005930.KS, 000660.KS",
        "sidebar.start": "시작일",
        "sidebar.end": "종료일",
        "sidebar.strategy": "전략",
        "sidebar.initial_cash": "초기 자금",
        "sidebar.initial_cash_help": (
            "티커의 통화 기준이에요. 자금은 종목별로 똑같이 나뉘므로 종목마다 "
            "최소 1주는 살 수 있어야 해요 (원화·엔화 종목은 수백만 단위가 필요해요)."
        ),
        "sidebar.commission": "주당 수수료",
        "sidebar.run": "백테스트 실행",
        "strategy.buy_and_hold": "매수 후 보유",
        "strategy.ma_cross": "이동평균 크로스",
        "strategy.rsi": "RSI 역추세",
        "param.fast": "단기 이동평균 기간",
        "param.slow": "장기 이동평균 기간",
        "param.period": "RSI 기간",
        "param.lower": "RSI 이하에서 매수",
        "param.upper": "RSI 이상에서 매도",
        "result.spinner": "데이터를 받고 엔진을 실행하는 중…",
        "result.metrics": "전략 vs 매수 후 보유",
        "result.equity": "자산 곡선",
        "result.orders": "주문",
        "result.fills": "체결",
        "result.warnings": "경고",
        "result.info": "실행 정보",
        "result.none": "없음",
        "column.strategy": "전략",
        "column.benchmark": "매수 후 보유",
        "metric.total_return": "총수익률",
        "metric.cagr": "연평균 수익률",
        "metric.volatility": "변동성",
        "metric.sharpe": "샤프 지수",
        "metric.sortino": "소르티노 지수",
        "metric.maximum_drawdown": "최대 낙폭",
        "metric.win_rate": "승률",
        "metric.turnover": "회전율",
        "error.no_tickers": "티커를 하나 이상 입력하세요.",
        "error.too_many_tickers": "티커는 최대 {detail}개까지 입력할 수 있어요.",
        "error.empty_history": "다음 티커의 데이터를 찾지 못했어요: {detail}.",
        "error.mixed_currency": ("모든 티커의 통화가 같아야 해요 (현재 {detail})."),
        "error.insufficient_history": (
            "이 전략에 필요한 봉 수가 부족해요 ({detail}). 기간을 늘려주세요."
        ),
        "error.invalid_period": "시작일은 종료일보다 앞서야 해요.",
        "error.rate_limited": (
            "야후 파이낸스가 요청을 제한하고 있어요. 잠시 후 다시 시도해 주세요."
        ),
        "error.insufficient_cash": (
            "초기 자금이 너무 적어서 종목마다 1주도 살 수 없어요 "
            "(가격 > 종목당 자금): {detail}. 초기 자금을 늘려주세요."
        ),
        "error.engine": "엔진이 이 설정을 거부했어요: {detail}",
        "error.unexpected": "예상치 못한 오류가 발생했어요.",
        "error.details": "상세 정보",
    },
}


def t(key: str, lang: str, **values: object) -> str:
    """Return the ``lang`` string for ``key`` with ``values`` interpolated."""
    text = STRINGS[lang][key]
    return text.format(**values) if values else text
