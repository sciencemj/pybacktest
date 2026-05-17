# Streamlit Binance Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Streamlit "Save Changes doesn't update JSON view" bug (symptom 1) and reskin the strategy editor and backtest pages with a Binance-inspired dark theme, replacing the raw JSON viewer with strategy summary cards.

**Architecture:** Extract shared UI helpers into a new `streamlit_ui/` package (theme, components, forms, i18n). Page files (`streamlit_page_en.py`, `streamlit_page_ko.py`) become thin localization wrappers that import the shared logic. Bug fix lives in `forms.py`: switch the Save button to an `on_click` callback, drop ticker from widget keys, purge stale widget state on JSON upload, always rebuild the strategy dict.

**Tech Stack:** Streamlit ≥ 1.50, pandas, pydantic v2, pytest, uv (package manager). CSS injected via `st.markdown(unsafe_allow_html=True)`. Inter and JetBrains Mono loaded via Google Fonts `@import`.

**Spec:** `docs/superpowers/specs/2026-05-17-streamlit-binance-redesign-design.md`

---

## File Structure

**Files to create:**

| Path | Responsibility |
|---|---|
| `.streamlit/config.toml` | Base dark theme + Binance Yellow primary color |
| `streamlit_ui/__init__.py` | Package marker, public API re-exports |
| `streamlit_ui/theme.py` | CSS string + `inject_global_styles()`; color/spacing/radius constants |
| `streamlit_ui/i18n.py` | `LABELS` dict (en + ko) + `T(key, lang)` helper |
| `streamlit_ui/forms.py` | `_collect_form_dict`, `_apply_uploaded_json`, `_extract_defaults`, `_maybe_reseed_widgets`, `input_strategy_details`, `_save_strategy_callback` |
| `streamlit_ui/components.py` | `_describe_rule`, `render_section_header`, `render_metric`, `render_buy_sell_selector`, `render_strategy_card` |
| `tests/test_streamlit_theme.py` | Theme module smoke test |
| `tests/test_streamlit_i18n.py` | EN/KO key parity test |
| `tests/test_streamlit_forms.py` | Form state-machine tests |
| `tests/test_streamlit_components.py` | `_describe_rule` cases |

**Files to modify:**

| Path | Change |
|---|---|
| `streamlit_page_en.py` | Rewrite as thin wrapper using `streamlit_ui` modules; pass `lang="en"` |
| `streamlit_page_ko.py` | Rewrite as thin wrapper using `streamlit_ui` modules; pass `lang="ko"` |
| `streamlit_page.py` | No change (entry router stays the same) |
| `.gitignore` | Add `.streamlit/secrets.toml` if not already covered (defensive) |

---

## Task 1: Scaffolding + Theme Module

**Files:**
- Create: `.streamlit/config.toml`
- Create: `streamlit_ui/__init__.py`
- Create: `streamlit_ui/theme.py`
- Create: `tests/test_streamlit_theme.py`

- [ ] **Step 1: Create the package directory and config**

```bash
mkdir -p streamlit_ui
mkdir -p .streamlit
```

- [ ] **Step 2: Write the failing test**

`tests/test_streamlit_theme.py`:
```python
"""Tests for streamlit_ui.theme module."""

from streamlit_ui import theme


def test_inject_global_styles_is_callable():
    assert callable(theme.inject_global_styles)


def test_css_block_contains_binance_yellow():
    css = theme.CSS_BLOCK
    assert "#FCD535" in css, "Binance Yellow primary color must appear in CSS"


def test_css_block_contains_trading_colors():
    css = theme.CSS_BLOCK
    assert "#0ecb81" in css, "Trading-up green must appear in CSS"
    assert "#f6465d" in css, "Trading-down red must appear in CSS"


def test_css_block_loads_inter_and_jetbrains_mono():
    css = theme.CSS_BLOCK
    assert "Inter" in css
    assert "JetBrains+Mono" in css or "JetBrains Mono" in css


def test_color_constants_exposed():
    assert theme.PRIMARY == "#FCD535"
    assert theme.CANVAS_DARK == "#0b0e11"
    assert theme.SURFACE_CARD_DARK == "#1e2329"
    assert theme.TRADING_UP == "#0ecb81"
    assert theme.TRADING_DOWN == "#f6465d"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_streamlit_theme.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'streamlit_ui'".

- [ ] **Step 4: Create `streamlit_ui/__init__.py`**

```python
"""Streamlit UI helpers for the pybacktest strategy editor and backtest page."""

from streamlit_ui.theme import inject_global_styles

__all__ = ["inject_global_styles"]
```

- [ ] **Step 5: Create `.streamlit/config.toml`**

```toml
[theme]
base = "dark"
primaryColor = "#FCD535"
backgroundColor = "#0b0e11"
secondaryBackgroundColor = "#1e2329"
textColor = "#eaecef"
font = "sans serif"
```

- [ ] **Step 6: Create `streamlit_ui/theme.py`**

```python
"""Binance-inspired visual theme for the pybacktest Streamlit pages.

Exports color/spacing/radius constants and an ``inject_global_styles``
function that emits one ``<style>`` block per page render via
``st.markdown(unsafe_allow_html=True)``.
"""

from __future__ import annotations

import streamlit as st

# Color tokens (Binance design system)
PRIMARY = "#FCD535"
PRIMARY_ACTIVE = "#f0b90b"
CANVAS_DARK = "#0b0e11"
SURFACE_CARD_DARK = "#1e2329"
SURFACE_ELEVATED_DARK = "#2b3139"
HAIRLINE_DARK = "#2b3139"
BODY_ON_DARK = "#eaecef"
MUTED = "#707a8a"
TRADING_UP = "#0ecb81"
TRADING_DOWN = "#f6465d"
ON_PRIMARY = "#181a20"

# Radius tokens
RADIUS_SM = "4px"
RADIUS_MD = "6px"
RADIUS_LG = "8px"
RADIUS_XL = "12px"
RADIUS_PILL = "9999px"


CSS_BLOCK = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
  --bn-primary: #FCD535;
  --bn-primary-active: #f0b90b;
  --bn-canvas: #0b0e11;
  --bn-surface-card: #1e2329;
  --bn-surface-elevated: #2b3139;
  --bn-hairline: #2b3139;
  --bn-body: #eaecef;
  --bn-muted: #707a8a;
  --bn-trading-up: #0ecb81;
  --bn-trading-down: #f6465d;
  --bn-on-primary: #181a20;
}

html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.bn-num, [data-bn-num] {
  font-family: 'JetBrains Mono', monospace;
  font-feature-settings: 'tnum';
}

/* Primary buttons */
.stButton > button {
  background: var(--bn-primary);
  color: var(--bn-on-primary);
  border: none;
  border-radius: 6px;
  font-weight: 600;
  font-family: 'Inter', sans-serif;
}
.stButton > button:hover {
  background: var(--bn-primary-active);
  color: var(--bn-on-primary);
}
.stButton > button:disabled {
  background: #3a3a1f;
  color: var(--bn-muted);
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox > div > div,
.stDateInput input {
  background: var(--bn-surface-card) !important;
  border: 1px solid var(--bn-hairline) !important;
  border-radius: 8px !important;
  color: var(--bn-body) !important;
}

/* Section header with yellow accent rule */
.bn-section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 16px 0 8px 0;
}
.bn-section-header::before {
  content: '';
  display: inline-block;
  width: 4px;
  height: 24px;
  background: var(--bn-primary);
  border-radius: 2px;
}
.bn-section-header h2 {
  font-size: 24px;
  font-weight: 600;
  color: var(--bn-body);
  margin: 0;
}

/* Strategy card */
.bn-strategy-card {
  background: var(--bn-surface-card);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 16px;
  border: 1px solid var(--bn-hairline);
}
.bn-strategy-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.bn-strategy-card-ticker {
  font-size: 20px;
  font-weight: 600;
  color: var(--bn-body);
}
.bn-strategy-card-weight {
  background: var(--bn-surface-elevated);
  color: var(--bn-primary);
  padding: 4px 12px;
  border-radius: 9999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 600;
}
.bn-side-block {
  border-left: 3px solid;
  padding-left: 12px;
  margin: 12px 0;
}
.bn-side-block.buy { border-color: var(--bn-trading-up); }
.bn-side-block.sell { border-color: var(--bn-trading-down); }
.bn-side-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
}
.bn-side-label.buy { color: var(--bn-trading-up); }
.bn-side-label.sell { color: var(--bn-trading-down); }
.bn-side-condition,
.bn-side-action {
  font-size: 14px;
  color: var(--bn-body);
  line-height: 1.5;
}

/* Metric callout */
.bn-metric {
  background: var(--bn-surface-card);
  border-radius: 12px;
  padding: 20px 24px;
  border: 1px solid var(--bn-hairline);
}
.bn-metric-label {
  font-size: 12px;
  color: var(--bn-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}
.bn-metric-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 32px;
  font-weight: 700;
  font-feature-settings: 'tnum';
  line-height: 1.1;
}
.bn-metric-value.up { color: var(--bn-trading-up); }
.bn-metric-value.down { color: var(--bn-trading-down); }
.bn-metric-value.neutral { color: var(--bn-body); }
.bn-metric-value.accent { color: var(--bn-primary); }

/* Page header band */
.bn-page-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 0;
  border-bottom: 1px solid var(--bn-hairline);
  margin-bottom: 32px;
}
.bn-page-header-title {
  font-size: 40px;
  font-weight: 700;
  color: var(--bn-body);
  margin: 0;
}
</style>
"""


def inject_global_styles() -> None:
    """Emit the page-level CSS block. Safe to call multiple times per session."""
    st.markdown(CSS_BLOCK, unsafe_allow_html=True)
```

- [ ] **Step 7: Run test to verify it passes**

```bash
uv run pytest tests/test_streamlit_theme.py -v
```

Expected: 5 passed.

- [ ] **Step 8: Run full test suite to confirm no regressions**

```bash
uv run pytest -q
```

Expected: all existing tests pass + 5 new pass.

- [ ] **Step 9: Commit**

```bash
git add streamlit_ui/__init__.py streamlit_ui/theme.py .streamlit/config.toml tests/test_streamlit_theme.py
git commit -m "feat: add binance-inspired theme module and config"
```

---

## Task 2: i18n Module

**Files:**
- Create: `streamlit_ui/i18n.py`
- Create: `tests/test_streamlit_i18n.py`

- [ ] **Step 1: Write the failing test**

`tests/test_streamlit_i18n.py`:
```python
"""Tests for streamlit_ui.i18n module."""

import pytest

from streamlit_ui import i18n


def test_languages_present():
    assert "en" in i18n.LABELS
    assert "ko" in i18n.LABELS


def test_languages_have_same_keys():
    en_keys = set(i18n.LABELS["en"].keys())
    ko_keys = set(i18n.LABELS["ko"].keys())
    missing_in_ko = en_keys - ko_keys
    missing_in_en = ko_keys - en_keys
    assert not missing_in_ko, f"Korean missing keys: {missing_in_ko}"
    assert not missing_in_en, f"English missing keys: {missing_in_en}"


def test_t_returns_english_string():
    assert i18n.T("page_title", "en") == "Automated Trading Strategy"


def test_t_returns_korean_string():
    # Korean is a non-ASCII string; just confirm it's present and non-empty.
    value = i18n.T("page_title", "ko")
    assert isinstance(value, str)
    assert len(value) > 0


def test_t_raises_on_missing_key():
    with pytest.raises(KeyError):
        i18n.T("nonexistent_key_xyz", "en")


def test_t_raises_on_missing_lang():
    with pytest.raises(KeyError):
        i18n.T("page_title", "fr")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_streamlit_i18n.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'streamlit_ui.i18n'".

- [ ] **Step 3: Read existing Korean translations from streamlit_page_ko.py**

```bash
grep -E '(st\.title|st\.subheader|st\.button|st\.text_input|st\.caption|st\.markdown).*"' streamlit_page_ko.py | head -50
```

Use the output to populate Korean strings for existing UI elements. For NEW card-UI strings (Portfolio Composition, Editing existing, Creating new, Edit, Delete, View Raw JSON, etc.), use the placeholders below and ask the user for confirmation in Task 9.

- [ ] **Step 4: Create `streamlit_ui/i18n.py`**

```python
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
        "apply_data_button": "데이터 적용",
        "reset_all_button": "전체 초기화",
        "json_loaded_success": "JSON 파일을 성공적으로 불러왔습니다!",
        "invalid_json_format": "잘못된 JSON 형식입니다. (루트는 dict 이어야 합니다)",
        "invalid_json_file": "잘못된 JSON 파일입니다.",
        "error_occurred": "오류가 발생했습니다: {error}",
        # Editor
        "edit_strategy_header": "전략 편집",
        "main_ticker_label": "메인 티커",
        "main_ticker_help": "편집할 티커 입력",
        "editing_existing": "[{ticker}] 기존 데이터 편집 중",
        "creating_new": "[{ticker}] 신규 전략 작성 중",
        "please_enter_ticker_warning": "티커를 입력해 주세요.",
        "portfolio_weight_label": "목표 포트폴리오 비중 (리밸런싱용)",
        "buy_pill_label": "🔵 매수",
        "sell_pill_label": "🔴 매도",
        # Form fields
        "target_ticker_label": "대상 티커",
        "base_group_label": "기준",
        "aggregation_method_label": "집계 방식",
        "field_label": "필드",
        "price_point_label": "매수 기준 가격",
        "period_group_label": "기간",
        "use_period_label": "기간 설정 사용",
        "period_days_label": "기간 (일)",
        "criteria_group_label": "조건",
        "criteria_type_label": "조건 유형",
        "criteria_value_label": "조건 값",
        "quantity_group_label": "수량",
        "quantity_unit_label": "단위",
        "quantity_value_label": "수량 값",
        "save_changes_button": "💾 변경사항 저장",
        "add_strategy_button": "➕ 전략 추가",
        "strategy_updated_toast": "[{ticker}] 전략이 업데이트되었습니다",
        # Right column
        "portfolio_composition_header": "포트폴리오 구성",
        "summary_strategies_label": "전략 수",
        "summary_weight_label": "총 비중",
        "card_edit_button": "편집",
        "card_delete_button": "삭제",
        "view_raw_json_expander": "원본 JSON 보기",
        "download_json_button": "JSON 파일 다운로드",
        "json_empty_message": "데이터가 비어 있습니다. 왼쪽에서 전략을 추가하거나 JSON 파일을 업로드하세요.",
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
        "initial_capital_label": "초기 자본",
        "start_backtest_button": "백테스트 시작!",
        "trade_history_header": "거래 기록",
        "portfolio_value_at_date_header": "특정 시점의 포트폴리오 가치",
        "monthly_snapshot_header": "월간 포트폴리오 스냅샷",
        "value_at_date_template": "{date} 가치: **${value:,.2f}**",
        "no_monthly_data_message": "월간 데이터가 없습니다.",
        # Backtest metrics
        "final_value_metric": "최종 가치",
        "profit_rate_metric": "수익률",
        "trade_count_metric": "거래 수",
        "ticker_count_metric": "종목 수",
        "final_profit_rate_label": "최종 수익률: {rate:.3f}",
    },
}


def T(key: str, lang: str) -> str:
    """Return the label for ``key`` in ``lang``. Raises KeyError if missing."""
    return LABELS[lang][key]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_streamlit_i18n.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add streamlit_ui/i18n.py tests/test_streamlit_i18n.py
git commit -m "feat: add bilingual labels module with parity test"
```

---

## Task 3: forms.py — Pure Functions

**Files:**
- Create: `streamlit_ui/forms.py`
- Create: `tests/test_streamlit_forms.py`

- [ ] **Step 1: Write the failing test**

`tests/test_streamlit_forms.py`:
```python
"""Tests for streamlit_ui.forms pure functions."""

import pytest

from pybacktest.strategy import StrategyWrapper
from streamlit_ui.forms import (
    _apply_uploaded_json,
    _collect_form_dict,
    _extract_defaults,
)


def _seed_state(prefix: str, **overrides) -> dict:
    """Build a fake widget-state dict with sensible defaults."""
    state = {
        f"{prefix}_ticker": "AAPL",
        f"{prefix}_by_agg": "current",
        f"{prefix}_by_field": "Close",
        f"{prefix}_trade_as": "Close",
        f"{prefix}_use_period": False,
        f"{prefix}_period_val": 3,
        f"{prefix}_crit_type": "percent-change",
        f"{prefix}_crit_val": -0.5,
        f"{prefix}_qty_type": "count",
        f"{prefix}_qty_val": 10.0,
    }
    state.update(overrides)
    return state


def test_collect_form_dict_assembles_schema():
    state = _seed_state("en_buy")
    result = _collect_form_dict("en_buy", ("count", "percent", "value"), state=state)
    assert result == {
        "ticker": "AAPL",
        "indicator": ["current", "Close"],
        "window": False,
        "threshold": ["percent-change", -0.5],
        "quantity": ["count", 10],
        "price_point": "Close",
    }


def test_collect_form_dict_coerces_count_to_int():
    state = _seed_state("en_buy", **{
        "en_buy_qty_type": "count",
        "en_buy_qty_val": 10.0,
    })
    result = _collect_form_dict("en_buy", ("count", "percent"), state=state)
    assert result["quantity"] == ["count", 10]
    assert isinstance(result["quantity"][1], int)


def test_collect_form_dict_coerces_split_to_int():
    state = _seed_state("en_buy", **{
        "en_buy_qty_type": "split",
        "en_buy_qty_val": 4.0,
    })
    result = _collect_form_dict("en_buy", ("count", "split"), state=state)
    assert result["quantity"] == ["split", 4]
    assert isinstance(result["quantity"][1], int)


def test_collect_form_dict_period_disabled_returns_false():
    state = _seed_state("en_buy", **{"en_buy_use_period": False})
    result = _collect_form_dict("en_buy", ("count",), state=state)
    assert result["window"] is False


def test_collect_form_dict_period_enabled_returns_int():
    state = _seed_state("en_buy", **{
        "en_buy_use_period": True,
        "en_buy_period_val": 7,
    })
    result = _collect_form_dict("en_buy", ("count",), state=state)
    assert result["window"] == 7
    assert isinstance(result["window"], int)


def test_apply_uploaded_json_clears_widget_state():
    state = {
        "en_buy_ticker": "OLD",
        "en_buy_crit_val": 99.0,
        "en_sell_qty_val": 5.0,
        "en_weight_AAPL": 0.3,
        "ko_buy_ticker": "OLD",
        "strategies": {"OLD": {}},
        "backtest": None,
        "unrelated_key": "keep me",
    }
    _apply_uploaded_json({"NEW": {"buy": {}}}, state=state)
    assert "en_buy_ticker" not in state
    assert "en_buy_crit_val" not in state
    assert "en_sell_qty_val" not in state
    assert "en_weight_AAPL" not in state
    assert "ko_buy_ticker" not in state
    assert state["strategies"] == {"NEW": {"buy": {}}}
    assert state["unrelated_key"] == "keep me"  # untouched
    assert state["backtest"] is None  # untouched


def test_saved_strategy_validates_against_StrategyWrapper():
    state = {
        **_seed_state("en_buy"),
        **_seed_state("en_sell", **{
            "en_sell_qty_type": "percent",
            "en_sell_qty_val": 30.0,
            "en_sell_crit_val": 2.0,
        }),
    }
    buy = _collect_form_dict("en_buy", ("count", "percent", "value", "split"), state=state)
    sell = _collect_form_dict("en_sell", ("count", "percent", "value"), state=state)
    candidate = {"AAPL": {"buy": buy, "sell": sell, "portfolio_weight": 0.5}}
    wrapper = StrategyWrapper.model_validate(candidate)
    assert "AAPL" in wrapper.root
    assert wrapper["AAPL"].buy.ticker == "AAPL"
    assert wrapper["AAPL"].portfolio_weight == 0.5


def test_extract_defaults_handles_missing_saved_data():
    result = _extract_defaults(None, default_ticker="TSLA")
    assert result["ticker"] == "TSLA"
    assert result["by_agg"] == "current"
    assert result["by_field"] == "Close"  # safe default
    assert result["use_period"] is False
    assert result["period_val"] == 3
    assert result["crit_type"] == "percent-change"
    assert result["qty_type"] == "count"


def test_extract_defaults_window_false_disables_period():
    saved = {"window": False, "indicator": ["current", "Close"]}
    result = _extract_defaults(saved, default_ticker="AAPL")
    assert result["use_period"] is False


def test_extract_defaults_window_int_enables_period():
    saved = {"window": 5, "indicator": ["current", "Close"]}
    result = _extract_defaults(saved, default_ticker="AAPL")
    assert result["use_period"] is True
    assert result["period_val"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_streamlit_forms.py -v
```

Expected: FAIL with "ImportError: cannot import name '_collect_form_dict' from 'streamlit_ui.forms'".

- [ ] **Step 3: Create `streamlit_ui/forms.py` with pure functions**

```python
"""Form state machine + save logic for the Streamlit strategy editor.

Pure functions in this module (_collect_form_dict, _apply_uploaded_json,
_extract_defaults) accept an optional ``state`` dict so they are testable
without a Streamlit runtime. The runtime helpers (input_strategy_details,
_save_strategy_callback, _maybe_reseed_widgets) are added in Task 4.
"""

from __future__ import annotations

from typing import Optional, Sequence

import streamlit as st

# Prefixes of widget keys that belong to the strategy form.
WIDGET_KEY_PREFIXES: tuple[str, ...] = (
    "en_buy_", "en_sell_", "ko_buy_", "ko_sell_",
    "en_weight_", "ko_weight_",
    "en_active_side", "ko_active_side",
    "en_prev_main_ticker", "ko_prev_main_ticker",
)


def _collect_form_dict(
    prefix: str,
    allowed_qty: Sequence[str],
    state: Optional[dict] = None,
) -> dict:
    """Read widget state by prefix; return a dict matching TradeAction schema.

    Parameters
    ----------
    prefix : str
        The shared key prefix for widget state (e.g. ``"en_buy"``).
    allowed_qty : Sequence[str]
        Allowed quantity types for this side. Used to clamp ``qty_type``
        to a valid value if widget state is somehow out of range.
    state : dict, optional
        Override for ``st.session_state``. Useful in tests.
    """
    if state is None:
        state = st.session_state
    ticker = state[f"{prefix}_ticker"]
    by_agg = state[f"{prefix}_by_agg"]
    by_field = state[f"{prefix}_by_field"]
    use_period = bool(state[f"{prefix}_use_period"])
    if use_period:
        period_val: int | bool = int(state[f"{prefix}_period_val"])
    else:
        period_val = False
    crit_type = state[f"{prefix}_crit_type"]
    crit_val = float(state[f"{prefix}_crit_val"])
    qty_type = state[f"{prefix}_qty_type"]
    if qty_type not in allowed_qty:
        qty_type = allowed_qty[0]
    qty_val_raw = state[f"{prefix}_qty_val"]
    if qty_type in ("count", "split"):
        qty_val: int | float = int(qty_val_raw)
    else:
        qty_val = float(qty_val_raw)
    price_point = state[f"{prefix}_trade_as"]
    return {
        "ticker": ticker,
        "indicator": [by_agg, by_field],
        "window": period_val,
        "threshold": [crit_type, crit_val],
        "quantity": [qty_type, qty_val],
        "price_point": price_point,
    }


def _apply_uploaded_json(
    loaded_data: dict,
    state: Optional[dict] = None,
) -> None:
    """Clear stale form widget state, then set strategies.

    Fix for symptom-1 bug: widget keys persist across JSON uploads, causing
    the form to display pre-upload values. Purging the form widget keys forces
    the next render to re-seed widgets from the new strategies dict.
    """
    if state is None:
        state = st.session_state
    keys_to_delete = [
        key for key in list(state.keys())
        if any(key.startswith(p) for p in WIDGET_KEY_PREFIXES)
    ]
    for key in keys_to_delete:
        del state[key]
    state["strategies"] = loaded_data


def _extract_defaults(saved_data: Optional[dict], default_ticker: str) -> dict:
    """Extract widget default values from a saved strategy buy/sell dict."""
    saved_data = saved_data or {}
    saved_by = saved_data.get("indicator", ["current", "Close"])
    saved_period = saved_data.get("window", False)
    # bool is a subclass of int in Python — explicitly exclude bool.
    use_period = (
        isinstance(saved_period, int)
        and not isinstance(saved_period, bool)
        and saved_period >= 1
    )
    saved_crit = saved_data.get("threshold", ["percent-change", -0.5])
    saved_qty = saved_data.get("quantity", ["count", 10])
    return {
        "ticker": saved_data.get("ticker", default_ticker),
        "by_agg": saved_by[0],
        "by_field": saved_by[1],
        "trade_as": saved_data.get("price_point", "Close"),
        "use_period": use_period,
        "period_val": int(saved_period) if use_period else 3,
        "crit_type": saved_crit[0],
        "crit_val": float(saved_crit[1]),
        "qty_type": saved_qty[0],
        "qty_val": float(saved_qty[1]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_streamlit_forms.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
uv run pytest -q
```

Expected: all previous tests + 10 new pass.

- [ ] **Step 6: Commit**

```bash
git add streamlit_ui/forms.py tests/test_streamlit_forms.py
git commit -m "feat: add form state pure functions with save-bug fix"
```

---

## Task 4: forms.py — Runtime Helpers

**Files:**
- Modify: `streamlit_ui/forms.py` (add runtime functions)

- [ ] **Step 1: Add runtime helpers to `streamlit_ui/forms.py`**

Append the following to `streamlit_ui/forms.py`:

```python
# -- Runtime helpers (require Streamlit runtime; not unit-tested) --------------

def _maybe_reseed_widgets(lang: str, main_ticker: str) -> None:
    """If main_ticker changed since last render, re-seed all form widgets."""
    prev_key = f"{lang}_prev_main_ticker"
    if st.session_state.get(prev_key) == main_ticker:
        return
    current = st.session_state.get("strategies", {}).get(main_ticker, {})
    for side in ("buy", "sell"):
        defaults = _extract_defaults(current.get(side), main_ticker)
        for suffix, val in defaults.items():
            st.session_state[f"{lang}_{side}_{suffix}"] = val
    st.session_state[f"{lang}_weight_{main_ticker}"] = float(
        current.get("portfolio_weight", 0.0)
    )
    st.session_state[prev_key] = main_ticker


def input_strategy_details(
    state_key_prefix: str,
    allowed_qty_types: Sequence[str],
    lang: str,
) -> None:
    """Render strategy-side form widgets with stable per-language keys.

    ``state_key_prefix`` is ``f"{lang}_buy"`` or ``f"{lang}_sell"``.
    Widget state seeding happens in ``_maybe_reseed_widgets`` before this is
    called. This function only renders.
    """
    from streamlit_ui.i18n import T

    opts_agg = ["current", "average"]
    opts_field = ["Close", "Change_Pct", "Change", "Open", "High", "Low"]
    opts_trade_as = ["Close", "Open", "High", "Low"]
    opts_crit = ["percent-change", "profit-rate", "point", "value"]
    opts_qty = list(allowed_qty_types)

    col1, col2 = st.columns(2)
    with col1:
        st.text_input(T("target_ticker_label", lang), key=f"{state_key_prefix}_ticker")
        st.caption(T("base_group_label", lang))
        c1, c2 = st.columns(2)
        c1.selectbox(
            T("aggregation_method_label", lang),
            opts_agg,
            key=f"{state_key_prefix}_by_agg",
        )
        c2.selectbox(
            T("field_label", lang),
            opts_field,
            key=f"{state_key_prefix}_by_field",
        )
        st.selectbox(
            T("price_point_label", lang),
            opts_trade_as,
            key=f"{state_key_prefix}_trade_as",
        )
    with col2:
        st.caption(T("period_group_label", lang))
        use_period = st.checkbox(
            T("use_period_label", lang),
            key=f"{state_key_prefix}_use_period",
        )
        if use_period:
            st.number_input(
                T("period_days_label", lang),
                min_value=1,
                step=1,
                key=f"{state_key_prefix}_period_val",
            )

    col3, col4 = st.columns(2)
    with col3:
        st.caption(T("criteria_group_label", lang))
        c3, c4 = st.columns(2)
        c3.selectbox(
            T("criteria_type_label", lang),
            opts_crit,
            key=f"{state_key_prefix}_crit_type",
        )
        c4.number_input(
            T("criteria_value_label", lang),
            step=0.1,
            format="%.2f",
            key=f"{state_key_prefix}_crit_val",
        )
    with col4:
        st.caption(T("quantity_group_label", lang))
        c5, c6 = st.columns(2)
        c5.selectbox(
            T("quantity_unit_label", lang),
            opts_qty,
            key=f"{state_key_prefix}_qty_type",
        )
        c6.number_input(
            T("quantity_value_label", lang),
            step=1.0,
            key=f"{state_key_prefix}_qty_val",
        )


def _save_strategy_callback(lang: str, main_ticker: str) -> None:
    """on_click callback for the Save button.

    Reads widget state directly via ``_collect_form_dict`` and writes a fresh
    dict to ``st.session_state["strategies"][main_ticker]``.
    """
    from streamlit_ui.i18n import T

    buy = _collect_form_dict(f"{lang}_buy", ("count", "percent", "value", "split"))
    sell = _collect_form_dict(f"{lang}_sell", ("count", "percent", "value"))
    weight = float(st.session_state.get(f"{lang}_weight_{main_ticker}", 0.0))
    st.session_state["strategies"][main_ticker] = {
        "buy": buy,
        "sell": sell,
        "portfolio_weight": weight,
    }
    st.toast(
        T("strategy_updated_toast", lang).format(ticker=main_ticker),
        icon="✅",
    )


def _delete_strategy_callback(ticker: str) -> None:
    """on_click callback for the Delete button on a strategy card."""
    if ticker in st.session_state.get("strategies", {}):
        del st.session_state["strategies"][ticker]


def _edit_strategy_callback(lang: str, ticker: str) -> None:
    """on_click callback for the Edit button on a strategy card.

    Sets the main ticker input to ``ticker`` so ``_maybe_reseed_widgets``
    will pick it up on the next render and re-seed form widgets.
    """
    st.session_state[f"{lang}_main_ticker"] = ticker
```

- [ ] **Step 2: Run existing tests to confirm runtime helpers don't break pure-function tests**

```bash
uv run pytest tests/test_streamlit_forms.py -v
```

Expected: 10 passed (still).

- [ ] **Step 3: Commit**

```bash
git add streamlit_ui/forms.py
git commit -m "feat: add form runtime helpers with reseed + save callbacks"
```

---

## Task 5: components.py — `_describe_rule`

**Files:**
- Create: `streamlit_ui/components.py`
- Create: `tests/test_streamlit_components.py`

- [ ] **Step 1: Write the failing test**

`tests/test_streamlit_components.py`:
```python
"""Tests for streamlit_ui.components pure functions."""

import pytest

from streamlit_ui.components import _describe_rule


def test_describe_rule_buy_percent_change_negative():
    side = {
        "indicator": ["current", "Close"],
        "threshold": ["percent-change", -0.5],
        "quantity": ["count", 10],
        "price_point": "Close",
    }
    condition, action = _describe_rule(side, side_kind="buy", lang="en")
    assert "drops" in condition
    assert "0.50" in condition
    assert "Close" in condition
    assert "Buy" in action
    assert "10" in action
    assert "shares" in action


def test_describe_rule_buy_percent_change_positive():
    side = {
        "indicator": ["current", "Close"],
        "threshold": ["percent-change", 2.0],
        "quantity": ["count", 10],
        "price_point": "Close",
    }
    condition, _ = _describe_rule(side, side_kind="buy", lang="en")
    assert "rises" in condition
    assert "2.00" in condition


def test_describe_rule_sell_percent_quantity():
    side = {
        "indicator": ["current", "Close"],
        "threshold": ["profit-rate", 5.0],
        "quantity": ["percent", 30],
        "price_point": "Close",
    }
    _, action = _describe_rule(side, side_kind="sell", lang="en")
    assert "Sell" in action
    assert "30%" in action


def test_describe_rule_buy_value_quantity():
    side = {
        "indicator": ["average", "Close"],
        "threshold": ["point", 100.0],
        "quantity": ["value", 500],
        "price_point": "Open",
    }
    condition, action = _describe_rule(side, side_kind="buy", lang="en")
    assert "average" in condition
    assert "$500" in action or "500" in action
    assert "Open" in action


def test_describe_rule_buy_split_quantity():
    side = {
        "indicator": ["current", "Close"],
        "threshold": ["percent-change", -0.5],
        "quantity": ["split", 4],
        "price_point": "Close",
    }
    _, action = _describe_rule(side, side_kind="buy", lang="en")
    assert "split" in action.lower() or "4" in action


def test_describe_rule_korean_returns_korean_strings():
    side = {
        "indicator": ["current", "Close"],
        "threshold": ["percent-change", -0.5],
        "quantity": ["count", 10],
        "price_point": "Close",
    }
    condition, action = _describe_rule(side, side_kind="buy", lang="ko")
    assert "매수" in action
    assert "하락" in condition
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_streamlit_components.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'streamlit_ui.components'".

- [ ] **Step 3: Create `streamlit_ui/components.py` with `_describe_rule`**

```python
"""Render helpers and rule-description pure functions for the Streamlit UI."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from streamlit_ui import theme
from streamlit_ui.i18n import T


def _describe_rule(
    side: dict,
    side_kind: Literal["buy", "sell"],
    lang: str,
) -> tuple[str, str]:
    """Translate a buy/sell rule dict into (condition, action) human-readable strings.

    Examples (English)::

        buy={"indicator": ["current", "Close"],
             "threshold": ["percent-change", -0.5],
             "quantity": ["count", 10],
             "price_point": "Close"}
        -> ("When current Close drops 0.50%", "Buy 10 shares at Close")
    """
    agg_raw, field = side["indicator"]
    crit_type, crit_val = side["threshold"]
    qty_type, qty_val = side["quantity"]
    price_point = side.get("price_point", "Close")

    agg = T(f"describe_{agg_raw}", lang) if agg_raw in ("current", "average") else agg_raw

    if crit_val < 0:
        direction = T("describe_drops", lang)
    elif crit_val > 0:
        direction = T("describe_rises", lang)
    else:
        direction = T("describe_reaches", lang)

    if crit_type in ("percent-change", "profit-rate"):
        crit_str = f"{abs(crit_val):.2f}%"
    else:
        crit_str = f"{abs(crit_val):.2f}"

    condition = f"{T('describe_when', lang)} {agg} {field} {direction} {crit_str}"

    verb_key = "describe_buy_prefix" if side_kind == "buy" else "describe_sell_prefix"
    verb = T(verb_key, lang)

    if qty_type == "count":
        qty_str = f"{int(qty_val)} {T('describe_shares', lang)}"
    elif qty_type == "value":
        qty_str = f"${qty_val:,.0f}"
    elif qty_type == "percent":
        qty_str = f"{qty_val:.0f}%"
    elif qty_type == "split":
        qty_str = T("describe_split_parts", lang).format(parts=int(qty_val))
    else:
        qty_str = str(qty_val)

    action = f"{verb} {qty_str} {T('describe_at_price', lang)} {price_point}"
    return condition, action
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_streamlit_components.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: all previous + 6 new pass.

- [ ] **Step 6: Commit**

```bash
git add streamlit_ui/components.py tests/test_streamlit_components.py
git commit -m "feat: add rule-description helper with bilingual support"
```

---

## Task 6: components.py — Render Helpers

**Files:**
- Modify: `streamlit_ui/components.py` (add render functions)

- [ ] **Step 1: Append render helpers to `streamlit_ui/components.py`**

```python
# -- Render helpers (Streamlit-runtime; not unit-tested) -----------------------

import html


def render_page_header(title: str, icon: str = "📈") -> None:
    """Render the page header band with yellow icon + title."""
    safe_title = html.escape(title)
    st.markdown(
        f"""
        <div class="bn-page-header">
          <div style="font-size: 40px;">{icon}</div>
          <h1 class="bn-page-header-title">{safe_title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(text: str) -> None:
    """Render an h2 section header with the yellow accent rule."""
    safe = html.escape(text)
    st.markdown(
        f'<div class="bn-section-header"><h2>{safe}</h2></div>',
        unsafe_allow_html=True,
    )


def render_metric(
    label: str,
    value: str,
    kind: Literal["neutral", "up", "down", "accent"] = "neutral",
) -> None:
    """Render a metric callout card."""
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    st.markdown(
        f"""
        <div class="bn-metric">
          <div class="bn-metric-label">{safe_label}</div>
          <div class="bn-metric-value {kind}">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_buy_sell_selector(state_key: str, lang: str) -> Literal["buy", "sell"]:
    """Render the two-pill Buy/Sell selector. Returns the active side."""
    if state_key not in st.session_state:
        st.session_state[state_key] = "buy"
    active = st.session_state[state_key]

    col_buy, col_sell = st.columns(2)
    if col_buy.button(
        T("buy_pill_label", lang),
        use_container_width=True,
        key=f"{state_key}_buy_btn",
        type="primary" if active == "buy" else "secondary",
    ):
        st.session_state[state_key] = "buy"
        st.rerun()
    if col_sell.button(
        T("sell_pill_label", lang),
        use_container_width=True,
        key=f"{state_key}_sell_btn",
        type="primary" if active == "sell" else "secondary",
    ):
        st.session_state[state_key] = "sell"
        st.rerun()
    return st.session_state[state_key]


def render_strategy_card(
    ticker: str,
    strategy: dict,
    lang: str,
) -> None:
    """Render a strategy summary card + Edit / Delete buttons."""
    from streamlit_ui.forms import _delete_strategy_callback, _edit_strategy_callback

    weight = strategy.get("portfolio_weight", 0.0)
    weight_pct = f"{weight * 100:.0f}%"
    safe_ticker = html.escape(ticker)

    sides_html_parts = []
    for side_kind in ("buy", "sell"):
        side = strategy.get(side_kind)
        if not side:
            continue
        try:
            condition, action = _describe_rule(side, side_kind=side_kind, lang=lang)
        except (KeyError, TypeError):
            condition, action = "?", "?"
        label_text = T(f"{side_kind}_pill_label", lang)
        safe_label = html.escape(label_text)
        sides_html_parts.append(
            f"""
            <div class="bn-side-block {side_kind}">
              <div class="bn-side-label {side_kind}">{safe_label}</div>
              <div class="bn-side-condition">{html.escape(condition)}</div>
              <div class="bn-side-action">{html.escape(action)}</div>
            </div>
            """
        )
    sides_html = "".join(sides_html_parts)

    st.markdown(
        f"""
        <div class="bn-strategy-card">
          <div class="bn-strategy-card-header">
            <div class="bn-strategy-card-ticker">{safe_ticker}</div>
            <div class="bn-strategy-card-weight">{weight_pct}</div>
          </div>
          {sides_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_edit, col_delete = st.columns(2)
    col_edit.button(
        T("card_edit_button", lang),
        key=f"card_edit_{lang}_{ticker}",
        use_container_width=True,
        on_click=_edit_strategy_callback,
        args=(lang, ticker),
    )
    col_delete.button(
        T("card_delete_button", lang),
        key=f"card_delete_{lang}_{ticker}",
        use_container_width=True,
        on_click=_delete_strategy_callback,
        args=(ticker,),
    )
```

- [ ] **Step 2: Update `streamlit_ui/__init__.py` to re-export the public API**

```python
"""Streamlit UI helpers for the pybacktest strategy editor and backtest page."""

from streamlit_ui.components import (
    render_buy_sell_selector,
    render_metric,
    render_page_header,
    render_section_header,
    render_strategy_card,
)
from streamlit_ui.forms import (
    _apply_uploaded_json,
    _maybe_reseed_widgets,
    _save_strategy_callback,
    input_strategy_details,
)
from streamlit_ui.i18n import T
from streamlit_ui.theme import inject_global_styles

__all__ = [
    "T",
    "inject_global_styles",
    "render_buy_sell_selector",
    "render_metric",
    "render_page_header",
    "render_section_header",
    "render_strategy_card",
    "_apply_uploaded_json",
    "_maybe_reseed_widgets",
    "_save_strategy_callback",
    "input_strategy_details",
]
```

- [ ] **Step 3: Run pytest to ensure no regressions from imports**

```bash
uv run pytest -q
```

Expected: all tests pass (the new render helpers have no unit tests).

- [ ] **Step 4: Commit**

```bash
git add streamlit_ui/components.py streamlit_ui/__init__.py
git commit -m "feat: add binance-styled render helpers for cards and metrics"
```

---

## Task 7: Refactor `streamlit_page_en.py` + Reproduce & Verify Bug Fix

**Files:**
- Modify: `streamlit_page_en.py` (rewrite)

- [ ] **Step 1: Reproduce the symptom-1 bug first (BEFORE rewriting)**

Run the app:
```bash
uv run streamlit run streamlit_page.py
```

In the browser:
1. Sidebar → upload `strategy_test_format.json` → click "Apply Data".
2. Confirm AAPL data appears in the right-hand JSON view.
3. In the form, click "Buy" tab if available, edit "Criteria Value" from 300 to 500.
4. Click "💾 Save Changes".
5. **Observe the JSON view does NOT update to 500.**

Take a screenshot or record the steps. If the bug does NOT reproduce as described, STOP and report — the spec's hypothesis is wrong and the fix may be unnecessary.

- [ ] **Step 2: Rewrite `streamlit_page_en.py`**

Replace the entire file contents with:

```python
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
```

- [ ] **Step 3: Run the app and verify the bug is fixed**

```bash
uv run streamlit run streamlit_page.py
```

Walk through the smoke tests for the EN page:
1. **Fresh load** — open page, enter AAPL, fill buy + sell, click Add Strategy → card appears.
2. **Edit existing** — change buy criteria from 0.5 to 1.5, Save → card + JSON show 1.5.
3. **Upload JSON** — upload `strategy_test_format.json`, Apply Data → cards for AAPL and TQQQ appear; form populates from AAPL.
4. **Upload → edit → save (BUG FIX VERIFICATION)** — change AAPL's quantity from 10 to 25 → Save → JSON expander shows 25. **If JSON does NOT show 25, the fix is incomplete — debug before continuing.**
8. **Backtest** — run a backtest; metric cards render; profit rate colored by direction.

Stop the app once all 5 smoke tests pass.

- [ ] **Step 4: Run full pytest suite**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add streamlit_page_en.py
git commit -m "feat: refactor english page with binance reskin and save-bug fix

Symptom-1 reproduced: editing an existing strategy from an uploaded JSON
no longer triggered a JSON-view update. Root cause was widget keys
prefixed with the ticker (e.g. buy_AAPL_en_crit_val) persisting through
JSON uploads. Fix:

* Drop ticker from widget keys; reseed on ticker change via
  _maybe_reseed_widgets.
* Wire Save through an on_click callback (_save_strategy_callback) that
  reads st.session_state directly.
* Purge form widget keys on JSON upload (_apply_uploaded_json)."
```

---

## Task 8: Refactor `streamlit_page_ko.py`

**Files:**
- Modify: `streamlit_page_ko.py` (rewrite)

- [ ] **Step 1: Read existing streamlit_page_ko.py for any KO-specific behavior**

```bash
diff streamlit_page_en.py streamlit_page_ko.py | head -100
```

The KO page on `main` is structurally identical to EN. Any Korean-specific tweaks (different default ticker? different help text?) should be preserved in the refactor.

- [ ] **Step 2: Rewrite `streamlit_page_ko.py`**

Replace the entire file contents with a Korean variant of the EN page. The only difference is `_LANG = "ko"` and widget keys prefixed with `ko_` instead of `en_`. The full file:

```python
"""Korean-language Streamlit page for the pybacktest strategy editor."""

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


_LANG = "ko"


def _sidebar() -> None:
    with st.sidebar:
        uploaded_file = st.file_uploader(
            T("load_json_label", _LANG), type=["json"], key="ko_upload"
        )
        if uploaded_file is not None:
            if st.button(
                T("apply_data_button", _LANG),
                type="primary",
                use_container_width=True,
                key="ko_apply",
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
            key="ko_reset",
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
        key=f"ko_save_{main_ticker_input}",
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
            key="ko_download",
        )


def _backtest_tab() -> None:
    render_section_header(T("backtest_header", _LANG))
    col_form, col_results = st.columns([0.4, 0.6])
    with col_form:
        with st.form("backtest_form_ko"):
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
                    key="ko_slider",
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


def show_korean_page() -> None:
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
```

- [ ] **Step 3: Test the language toggle (smoke test 7 from spec)**

```bash
uv run streamlit run streamlit_page.py
```

In the browser:
1. Start in English → add a strategy for AAPL.
2. Toggle language to 한국어 via the sidebar radio.
3. Confirm: AAPL data persists; all labels render in Korean; yellow CTAs unchanged.
4. Edit AAPL on the Korean page → Save → JSON updates.
5. Toggle back to English → AAPL changes persist.

If translations look wrong (awkward phrasing), make a note for Task 9. Do NOT block on translation polish here; the structural bug is what matters.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add streamlit_page_ko.py
git commit -m "feat: refactor korean page to share streamlit_ui module"
```

---

## Task 9: Full Smoke Test Pass + Final Cleanup

**Files:**
- Modify (if needed): `streamlit_ui/i18n.py`, page files (small polish)

- [ ] **Step 1: Run the app and walk through all 9 smoke tests from spec section 9.2**

```bash
uv run streamlit run streamlit_page.py
```

Smoke tests:
1. **Fresh load** — Add AAPL → card + JSON populate.
2. **Edit existing** — Change buy criteria → Save → card + JSON reflect.
3. **Upload JSON** — Apply `strategy_test_format.json` → AAPL + TQQQ cards appear.
4. **Upload → edit → save** — Edit AAPL quantity → Save → JSON shows new value (symptom-1 regression check).
5. **Edit from card** — Click Edit on TQQQ card → Main Ticker = TQQQ; form shows TQQQ values.
6. **Delete from card** — Click Delete on AAPL → AAPL removed from cards + JSON.
7. **Language toggle** — Switch to Korean → data persists; KO strings render; yellow CTAs unchanged.
8. **Backtest** — Run backtest; metric cards render; profit rate colored by direction; chart renders.
9. **Narrow viewport (~600px)** — Resize browser; layout doesn't break; cards stack vertically.

- [ ] **Step 2: If translations are awkward, ask the user**

The Korean strings in `streamlit_ui/i18n.py` for NEW card UI (Portfolio Composition, Editing existing, etc.) were translated by the implementer. Ask the user to review and provide corrections before merging:

> "I added Korean translations for the new card-UI strings (Portfolio Composition, Editing existing, etc.). Please skim `streamlit_ui/i18n.py` LABELS['ko'] and let me know if any phrasing should change before we wrap up."

Apply any corrections returned by the user.

- [ ] **Step 3: Check for orphaned imports / unused code**

```bash
uv run ruff check streamlit_ui/ streamlit_page_en.py streamlit_page_ko.py streamlit_page.py
```

Fix any reported issues. If `ruff` is not installed, install via `uv add --dev ruff` first, or skip this step if formatting was kept clean during implementation.

- [ ] **Step 4: Final test suite run**

```bash
uv run pytest -q
```

Expected: all tests pass (original 28 + ~30 new from Tasks 1, 2, 3, 5).

- [ ] **Step 5: Manual verification of CSS rendering**

Visit the running Streamlit page one more time:
- Confirm yellow #FCD535 appears on all primary buttons.
- Confirm trading-up green and trading-down red appear on Buy/Sell tab indicators and strategy card borders.
- Confirm JetBrains Mono renders for numeric values (final value, profit rate, weight pills).
- Confirm Inter renders for body / headlines.
- Take a screenshot of the redesigned EN page for the PR description.

- [ ] **Step 6: Commit any final polish**

```bash
git add -u
git commit -m "polish: address smoke test findings"
```

(If no polish needed, skip this step — empty commits are not allowed.)

- [ ] **Step 7: Final verification**

```bash
git log --oneline -15
uv run pytest -q
```

Expected:
- 9 commits since the spec commit (one per task).
- All tests pass.

---

## Self-Review Notes

The plan covers every spec section:
- §1 Goals: bug fix (Task 7), reskin (Tasks 1, 6, 7, 8), layout (Tasks 6, 7, 8), refactor (Tasks 1-6).
- §3 Architecture / module layout: Task 1 (theme + scaffolding), Task 2 (i18n), Task 3-4 (forms), Task 5-6 (components).
- §4 Visual system: Task 1 CSS_BLOCK contains all tokens.
- §5 Layout & components: Task 6 (render helpers), Tasks 7-8 (page integration).
- §6 Bug fix: Task 3 (pure functions for the fix), Task 7 (verification on real app).
- §7 Data flow: Tasks 3-7 implement each flow node.
- §8 Error handling: invalid JSON handled in Tasks 7-8 sidebar code.
- §9 Testing: Tasks 1-5 add automated tests; Tasks 7, 8, 9 run manual smoke.
- §10 Risks: bug-fix verification gated by reproduce-first step (Task 7 Step 1).

No placeholders, all code blocks contain real code, all function signatures referenced in later tasks are defined in earlier tasks (theme constants in Task 1 → CSS class names in Task 6 → page usage in Task 7).
