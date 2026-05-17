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
PRIMARY_DISABLED = "#3a3a1f"
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
  --bn-primary-disabled: #3a3a1f;
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
  background: var(--bn-primary-disabled);
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
