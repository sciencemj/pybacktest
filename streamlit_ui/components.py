"""Render helpers and rule-description pure functions for the Streamlit UI."""

from __future__ import annotations

import html
from textwrap import dedent
from typing import Literal

import streamlit as st

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


# -- Render helpers (Streamlit-runtime; not unit-tested) -----------------------


def render_page_header(title: str, icon: str = "📈") -> None:
    """Render the page header band with yellow icon + title."""
    safe_title = html.escape(title)
    safe_icon = html.escape(icon)
    st.markdown(
        dedent(f"""
            <div class="bn-page-header">
              <div style="font-size: 40px;">{safe_icon}</div>
              <h1 class="bn-page-header-title">{safe_title}</h1>
            </div>
        """),
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
        dedent(f"""
            <div class="bn-metric">
              <div class="bn-metric-label">{safe_label}</div>
              <div class="bn-metric-value {kind}">{safe_value}</div>
            </div>
        """),
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

    weight = float(strategy.get("portfolio_weight", 0.0))
    weight_pct = f"{weight * 100:.0f}%"
    safe_ticker = html.escape(ticker)

    sides_html_parts = []
    for side_kind in ("buy", "sell"):
        side = strategy.get(side_kind)
        if not side:
            continue
        try:
            condition, action = _describe_rule(side, side_kind=side_kind, lang=lang)
        except (KeyError, TypeError, ValueError):
            condition, action = "?", "?"
        label_text = T(f"{side_kind}_pill_label", lang)
        safe_label = html.escape(label_text)
        sides_html_parts.append(
            dedent(f"""
                <div class="bn-side-block {side_kind}">
                  <div class="bn-side-label {side_kind}">{safe_label}</div>
                  <div class="bn-side-condition">{html.escape(condition)}</div>
                  <div class="bn-side-action">{html.escape(action)}</div>
                </div>
            """).strip()
        )
    sides_html = "".join(sides_html_parts)

    st.markdown(
        dedent(f"""
            <div class="bn-strategy-card">
              <div class="bn-strategy-card-header">
                <div class="bn-strategy-card-ticker">{safe_ticker}</div>
                <div class="bn-strategy-card-weight">{weight_pct}</div>
              </div>
              {sides_html}
            </div>
        """),
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
