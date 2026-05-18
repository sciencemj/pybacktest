"""Form state machine + save logic for the Streamlit strategy editor.

Pure functions in this module (_collect_form_dict, _apply_uploaded_json,
_extract_defaults) accept an optional ``state`` dict so they are testable
without a Streamlit runtime. The runtime helpers (input_strategy_details,
_save_strategy_callback, _maybe_reseed_widgets) are added in Task 4.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any

import streamlit as st

# Prefixes of widget keys that belong to the strategy form (each ends with _).
WIDGET_KEY_PREFIXES: tuple[str, ...] = (
    "en_buy_", "en_sell_", "ko_buy_", "ko_sell_",
    "en_weight_", "ko_weight_",
)

# Exact widget keys (no over-match via startswith).
WIDGET_KEY_EXACT: frozenset[str] = frozenset({
    "en_active_side", "ko_active_side",
    "en_prev_main_ticker", "ko_prev_main_ticker",
})


def _collect_form_dict(
    prefix: str,
    allowed_qty: Sequence[str],
    state: MutableMapping[str, Any] | None = None,
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
    if not allowed_qty:
        raise ValueError("allowed_qty must not be empty")
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
    state: MutableMapping[str, Any] | None = None,
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
        or key in WIDGET_KEY_EXACT
    ]
    for key in keys_to_delete:
        del state[key]
    state["strategies"] = loaded_data


def _extract_defaults(saved_data: dict | None, default_ticker: str) -> dict:
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
    # "strategies" is guaranteed initialized by streamlit_page.py entry router.
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

    Sets the main ticker input to ``ticker`` and clears the prev-ticker
    sentinel so ``_maybe_reseed_widgets`` will reseed on the next render.
    Clearing the sentinel is required so re-clicking Edit on the same
    ticker (e.g. after saving) still forces a refresh of widget state.
    """
    st.session_state[f"{lang}_main_ticker"] = ticker
    st.session_state.pop(f"{lang}_prev_main_ticker", None)
