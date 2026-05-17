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
