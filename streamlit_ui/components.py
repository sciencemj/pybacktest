"""Render helpers and rule-description pure functions for the Streamlit UI."""

from __future__ import annotations

from typing import Literal

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
