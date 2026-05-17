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
