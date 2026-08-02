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
    assert "4" in action
    assert "split" in action.lower() or "분할" in action


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
    assert "0.50" in condition


def test_describe_rule_missing_quantity_uses_pydantic_default():
    """Match pybacktest.strategy.TradeAction default (quantity: ["percent", 100])."""
    side = {
        "indicator": ["current", "Close"],
        "threshold": ["percent-change", -3.0],
        # quantity intentionally omitted
        # price_point intentionally omitted
    }
    condition, action = _describe_rule(side, side_kind="sell", lang="en")
    assert "drops" in condition
    assert "Sell" in action
    assert "100%" in action  # default quantity is ["percent", 100]
    assert "Close" in action  # default price_point is "Close"
