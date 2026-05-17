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
