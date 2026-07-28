"""Streamlit UI helpers for the pybacktest strategy editor and backtest page."""

from streamlit_ui.components import (
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
    "render_metric",
    "render_page_header",
    "render_section_header",
    "render_strategy_card",
    "_apply_uploaded_json",
    "_maybe_reseed_widgets",
    "_save_strategy_callback",
    "input_strategy_details",
]
