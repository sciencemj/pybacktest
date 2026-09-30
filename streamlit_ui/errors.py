"""Typed input errors the demo reports with a localized message."""

from __future__ import annotations


class DemoInputError(ValueError):
    """A user input problem detected before the engine runs.

    ``code`` selects the localized message (``error.<code>`` in
    :mod:`streamlit_ui.i18n`); ``detail`` is interpolated into it.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail
