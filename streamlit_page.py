"""Streamlit Community Cloud entrypoint for the Pybacktest demo."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from streamlit_ui.app import main  # noqa: E402

main()
