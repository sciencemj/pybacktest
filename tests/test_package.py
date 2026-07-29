import json
import os
import subprocess
import sys
from pathlib import Path

OPTIONAL_PACKAGES = (
    "matplotlib",
    "mcp",
    "pandas",
    "pyarrow",
    "streamlit",
    "yfinance",
)
"""Optional or out-of-core packages a core ``import pybacktest`` must not load."""

NEVER_IMPORTED_BY_THE_SUITE = ("matplotlib", "mcp", "streamlit", "yfinance")
"""The subset no other test imports, so an in-process check stays meaningful.

``pandas`` and ``pyarrow`` are dev dependencies that the adapter and README
tests import deliberately, so only the fresh-subprocess probe below can prove
that ``import pybacktest`` alone leaves them out.
"""

_PROBE = """
import json, sys
import pybacktest
print(json.dumps({{
    "optional": sorted(name for name in {optional!r} if name in sys.modules),
    "data_adapters_loaded": "pybacktest.adapters.data" in sys.modules,
    "version": pybacktest.__version__,
}}))
"""


def _fresh_subprocess_import_report() -> dict[str, object]:
    """Import the package alone and report what that pulled in."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.format(optional=OPTIONAL_PACKAGES)],
        capture_output=True,
        check=True,
        cwd=source_root.parent,
        env=environment,
        text=True,
    )
    return json.loads(completed.stdout)


def test_v2_package_has_version_without_optional_imports():
    import pybacktest

    assert pybacktest.__version__ == "0.2.0"
    for optional in NEVER_IMPORTED_BY_THE_SUITE:
        assert optional not in sys.modules


def test_importing_pybacktest_loads_no_optional_package_in_a_fresh_process():
    report = _fresh_subprocess_import_report()

    assert report["version"] == "0.2.0"
    assert report["optional"] == []
    assert report["data_adapters_loaded"] is False
