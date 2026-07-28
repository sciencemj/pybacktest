import sys


def test_v2_package_has_version_without_optional_imports():
    import pybacktest

    assert pybacktest.__version__ == "0.2.0"
    assert "yfinance" not in sys.modules
    assert "matplotlib" not in sys.modules
    assert "streamlit" not in sys.modules
    assert "mcp" not in sys.modules
