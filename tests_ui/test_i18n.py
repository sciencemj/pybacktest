import pytest

from streamlit_ui.i18n import LANGUAGES, STRINGS, t
from streamlit_ui.runner import METRIC_ROWS, STRATEGY_NAMES

ERROR_CODES = (
    "no_tickers",
    "too_many_tickers",
    "empty_history",
    "mixed_currency",
    "insufficient_history",
    "invalid_period",
)


def test_languages_have_identical_keys():
    assert set(STRINGS) == set(LANGUAGES)
    assert set(STRINGS["en"]) == set(STRINGS["ko"])


@pytest.mark.parametrize("lang", list(LANGUAGES))
def test_every_referenced_key_exists(lang):
    for name in STRATEGY_NAMES:
        t(f"strategy.{name}", lang)
    for metric in METRIC_ROWS:
        t(f"metric.{metric.value}", lang)
    for code in ERROR_CODES:
        t(f"error.{code}", lang, detail="x")


def test_missing_key_raises():
    with pytest.raises(KeyError):
        t("does.not.exist", "en")


def test_interpolation():
    assert t("error.too_many_tickers", "en", detail="5") == "Enter at most 5 tickers."
