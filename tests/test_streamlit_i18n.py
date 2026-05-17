"""Tests for streamlit_ui.i18n module."""

import pytest

from streamlit_ui import i18n


def test_languages_present():
    assert "en" in i18n.LABELS
    assert "ko" in i18n.LABELS


def test_languages_have_same_keys():
    en_keys = set(i18n.LABELS["en"].keys())
    ko_keys = set(i18n.LABELS["ko"].keys())
    missing_in_ko = en_keys - ko_keys
    missing_in_en = ko_keys - en_keys
    assert not missing_in_ko, f"Korean missing keys: {missing_in_ko}"
    assert not missing_in_en, f"English missing keys: {missing_in_en}"


def test_t_returns_english_string():
    assert i18n.T("page_title", "en") == "Automated Trading Strategy"


def test_t_returns_korean_string():
    # Korean is a non-ASCII string; just confirm it's present and non-empty.
    value = i18n.T("page_title", "ko")
    assert isinstance(value, str)
    assert len(value) > 0


def test_t_raises_on_missing_key():
    with pytest.raises(KeyError):
        i18n.T("nonexistent_key_xyz", "en")


def test_t_raises_on_missing_lang():
    with pytest.raises(KeyError):
        i18n.T("page_title", "fr")
