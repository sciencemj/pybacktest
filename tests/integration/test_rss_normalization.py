"""Unit coverage for the benchmark's peak-RSS normalization helper.

``resource.getrusage`` reports ``ru_maxrss`` in bytes on Darwin and in KiB on
Linux, so the performance gate would silently compare numbers a factor of
1024 apart if the platform were ignored. The helper takes the platform name as
an argument so this test can drive every branch without monkeypatching.
"""

import pytest

from benchmarks._rss import normalized_peak_rss_bytes
from pybacktest import ConfigurationError


@pytest.mark.parametrize("raw", [True, False, 1.0, "1024", None, 2**20 + 0.5])
def test_non_integer_readings_are_rejected(raw: object) -> None:
    with pytest.raises(ConfigurationError) as raised:
        normalized_peak_rss_bytes(raw, "Darwin")

    assert raised.value.code == "invalid_rss_reading"


@pytest.mark.parametrize("raw", [0, -1, -1024])
def test_non_positive_readings_are_rejected(raw: int) -> None:
    with pytest.raises(ConfigurationError) as raised:
        normalized_peak_rss_bytes(raw, "Linux")

    assert raised.value.code == "invalid_rss_reading"


def test_darwin_readings_are_already_bytes() -> None:
    assert normalized_peak_rss_bytes(423_493_632, "Darwin") == 423_493_632


def test_linux_readings_are_kibibytes() -> None:
    assert normalized_peak_rss_bytes(413_568, "Linux") == 413_568 * 1024


@pytest.mark.parametrize("system", ["Windows", "darwin", "linux", "", "Java"])
def test_unsupported_platforms_are_rejected(system: str) -> None:
    with pytest.raises(ConfigurationError) as raised:
        normalized_peak_rss_bytes(1024, system)

    assert raised.value.code == "unsupported_rss_platform"


@pytest.mark.parametrize("system", [None, 11, b"Linux"])
def test_non_text_platforms_are_rejected(system: object) -> None:
    with pytest.raises(ConfigurationError) as raised:
        normalized_peak_rss_bytes(1024, system)

    assert raised.value.code == "unsupported_rss_platform"
