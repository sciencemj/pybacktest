"""Normalize a platform-dependent peak-RSS reading to bytes."""

from pybacktest import ConfigurationError

_KIBIBYTE = 1024


def normalized_peak_rss_bytes(raw: object, system: object) -> int:
    """Return ``resource.getrusage(...).ru_maxrss`` expressed in bytes.

    Darwin reports bytes and Linux reports kibibytes, so comparing a raw
    reading against a byte budget would be wrong by a factor of 1024 on one of
    them. The platform name is a parameter rather than a ``platform.system()``
    call so every branch stays directly testable.
    """
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ConfigurationError(
            "peak RSS reading must be a positive integer.",
            code="invalid_rss_reading",
        )
    if system == "Darwin":
        return raw
    if system == "Linux":
        return raw * _KIBIBYTE
    raise ConfigurationError(
        f"unsupported platform for RSS normalization: {system!r}.",
        code="unsupported_rss_platform",
    )


__all__ = ["normalized_peak_rss_bytes"]
