"""Stable content fingerprints for validated market datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING

from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.time import Timeframe

if TYPE_CHECKING:
    from pybacktest.data.dataset import BarSeries

_ARRAY_FIELDS = ("timestamps", "open", "high", "low", "close", "volume")
_FINGERPRINT_VERSION = 1


def compute_dataset_fingerprint(
    series: Mapping[InstrumentId, BarSeries],
    instruments: Mapping[InstrumentId, Instrument],
    timeframe: Timeframe,
) -> str:
    """Hash canonical metadata and every immutable array byte."""
    ordered_ids = sorted(series, key=str)
    metadata = {
        "fingerprint_version": _FINGERPRINT_VERSION,
        "timeframe": {
            "count": timeframe.count,
            "unit": timeframe.unit.value,
        },
        "instruments": [
            {
                "id": str(instrument_id),
                "lot_size": _canonical_decimal(instruments[instrument_id].lot_size),
                "quote_currency": instruments[instrument_id].quote_currency,
                "tick_size": _canonical_decimal(instruments[instrument_id].tick_size),
                "timezone": str(instruments[instrument_id].timezone),
            }
            for instrument_id in ordered_ids
        ],
    }
    digest = hashlib.sha256()
    _update_length_prefixed(
        digest,
        json.dumps(
            metadata,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    )
    for instrument_id in ordered_ids:
        _update_length_prefixed(digest, str(instrument_id).encode("utf-8"))
        bar_series = series[instrument_id]
        for field_name in _ARRAY_FIELDS:
            _update_length_prefixed(digest, field_name.encode("ascii"))
            values = getattr(bar_series, field_name)
            _update_length_prefixed(digest, values.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _update_length_prefixed(
    digest: hashlib._Hash,
    value: bytes,
) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)
