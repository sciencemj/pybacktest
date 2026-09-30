"""Lazy PyArrow-backed adapter for versioned Parquet datasets."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from os import PathLike
from pathlib import Path
from types import MappingProxyType

from pybacktest.data.dataset import MarketDataSet
from pybacktest.data.validation import copy_instrument_mapping
from pybacktest.domain.errors import AdapterContractError
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.time import DateRange, Timeframe

from .pandas import PandasDataSource

_PARQUET_COLUMNS = (
    "timestamp",
    "instrument",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(frozen=True, slots=True, init=False)
class ParquetDataSource:
    """Store validated configuration and perform all Parquet I/O in load."""

    path: Path
    _instruments: Mapping[InstrumentId, Instrument]

    def __init__(
        self,
        path: str | PathLike[str],
        instruments: Mapping[InstrumentId, Instrument],
    ) -> None:
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(
            self,
            "_instruments",
            MappingProxyType(copy_instrument_mapping(instruments)),
        )

    def load(
        self,
        universe: Sequence[InstrumentId],
        period: DateRange,
        timeframe: Timeframe,
    ) -> MarketDataSet:
        """Read only requested rows and delegate canonical validation."""
        try:
            from pyarrow import dataset as pyarrow_dataset
        except ImportError as exc:
            raise AdapterContractError(
                "ParquetDataSource requires the optional 'pyarrow' dependency.",
                code="optional_dependency_missing",
            ) from exc

        requested = tuple(universe)
        start = period.start.astimezone(UTC)
        end = period.end.astimezone(UTC)
        row_filter = (
            (pyarrow_dataset.field("timestamp") >= start)
            & (pyarrow_dataset.field("timestamp") < end)
            & pyarrow_dataset.field("instrument").isin(
                [str(instrument_id) for instrument_id in requested]
            )
        )
        arrow_dataset = pyarrow_dataset.dataset(
            self.path,
            format="parquet",
        )
        table = arrow_dataset.to_table(
            columns=list(_PARQUET_COLUMNS),
            filter=row_filter,
        )
        frame = table.to_pandas().set_index("timestamp")
        return PandasDataSource(
            frame,
            instruments=self._instruments,
        ).load(requested, period, timeframe)
