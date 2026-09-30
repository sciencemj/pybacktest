import builtins
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from pybacktest.domain.errors import AdapterContractError, DataValidationError
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.time import DateRange, Timeframe

pd = pytest.importorskip("pandas")

from pybacktest.adapters.data.pandas import PandasDataSource  # noqa: E402
from pybacktest.adapters.data.parquet import ParquetDataSource  # noqa: E402

AAPL = Instrument(
    id=InstrumentId.parse("XNAS:AAPL"),
    quote_currency="USD",
    tick_size=Decimal("0.01"),
    lot_size=Decimal("1"),
    timezone=ZoneInfo("America/New_York"),
)
MSFT = Instrument(
    id=InstrumentId.parse("XNAS:MSFT"),
    quote_currency="USD",
    tick_size=Decimal("0.01"),
    lot_size=Decimal("1"),
    timezone=ZoneInfo("America/New_York"),
)
INSTRUMENTS = {AAPL.id: AAPL, MSFT.id: MSFT}


def frame(
    index: pd.DatetimeIndex,
    instrument: InstrumentId = AAPL.id,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument": [str(instrument)] * len(index),
            "open": np.full(len(index), 100.0),
            "high": np.full(len(index), 101.0),
            "low": np.full(len(index), 99.0),
            "close": np.full(len(index), 100.5),
            "volume": np.full(len(index), 1_000.0),
        },
        index=index,
    )


def period_for(index: pd.DatetimeIndex) -> DateRange:
    return DateRange(
        index[0].to_pydatetime(),
        (index[-1] + pd.Timedelta(days=1)).to_pydatetime(),
    )


def test_pandas_source_normalizes_to_read_only_float64_arrays():
    index = pd.date_range("2024-01-02", periods=2, tz="UTC", freq="D")
    source = PandasDataSource(frame(index), instruments={AAPL.id: AAPL})

    dataset = source.load(
        [AAPL.id],
        period_for(index),
        Timeframe.days(1),
    )

    series = dataset.series[AAPL.id]
    assert series.close.dtype == np.float64
    assert not series.close.flags.writeable


def test_pandas_source_snapshots_caller_owned_frame_and_metadata_mapping():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    caller_frame = frame(index)
    caller_instruments = {AAPL.id: AAPL}
    source = PandasDataSource(caller_frame, instruments=caller_instruments)
    caller_frame.loc[index[0], "close"] = 999.0
    caller_instruments.clear()

    dataset = source.load(
        [AAPL.id],
        period_for(index),
        Timeframe.days(1),
    )

    assert dataset.series[AAPL.id].close.tolist() == [100.5]
    assert dataset.instruments[AAPL.id] == AAPL


def test_duplicate_timestamp_is_rejected():
    index = pd.DatetimeIndex(
        [
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ]
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        PandasDataSource(frame(index), instruments={AAPL.id: AAPL}).load(
            [AAPL.id],
            DateRange(index[0], datetime(2024, 1, 3, tzinfo=UTC)),
            Timeframe.days(1),
        )


def test_high_low_invariant_is_rejected():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    invalid = frame(index)
    invalid.loc[index[0], "high"] = 90.0
    with pytest.raises(DataValidationError, match="high"):
        PandasDataSource(invalid, instruments={AAPL.id: AAPL}).load(
            [AAPL.id],
            DateRange(
                index[0].to_pydatetime(),
                datetime(2024, 1, 3, tzinfo=UTC),
            ),
            Timeframe.days(1),
        )


def test_source_never_loads_at_or_after_exclusive_end():
    index = pd.date_range("2024-01-01", periods=5, tz="UTC", freq="D")
    source = PandasDataSource(frame(index), instruments={AAPL.id: AAPL})

    dataset = source.load(
        [AAPL.id],
        DateRange(index[0].to_pydatetime(), index[3].to_pydatetime()),
        Timeframe.days(1),
    )

    np.testing.assert_array_equal(
        dataset.series[AAPL.id].timestamps,
        index[:3].to_numpy(dtype="datetime64[ns]"),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.drop(columns=["volume"]),
        lambda data: data.rename(columns={"open": "Open"}),
        lambda data: data.rename(columns={"open": "price"}),
        lambda data: data.assign(extra=1.0),
    ],
)
def test_pandas_source_rejects_non_exact_schema(mutate):
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")

    with pytest.raises(DataValidationError, match="columns"):
        PandasDataSource(
            mutate(frame(index)),
            instruments={AAPL.id: AAPL},
        )


def test_pandas_source_accepts_exact_columns_in_any_order():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    reordered = frame(index)[["volume", "close", "instrument", "low", "high", "open"]]

    result = PandasDataSource(
        reordered,
        instruments={AAPL.id: AAPL},
    ).load([AAPL.id], period_for(index), Timeframe.days(1))

    assert result.series[AAPL.id].open.tolist() == [100.0]


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2024-01-02", periods=1, freq="D"),
        pd.date_range(
            "2024-01-02",
            periods=1,
            tz="America/New_York",
            freq="D",
        ),
    ],
)
def test_pandas_source_rejects_naive_or_non_utc_index(
    index: pd.DatetimeIndex,
):
    with pytest.raises(DataValidationError, match="UTC"):
        PandasDataSource(frame(index), instruments={AAPL.id: AAPL})


def test_pandas_source_rejects_non_datetime_index():
    invalid = frame(pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D"))
    invalid.index = pd.Index([1])

    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        PandasDataSource(invalid, instruments={AAPL.id: AAPL})


def test_pandas_source_rejects_nat_index_before_period_filtering():
    index = pd.DatetimeIndex(
        [
            datetime(2024, 1, 2, tzinfo=UTC),
            pd.NaT,
        ]
    )

    with pytest.raises(DataValidationError, match="NaT"):
        PandasDataSource(frame(index), instruments={AAPL.id: AAPL})


def test_pandas_source_rejects_non_numeric_ohlcv_without_coercion():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    invalid = frame(index)
    invalid["close"] = "100.5"

    with pytest.raises(DataValidationError, match="numeric"):
        PandasDataSource(invalid, instruments={AAPL.id: AAPL})


def test_pandas_source_rejects_unknown_frame_instrument():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")

    with pytest.raises(DataValidationError, match="unknown"):
        PandasDataSource(frame(index, MSFT.id), instruments={AAPL.id: AAPL})


def test_pandas_source_rejects_malformed_frame_instrument():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    invalid = frame(index)
    invalid["instrument"] = "not-an-instrument"

    with pytest.raises(DataValidationError, match="instrument"):
        PandasDataSource(invalid, instruments={AAPL.id: AAPL})


def test_pandas_source_rejects_unknown_requested_instrument():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    source = PandasDataSource(frame(index), instruments={AAPL.id: AAPL})

    with pytest.raises(DataValidationError, match="metadata"):
        source.load([MSFT.id], period_for(index), Timeframe.days(1))


def test_pandas_source_rejects_requested_instrument_missing_in_period():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    source = PandasDataSource(frame(index), instruments=INSTRUMENTS)

    with pytest.raises(DataValidationError, match="missing"):
        source.load([MSFT.id], period_for(index), Timeframe.days(1))


def test_pandas_source_rejects_empty_or_duplicate_universe():
    index = pd.date_range("2024-01-02", periods=1, tz="UTC", freq="D")
    source = PandasDataSource(frame(index), instruments={AAPL.id: AAPL})

    with pytest.raises(DataValidationError, match="at least one"):
        source.load([], period_for(index), Timeframe.days(1))
    with pytest.raises(DataValidationError, match="duplicate"):
        source.load(
            [AAPL.id, AAPL.id],
            period_for(index),
            Timeframe.days(1),
        )


def test_parquet_constructor_performs_no_path_io(tmp_path: Path):
    missing_path = tmp_path / "does-not-exist"

    source = ParquetDataSource(missing_path, instruments={AAPL.id: AAPL})

    assert source.path == missing_path


def test_parquet_source_path_is_read_only(tmp_path: Path):
    source = ParquetDataSource(
        tmp_path / "original.parquet",
        instruments={AAPL.id: AAPL},
    )

    with pytest.raises(AttributeError):
        source.path = tmp_path / "replacement.parquet"  # type: ignore[misc]
    assert source.path == tmp_path / "original.parquet"


def test_parquet_source_loads_filtered_half_open_period(tmp_path: Path):
    index = pd.date_range("2024-01-01", periods=5, tz="UTC", freq="D")
    parquet_path = tmp_path / "bars.parquet"
    stored = frame(index).rename_axis("timestamp").reset_index()
    stored.to_parquet(parquet_path, engine="pyarrow", index=False)
    source = ParquetDataSource(
        parquet_path,
        instruments={AAPL.id: AAPL},
    )

    dataset = source.load(
        [AAPL.id],
        DateRange(index[1].to_pydatetime(), index[3].to_pydatetime()),
        Timeframe.days(1),
    )

    np.testing.assert_array_equal(
        dataset.series[AAPL.id].timestamps,
        index[1:3].to_numpy(dtype="datetime64[ns]"),
    )


def test_missing_pyarrow_raises_typed_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    source = ParquetDataSource(
        tmp_path / "bars.parquet",
        instruments={AAPL.id: AAPL},
    )
    original_import = builtins.__import__

    def import_without_pyarrow(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_pyarrow)

    with pytest.raises(AdapterContractError) as caught:
        source.load(
            [AAPL.id],
            DateRange(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ),
            Timeframe.days(1),
        )

    assert caught.value.code == "optional_dependency_missing"
