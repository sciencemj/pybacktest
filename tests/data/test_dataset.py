from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from types import MappingProxyType
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from pybacktest.data.calendar import CalendarMode, CalendarPolicy
from pybacktest.data.dataset import BarSeries, MarketDataSet
from pybacktest.domain.errors import ConfigurationError, DataValidationError
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.market import BarView, MarketSlice
from pybacktest.domain.time import Timeframe
from tests.factories import bar_series as factory_bar_series
from tests.factories import instrument as factory_instrument
from tests.factories import market_dataset as factory_market_dataset
from tests.factories import one_instrument_dataset
from tests.factories import timestamp as factory_timestamp

AAPL = Instrument(
    id=InstrumentId.parse("XNAS:AAPL"),
    quote_currency="USD",
    tick_size=Decimal("0.01"),
    lot_size=Decimal("1"),
    timezone=ZoneInfo("America/New_York"),
)
MSFT = replace(AAPL, id=InstrumentId.parse("XNAS:MSFT"))


def bar_series(**overrides: object) -> BarSeries:
    arguments: dict[str, object] = {
        "timestamps": np.array(
            ["2024-01-02", "2024-01-03"], dtype="datetime64[D]"
        ),
        "open": np.array([100, 101], dtype=np.int64),
        "high": np.array([102, 103], dtype=np.float32),
        "low": [99, 100],
        "close": [101, 102],
        "volume": [1_000, 2_000],
    }
    arguments.update(overrides)
    return BarSeries(**arguments)


def dataset(
    *,
    series: BarSeries | None = None,
    instrument: Instrument = AAPL,
    timeframe: Timeframe | None = None,
) -> MarketDataSet:
    return MarketDataSet(
        series={instrument.id: series or bar_series()},
        instruments={instrument.id: instrument},
        timeframe=timeframe or Timeframe.days(1),
    )


def test_bar_series_normalizes_copies_and_freezes_every_array():
    source_close = np.array([101, 102], dtype=np.float32)
    series = bar_series(close=source_close)
    source_close[0] = 999

    assert series.timestamps.dtype == np.dtype("datetime64[ns]")
    assert series.close.tolist() == [101.0, 102.0]
    for values in (
        series.timestamps,
        series.open,
        series.high,
        series.low,
        series.close,
        series.volume,
    ):
        assert not values.flags.writeable
    for values in (
        series.open,
        series.high,
        series.low,
        series.close,
        series.volume,
    ):
        assert values.dtype == np.float64


def test_bar_series_is_frozen():
    series = bar_series()

    with pytest.raises(FrozenInstanceError):
        series.close = np.array([1.0])  # type: ignore[misc]


def test_bar_series_arrays_cannot_be_made_writeable_again():
    series = bar_series()

    with pytest.raises(ValueError):
        series.close.setflags(write=True)
    with pytest.raises(ValueError):
        series.timestamps.setflags(write=True)


def test_bar_series_rejects_unequal_column_lengths():
    with pytest.raises(DataValidationError, match="length"):
        bar_series(volume=[1_000])


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bar_series_rejects_nonfinite_ohlcv(
    column: str, value: float
):
    with pytest.raises(DataValidationError, match="finite"):
        bar_series(**{column: [value, 1.0]})


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_bar_series_rejects_non_positive_prices(column: str, value: float):
    with pytest.raises(DataValidationError, match="positive"):
        bar_series(**{column: [value, 1.0]})


def test_bar_series_rejects_negative_volume():
    with pytest.raises(DataValidationError, match="volume"):
        bar_series(volume=[-1.0, 1.0])


def test_bar_series_rejects_duplicate_timestamps():
    with pytest.raises(DataValidationError, match="duplicate"):
        bar_series(
            timestamps=np.array(
                ["2024-01-02", "2024-01-02"], dtype="datetime64[ns]"
            )
        )


def test_bar_series_rejects_decreasing_timestamps():
    with pytest.raises(DataValidationError, match="increasing"):
        bar_series(
            timestamps=np.array(
                ["2024-01-03", "2024-01-02"], dtype="datetime64[ns]"
            )
        )


def test_bar_series_rejects_high_below_open_or_close():
    with pytest.raises(DataValidationError, match="high"):
        bar_series(high=[100.0, 103.0])


def test_bar_series_rejects_low_above_open_or_close():
    with pytest.raises(DataValidationError, match="low"):
        bar_series(low=[102.0, 100.0])


def test_dataset_copies_and_freezes_series_and_instrument_mappings():
    series_mapping = {AAPL.id: bar_series()}
    instrument_mapping = {AAPL.id: AAPL}
    market_data = MarketDataSet(
        series=series_mapping,
        instruments=instrument_mapping,
        timeframe=Timeframe.days(1),
    )

    series_mapping.clear()
    instrument_mapping.clear()

    assert isinstance(market_data.series, MappingProxyType)
    assert isinstance(market_data.instruments, MappingProxyType)
    assert tuple(market_data.series) == (AAPL.id,)
    with pytest.raises(TypeError):
        market_data.series[AAPL.id] = bar_series()  # type: ignore[index]
    with pytest.raises(TypeError):
        market_data.instruments[AAPL.id] = AAPL  # type: ignore[index]


def test_dataset_rejects_empty_or_mismatched_series_and_metadata():
    with pytest.raises(DataValidationError, match="at least one"):
        MarketDataSet(
            series={},
            instruments={},
            timeframe=Timeframe.days(1),
        )

    with pytest.raises(DataValidationError, match="match"):
        MarketDataSet(
            series={AAPL.id: bar_series()},
            instruments={MSFT.id: MSFT},
            timeframe=Timeframe.days(1),
        )


def test_dataset_rejects_metadata_stored_under_the_wrong_key():
    with pytest.raises(DataValidationError, match="metadata"):
        MarketDataSet(
            series={AAPL.id: bar_series()},
            instruments={AAPL.id: MSFT},
            timeframe=Timeframe.days(1),
        )


def test_dataset_fingerprint_is_canonical_across_mapping_order():
    aapl_series = bar_series()
    msft_series = bar_series(
        open=[200.0, 201.0],
        high=[202.0, 203.0],
        low=[199.0, 200.0],
        close=[201.0, 202.0],
    )
    forward = MarketDataSet(
        series={AAPL.id: aapl_series, MSFT.id: msft_series},
        instruments={AAPL.id: AAPL, MSFT.id: MSFT},
        timeframe=Timeframe.days(1),
    )
    reverse = MarketDataSet(
        series={MSFT.id: msft_series, AAPL.id: aapl_series},
        instruments={MSFT.id: MSFT, AAPL.id: AAPL},
        timeframe=Timeframe.days(1),
    )

    assert forward.fingerprint == reverse.fingerprint
    assert len(forward.fingerprint) == 64


@pytest.mark.parametrize(
    ("field", "values"),
    [
        (
            "timestamps",
            np.array(["2024-01-02", "2024-01-04"], dtype="datetime64[ns]"),
        ),
        ("open", [100.0, 100.5]),
        ("high", [102.0, 102.5]),
        ("low", [99.0, 99.5]),
        ("close", [101.0, 101.5]),
        ("volume", [1_000.0, 2_001.0]),
    ],
)
def test_dataset_fingerprint_changes_with_any_bar_column(
    field: str, values: object
):
    assert dataset().fingerprint != dataset(
        series=bar_series(**{field: values})
    ).fingerprint


@pytest.mark.parametrize(
    "instrument",
    [
        replace(AAPL, quote_currency="KRW"),
        replace(AAPL, tick_size=Decimal("0.001")),
        replace(AAPL, lot_size=Decimal("10")),
        replace(AAPL, timezone=ZoneInfo("UTC")),
    ],
)
def test_dataset_fingerprint_changes_with_instrument_metadata(
    instrument: Instrument,
):
    assert dataset().fingerprint != dataset(instrument=instrument).fingerprint


def test_dataset_fingerprint_changes_with_typed_timeframe():
    assert dataset().fingerprint != dataset(
        timeframe=Timeframe.days(2)
    ).fingerprint


def test_bar_view_and_market_slice_are_immutable_current_timestamp_views():
    timestamp = np.datetime64("2024-01-02T00:00:00", "ns")
    view = BarView(
        timestamp=timestamp,
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1_000.0,
    )
    source_bars = {AAPL.id: view}
    market_slice = MarketSlice(timestamp=timestamp, bars=source_bars)
    source_bars.clear()

    assert tuple(market_slice.bars) == (AAPL.id,)
    assert market_slice.bars[AAPL.id].close == 101.0
    with pytest.raises(TypeError):
        market_slice.bars[AAPL.id] = view  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        view.close = 999.0  # type: ignore[misc]


def test_market_slice_rejects_stale_bar_as_tradable():
    timestamp = np.datetime64("2024-01-03", "ns")
    stale_view = BarView(
        timestamp=np.datetime64("2024-01-02", "ns"),
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=1_000.0,
    )

    with pytest.raises(DataValidationError, match="current timestamp"):
        MarketSlice(timestamp=timestamp, bars={AAPL.id: stale_view})


def test_union_calendar_keeps_dates_seen_by_only_one_instrument():
    timestamps = {
        AAPL.id: np.array(
            ["2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
        ),
        MSFT.id: np.array(
            ["2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
        ),
    }
    calendar = CalendarPolicy.union(max_staleness_bars=1).build(timestamps)

    assert calendar.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert not calendar.flags.writeable


def test_calendar_array_cannot_be_made_writeable_again():
    calendar = CalendarPolicy.union().build(
        {
            AAPL.id: np.array(
                ["2024-01-02", "2024-01-03"],
                dtype="datetime64[ns]",
            )
        }
    )

    with pytest.raises(ValueError):
        calendar.setflags(write=True)


def test_intersection_calendar_keeps_only_shared_dates():
    timestamps = {
        AAPL.id: np.array(
            ["2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
        ),
        MSFT.id: np.array(
            ["2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
        ),
    }

    calendar = CalendarPolicy.intersection().build(timestamps)

    assert calendar.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-03"
    ]
    assert CalendarPolicy.intersection().max_staleness_bars == 0


def test_calendar_policy_keeps_staleness_separate_from_trading_dates():
    timestamps = {
        AAPL.id: np.array(["2024-01-02"], dtype="datetime64[ns]"),
        MSFT.id: np.array(["2024-01-03"], dtype="datetime64[ns]"),
    }

    fresh_only = CalendarPolicy.union(max_staleness_bars=0)
    allow_marking = CalendarPolicy.union(max_staleness_bars=2)

    np.testing.assert_array_equal(
        fresh_only.build(timestamps),
        allow_marking.build(timestamps),
    )
    assert fresh_only.max_staleness_bars == 0
    assert allow_marking.max_staleness_bars == 2


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_union_calendar_rejects_invalid_staleness(value: object):
    with pytest.raises(ConfigurationError, match="staleness"):
        CalendarPolicy.union(max_staleness_bars=value)  # type: ignore[arg-type]


def test_calendar_policy_requires_typed_mode():
    with pytest.raises(ConfigurationError, match="mode"):
        CalendarPolicy(mode="union", max_staleness_bars=0)  # type: ignore[arg-type]

    assert CalendarPolicy.union().mode is CalendarMode.UNION


def test_calendar_rejects_empty_instrument_mapping():
    with pytest.raises(DataValidationError, match="at least one"):
        CalendarPolicy.union().build({})


def test_dataset_union_timestamps_are_sorted_and_immutable():
    aapl = factory_instrument("AAPL")
    msft = factory_instrument("MSFT")
    market_data = factory_market_dataset(
        {
            aapl: factory_bar_series(
                closes=[10, 12],
                timestamps=[factory_timestamp(0), factory_timestamp(2)],
            ),
            msft: factory_bar_series(
                closes=[20, 21],
                timestamps=[factory_timestamp(1), factory_timestamp(2)],
            ),
        }
    )

    np.testing.assert_array_equal(
        market_data.timestamps,
        [
            factory_timestamp(0),
            factory_timestamp(1),
            factory_timestamp(2),
        ],
    )
    assert not market_data.timestamps.flags.writeable
    with pytest.raises(ValueError):
        market_data.timestamps.setflags(write=True)


def test_dataset_prefix_slices_each_instrument_at_union_boundary():
    aapl = factory_instrument("AAPL")
    msft = factory_instrument("MSFT")
    market_data = factory_market_dataset(
        {
            aapl: factory_bar_series(
                closes=[10, 12],
                timestamps=[factory_timestamp(0), factory_timestamp(2)],
            ),
            msft: factory_bar_series(
                closes=[20, 21],
                timestamps=[factory_timestamp(1), factory_timestamp(2)],
            ),
        }
    )

    prefix = market_data.prefix(2)

    np.testing.assert_array_equal(
        prefix.timestamps,
        [factory_timestamp(0), factory_timestamp(1)],
    )
    np.testing.assert_array_equal(
        prefix.series[aapl.id].timestamps,
        [factory_timestamp(0)],
    )
    np.testing.assert_array_equal(
        prefix.series[msft.id].timestamps,
        [factory_timestamp(1)],
    )


@pytest.mark.parametrize("count", [0, -1, True, 4])
def test_dataset_prefix_rejects_invalid_counts(count: object):
    market_data, _ = one_instrument_dataset(closes=[1, 2, 3])

    with pytest.raises(ConfigurationError, match="prefix"):
        market_data.prefix(count)  # type: ignore[arg-type]
