from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from pybacktest.domain.errors import ConfigurationError
from pybacktest.domain.identifiers import FillId, OrderId, RunId
from pybacktest.domain.instruments import Instrument, InstrumentId
from pybacktest.domain.money import Money, Quantity
from pybacktest.domain.time import DateRange, Timeframe, TimeframeUnit


def test_instrument_id_round_trips_canonical_form():
    instrument_id = InstrumentId.parse("XNAS:AAPL")
    assert instrument_id.venue == "XNAS"
    assert instrument_id.symbol == "AAPL"
    assert str(instrument_id) == "XNAS:AAPL"


def test_instrument_id_direct_construction_normalizes_canonical_parts():
    direct = InstrumentId("xnas", "aapl")
    parsed = InstrumentId.parse("xnas:aapl")

    assert direct == parsed
    assert str(direct) == "XNAS:AAPL"


@pytest.mark.parametrize("value", ["XNAS", "XNAS:AAPL:OPTION", ":AAPL", "XNAS:"])
def test_instrument_id_rejects_noncanonical_forms(value: str):
    with pytest.raises(ConfigurationError):
        InstrumentId.parse(value)


def test_instrument_rejects_non_positive_tick_and_lot():
    with pytest.raises(ConfigurationError, match="tick_size"):
        Instrument(
            id=InstrumentId.parse("XNAS:AAPL"),
            quote_currency="USD",
            tick_size=Decimal("0"),
            lot_size=Decimal("1"),
            timezone=ZoneInfo("America/New_York"),
        )

    with pytest.raises(ConfigurationError, match="lot_size"):
        Instrument(
            id=InstrumentId.parse("XNAS:AAPL"),
            quote_currency="USD",
            tick_size=Decimal("0.01"),
            lot_size=Decimal("0"),
            timezone=ZoneInfo("America/New_York"),
        )


def test_instrument_normalizes_quote_currency_and_is_immutable():
    instrument = Instrument(
        id=InstrumentId.parse("xnas:aapl"),
        quote_currency="usd",
        tick_size=Decimal("0.01"),
        lot_size=Decimal("1"),
        timezone=ZoneInfo("America/New_York"),
    )

    assert instrument.quote_currency == "USD"
    with pytest.raises(FrozenInstanceError):
        instrument.lot_size = Decimal("2")  # type: ignore[misc]


def test_money_requires_matching_currency_for_addition():
    with pytest.raises(ConfigurationError, match="currency"):
        Money.usd("1") + Money.of("1", "KRW")


def test_money_normalizes_currency_and_preserves_decimal_input_precision():
    money = Money.of(0.1, "usd")
    assert money.amount == Decimal("0.1")
    assert money.currency == "USD"


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-Infinity"])
def test_money_rejects_nonfinite_amounts(amount: str):
    with pytest.raises(ConfigurationError, match="finite"):
        Money.of(amount, "USD")


def test_quantity_quantizes_down_to_lot_size():
    quantity = Quantity.of("1.234").quantized(Decimal("0.01"))
    assert quantity.value == Decimal("1.23")


def test_quantity_quantizes_to_a_lot_multiple():
    quantity = Quantity.of("1.234").quantized(Decimal("0.05"))
    assert quantity.value == Decimal("1.20")


def test_quantity_quantizes_negative_values_toward_zero():
    quantity = Quantity.of("-1.234").quantized(Decimal("0.01"))
    assert quantity.value == Decimal("-1.23")


@pytest.mark.parametrize("value", ["NaN", "Infinity"])
def test_quantity_rejects_nonfinite_values(value: str):
    with pytest.raises(ConfigurationError):
        Quantity.of(value)


@pytest.mark.parametrize("lot_size", ["0", "-0.01", "NaN", "Infinity"])
def test_quantity_rejects_non_positive_or_nonfinite_lot_sizes(lot_size: str):
    with pytest.raises(ConfigurationError, match="lot_size"):
        Quantity.of("1").quantized(Decimal(lot_size))


def test_date_range_is_start_inclusive_end_exclusive():
    period = DateRange(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
    )
    assert period.contains(datetime(2024, 1, 1, tzinfo=UTC))
    assert not period.contains(datetime(2024, 2, 1, tzinfo=UTC))


def test_date_range_rejects_naive_or_empty_ranges():
    aware_start = datetime(2024, 1, 1, tzinfo=UTC)
    aware_end = datetime(2024, 2, 1, tzinfo=UTC)

    with pytest.raises(ConfigurationError, match="timezone-aware"):
        DateRange(datetime(2024, 1, 1), aware_end)
    with pytest.raises(ConfigurationError, match="start"):
        DateRange(aware_end, aware_start)


def test_date_range_rejects_naive_query_timestamp():
    period = DateRange(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
    )

    with pytest.raises(ConfigurationError, match="timezone-aware"):
        period.contains(datetime(2024, 1, 15))


def test_timeframe_rejects_zero_count():
    with pytest.raises(ConfigurationError, match="count"):
        Timeframe.minutes(0)


def test_timeframe_accepts_only_supported_positive_units():
    assert Timeframe.minutes(5) == Timeframe(TimeframeUnit.MINUTE, 5)
    assert Timeframe.days(1) == Timeframe(TimeframeUnit.DAY, 1)

    with pytest.raises(ConfigurationError, match="unit"):
        Timeframe("hour", 1)
    with pytest.raises(ConfigurationError, match="unit"):
        Timeframe("minute", 1)
    with pytest.raises(ConfigurationError, match="count"):
        Timeframe.days(-1)


@pytest.mark.parametrize(
    ("identifier_type", "value"),
    [
        (RunId, "run_" + "a" * 32),
        (OrderId, "order_" + "b" * 32),
        (FillId, "fill_" + "c" * 32),
    ],
)
def test_identifiers_parse_canonical_values(identifier_type, value: str):
    assert str(identifier_type.parse(value)) == value


@pytest.mark.parametrize(
    ("identifier_type", "value"),
    [
        (RunId, "run_" + "A" * 32),
        (OrderId, "fill_" + "a" * 32),
        (FillId, "fill_" + "a" * 31),
        (RunId, "run_../" + "a" * 28),
    ],
)
def test_identifiers_reject_malformed_or_cross_type_values(identifier_type, value: str):
    with pytest.raises(ConfigurationError):
        identifier_type.parse(value)


@pytest.mark.parametrize(
    ("identifier_type", "prefix"),
    [(RunId, "run_"), (OrderId, "order_"), (FillId, "fill_")],
)
def test_new_identifiers_use_type_specific_canonical_prefix(
    identifier_type, prefix: str
):
    value = str(identifier_type.new())
    assert value.startswith(prefix)
    assert len(value) == len(prefix) + 32
