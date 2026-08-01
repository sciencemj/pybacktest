"""Load and validate the one immutable dataset bound to a simulation."""

from __future__ import annotations

from pybacktest.application.requests import SimulationRequest
from pybacktest.data.dataset import MarketDataSet
from pybacktest.domain.errors import AdapterContractError, DataValidationError
from pybacktest.engine._time import _as_np_datetime
from pybacktest.ports.data import MarketDataSource


def _load_dataset(
    data_source: MarketDataSource,
    simulation: SimulationRequest,
) -> MarketDataSet:
    """Read the adapter once and enforce its complete request contract."""
    try:
        load = data_source.load
    except AttributeError as exc:
        raise AdapterContractError(
            "data source must expose load().",
            code="invalid_data_source",
        ) from exc
    dataset = load(
        simulation.universe,
        simulation.period,
        simulation.timeframe,
    )
    if not isinstance(dataset, MarketDataSet):
        raise AdapterContractError(
            "data source must return MarketDataSet.",
            code="invalid_dataset_type",
        )
    if dataset.timeframe != simulation.timeframe:
        raise DataValidationError("dataset timeframe does not match the request.")
    if set(dataset.instruments) != set(simulation.universe):
        raise DataValidationError("dataset universe does not match the request.")
    currencies = {
        instrument.quote_currency for instrument in dataset.instruments.values()
    }
    if currencies != {simulation.initial_cash.currency}:
        raise DataValidationError(
            "dataset instruments must share the initial-cash currency."
        )
    start = _as_np_datetime(simulation.period.start)
    end = _as_np_datetime(simulation.period.end)
    for series in dataset.series.values():
        if (
            len(series.timestamps) == 0
            or series.timestamps[0] < start
            or series.timestamps[-1] >= end
        ):
            raise DataValidationError(
                "dataset timestamps must fall inside the requested period."
            )
    return dataset
